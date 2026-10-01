"""A small Git-based skill marketplace with workspace-local, explicit installation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings
from .extensions import safe_environment
from .tools import ToolError


@dataclass(frozen=True)
class PluginSpec:
    name: str
    repository: str
    description: str = "Skill collection"
    skills_path: str = "skills"
    bootstrap: str = ""


MARKETPLACE = {
    "superpowers": PluginSpec(
        "superpowers",
        "https://github.com/obra/superpowers.git",
        "Development workflows: brainstorming, planning, debugging, TDD, and review.",
        bootstrap="using-superpowers",
    ),
}


class PluginManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.workspace / ".agent-tui"
        self.path = self.root / "plugins.json"
        self.packages = self.root / "plugins"

    @staticmethod
    def _name(name: str) -> str:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
            raise ToolError("Invalid plugin name")
        return name

    def _check(self) -> None:
        if any(p.is_symlink() for p in (self.root, self.path, self.packages)):
            raise ToolError("Plugin storage must not use symlinks")

    def _writable(self) -> None:
        self._check()
        if self.settings.read_only or self.settings.demo:
            raise ToolError("Plugin changes are disabled in read-only and demo modes")

    def catalog(self) -> dict[str, PluginSpec]:
        self._check()
        catalog = dict(MARKETPLACE)
        path = self.root / "marketplace.json"
        if path.exists():
            if path.is_symlink() or path.stat().st_size > 128000:
                raise ToolError("Invalid marketplace file")
            try:
                entries = json.loads(path.read_text())
                if not isinstance(entries, list):
                    raise ValueError("expected a list")
                for entry in entries:
                    spec = PluginSpec(**entry)
                    self._validate_spec(spec)
                    catalog[spec.name] = spec
            except (TypeError, ValueError) as exc:
                raise ToolError(f"Invalid marketplace.json: {exc}") from None
        return catalog

    def _validate_spec(self, spec: PluginSpec) -> None:
        if not all(isinstance(value, str) for value in asdict(spec).values()):
            raise ToolError("Plugin metadata fields must be strings")
        self._name(spec.name)
        if spec.bootstrap:
            self._name(spec.bootstrap)
        if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+(?:\.git)?", spec.repository):
            raise ToolError("Plugin repository must be a public GitHub HTTPS URL")
        path = Path(spec.skills_path)
        if path.is_absolute() or ".." in path.parts:
            raise ToolError("Plugin skills_path must stay inside its repository")

    def installed(self) -> dict[str, dict]:
        self._check()
        if not self.path.exists():
            return {}
        try:
            if self.path.stat().st_size > 128000:
                raise ValueError("manifest too large")
            entries = json.loads(self.path.read_text())
            if not isinstance(entries, dict):
                raise ValueError("expected a mapping")
            for name, entry in entries.items():
                self._name(name)
                if not isinstance(entry, dict) or type(entry.get("enabled")) is not bool:
                    raise ValueError("invalid installed plugin record")
                spec = PluginSpec(**entry["spec"])
                self._validate_spec(spec)
                if spec.name != name:
                    raise ValueError("plugin name mismatch")
                if not isinstance(entry.get("commit"), str) or not re.fullmatch(
                    r"[a-f0-9]{40,64}", entry["commit"]
                ):
                    raise ValueError("invalid plugin commit")
            return entries
        except (TypeError, ValueError, KeyError) as exc:
            raise ToolError(f"Invalid plugins.json: {exc}") from None

    def _save(self, entries: dict[str, dict]) -> None:
        self._writable()
        self.root.mkdir(mode=0o700, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.root, delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(entries, file, indent=2)
        try:
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def skill_roots(self) -> list[tuple[str, Path]]:
        result = []
        for name, entry in self.installed().items():
            if entry["enabled"]:
                package = self.packages / name
                root = package / entry["spec"]["skills_path"]
                if package.is_symlink() or not root.resolve().is_relative_to(package.resolve()):
                    raise ToolError(f"Invalid plugin skill directory: {name}")
                if not root.is_dir():
                    raise ToolError(f"Plugin {name} is missing its skills; reinstall it")
                result.append((name, root))
        return result

    def bootstrap_skills(self) -> list[str]:
        return [
            f"{name}:{entry['spec']['bootstrap']}"
            for name, entry in self.installed().items()
            if entry["enabled"] and entry["spec"].get("bootstrap")
        ]

    async def _git(self, *args: str) -> str:
        # No shell, credentials, repository hooks, submodules, or package install scripts.
        process = await asyncio.create_subprocess_exec(
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "credential.helper=",
            "-c",
            "protocol.file.allow=never",
            *args,
            env={
                **safe_environment(),
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            async with asyncio.timeout(90):
                stdout, stderr = await process.communicate()
            if process.returncode:
                raise ToolError(
                    "Plugin download failed: " + stderr.decode(errors="replace")[-1000:]
                )
            return stdout.decode().strip()
        except TimeoutError:
            raise ToolError("Plugin download timed out") from None
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                await process.wait()

    async def install(self, name: str) -> str:
        self._writable()
        self._name(name)
        spec = self.catalog().get(name)
        if spec is None:
            raise ToolError(f"Plugin not found in marketplace: {name}")
        entries = self.installed()
        if name in entries:
            raise ToolError(
                f"{name} is already installed; enable it or uninstall before reinstalling"
            )
        self.packages.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination = self.packages / name
        if destination.exists() or destination.is_symlink():
            raise ToolError(f"Plugin destination already exists: {destination}")
        with tempfile.TemporaryDirectory(prefix=".install-", dir=self.packages) as temporary:
            checkout = Path(temporary) / "checkout"
            await self._git(
                "clone", "--quiet", "--depth", "1", "--", spec.repository, str(checkout)
            )
            root = checkout / spec.skills_path
            if (
                not root.is_dir()
                or not root.resolve().is_relative_to(checkout.resolve())
                or not any(root.glob("*/SKILL.md"))
            ):
                raise ToolError("Plugin contains no usable skills directory")
            from .skills import SkillManager

            catalog = SkillManager(self.settings, [(name, root)])
            if not any(key.startswith(name + ":") for key in catalog.skills):
                raise ToolError("Plugin contains no valid SKILL.md files")
            if spec.bootstrap and f"{name}:{spec.bootstrap}" not in catalog.skills:
                raise ToolError("Plugin bootstrap skill is missing or invalid")
            commit = await self._git("-C", str(checkout), "rev-parse", "HEAD")
            checkout.rename(destination)
            entries[name] = {"spec": asdict(spec), "commit": commit, "enabled": True}
            try:
                self._save(entries)
            except BaseException:
                destination.rename(checkout)
                raise
        return commit

    def enable(self, name: str, enabled: bool) -> None:
        self._writable()
        entries = self.installed()
        if name not in entries:
            raise ToolError(f"Plugin is not installed: {name}")
        entries[name]["enabled"] = enabled
        self._save(entries)

    def uninstall(self, name: str) -> None:
        self._writable()
        self._name(name)
        entries = self.installed()
        if name not in entries:
            raise ToolError(f"Plugin is not installed: {name}")
        package = self.packages / name
        if package.is_symlink():
            raise ToolError("Plugin directory must not be a symlink")
        # Disable first so a failed removal cannot leave active, partially deleted skills.
        entries[name]["enabled"] = False
        self._save(entries)
        if package.exists():
            shutil.rmtree(package)
        del entries[name]
        self._save(entries)
