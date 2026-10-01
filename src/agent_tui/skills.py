"""Discover SKILL.md metadata and activate instructions progressively."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Settings
from .relevance import is_follow_up, keywords
from .tools import ToolError


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    instructions: str


class SkillManager:
    def __init__(
        self, settings: Settings, plugin_roots: list[tuple[str, Path]] | None = None
    ) -> None:
        self.skills: dict[str, Skill] = {}
        self.active: dict[str, Skill] = {}
        self.pinned: set[str] = set()
        self.task_goal = ""
        self.warnings: list[str] = []
        roots = [
            ("", Path(__file__).with_name("builtin_skills")),
            *(plugin_roots or []),
            ("", settings.workspace / ".agents" / "skills"),
            ("", settings.workspace / ".agent-tui" / "skills"),
            *(("", root) for root in settings.skills_dirs),
        ]
        for namespace, root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*/SKILL.md")):
                try:
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError("skill symlink escapes its collection")
                    if path.stat().st_size > 128000:
                        raise ValueError("SKILL.md exceeds 128 KB")
                    text = path.read_text(encoding="utf-8")
                    match = re.match(r"\A---\r?\n(.*?)\r?\n---\s*\r?\n(.*)\Z", text, re.S)
                    if not match:
                        raise ValueError("expected YAML frontmatter and Markdown instructions")
                    meta = yaml.safe_load(match[1])
                    if not isinstance(meta, dict):
                        raise ValueError("frontmatter must be a mapping")
                    name, description = meta.get("name"), meta.get("description")
                    if (
                        not isinstance(name, str)
                        or len(name) > 64
                        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                        or name != path.parent.name
                    ):
                        raise ValueError("invalid name or name does not match directory")
                    if (
                        not isinstance(description, str)
                        or not description.strip()
                        or len(description) > 1024
                        or not match[2].strip()
                    ):
                        raise ValueError("non-empty description and instructions required")
                    key = f"{namespace}:{name}" if namespace else name
                    self.skills[key] = Skill(key, description, path.resolve(), match[2].strip())
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
                    self.warnings.append(f"Skipped skill {path}: {exc}")

    def activate(self, name: str, *, persistent: bool = True) -> str:
        if name not in self.skills:
            raise ToolError(f"Unknown skill: {name}")
        if name not in self.active and len(self.active) >= 4:
            raise ToolError("At most four skills may be active; use /skill off NAME")
        self.active[name] = self.skills[name]
        if persistent:
            self.pinned.add(name)
        scope = "until unloaded" if name in self.pinned else "for this task and related follow-ups"
        return f"Activated {name} {scope}. Instructions will be included in model requests."

    def deactivate(self, name: str) -> None:
        if self.active.pop(name, None) is None:
            raise ToolError(f"Skill is not active: {name}")
        self.pinned.discard(name)

    def clear(self) -> None:
        self.active.clear()
        self.pinned.clear()
        self.task_goal = ""

    def begin_task(self, goal: str) -> None:
        query = keywords(goal)
        related = is_follow_up(goal) or bool(query & keywords(self.task_goal))
        for name, skill in list(self.active.items()):
            if (
                name not in self.pinned
                and not related
                and not query & keywords(f"{skill.name} {skill.description}")
            ):
                self.deactivate(name)
        self.select_mentions(goal)
        if not is_follow_up(goal) or not self.task_goal:
            self.task_goal = goal

    def select_mentions(self, goal: str) -> None:
        for name in re.findall(r"(?<!\w)\$([a-z0-9][a-z0-9:-]*)(?![\w:-])", goal):
            self.activate(name, persistent=False)

    def read_resource(self, name: str, path: str) -> str:
        if name not in self.active:
            raise ToolError("Activate the skill before reading its resources")
        root = self.active[name].path.parent
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ToolError("Resource path must be a file inside the active skill directory")
        if target.stat().st_size > 64000:
            raise ToolError("Skill resource exceeds 64 KB")
        text = target.read_text(encoding="utf-8")
        if "\0" in text:
            raise ToolError("Skill resources must be text")
        return text

    def catalog(self, query: str = "") -> str:
        terms = keywords(query)

        def rank(skill: Skill) -> tuple[int, str]:
            score = 2 * len(terms & keywords(skill.name)) + len(terms & keywords(skill.description))
            return -score, skill.name

        selected = sorted(self.skills.values(), key=rank)[:6]
        entries = [
            f"{skill.name}: {' '.join(skill.description.split())[:240]}" for skill in selected
        ]
        return (
            "Available skills (use skill_load to activate for this task):\n"
            + "\n".join(entries)
            + f"\nShowing {len(selected)} of {len(self.skills)} skills. "
            "Use skill_search with keywords to find other workflows."
        )

    def instructions(self) -> str:
        return "\n\n".join(
            f"Active skill: {s.name}\nBase directory: {s.path.parent}\n{s.instructions}"
            for s in self.active.values()
        )
