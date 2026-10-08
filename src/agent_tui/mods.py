"""Explicit, per-agent component factories loaded from Python or a TOML manifest."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import inspect
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings

MOD_API_VERSION = 1

# These are component boundaries, not inheritance requirements. Any object implementing
# the listed methods can fill a slot. Stateful instances belong to one runtime only.
CONTRACTS = {
    "agent": ("run", "aclose"),
    "app": ("run",),
    "router": ("route",),
    "executor": ("complete",),
    "planner": ("plan",),
    "tools": (
        "register",
        "unregister",
        "validate",
        "execute",
        "schemas",
        "descriptors",
        "requires_approval",
        "preview",
        "sanitize",
    ),
    "context": ("build", "clear", "remember_turn", "compact"),
    "memory": ("context", "record_run", "search", "put", "forget"),
    "plugins": ("skill_roots", "bootstrap_skills", "installed", "catalog"),
    "skills": (
        "activate",
        "deactivate",
        "begin_task",
        "clear",
        "instructions",
        "catalog",
        "read_resource",
    ),
    "extensions": (),
    "middleware": ("authorize", "dispatch"),
    "mcp": ("connect", "aclose"),
    "subagents": ("begin_run", "end_run", "execute"),
    "prompts": ("build",),
    "code_runtime": ("execute", "prompt", "adapt"),
    "tool_policy": ("execute",),
    "tool_bindings": ("register", "refresh"),
    "responses": ("as_dict", "message"),
    "loop": ("run",),
}
OPTIONAL_MODS = {"planner", "subagents"}
Factory = Callable[["ModContext"], Any]


def _disabled(context: ModContext) -> None:
    return None


class ModError(ValueError):
    """Invalid configuration or a component that does not meet its contract."""


@dataclass(frozen=True)
class ModSpec:
    factory: str | Factory = "default"
    options: Mapping[str, Any] = field(default_factory=dict)


class HarnessMods:
    """Reusable factory definitions; create a new runtime for every agent."""

    def __init__(
        self,
        specs: Mapping[str, ModSpec | str | Factory] | None = None,
        *,
        base: Path | None = None,
    ) -> None:
        self.base = (base or Path.cwd()).resolve()
        self.specs: dict[str, ModSpec] = {}
        self._factories: dict[str, Factory] = {}
        for name, spec in (specs or {}).items():
            if name not in CONTRACTS:
                raise ModError(f"Unknown mod slot {name!r}; choose from {', '.join(CONTRACTS)}")
            if not isinstance(spec, ModSpec):
                spec = ModSpec(spec)
            if not isinstance(spec.options, Mapping):
                raise ModError(f"Options for mod {name!r} must be a mapping")
            target = spec.factory
            if isinstance(target, str):
                if target == "none":
                    if name not in OPTIONAL_MODS:
                        raise ModError(f"Mod {name!r} cannot be disabled; provide a replacement")
                elif target != "default":
                    module, separator, attribute = target.rpartition(":")
                    if (
                        not separator
                        or not module
                        or not all(part.isidentifier() for part in attribute.split("."))
                    ):
                        raise ModError(f"Mod {name!r} needs module:factory or path.py:factory")
            elif not callable(target):
                raise ModError(f"Factory for mod {name!r} must be a callable or import target")
            self.specs[name] = ModSpec(target, copy.deepcopy(dict(spec.options)))

    @classmethod
    def load(cls, settings: Settings) -> HarnessMods:
        specs: dict[str, ModSpec] = {}
        base = settings.workspace
        if settings.mods_path is not None:
            path = settings.mods_path
            base = path.parent
            try:
                if path.stat().st_size > 128_000:
                    raise ValueError("manifest exceeds 128 KB")
                data = tomllib.loads(path.read_text(encoding="utf-8"))
                if set(data) - {"version", "mods", "options"}:
                    raise ValueError("allowed manifest fields are version, mods and options")
                if type(data.get("version")) is not int or data["version"] != MOD_API_VERSION:
                    raise ValueError(f"version must be {MOD_API_VERSION}")
                targets, options = data.get("mods", {}), data.get("options", {})
                if not isinstance(targets, dict) or not isinstance(options, dict):
                    raise ValueError("mods and options must be tables")
                for name in targets.keys() | options.keys():
                    target = targets.get(name, "default")
                    if not isinstance(target, str):
                        raise ValueError(f"factory target for {name!r} must be a string")
                    specs[name] = ModSpec(target, options.get(name, {}))
            except (OSError, UnicodeError, ValueError) as exc:
                raise ModError(f"Invalid mods manifest {path}: {exc}") from None
        for override in settings.mod_overrides:
            name, separator, target = override.partition("=")
            if not separator or not name or not target:
                raise ModError("--mod must be SLOT=module:factory (or SLOT=default/none)")
            specs[name] = ModSpec(target, specs.get(name, ModSpec()).options)
        return cls(specs, base=base)

    def describe(self) -> dict[str, str]:
        """Inspect configuration without importing or executing any custom code."""
        descriptions = {}
        for name in CONTRACTS:
            target = self.specs.get(name, ModSpec()).factory
            if isinstance(target, str):
                descriptions[name] = target
            else:
                owner = target if hasattr(target, "__qualname__") else type(target)
                descriptions[name] = f"{owner.__module__}:{owner.__qualname__}"
        return descriptions

    def factory(self, name: str) -> Factory:
        if name not in CONTRACTS:
            raise ModError(f"Unknown mod slot: {name}")
        if name in self._factories:
            return self._factories[name]
        target = self.specs.get(name, ModSpec()).factory
        if target == "default":
            from .defaults import FACTORIES

            factory = FACTORIES[name]
        elif target == "none":
            factory = _disabled
        elif callable(target):
            factory = target
        else:
            module_name, _, attribute = target.rpartition(":")
            try:
                if module_name.endswith(".py"):
                    path = (self.base / Path(module_name).expanduser()).resolve()
                    key = "_agent_tui_mod_" + hashlib.sha256(str(path).encode()).hexdigest()[:24]
                    module = sys.modules.get(key)
                    if module is None:
                        spec = importlib.util.spec_from_file_location(key, path)
                        if spec is None or spec.loader is None:
                            raise ValueError("cannot create a Python module loader")
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[key] = module
                        try:
                            spec.loader.exec_module(module)
                        except BaseException:
                            sys.modules.pop(key, None)
                            raise
                else:
                    module = importlib.import_module(module_name)
                factory = module
                for part in attribute.split("."):
                    factory = getattr(factory, part)
            except Exception as exc:
                raise ModError(f"Cannot load mod {name!r} from {target!r}: {exc}") from exc
        if not callable(factory):
            raise ModError(f"Factory for mod {name!r} is not callable")
        try:
            inspect.signature(factory).bind(None)
        except (TypeError, ValueError) as exc:
            raise ModError(f"Factory for mod {name!r} must accept one ModContext: {exc}") from exc
        self._factories[name] = factory
        return factory


@dataclass(frozen=True)
class ModContext:
    runtime: ModRuntime
    name: str
    options: dict[str, Any]

    @property
    def settings(self) -> Settings:
        return self.runtime.settings

    def get(self, name: str) -> Any:
        return self.runtime.get(name)

    def default(self) -> Any:
        """Build this slot's default using the same configured dependencies."""
        from .defaults import FACTORIES

        return FACTORIES[self.name](self)


class ModRuntime:
    def __init__(
        self,
        settings: Settings,
        mods: HarnessMods | None = None,
        *,
        instances: Mapping[str, Any] | None = None,
        allow_delegation: bool = True,
        initial_prompt: str | None = None,
    ) -> None:
        self.settings = settings
        self.mods = mods if mods is not None else HarnessMods.load(settings)
        self.allow_delegation = allow_delegation
        self.initial_prompt = initial_prompt
        self.child_factory: Callable[[], Any] | None = None
        self._instances = dict(instances or {})
        unknown = self._instances.keys() - CONTRACTS.keys()
        if unknown:
            raise ModError(f"Unknown mod instances: {', '.join(sorted(unknown))}")
        self._building: list[str] = []
        self._owned = list(self._instances.values())
        self._closed = False

    def get(self, name: str) -> Any:
        if self._closed:
            raise ModError("This mod runtime is closed")
        if name not in CONTRACTS:
            raise ModError(f"Unknown mod slot: {name}")
        if name == "subagents" and not self.allow_delegation:
            return None
        if name in self._instances:
            return self._instances[name]
        if name in self._building:
            raise ModError("Mod dependency cycle: " + " -> ".join([*self._building, name]))
        self._building.append(name)
        try:
            spec = self.mods.specs.get(name, ModSpec())
            context = ModContext(self, name, copy.deepcopy(dict(spec.options)))
            component = self.mods.factory(name)(context)
            if inspect.isawaitable(component):
                if inspect.iscoroutine(component):
                    component.close()
                raise ModError(
                    f"Mod {name!r} factory must be synchronous; use async methods for I/O"
                )
            self._owned.append(component)
            if component is None and name in OPTIONAL_MODS:
                self._instances[name] = None
                return None
            missing = [
                method
                for method in CONTRACTS[name]
                if not callable(getattr(component, method, None))
            ]
            if component is None or missing:
                detail = ", ".join(missing) or "component is None"
                raise ModError(f"Mod {name!r} is missing required methods: {detail}")
            self._instances[name] = component
            return component
        except ModError as exc:
            raise ModError(self.settings.redact(str(exc))) from None
        except Exception as exc:
            raise ModError(
                f"Cannot construct mod {name!r}: {self.settings.redact(str(exc))}"
            ) from exc
        finally:
            self._building.pop()

    def fork(self) -> ModRuntime:
        # Share factory definitions, never mutable component instances or conversation state.
        if self._closed:
            raise ModError("This mod runtime is closed")
        child = ModRuntime(self.settings, self.mods, allow_delegation=False)
        # Also retain failed/partially constructed children for eventual cleanup.
        self._owned.append(child)
        return child

    def rebuild(self, name: str) -> Any:
        """Recreate a catalog component when the idle agent refreshes installed skills."""
        previous = self._instances.pop(name, None)
        try:
            return self.get(name)
        except BaseException:
            if previous is not None:
                self._instances[name] = previous
            raise

    async def __aenter__(self) -> ModRuntime:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        seen: set[int] = set()
        errors: list[BaseException] = []
        for component in reversed(self._owned):
            if component is None or id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "aclose", None)
            if close is not None:
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except BaseException as exc:
                    errors.append(exc)
        self._instances.clear()
        self._owned.clear()
        self.child_factory = None
        if errors:
            raise BaseExceptionGroup("Harness mod cleanup failed", errors)


def create_agent(settings: Settings, mods: HarnessMods | None = None, **kwargs: Any) -> Any:
    return ModRuntime(settings, mods, **kwargs).get("agent")


def create_app(
    settings: Settings, initial_prompt: str | None = None, mods: HarnessMods | None = None
) -> Any:
    return ModRuntime(settings, mods, initial_prompt=initial_prompt).get("app")
