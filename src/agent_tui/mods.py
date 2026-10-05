"""Composable in-process mods for changing PyLattice behavior.

A mod is a trusted Python module exporting:

    async def mod(event, payload, next):
        ...

Calling next(payload) delegates to the next mod / built-in behavior.
Skipping it replaces the built-in behavior. Code before and after next wraps it.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import Settings
from .tools import ToolError

MOD_EVENTS = ("model", "tool", "permission", "render")
Payload = dict[str, Any]
Next = Callable[[Payload], Awaitable[Any]]
ModHandler = Callable[[str, Payload, Next], Awaitable[Any]]


class ModError(ToolError):
    """A mod failed or returned an invalid value."""


@dataclass(frozen=True)
class LoadedMod:
    name: str
    path: Path
    handler: ModHandler
    events: frozenset[str]

    def handles(self, event: str) -> bool:
        return "*" in self.events or event in self.events


class ModManager:
    """Load trusted Python mods and compose them around built-in operations."""

    def __init__(
        self,
        settings: Settings,
        plugin_roots: Sequence[tuple[str, Path]] = (),
    ) -> None:
        self.settings = settings
        self.workspace_root = settings.workspace / ".agent-tui" / "mods"
        self.plugin_roots = tuple(plugin_roots)
        self.mods: list[LoadedMod] = []
        self.reload()

    def reload(self) -> None:
        loaded: list[LoadedMod] = []
        if self.workspace_root.is_dir():
            loaded.extend(self._load_root("workspace", self.workspace_root))
        for plugin, root in self.plugin_roots:
            if root.is_dir():
                loaded.extend(self._load_root(plugin, root))
        self.mods = loaded

    def names(self) -> list[str]:
        return [mod.name for mod in self.mods]

    async def invoke(self, event: str, payload: Payload, terminal: Next) -> Any:
        if event not in MOD_EVENTS:
            raise ModError(f"Unknown mod event: {event}")

        chain = [mod for mod in self.mods if mod.handles(event)]

        async def call(index: int, current: Payload) -> Any:
            if not isinstance(current, dict):
                raise ModError(f"Mod {event} payload must be a dict")
            if index == len(chain):
                return await terminal(current)

            loaded = chain[index]

            async def next_handler(updated: Payload | None = None) -> Any:
                return await call(index + 1, current if updated is None else updated)

            try:
                result = loaded.handler(event, deepcopy(current), next_handler)
                if not inspect.isawaitable(result):
                    raise ModError(f"Mod {loaded.name} must be async")
                return await result
            except ModError:
                raise
            except Exception as exc:
                raise ModError(f"Mod {loaded.name} failed during {event}: {exc}") from exc

        return await call(0, deepcopy(payload))

    def _load_root(self, namespace: str, root: Path) -> list[LoadedMod]:
        if root.is_symlink():
            raise ModError(f"Mod directory must not be a symlink: {root}")
        resolved = root.resolve()
        result: list[LoadedMod] = []
        for path in sorted(root.glob("*.py")):
            if path.name.startswith("_"):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(resolved):
                raise ModError(f"Invalid mod path: {path}")
            result.append(self._load_file(namespace, path))
        return result

    def _load_file(self, namespace: str, path: Path) -> LoadedMod:
        key = f"agent_tui_mod_{namespace}_{path.stem}_{abs(hash(path.resolve()))}"
        spec = importlib.util.spec_from_file_location(key, path)
        if spec is None or spec.loader is None:
            raise ModError(f"Could not load mod: {path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise ModError(f"Could not import mod {path.name}: {exc}") from exc
        return self._validate_module(namespace, path, module)

    @staticmethod
    def _validate_module(namespace: str, path: Path, module: ModuleType) -> LoadedMod:
        handler = getattr(module, "mod", None)
        if not callable(handler):
            raise ModError(f"Mod {path.name} must export async def mod(event, payload, next)")
        events = getattr(module, "EVENTS", ("*",))
        if not isinstance(events, (tuple, list, set, frozenset)) or not all(
            isinstance(event, str) for event in events
        ):
            raise ModError(f"Mod {path.name} EVENTS must be a sequence of strings")
        unknown = set(events) - set(MOD_EVENTS) - {"*"}
        if unknown:
            raise ModError(f"Mod {path.name} has unknown events: {sorted(unknown)}")
        return LoadedMod(
            name=f"{namespace}:{path.stem}",
            path=path,
            handler=handler,
            events=frozenset(events),
        )
