"""Deterministic zero-LLM fast paths for obvious local actions."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FastAction:
    tool: str
    arguments: dict[str, Any]
    reason: str


_PATH = re.compile(r"(?P<path>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+)")


class FastPathResolver:
    """Resolve obvious requests without a router or executor model call."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def resolve(self, goal: str, available: set[str]) -> FastAction | None:
        text = goal.strip()
        lowered = text.casefold()

        path = self._existing_path(text)
        if path and "read_file" in available and self._looks_like_read(lowered):
            return FastAction("read_file", {"path": path}, "explicit file read")

        if "list_files" in available and self._looks_like_list(lowered):
            return FastAction("list_files", {}, "explicit file listing")

        if "run_command" in available and self._looks_like_tests(lowered):
            return FastAction("run_command", {"argv": self._pytest_argv()}, "explicit test run")

        search = self._search_query(text)
        if search and "search_files" in available:
            return FastAction("search_files", {"query": search}, "explicit literal search")

        return None

    def verification_for(self, changed_path: str, available: set[str]) -> FastAction | None:
        if "run_command" not in available:
            return None
        relative = Path(changed_path)
        candidates: list[Path] = []
        if relative.parts[:2] == ("src", "agent_tui") and relative.suffix == ".py":
            candidates.append(Path("tests") / f"test_{relative.stem}.py")
        if relative.parts and relative.parts[0] != "tests" and relative.suffix == ".py":
            candidates.append(Path("tests") / f"test_{relative.stem}.py")
        for candidate in candidates:
            if (self.workspace / candidate).is_file():
                return FastAction(
                    "run_command",
                    {"argv": [*self._pytest_argv(), str(candidate)]},
                    f"verify edit with {candidate}",
                )
        return None

    def _existing_path(self, text: str) -> str | None:
        for match in _PATH.finditer(text):
            candidate = match.group("path").rstrip(".,:;)")
            path = self.workspace / candidate
            if path.is_file() and path.resolve().is_relative_to(self.workspace.resolve()):
                return candidate
        return None

    @staticmethod
    def _looks_like_read(text: str) -> bool:
        return any(word in text for word in ("read ", "show ", "open ", "inspect ", "cat "))

    @staticmethod
    def _looks_like_list(text: str) -> bool:
        return any(
            phrase in text
            for phrase in ("list files", "show files", "project structure", "repo structure")
        )

    @staticmethod
    def _looks_like_tests(text: str) -> bool:
        return text in {"test", "tests", "run tests", "run the tests", "pytest"} or (
            ("run" in text or "execute" in text) and ("pytest" in text or "tests" in text)
        )

    @staticmethod
    def _search_query(text: str) -> str | None:
        match = re.fullmatch(r'(?:search|find)\s+(?:for\s+)?["\'](.+?)["\']', text.strip(), re.I)
        return match.group(1) if match else None

    def _pytest_argv(self) -> list[str]:
        if (self.workspace / "uv.lock").exists():
            return ["uv", "run", "pytest"]
        return [sys.executable, "-m", "pytest"]
