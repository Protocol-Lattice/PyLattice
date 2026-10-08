"""Application configuration. Credentials are read only from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONTEXT_CHARS = 64_000


@dataclass(frozen=True)
class Settings:
    workspace: Path
    api_key: str = field(default="", repr=False)
    model: str = "openrouter:free"
    router_model: str = "typesafe/jev-1.13"
    max_steps: int = 1024
    command_timeout: float = 60.0
    request_timeout: float = 90.0
    router_timeout: float = 8.0
    max_output_chars: int = 16000
    context_chars: int = DEFAULT_CONTEXT_CHARS
    history_turns: int = 12
    memory_enabled: bool = True
    skills_dirs: tuple[Path, ...] = ()
    extensions_path: Path | None = None
    mods_path: Path | None = None
    mod_overrides: tuple[str, ...] = ()
    max_tokens: int = 4096
    code_mode: bool = True
    planning: bool = True
    routing: str = "mcts"
    mcts_simulations: int = 64
    mcts_depth: int = 3
    read_only: bool = False
    auto_approve: bool = False
    demo: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", self.workspace.expanduser().resolve())
        object.__setattr__(
            self,
            "skills_dirs",
            tuple((self.workspace / path.expanduser()).resolve() for path in self.skills_dirs),
        )
        if self.extensions_path is not None:
            object.__setattr__(
                self,
                "extensions_path",
                (self.workspace / self.extensions_path.expanduser()).resolve(),
            )
        if self.mods_path is not None:
            object.__setattr__(
                self, "mods_path", (self.workspace / Path(self.mods_path).expanduser()).resolve()
            )
        # Accept the spelling sometimes used for the free router, but send the real API ID.
        model = self.model.strip()
        object.__setattr__(
            self, "model", "openrouter/free" if model == "openrouter:free" else model
        )
        if not self.workspace.is_dir():
            raise ValueError(f"Workspace is not a directory: {self.workspace}")
        if not self.model.strip() or not self.router_model.strip():
            raise ValueError("Model names cannot be empty")
        if self.routing not in {"jev", "mcts"}:
            raise ValueError("routing must be jev or mcts")
        if not self.code_mode and self.routing == "mcts" and not self.planning:
            raise ValueError("MCTS needs the planner; remove --no-planner")
        if not 1 <= self.mcts_simulations <= 1024 or not 1 <= self.mcts_depth <= 5:
            raise ValueError("MCTS needs 1–1024 simulations and depth 1–5")
        for name in (
            "max_steps",
            "max_output_chars",
            "context_chars",
            "max_tokens",
            "history_turns",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("command_timeout", "request_timeout", "router_timeout"):
            if not 0 < getattr(self, name) <= 3600:
                raise ValueError(f"{name} must be between 0 and 3600 seconds")

    @classmethod
    def from_env(cls, workspace: Path, **overrides: object) -> Settings:
        values = {
            "workspace": workspace,
            "api_key": os.getenv("OPENROUTER_API_KEY", "").strip(),
            "model": os.getenv("AGENT_TUI_MODEL", "openrouter/free"),
            "router_model": os.getenv("AGENT_TUI_ROUTER_MODEL", "typesafe/jev-1.13"),
            "max_steps": int(os.getenv("AGENT_TUI_MAX_STEPS", "24")),
            "command_timeout": float(os.getenv("AGENT_TUI_COMMAND_TIMEOUT", "60")),
            "routing": os.getenv("AGENT_TUI_ROUTING", "mcts"),
            "code_mode": os.getenv("AGENT_TUI_CODE_MODE", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            "context_chars": int(os.getenv("AGENT_TUI_CONTEXT_CHARS", str(DEFAULT_CONTEXT_CHARS))),
            "mods_path": Path(os.environ["AGENT_TUI_MODS"])
            if os.getenv("AGENT_TUI_MODS")
            else None,
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def redact(self, text: str) -> str:
        return text.replace(self.api_key, "[REDACTED]") if self.api_key else text
