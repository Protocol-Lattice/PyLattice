"""Validated, workspace-local configuration for external MCP servers and hooks."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator, ValidationError

from .config import Settings

HOOK_EVENTS = ("before_run", "before_model", "before_tool", "after_tool", "after_run", "on_error")
STRING = {"type": "string", "minLength": 1}
STRINGS = {"type": "array", "items": STRING}
COMMAND = {**STRINGS, "minItems": 1}
TIMEOUT = {"type": "number", "exclusiveMinimum": 0, "maximum": 3600}


def _object(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


CONFIG_SCHEMA = _object(
    {
        "mcp_servers": {
            "type": "object",
            "propertyNames": {"pattern": "^[a-zA-Z0-9_-]{1,40}$"},
            "additionalProperties": _object(
                {
                    "transport": {"enum": ["stdio", "http"]},
                    "command": STRING,
                    "args": STRINGS,
                    "url": STRING,
                    "env": {"type": "object", "additionalProperties": STRING},
                    "headers": {"type": "object", "additionalProperties": STRING},
                    "timeout": TIMEOUT,
                    "enabled": {"type": "boolean"},
                    "read_only_tools": STRINGS,
                }
            ),
        },
        "hooks": {
            "type": "array",
            "items": _object(
                {
                    "event": {"enum": list(HOOK_EVENTS)},
                    "command": COMMAND,
                    "timeout": TIMEOUT,
                    "enabled": {"type": "boolean"},
                },
                ["event", "command"],
            ),
        },
    }
)


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    env: dict[str, str] = field(default_factory=dict, repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    timeout: float = 30
    enabled: bool = True
    read_only_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class HookConfig:
    event: str
    command: tuple[str, ...]
    timeout: float = 10
    enabled: bool = True


@dataclass(frozen=True)
class ExtensionConfig:
    servers: tuple[MCPServerConfig, ...] = ()
    hooks: tuple[HookConfig, ...] = ()

    @classmethod
    def load(cls, settings: Settings) -> ExtensionConfig:
        path = settings.extensions_path or settings.workspace / ".agent-tui" / "extensions.toml"
        if not path.exists() and settings.extensions_path is None:
            return cls()
        try:
            if path.stat().st_size > 128_000:
                raise ValueError("Extension configuration exceeds 128 KB")
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            Draft202012Validator(CONFIG_SCHEMA).validate(data)
            servers = []
            for name, raw in data.get("mcp_servers", {}).items():
                server = MCPServerConfig(name=name, **raw)
                if server.transport == "stdio" and (not server.command or server.url):
                    raise ValueError(f"MCP {name}: stdio requires command and no url")
                if server.transport == "http" and (
                    not server.url.startswith(("http://", "https://")) or server.command
                ):
                    raise ValueError(f"MCP {name}: http requires an HTTP(S) url and no command")
                servers.append(server)
            return cls(tuple(servers), tuple(HookConfig(**raw) for raw in data.get("hooks", [])))
        except (OSError, UnicodeError, ValueError, ValidationError) as exc:
            detail = exc.message if isinstance(exc, ValidationError) else str(exc)
            raise ValueError(f"Invalid extension configuration {path}: {detail}") from None


def expand_environment(values: dict[str, str]) -> dict[str, str]:
    def replace(match: re.Match) -> str:
        name = match[1]
        if name not in os.environ:
            raise ValueError(f"Missing environment variable: {name}")
        return os.environ[name]

    return {
        key: re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
        for key, value in values.items()
    }


def safe_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
