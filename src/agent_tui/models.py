from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str | dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        # OpenAI-compatible chat transcripts require a JSON *string* here, even
        # when a custom executor provides already-decoded argument objects.
        arguments = (
            self.arguments
            if isinstance(self.arguments, str)
            else json.dumps(self.arguments, ensure_ascii=False, allow_nan=False)
        )
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": arguments},
        }


@dataclass
class Completion:
    content: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    tokens: int = 0

    def as_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.calls:
            message["tool_calls"] = [call.as_dict() for call in self.calls]
        return message


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    text: str = ""
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)


EventSink = Callable[[AgentEvent], Awaitable[None]]
TokenSink = Callable[[str], Awaitable[None]]
Approval = Callable[[str, str], Awaitable[bool]]


@dataclass(frozen=True)
class RunResult:
    status: str
    message: str
    steps: int
