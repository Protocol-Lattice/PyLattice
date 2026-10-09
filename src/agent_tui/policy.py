"""Default tool validation, middleware, approval, repeat guards and notifications."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .models import AgentEvent, Approval, EventSink, ToolCall
from .tools import ToolError, ToolResult


@dataclass
class RunTools:
    repeats: Counter[str] = field(default_factory=Counter)
    sequence: int = 0


class DefaultToolPolicy:
    @staticmethod
    async def execute(
        agent,
        call: ToolCall,
        run_tools: RunTools,
        emit: EventSink,
        approve: Approval,
        step: int,
        exchange: list[dict[str, Any]] | None = None,
        allowed: set[str] | None = None,
    ) -> tuple[ToolResult, dict[str, Any]]:
        """The same policy boundary for top-level calls and calls from sandboxed programs."""
        arguments: dict[str, Any] = {}
        run_tools.sequence += 1
        event_id = run_tools.sequence
        try:
            if allowed is not None and call.name not in allowed:
                raise ToolError(
                    f"Return a call to one of: {', '.join(sorted(allowed))}", code="wrong_tool"
                )
            arguments = agent.registry.validate(call.name, call.arguments)
            hook = await agent.middleware.dispatch(
                "before_tool", {"tool": call.name, "arguments": arguments, "step": step}
            )
            arguments = agent.registry.validate(call.name, json.dumps(hook["arguments"]))
            if exchange is not None:
                exchange[0]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
            fingerprint = call.name + json.dumps(arguments, sort_keys=True)
            run_tools.repeats[fingerprint] += 1
            if run_tools.repeats[fingerprint] > 2:
                raise ToolError(
                    "Repeated identical action blocked. Choose a different "
                    "approach or use finish to report the blocker.",
                    code="repeated_action",
                )
            await emit(
                AgentEvent(
                    "tool_start", call.name, step, {"arguments": arguments, "call_id": event_id}
                )
            )
            if agent.registry.requires_approval(call.name):
                preview = agent.registry.preview(call.name, arguments)
                await emit(AgentEvent("approval", call.name, step))
                if not await approve(call.name, agent.settings.redact(preview)):
                    raise ToolError(
                        "User denied this action. Do not bypass the denial.",
                        code="approval_denied",
                        retry_hint="Respect the denial; ask the user if essential input is needed.",
                    )
                if preview != agent.registry.preview(call.name, arguments):
                    raise ToolError(
                        "File changed while approval was pending; inspect and retry",
                        code="file_changed",
                    )
            tool_result = await agent.registry.execute(call.name, arguments)
            # Reads must stay usable after edits or context pruning. Successful Code Mode
            # wrappers can repeat too; their individual actions retain the guard above.
            if tool_result.ok and call.name in {
                "read_file",
                "list_files",
                "search_files",
                "context_collect",
                "execute_code",
            }:
                run_tools.repeats.pop(fingerprint, None)
        except (ToolError, OSError, UnicodeError, ValueError) as exc:
            tool_result = ToolResult.from_error(exc)
        except asyncio.CancelledError:
            if exchange is not None:
                exchange.append(
                    agent._tool_message(
                        call,
                        ToolResult.from_error(
                            ToolError(
                                "Execution cancelled. An action may have partially completed; "
                                "inspect current state before trying again.",
                                code="cancelled",
                            )
                        ),
                    )
                )
            raise
        tool_result = agent.registry.sanitize(tool_result, call.name)
        reply = agent.responses.as_dict(tool_result)
        if exchange is not None:
            exchange.append(agent._tool_message(call, tool_result))
        # Record the reply before notification hooks can fail or be cancelled.
        await agent.middleware.dispatch(
            "after_tool",
            {
                "tool": call.name,
                "arguments": arguments,
                "step": step,
                **reply,
            },
        )
        await emit(
            AgentEvent(
                "tool_result",
                tool_result.content,
                step,
                {"tool": call.name, "call_id": event_id, **reply},
            )
        )
        return tool_result, arguments
