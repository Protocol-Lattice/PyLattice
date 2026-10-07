"""Sandboxed Python programs that compose policy-checked tools without model round trips."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_monty import AsyncMonty, CollectString, MontyError

from .config import Settings
from .models import Completion, ToolCall
from .tools import ToolError, ToolRegistry, ToolResult, ToolSpec, object_schema

MAX_CODE_CHARS = 24_000
MAX_TOOL_CALLS = 32
EXECUTION_SECONDS = 5.0
MEMORY_BYTES = 64 * 1024 * 1024

CODE_SPEC = ToolSpec(
    "execute_code",
    "Execute a sandboxed Python program to compose multiple tools in one model turn. "
    "Use await call_tool(name, arguments) and return the useful data as the final expression. "
    "Tool calls retain validation, hooks and approvals. A failed tool stops further calls. "
    "Programs are independent; variables do not persist between calls.",
    object_schema(
        {"code": {"type": "string", "minLength": 1, "maxLength": MAX_CODE_CHARS}}, ["code"]
    ),
    "orchestrate",
)

CODE_PROMPT = """Use Code Mode to carry out related actions in one execute_code call.
Write Python using variables, loops, conditions and comprehensions. No planning or routing
model runs between tools. Call tools with await call_tool("name", {"argument": value}).
Catalog names such as read_file and skill_search are call_tool arguments inside Python,
not top-level function calls. The only top-level tools are execute_code and finish.
The returned dict has ok and output; JSON tool output is already decoded, other output is
a string. Calls execute serially, including calls scheduled with asyncio.gather.
Return only useful evidence as the program's final expression, or use print for short output.
Intermediate tool results stay in the program. Each program starts fresh; variables do not
persist. Use a new program only when you need to reason about the previous result.
All workspace, command, MCP, skill and memory access must go through call_tool. The sandbox
has no host filesystem or network access. Do not use open, subprocess or third-party packages.
Each program permits at most 32 tool calls, 5 seconds of computation and 64 MiB of memory.
Waiting for a tool or user approval does not consume the computation budget. On any tool
failure, further tool calls in that program are blocked, even if you catch the exception.
Earlier successful actions are not rolled back. Inspect current state before retrying.
execute_code and finish cannot be called inside a program. Use finish or answer directly
after inspecting the program's output. Use exactly one top-level tool call per response.
Example:
listing = await call_tool("list_files", {"path": "src"})
results = []
for path in listing["output"]["files"][:3]:
    result = await call_tool("read_file", {"path": path})
    results.append(result["output"])
results
"""

ToolRunner = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


def adapt_tool_calls(completion: Completion, registry: ToolRegistry) -> Completion:
    """Accommodate models that emit catalog tools directly without another model turn."""
    if not completion.calls or any(
        call.name not in registry.specs or call.name in {"execute_code", "finish"}
        for call in completion.calls
    ):
        return completion
    lines = ["results = []"]
    for call in completion.calls:
        arguments = registry.validate(call.name, call.arguments)
        # Arguments must remain data, including quotes, newlines and Python-looking text.
        try:
            json.dumps(arguments, allow_nan=False)
        except ValueError as exc:
            raise ToolError(f"Invalid arguments for {call.name}: {exc}") from None
        lines.append(f"results.append(await call_tool({call.name!r}, {arguments!r}))")
    lines.append("results[0]" if len(completion.calls) == 1 else "results")
    return Completion(
        content=completion.content,
        calls=[
            ToolCall(
                completion.calls[0].id,
                "execute_code",
                json.dumps(
                    {
                        "code": "\n".join(lines),
                    }
                ),
            )
        ],
        model=completion.model,
        tokens=completion.tokens,
    )


async def execute_code(code: str, settings: Settings, run_tool: ToolRunner) -> ToolResult:
    """Expose only a JSON tool bridge; never share host objects, mounts or OS handlers."""
    if not code.strip() or len(code) > MAX_CODE_CHARS:
        raise ToolError(f"Code must contain 1–{MAX_CODE_CHARS} characters")
    calls: list[dict[str, Any]] = []
    failure = ""
    active = True
    lock = asyncio.Lock()
    callbacks: set[asyncio.Task] = set()
    printed = CollectString(max_bytes=settings.max_output_chars)
    output = None

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal failure
        task = asyncio.current_task()
        if task is not None:
            callbacks.add(task)
        try:
            async with lock:
                if not active or failure:
                    raise ToolError(failure or "Code execution has ended")
                try:
                    if len(calls) >= MAX_TOOL_CALLS:
                        raise ToolError(f"Code Mode reached its {MAX_TOOL_CALLS}-tool-call limit")
                    if not isinstance(name, str) or name in {"execute_code", "finish"}:
                        raise ToolError(
                            "Use a workspace tool name; execution and finish cannot nest"
                        )
                    if not isinstance(arguments, dict):
                        raise ToolError("Tool arguments must be a JSON object")
                    # Reject non-JSON/cyclic values before they cross the tool boundary.
                    arguments = json.loads(json.dumps(arguments, allow_nan=False))
                    entry = {"tool": name, "ok": False}
                    calls.append(entry)
                    result = await run_tool(name, arguments)
                    entry["ok"] = result.ok
                    if not result.ok:
                        raise ToolError(result.content)
                    try:
                        value = json.loads(result.content)
                    except ValueError:
                        value = result.content
                    return {"ok": True, "output": value}
                except Exception as exc:
                    failure = settings.redact(str(exc) or type(exc).__name__)
                    raise ToolError(failure) from None
        finally:
            if task is not None:
                callbacks.discard(task)

    try:
        async with (
            AsyncMonty(max_processes=1, request_timeout=EXECUTION_SECONDS + 2) as pool,
            pool.checkout(
                limits={
                    "max_feed_duration_secs": EXECUTION_SECONDS,
                    "max_memory": MEMORY_BYTES,
                    "max_recursion_depth": 100,
                    "max_suspensions": 256,
                },
                os_policy={"sleep": "zero"},
            ) as session,
        ):
            output = await session.feed_run(
                code, external_lookup={"call_tool": call_tool}, print_callback=printed
            )
    except (MontyError, ValueError, OSError) as exc:
        failure = failure or settings.redact(str(exc) or type(exc).__name__)
    finally:
        active = False
        pending = [task for task in callbacks if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    payload = {"output": output, "stdout": printed.output, "tools": calls}
    if failure:
        payload["error"] = failure
    try:
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        failure = "Return JSON-compatible data from the final expression"
        text = json.dumps({"error": failure, "tools": calls})
    if len(text) > settings.max_output_chars:
        text = json.dumps(
            {
                "output_excerpt": text[: settings.max_output_chars // 3],
                "truncated": True,
                "tool_calls": len(calls),
                "error": failure[:160] if failure else None,
            },
            ensure_ascii=False,
        )
    # The registry also applies its common output bound and credential redaction.
    return ToolResult(not failure, text)
