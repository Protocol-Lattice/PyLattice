"""Sandboxed Python programs that compose policy-checked tools without model round trips."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any

from pydantic_monty import AsyncMonty, CollectString, MontyError, MontySyntaxError, MontyTypingError

from .config import Settings
from .models import Completion, ToolCall
from .tools import (
    ToolError,
    ToolFailure,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    bounded_json,
    object_schema,
)

MAX_CODE_CHARS = 96_000
MAX_TOOL_CALLS = 1024
EXECUTION_SECONDS = 5.0
MEMORY_BYTES = 64 * 1024 * 1024

# The same bridge signature is given to the model and Monty's pre-execution checker.
# Output varies by catalog tool; it is decoded JSON data or a plain text string.
TOOL_BRIDGE_STUBS = """from typing import Any, TypedDict

class ToolFailure(TypedDict):
    code: str
    message: str
    retry_hint: str | None

class ToolReply(TypedDict):
    ok: bool
    output: Any
    error: ToolFailure | None
    truncated: bool

async def call_tool(name: str, arguments: dict[str, Any]) -> ToolReply: ...
"""

CODE_SPEC = ToolSpec(
    "execute_code",
    "Execute a sandboxed Python program to inspect files, apply needed edits or writes, "
    "and verify results. Continue with this tool after reading when requested changes remain. "
    "The code argument must be a raw Python source string, without Markdown fences or prose. "
    "Use await call_tool(name, arguments) and return useful JSON data as the final expression. "
    "Syntax and types are checked before any tool runs. "
    "Tool calls retain validation, hooks and approvals. A failed tool stops further calls. "
    "Host files are available only through call_tool, never Python open(). "
    "Programs are independent; variables do not persist between calls.",
    object_schema(
        {
            "code": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_CODE_CHARS,
                "description": (
                    "Raw Python source with top-level await call_tool(name, arguments). "
                    "Use Python True, False and None, not JSON true, false and null. "
                    "End with a JSON-compatible expression or print a short result. "
                    "No Markdown fences, host imports, or asyncio.run wrapper."
                ),
            }
        },
        ["code"],
    ),
    "orchestrate",
)

CODE_PROMPT = """Code Mode generation contract:
1. The only top-level tools are execute_code and finish. For actions, emit exactly one tool
call with arguments matching its JSON schema. Obey the selected tool for this turn.
Catalog tools are available ONLY through call_tool inside execute_code; never emit them as
top-level calls or call them as Python functions. Never nest execute_code or finish.
2. execute_code arguments are {"code": "<Python source>"}. The code value is raw Python,
not Markdown, explanatory prose, a JSON object, or a quoted/JSON-encoded second copy of the
program. Escape quotes and newlines once in the outer JSON arguments. Inside Python use
True, False and None; JSON true, false and null are not Python literals.
3. Write a short, self-contained program with top-level await, variables, loops, conditions
and comprehensions. Every call_tool must be awaited before using its result. Use exactly
await call_tool("catalog_name", {"argument": value}); do not pass tool parameters as keyword
arguments to call_tool or pass a JSON string instead of a dict. Use only names and argument
keys from the current catalog, including required arguments and their exact types.
4. This is the Monty Python subset. Prefer builtins, strings, lists and dicts. Do not import
workspace modules or third-party packages. There is no host filesystem, environment or
network access: do not use open, pathlib for file access, os, subprocess, requests or shell
syntax. Use call_tool for all external actions. json, re and math support basic data work;
do not assume the full Python standard library is available. No asyncio.run/main wrapper,
create_task, background jobs, eval or exec. Serial awaits are simplest; asyncio.gather is
supported but tool calls still execute serially. All variables and imports start fresh in
each execute_code call. Syntax and type errors reject the program before tools execute.
5. Tool replies share {"ok": bool, "output": data, "error": object_or_None,
"truncated": bool}. output is already decoded; never json.loads an object again. Plain
text output remains a string. error is None on success; otherwise it contains code,
message and retry_hint. Check result["truncated"] before relying on partial data.
MCP output has content (typed text/omission blocks) and structured_content (JSON or None).
Inspect unknown extension data before assuming its shape. Failed tools raise an exception,
and all later tool calls in that program are blocked even if you catch it. Do not write
fallback actions in an except block to work around a tool failure or denied approval.
6. Built-in output shapes (under result["output"]):
   list_files: {"files": ["relative/path"], "truncated": bool}; entries are strings.
   read_file: {"path": str, "content": str, "total_lines": int, "has_more": bool}.
   content includes display prefixes like "12: text"; those prefixes are not file content.
   search_files: {"matches": [{"path": str, "line": int, "text": str}],
                  "truncated": bool, "skipped": int}; query is literal text, not regex.
   context_collect: {"files": [{"path": str, "sha256": str, "content": str,
                      "start_line": int, "total_lines": int, "excerpt": bool}],
                      "disk_reads": int, "index_hits": int, "truncated": bool}.
   apply_patchset: {"changes": [{"path": str, "bytes": int}], "count": int}.
   edit_file/write_file: {"path": str, "bytes": int}.
   run_command: {"exit_code": int, "output": str, "truncated": bool, "timed_out": bool}.
Use result["output"]["content"] for file text and result["output"]["output"] for command
text. Check result["truncated"] and has_more before relying on an excerpt. When an
older result was compacted from context its output may be None; re-read the source
instead of assuming missing data means an empty result.
Paths are workspace-relative. run_command takes an argv list, not a shell command string.
7. End with a small JSON-compatible expression (dict, list, string, number, bool or None),
or print short evidence. Do not end with only assignments, an unawaited coroutine, a set,
bytes or a custom object. Do not use a top-level return. Intermediate results are not shown
to the next model turn unless included in the final expression or printed.
8. For repository work, prefer context_collect over repeated list_files/search_files/
read_file calls: use {"query": "term", "max_files": 6} or {"paths": ["src/a.py",
"src/b.py"]}. Files are cached in Context Manager across independent Code Mode programs
and relevant verified excerpts are added to later model contexts. Returned content has
no line-number prefixes; use exact old_text from it. Excerpts are not whole files.
For refactors or new files, use apply_patchset with a list of actions. An edit is
{"action": "edit", "path": "src/a.py", "old_text": "exact old block",
"new_text": "replacement", "expected_sha256": "digest from context_collect"}.
A create is {"action": "create", "path": "src/new.py", "content": "source"}.
The patchset validates all files before writing and requires the usual approval.
Do not overwrite existing files with create. Check read-only tool availability.
For single edits, edit_file still works; write_file is for new files or requested
full replacements. Never reconstruct a full file from a partial excerpt. Verify
changed files with context_collect or a test command before finishing.
9. Keep output and loops bounded. A program has at most 1024 tool calls, 5 seconds of
computation (excluding tools/approvals), 64 MiB memory and 96,000 source characters.
On failure, read error, phase, tools and retry_hint. Correct the cause in a NEW program.
Validation errors run no tools. Runtime errors may follow successful actions, which are
not rolled back: inspect current state and never blindly replay earlier mutations.
10. Use finish only when complete or essential user input is required. Report verification
and blockers honestly. A direct final answer is allowed only when finish is selected or
routing falls back and no work remains; never answer directly when execute_code is selected.

Example Python source for bounded inspection in one host tool call:
result = await call_tool("context_collect", {"query": "example", "max_files": 4})
{"files": result["output"]["files"], "truncated": result["output"]["truncated"]}
"""


def code_mode_prompt(registry: ToolRegistry, selected: str | None) -> str:
    catalog = [
        schema["function"]
        for schema in registry.schemas()
        if schema["function"]["name"] not in {"execute_code", "finish"}
    ]
    instruction = {
        "execute_code": (
            "For this response, call execute_code exactly once with a code string. "
            "Do not return a final answer or a finish call."
        ),
        "finish": "For this response, finish with a concise summary or essential question.",
        None: "Routing fallback: choose execute_code or finish according to remaining work.",
    }[selected]
    return (
        CODE_PROMPT
        + "\nProvided bridge signature (already defined; do not redefine it):\n"
        + TOOL_BRIDGE_STUBS
        + "\ncall_tool catalog (JSON schemas, not Python source):\n"
        + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        + instruction
    )


ToolRunner = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


def adapt_tool_calls(completion: Completion, registry: ToolRegistry) -> Completion:
    """Accommodate models that emit catalog tools directly without another model turn."""
    if not completion.calls or any(
        call.name not in registry.specs or call.name in {"execute_code", "finish"}
        for call in completion.calls
    ):
        return completion
    if len(completion.calls) > MAX_TOOL_CALLS:
        raise ToolError(f"Code Mode permits at most {MAX_TOOL_CALLS} tool calls per program")
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


async def execute_code(
    code: str,
    settings: Settings,
    run_tool: ToolRunner,
    *,
    format_result: Callable[[ToolResult], dict[str, Any]] = ToolResult.as_dict,
    pool: AsyncMonty | None = None,
) -> ToolResult:
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
    phase = "runtime"
    tool_failure: ToolFailure | None = None

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal failure, tool_failure
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
                    reply = format_result(result)
                    entry["ok"] = result.ok
                    if not result.ok:
                        entry["error"] = reply["error"]
                        entry["output"] = reply["output"]
                        tool_failure = ToolFailure(**reply["error"])
                        raise ToolError(
                            tool_failure.message,
                            code=tool_failure.code,
                            retry_hint=tool_failure.retry_hint,
                        )
                    return reply
                except Exception as exc:
                    failure = settings.redact(str(exc) or type(exc).__name__)
                    if tool_failure is None:
                        tool_failure = ToolResult.from_error(exc).error
                    raise ToolError(failure) from None
        finally:
            if task is not None:
                callbacks.discard(task)

    try:
        async with (
            (
                nullcontext(pool)
                if pool is not None
                else AsyncMonty(max_processes=1, request_timeout=EXECUTION_SECONDS + 2)
            ) as runtime_pool,
            runtime_pool.checkout(
                type_check=True,
                type_check_stubs=TOOL_BRIDGE_STUBS,
                type_check_format="concise",
                limits={
                    "max_feed_duration_secs": EXECUTION_SECONDS,
                    "max_memory": MEMORY_BYTES,
                    "max_recursion_depth": 100,
                    "max_suspensions": MAX_TOOL_CALLS + 16,
                },
                os_policy={"sleep": "zero"},
            ) as session,
        ):
            output = await session.feed_run(
                code, external_lookup={"call_tool": call_tool}, print_callback=printed
            )
    except (MontyError, ValueError, OSError) as exc:
        if isinstance(exc, (MontySyntaxError, MontyTypingError)) and not calls:
            phase = "validation"
        if not failure:
            failure = settings.redact(str(exc) or type(exc).__name__)
            if isinstance(exc, MontyError) and (
                isinstance(exc.exception(), PermissionError)
                or isinstance(exc, MontyTypingError)
                and "Name `open` used when not defined" in failure
            ):
                failure += (
                    "\nCode Mode cannot access host files directly. Read files with "
                    'await call_tool("read_file", {"path": "workspace-relative/path.py"}). '
                    "Use edit_file for patches or write_file for new files through call_tool."
                )
    finally:
        active = False
        pending = [task for task in callbacks if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    try:
        json.dumps(output, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        failure = failure or "Return JSON-compatible data from the final expression"
        output = None
    payload = {"output": output, "stdout": printed.output, "tools": calls}
    retry_hint: str | None = None
    if failure:
        payload["error"] = failure
        payload["phase"] = phase
        retry_hint = (
            "No tools ran. Fix the reported Python error and submit a new execute_code call. "
            "Use raw Python without Markdown fences, await call_tool(name, arguments), "
            "Python True/False/None, and a final JSON-compatible expression."
            if phase == "validation"
            else "Inspect the tools log and current state before retrying in a new program. "
            "Earlier successful actions were not rolled back. Do not bypass denied approvals."
        )
        payload["retry_hint"] = retry_hint
    text = settings.redact(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    truncated = len(text) > settings.max_output_chars
    if truncated:
        summary = {
            "output_excerpt": "",
            "truncated": True,
            "tools": calls,
            "tool_calls": len(calls),
        }
        if failure:
            summary.update(error=failure[:160], phase=phase, retry_hint=payload["retry_hint"])
        if len(json.dumps(summary, ensure_ascii=False)) > settings.max_output_chars:
            summary.update(tools=[], tools_truncated=True)
        if failure and len(json.dumps(summary, ensure_ascii=False)) > settings.max_output_chars:
            # Escaped diagnostics can use far more characters than their source text.
            # Keep the phase and recovery guidance, shortening the error before excerpts.
            text = bounded_json(summary, "error", settings.max_output_chars)
        else:
            summary["output_excerpt"] = text
            text = bounded_json(summary, "output_excerpt", settings.max_output_chars)
    # The registry also applies its common output bound and credential redaction.
    error = None
    if failure:
        error = ToolFailure(
            tool_failure.code if tool_failure else f"code_{phase}_error",
            failure,
            tool_failure.retry_hint
            if tool_failure and tool_failure.retry_hint
            else retry_hint,
        )
    return ToolResult(not failure, text, error, truncated)
