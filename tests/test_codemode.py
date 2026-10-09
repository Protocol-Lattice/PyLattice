from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from harness_router import RouteDecision

from agent_tui.agent import Agent
from agent_tui.codemode import CODE_PROMPT, MAX_TOOL_CALLS, TOOL_BRIDGE_STUBS, execute_code
from agent_tui.models import Completion, ToolCall
from agent_tui.routing import HarnessDecisionLayer
from agent_tui.tools import ToolFailure, ToolRegistry, ToolResult, ToolSpec, object_schema


class CodeDecisions:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.states = []
        self.catalogs = []

    async def route(self, state, tools):
        self.states.append(copy.deepcopy(state))
        self.catalogs.append([tool.name for tool in tools])
        return next(self.decisions, RouteDecision.fallback_to_planner("test_fallback"))

    async def route_mcts(self, *_):
        pytest.fail("Code Mode must not call MCTS")

    async def plan(self, *_):
        pytest.fail("Code Mode must not call the planner")


class Executor:
    def __init__(self, *codes):
        self.completions = iter(
            [
                *(
                    Completion(calls=[ToolCall(str(i), "execute_code", json.dumps({"code": code}))])
                    for i, code in enumerate(codes)
                ),
                Completion(content="Done"),
            ]
        )
        self.requests = []

    async def complete(self, messages, schemas, selected, on_token):
        self.requests.append(copy.deepcopy((messages, schemas, selected)))
        return next(self.completions)


async def ignore(*_):
    pass


async def deny(*_):
    return False


def agent_for(settings, *codes, **overrides):
    settings = replace(settings, code_mode=True, **overrides)
    executor = Executor(*codes)
    return Agent(settings, CodeDecisions(), executor, planner=CodeDecisions()), executor


def result_before(executor, index=1):
    return json.loads(executor.requests[index][0][-1]["content"])


async def test_dependent_tools_share_one_model_turn_and_keep_intermediate_results_local(
    settings, tmp_path
):
    (tmp_path / "one.txt").write_text("one")
    (tmp_path / "two.txt").write_text("two")
    agent, executor = agent_for(
        settings,
        """
listing = await call_tool("list_files", {})
count = 0
for path in listing["output"]["files"]:
    r = await call_tool("read_file", {"path": path})
    count += r["output"]["total_lines"]
{"line_count": count}
""",
    )
    events = []

    async def emit(event):
        events.append(event)

    result = await agent.run("Count lines", emit, deny)
    assert result.status == "completed" and result.steps == 2
    assert len(executor.requests) == 2
    assert len(agent.router.states) == 2
    assert all(set(catalog) == {"execute_code", "finish"} for catalog in agent.router.catalogs)
    assert result_before(executor)["output"]["output"] == {"line_count": 2}
    assert "1: one" not in json.dumps(result_before(executor)["output"])
    assert all(selected is None for _, _, selected in executor.requests)
    assert {s["function"]["name"] for s in executor.requests[0][1]} == {"execute_code", "finish"}
    assert "read_files" not in executor.requests[0][0][0]["content"]
    starts = [e for e in events if e.kind == "tool_start"]
    assert [e.text for e in starts] == ["execute_code", "list_files", "read_file", "read_file"]
    assert len({e.data["call_id"] for e in starts}) == 4
    assert "execute_code" not in agent.registry.specs


async def test_configured_decision_model_routes_programs_from_latest_results(settings, tmp_path):
    (tmp_path / "file.py").write_text("observed source")
    agent, executor = agent_for(
        settings,
        'listing = await call_tool("list_files", {})\n'
        'await call_tool("read_file", {"path": listing["output"]["files"][0]})',
        router_model="test/decision-model",
    )
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        choice = "execute_code" if len(payloads) == 1 else "finish"
        return httpx.Response(
            200,
            json={
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.99,
                        "probabilities": {choice: 0.99},
                    }
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        agent.router = HarnessDecisionLayer(agent.settings, client)
        result = await agent.run("Inspect the source", ignore, deny)
    assert result.status == "completed"
    assert len(payloads) == 2
    assert all(payload["model"] == "test/decision-model" for payload in payloads)
    assert "1: observed source" in json.dumps(payloads[1])
    assert [selected for _, _, selected in executor.requests] == ["execute_code", "finish"]
    assert [
        [schema["function"]["name"] for schema in schemas] for _, schemas, _ in executor.requests
    ] == [["execute_code"], ["finish"]]


@pytest.mark.parametrize(
    "decision",
    [RouteDecision.fallback_to_planner("provider_error"), RouteDecision(tool="read_file")],
)
async def test_code_mode_routing_fallback_exposes_only_program_and_finish(settings, decision):
    agent, executor = agent_for(settings, 'await call_tool("list_files", {})')
    agent.router = CodeDecisions(decision)
    result = await agent.run("Inspect", ignore, deny)
    assert result.status == "completed" and result_before(executor)["ok"]
    _, schemas, selected = executor.requests[0]
    assert selected is None
    assert {schema["function"]["name"] for schema in schemas} == {"execute_code", "finish"}


@pytest.mark.parametrize(
    "selected,returned",
    [("finish", "execute_code"), ("finish", "write_file"), ("execute_code", "finish")],
)
async def test_code_mode_enforces_the_decision_before_execution(
    settings, tmp_path, selected, returned
):
    arguments = {
        "execute_code": {
            "code": 'await call_tool("write_file", {"path": "file", "content": "bad"})'
        },
        "write_file": {"path": "file", "content": "bad"},
        "finish": {"summary": "Premature finish"},
    }
    agent, executor = agent_for(settings, auto_approve=True)
    agent.router = CodeDecisions(RouteDecision(tool=selected), RouteDecision(tool="finish"))
    executor.completions = iter(
        [
            Completion(calls=[ToolCall("mismatch", returned, json.dumps(arguments[returned]))]),
            Completion(content="Done"),
        ]
    )
    result = await agent.run("Inspect", ignore, deny)
    assert result.status == "completed" and result.message == "Done"
    assert not (tmp_path / "file").exists()
    rejected = result_before(executor)
    assert (
        not rejected["ok"]
        and f"Return a call to one of: {selected}" in rejected["error"]["message"]
    )


async def test_cancellation_during_code_mode_routing_does_not_reach_executor(settings):
    entered = asyncio.Event()

    class WaitingRouter(CodeDecisions):
        async def route(self, state, tools):
            entered.set()
            await asyncio.Future()

    agent, executor = agent_for(settings)
    agent.router = WaitingRouter()
    task = asyncio.create_task(agent.run("Inspect", ignore, deny))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    result = await asyncio.wait_for(task, 2)
    assert result.status == "cancelled" and not agent.running
    assert not executor.requests
    assert "execute_code" not in agent.registry.specs


async def test_denied_action_halts_batch_even_if_program_catches_error(settings, tmp_path):
    agent, executor = agent_for(
        settings,
        """
try:
    await call_tool("write_file", {"path": "denied", "content": "bad"})
except Exception:
    pass
await call_tool("write_file", {"path": "bypass", "content": "bad"})
""",
    )
    previews = []

    async def review(name, preview):
        previews.append((name, preview))
        return False

    await agent.run("Write files", ignore, review)
    assert len(previews) == 1 and previews[0][0] == "write_file"
    assert "+bad" in previews[0][1]
    assert not (tmp_path / "denied").exists() and not (tmp_path / "bypass").exists()
    assert not result_before(executor)["ok"]
    assert "User denied" in result_before(executor)["error"]["message"]


async def test_hooks_modify_nested_arguments_before_approval_and_check_diff(settings, tmp_path):
    agent, executor = agent_for(
        settings,
        """
await call_tool("write_file", {"path": "file", "content": "original"})
""",
    )

    async def hook(event, payload):
        if event == "before_tool" and payload["tool"] == "write_file":
            return {"arguments": {"path": "file", "content": "changed by hook"}}

    agent.middleware.add(hook)

    async def review(name, preview):
        assert name == "write_file" and "+changed by hook" in preview
        (tmp_path / "file").write_text("user edit during approval")
        return True

    await agent.run("Write file", ignore, review)
    assert (tmp_path / "file").read_text() == "user edit during approval"
    assert "changed while approval" in result_before(executor)["error"]["message"]


@pytest.mark.parametrize(
    "code",
    [
        'await call_tool("write_file", {"path": "bad", "content": "x"})',
        'await call_tool("run_command", {"argv": ["touch", "bad"]})',
        'await call_tool("read_file", {"path": "../outside"})',
        'await call_tool("read_file", {"path": ".env"})',
        'await call_tool("read_file", {"path": 3})',
        'await call_tool("execute_code", {"code": "1"})',
        'await call_tool("finish", {"summary": "bypass"})',
        'open("bad", "w").write("x")',
        'import os\nos.environ["OPENROUTER_API_KEY"]',
        'import subprocess\nsubprocess.run(["touch", "bad"])',
    ],
)
async def test_sandbox_and_tool_policy_reject_unavailable_access(settings, tmp_path, code):
    agent, executor = agent_for(settings, code, read_only=True)
    await agent.run("Inspect", ignore, deny)
    assert not result_before(executor)["ok"]
    assert not (tmp_path / "bad").exists()
    prompt = executor.requests[0][0][0]["content"]
    assert '"name":"write_file"' not in prompt


async def test_extensions_available_and_tool_failure_stops_later_actions(settings, tmp_path):
    agent, executor = agent_for(
        settings,
        """
value = await call_tool("mcp__example__add", {"a": 2, "b": 3})
await call_tool("write_file", {"path": "first", "content": str(value["output"])})
await call_tool("read_file", {"path": "missing"})
await call_tool("write_file", {"path": "last", "content": "bad"})
""",
        auto_approve=True,
    )

    async def add(args):
        return ToolResult(True, str(args["a"] + args["b"]))

    agent.registry.register(
        ToolSpec(
            "mcp__example__add",
            "Add",
            object_schema(
                {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                ["a", "b"],
            ),
        ),
        add,
    )
    await agent.run("Compute and save", ignore, deny)
    assert (tmp_path / "first").read_text() == "5"
    assert not (tmp_path / "last").exists()
    assert not result_before(executor)["ok"]
    assert '"name":"mcp__example__add"' in executor.requests[0][0][0]["content"]


@pytest.mark.parametrize("direct", [False, True])
async def test_removed_bulk_read_is_rejected_in_code_mode(settings, direct):
    agent, executor = agent_for(
        settings,
        'await call_tool("read_files", {})',
    )
    if direct:
        executor.completions = iter(
            [
                Completion(calls=[ToolCall("removed", "read_files", "{}")]),
                Completion(content="Done"),
            ]
        )
    events = []

    async def emit(event):
        events.append(event)

    result = await agent.run("Inspect", emit, deny)
    assert result.status == "completed"
    assert "read_files" not in executor.requests[0][0][0]["content"]
    assert not result_before(executor)["ok"]
    expected = (
        "Return a call to one of: execute_code, finish"
        if direct
        else "Tool is not available: read_files"
    )
    assert expected in result_before(executor)["error"]["message"]
    assert not any(event.kind == "tool_start" and event.text == "read_files" for event in events)


async def test_repeat_guards_span_programs_but_reset_for_new_task(settings):
    code = 'await call_tool("write_file", {"path": "file", "content": "source"})'
    agent, executor = agent_for(
        settings,
        code,
        code,
        code,
        auto_approve=True,
    )
    await agent.run("Inspect", ignore, deny)
    assert result_before(executor, 1)["ok"] and result_before(executor, 2)["ok"]
    assert "Repeated identical action blocked" in result_before(executor, 3)["error"]["message"]
    failed_call = result_before(executor, 3)["output"]["tools"]
    assert len(failed_call) == 1
    assert failed_call[0]["tool"] == "write_file" and not failed_call[0]["ok"]
    assert failed_call[0]["error"]["code"] == "repeated_action"
    agent.executor = Executor(code)
    await agent.run("Inspect again", ignore, deny)
    assert result_before(agent.executor)["ok"]
    agent.executor = Executor("""
for i in range(3):
    await call_tool("write_file", {"path": "file", "content": "source"})
""")
    await agent.run("Repeat", ignore, deny)
    assert "Repeated identical action blocked" in result_before(agent.executor)["error"]["message"]


async def test_repeated_programs_can_read_patch_and_verify_current_contents(settings, tmp_path):
    file = tmp_path / "counter.txt"
    file.write_text("0")
    code = """
before = await call_tool("read_file", {"path": "counter.txt"})
value = before["output"]["content"].split(": ")[1]
await call_tool("edit_file", {
    "path": "counter.txt", "old_text": value, "new_text": str(int(value) + 1)
})
after = await call_tool("read_file", {"path": "counter.txt"})
after["output"]["content"]
"""
    agent, executor = agent_for(settings, code, code, code, auto_approve=True)
    result = await agent.run("Increment and verify three times", ignore, deny)
    assert result.status == "completed"
    assert file.read_text() == "3"
    for index in range(1, 4):
        reply = result_before(executor, index)
        assert reply["ok"]
        assert reply["output"]["output"] == f"1: {index}"


async def test_repeated_failed_reads_are_still_blocked(settings):
    agent, executor = agent_for(
        settings,
        *(f'{name} = await call_tool("read_file", {{"path": "missing"}})' for name in "abc"),
    )
    await agent.run("Inspect missing file", ignore, deny)
    assert "File not found" in result_before(executor, 1)["error"]["message"]
    assert "File not found" in result_before(executor, 2)["error"]["message"]
    assert "Repeated identical action blocked" in result_before(executor, 3)["error"]["message"]
    failed_call = result_before(executor, 3)["output"]["tools"]
    assert len(failed_call) == 1
    assert failed_call[0]["tool"] == "read_file" and not failed_call[0]["ok"]
    assert failed_call[0]["error"]["code"] == "repeated_action"


async def test_direct_file_access_error_explains_tool_bridge_and_agent_can_recover(
    settings, tmp_path
):
    file = tmp_path / "file.py"
    file.write_text("source")
    agent, executor = agent_for(
        settings,
        f"open({str(file)!r}).read()",
        'await call_tool("read_file", {"path": "file.py"})',
    )
    result = await agent.run("Read file.py", ignore, deny)
    assert result.status == "completed"
    denied = result_before(executor, 1)
    assert not denied["ok"]
    error = denied["output"]["error"]
    assert "Code Mode cannot access host files directly" in error
    assert 'await call_tool("read_file"' in error
    recovered = result_before(executor, 2)
    assert recovered["ok"] and "1: source" in recovered["output"]["output"]["output"]["content"]


async def test_cancellation_during_nested_approval_cleans_up_and_keeps_history(settings, tmp_path):
    agent, _ = agent_for(
        settings,
        """
await call_tool("write_file", {"path": "first", "content": "x"})
await call_tool("write_file", {"path": "second", "content": "x"})
""",
    )
    entered, exited = asyncio.Event(), asyncio.Event()

    async def review(*_):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            exited.set()

    task = asyncio.create_task(agent.run("Write files", ignore, review))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    result = await asyncio.wait_for(task, 5)
    assert result.status == "cancelled" and not agent.running
    assert exited.is_set()
    assert not (tmp_path / "first").exists() and not (tmp_path / "second").exists()
    assert agent.history[0][-2]["role"] == "tool"
    assert "cancelled" in agent.history[0][-2]["content"]


@pytest.mark.parametrize("code", ["while True:\n    pass", '"x" * 200_000_000'])
async def test_computation_and_memory_are_bounded(settings, monkeypatch, code):
    monkeypatch.setattr("agent_tui.codemode.EXECUTION_SECONDS", 0.05)
    result = await asyncio.wait_for(execute_code(code, settings, ignore), 5)
    assert not result.ok
    assert "error" in json.loads(result.content)


async def test_call_budget_and_print_output_are_bounded(settings, monkeypatch):
    monkeypatch.setattr("agent_tui.codemode.MAX_TOOL_CALLS", 2)
    calls = []

    async def tool(name, arguments):
        calls.append(arguments)
        return ToolResult(True, "true")

    result = await execute_code(
        """
for i in range(10):
    await call_tool("example", {"i": i})
""",
        settings,
        tool,
    )
    assert not result.ok and len(calls) == 2
    assert "tool-call limit" in result.content
    result = await execute_code(
        'print("x" * 10000)', replace(settings, max_output_chars=1000), tool
    )
    assert not result.ok


async def test_returned_output_is_bounded_and_redacted(settings):
    agent, executor = agent_for(
        settings, f'"{settings.api_key}" + "x" * 10000', max_output_chars=1000
    )
    await agent.run("Test output", ignore, deny)
    raw = result_before(executor)["output"]
    assert settings.api_key not in json.dumps(raw)
    assert len(json.dumps(raw, ensure_ascii=False)) <= 1000
    assert raw["truncated"]


async def test_gathered_calls_are_serial_and_stop_after_denial(settings, tmp_path):
    agent, executor = agent_for(
        settings,
        """
import asyncio
await asyncio.gather(
    call_tool("write_file", {"path": "first", "content": "x"}),
    call_tool("write_file", {"path": "second", "content": "x"}),
)
""",
    )
    reviewed = []

    async def review(name, preview):
        reviewed.append(name)
        await asyncio.sleep(0)
        return False

    await agent.run("Write files", ignore, review)
    assert reviewed == ["write_file"]
    assert not result_before(executor)["ok"]
    assert not (tmp_path / "first").exists() and not (tmp_path / "second").exists()


async def test_direct_catalog_calls_are_batched_without_model_correction_rounds(settings, tmp_path):
    agent, executor = agent_for(settings)
    content = 'A quote: "\\nawait call_tool("run_command", {})\nUnicode: 🐍'
    executor.completions = iter(
        [
            Completion(
                calls=[
                    ToolCall("one", "write_file", json.dumps({"path": "file", "content": content})),
                    ToolCall("two", "read_file", json.dumps({"path": "file"})),
                ]
            ),
            Completion(content="Done"),
        ]
    )
    reviews = []

    async def review(name, preview):
        reviews.append(name)
        return True

    result = await agent.run("Write and verify", ignore, review)
    assert result.status == "completed" and len(executor.requests) == 2
    assert (tmp_path / "file").read_text() == content
    assert reviews == ["write_file"]
    output = result_before(executor)["output"]
    assert [tool["tool"] for tool in output["tools"]] == ["write_file", "read_file"]
    messages = executor.requests[1][0]
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "execute_code"
    assert messages[-1]["tool_call_id"] == "one"


async def test_direct_call_preflight_rejects_invalid_batch_without_partial_effects(
    settings, tmp_path
):
    agent, executor = agent_for(settings, auto_approve=True)
    executor.completions = iter(
        [
            Completion(
                calls=[
                    ToolCall("one", "write_file", '{"path":"file","content":"x"}'),
                    ToolCall("two", "read_file", '{"path":3}'),
                ]
            ),
            Completion(content="Stopped"),
        ]
    )
    result = await agent.run("Invalid batch", ignore, deny)
    assert result.status == "completed"
    assert not (tmp_path / "file").exists()
    replies = [m for m in executor.requests[1][0] if m["role"] == "tool"]
    assert len(replies) == 2
    assert all(not json.loads(reply["content"])["ok"] for reply in replies)


@pytest.mark.parametrize(
    "invalid_code",
    [
        'call_tool("example", {})',
        'reply = call_tool("example", {})\nreply["output"]',
        'async def main():\n    return await call_tool("example", {})\nmain()',
        'await call_tool("example", path="file")',
        'await call_tool("example", "{}")',
        'await read_file({"path": "file"})',
        'reply = await call_tool("example", {})\nreply["content"]',
        '{"flag": true}',
        '{"flag": false}',
        '{"value": null}',
        "import subprocess",
        "import requests",
        'import asyncio\nasyncio.create_task(call_tool("example", {}))',
        "if True\n    pass",
    ],
)
async def test_generated_code_is_checked_before_any_tool_can_run(settings, invalid_code):
    called = []

    async def tool(name, arguments):
        called.append(name)
        return ToolResult(True, "{}")

    # Even an error after a valid mutating call must be rejected before that call.
    result = await execute_code(
        'await call_tool("write_file", {"path": "file", "content": "changed"})\n' + invalid_code,
        settings,
        tool,
    )
    assert not result.ok and called == []
    payload = json.loads(result.content)
    assert payload["phase"] == "validation" and payload["tools"] == []
    assert "main.py:" in payload["error"] or "SyntaxError" in payload["error"]
    assert "No tools ran" in payload["retry_hint"]


async def test_markdown_code_is_rejected_with_actionable_feedback(settings):
    result = await execute_code(
        '```python\nawait call_tool("list_files", {})\n```', settings, ignore
    )
    payload = json.loads(result.content)
    assert not result.ok and payload["phase"] == "validation"
    assert "without Markdown fences" in payload["retry_hint"]


async def test_agent_recovers_from_validation_error_without_replaying_a_write(settings, tmp_path):
    agent, executor = agent_for(
        settings,
        'await call_tool("write_file", {"path": "bad", "content": "bad"})\n'
        'reply = call_tool("read_file", {"path": "bad"})\nreply["output"]',
        'await call_tool("write_file", {"path": "good", "content": "verified"})\n'
        'reply = await call_tool("read_file", {"path": "good"})\nreply["output"]',
        auto_approve=True,
    )
    agent.router = CodeDecisions(
        RouteDecision(tool="execute_code"),
        RouteDecision(tool="execute_code"),
        RouteDecision(tool="finish"),
    )
    result = await agent.run("Write and verify", ignore, deny)
    assert result.status == "completed"
    assert not (tmp_path / "bad").exists()
    assert (tmp_path / "good").read_text() == "verified"
    failed = result_before(executor, 1)["output"]
    assert failed["phase"] == "validation" and failed["tools"] == []
    recovered = result_before(executor, 2)["output"]
    assert recovered["output"]["content"] == "1: verified"
    for messages, _, selected in executor.requests:
        prompt = messages[0]["content"]
        assert TOOL_BRIDGE_STUBS in prompt
        assert (
            "For this response, call execute_code exactly once"
            if selected == "execute_code"
            else "For this response, finish with a concise summary"
        ) in prompt


async def test_prompt_inspection_example_runs_against_real_tool_shapes(settings, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "example.py").write_text("example = True\n")
    code = CODE_PROMPT.split(
        "Example Python source for bounded inspection in one host tool call:\n", 1
    )[1]
    agent, executor = agent_for(settings, code)
    result = await agent.run("Inspect", ignore, deny)
    assert result.status == "completed" and result_before(executor)["ok"]
    output = result_before(executor)["output"]["output"]
    assert output["files"][0]["content"] == "example = True\n"
    assert output["files"][0]["sha256"]
    assert "src/example.py" in agent.context.workspace_sources
    assert output["truncated"] is False


async def test_type_checked_data_processing_and_async_helpers_remain_available(settings):
    async def tool(name, arguments):
        return ToolResult(True, '{"text":"item 9"}')

    result = await execute_code(
        """
import json
import math
import re

async def extract():
    reply = await call_tool("example", {"flag": True, "value": None})
    digits = re.findall("[0-9]+", reply["output"]["text"])
    return {"root": math.sqrt(int(digits[0])), "flag": False}

json.dumps(await extract())
""",
        settings,
        tool,
    )
    assert result.ok, result.content
    assert json.loads(json.loads(result.content)["output"]) == {"root": 3.0, "flag": False}


async def test_runtime_error_reports_completed_actions_for_safe_recovery(settings, tmp_path):
    agent, executor = agent_for(
        settings,
        'await call_tool("write_file", {"path": "first", "content": "kept"})\n'
        'reply = await call_tool("list_files", {})\nreply["output"]["missing_key"]',
        auto_approve=True,
    )
    await agent.run("Inspect", ignore, deny)
    failed = result_before(executor)["output"]
    assert failed["phase"] == "runtime"
    assert failed["tools"] == [
        {"tool": "write_file", "ok": True},
        {"tool": "list_files", "ok": True},
    ]
    assert (tmp_path / "first").read_text() == "kept"
    assert "not rolled back" in failed["retry_hint"]


async def test_truncated_file_output_can_still_be_consumed_by_generated_code(settings, tmp_path):
    (tmp_path / "large.txt").write_text(('quoted "text" and slashes \\\\ 🙂\n') * 200)
    agent, executor = agent_for(
        settings,
        'reply = await call_tool("read_file", {"path": "large.txt"})\n'
        '{"excerpt": reply["output"]["content"][:30], '
        '"truncated": reply["output"]["truncated"], '
        '"has_more": reply["output"]["has_more"]}',
        max_output_chars=1000,
    )
    await agent.run("Inspect large file", ignore, deny)
    reply = result_before(executor)
    assert reply["ok"]
    output = reply["output"]["output"]
    assert output["excerpt"].startswith('1: quoted "text"')
    assert output["truncated"] and output["has_more"]


@pytest.mark.parametrize("fragment", ['"\\\n🙂', "\0"])
async def test_truncated_program_error_keeps_recovery_metadata_and_valid_json(settings, fragment):
    async def tool(name, arguments):
        return ToolResult(False, "Failure with escaped characters: " + fragment * 1000)

    result = await execute_code(
        'await call_tool("example", {})', replace(settings, max_output_chars=1000), tool
    )
    assert not result.ok and len(result.content) <= 1000
    payload = json.loads(result.content)
    assert payload["truncated"] and payload["phase"] == "runtime"
    assert "not rolled back" in payload["retry_hint"]
    assert payload["tool_calls"] == 1


async def test_oversized_direct_call_batch_rejected_before_execution(settings, tmp_path):
    agent, executor = agent_for(settings, auto_approve=True)
    executor.completions = iter(
        [
            Completion(
                calls=[
                    ToolCall(str(i), "write_file", json.dumps({"path": str(i), "content": "x"}))
                    for i in range(MAX_TOOL_CALLS + 1)
                ]
            ),
            Completion(content="Stopped"),
        ]
    )
    result = await agent.run("Write files", ignore, deny)
    assert result.status == "completed"
    assert not (tmp_path / "0").exists()
    replies = [m for m in executor.requests[1][0] if m["role"] == "tool"]
    assert len(replies) == MAX_TOOL_CALLS + 1
    assert all(not json.loads(reply["content"])["ok"] for reply in replies)


@pytest.mark.parametrize("content", ['{"nested":{"items":[1,true,null]}}', "plain text"])
async def test_chat_reply_and_sandbox_bridge_share_the_same_response_shape(settings, content):
    registry = ToolRegistry(settings)

    async def handler(arguments):
        return ToolResult(True, content)

    registry.register(ToolSpec("example", "Example", object_schema({})), handler)
    tool_result = await registry.execute("example", {})
    from agent_tui.defaults import DefaultResponses

    message = DefaultResponses().message(ToolCall("example", "example", "{}"), tool_result)
    chat_reply = json.loads(message["content"])
    code_result = await execute_code('await call_tool("example", {})', settings, registry.execute)
    assert code_result.ok, code_result.content
    sandbox_reply = code_result.as_dict()["output"]["output"]
    assert sandbox_reply == chat_reply
    assert set(chat_reply) == {"ok", "output", "error", "truncated"}


async def test_failed_nested_tool_keeps_its_error_code_and_partial_output(settings):
    agent, executor = agent_for(settings, 'await call_tool("example", {})')

    async def handler(arguments):
        return ToolResult(
            False,
            '{"completed":2,"remaining":1}',
            ToolFailure("partial_failure", "The last item failed.", "Inspect completed items."),
        )

    agent.registry.register(ToolSpec("example", "Example", object_schema({})), handler)
    await agent.run("Inspect", ignore, deny)
    reply = result_before(executor)
    assert not reply["ok"] and reply["error"]["code"] == "partial_failure"
    assert reply["error"]["retry_hint"] == "Inspect completed items."
    nested = reply["output"]["tools"][0]
    assert nested["error"] == reply["error"]
    assert nested["output"] == {"completed": 2, "remaining": 1}


async def test_code_mode_reuses_one_monty_pool_and_repository_context_between_programs(
    settings, tmp_path, monkeypatch
):
    from agent_tui import defaults

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original_pool = defaults.AsyncMonty
    constructed = []

    def make_pool(*args, **kwargs):
        pool = original_pool(*args, **kwargs)
        constructed.append(pool)
        return pool

    monkeypatch.setattr(defaults, "AsyncMonty", make_pool)
    agent, executor = agent_for(
        settings,
        'result = await call_tool("context_collect", {"paths": ["source.py"]})\\n'
        'result["output"]',
        'result = await call_tool("apply_patchset", {"changes": ['
        '{"action": "edit", "path": "source.py", "old_text": "VALUE = 1", '
        '"new_text": "VALUE = 2", "expected_sha256": "' + digest + '"}, '
        '{"action": "create", "path": "new.py", "content": "NEW = True\\\\n"}'
        ']})\\nresult["output"]',
        'result = await call_tool("context_collect", {"paths": ["source.py", "new.py"]})\\n'
        'result["output"]',
        auto_approve=True,
    )
    result = await agent.run("Refactor VALUE", ignore, deny)
    assert result.status == "completed", result.message
    assert len(constructed) == 1
    assert agent.code_runtime._pool is None
    assert source.read_text() == "VALUE = 2\\n"
    assert (tmp_path / "new.py").read_text() == "NEW = True\\n"
    # The next model invocation receives a verified source cache reference.
    second_system = executor.requests[1][0][0]["content"]
    assert "Repository source cache" in second_system
    assert "source.py" in second_system
    assert "VALUE = 1" in second_system
    assert "VALUE = 2" in agent.context.workspace_context("VALUE")
