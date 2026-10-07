from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace

import pytest

from agent_tui.agent import Agent
from agent_tui.codemode import execute_code
from agent_tui.models import Completion, ToolCall
from agent_tui.tools import ToolResult, ToolSpec, object_schema


class UnusedDecisions:
    async def route(self, *_):
        pytest.fail("Code Mode must not call the router")

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
    return Agent(settings, UnusedDecisions(), executor, planner=UnusedDecisions()), executor


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
    assert json.loads(result_before(executor)["output"])["output"] == {"line_count": 2}
    assert "1: one" not in result_before(executor)["output"]
    assert all(selected is None for _, _, selected in executor.requests)
    assert {s["function"]["name"] for s in executor.requests[0][1]} == {"execute_code", "finish"}
    starts = [e for e in events if e.kind == "tool_start"]
    assert [e.text for e in starts] == ["execute_code", "list_files", "read_file", "read_file"]
    assert len({e.data["call_id"] for e in starts}) == 4
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
    assert "User denied" in result_before(executor)["output"]


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
    assert "changed while approval" in result_before(executor)["output"]


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


async def test_read_files_and_repeat_guards_span_programs_but_reset_for_new_task(settings):
    agent, executor = agent_for(
        settings,
        'await call_tool("read_files", {})',
        'await call_tool("read_files", {})',
    )
    await agent.run("Inspect", ignore, deny)
    assert "read_files already executed" in result_before(executor, 2)["output"]
    agent.executor = Executor('await call_tool("read_files", {})')
    await agent.run("Inspect again", ignore, deny)
    assert result_before(agent.executor)["ok"]
    agent.executor = Executor("""
for i in range(3):
    await call_tool("list_files", {})
""")
    await agent.run("Repeat", ignore, deny)
    assert "Repeated identical action blocked" in result_before(agent.executor)["output"]


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
    assert settings.api_key not in raw and len(raw) <= 1000
    assert json.loads(raw)["truncated"]


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
    output = json.loads(result_before(executor)["output"])
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
