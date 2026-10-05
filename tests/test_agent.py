from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace

import pytest
from harness_router import RouteDecision

from agent_tui.agent import Agent
from agent_tui.models import Completion, ToolCall


class FakeRouter:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.states = []
        self.catalogs = []

    async def route(self, state, tools):
        self.states.append(copy.deepcopy(state))
        self.catalogs.append([tool.name for tool in tools])
        return next(self.decisions)


class FakeExecutor:
    def __init__(self, *completions):
        self.completions = iter(completions)
        self.requests = []

    async def complete(self, messages, schemas, selected, on_token):
        self.requests.append(copy.deepcopy((messages, schemas, selected)))
        return next(self.completions)


def call(name, arguments, id="call"):
    return Completion(calls=[ToolCall(id, name, json.dumps(arguments))])


async def deny(*_):
    return False


async def allow(*_):
    return True


async def ignore(*_):
    pass


async def test_real_execution_updates_routing_and_executor_history(settings, tmp_path):
    (tmp_path / "hello.txt").write_text("hello world")
    router = FakeRouter(
        RouteDecision(tool="read_file", confidence=0.99),
        RouteDecision(tool="finish", confidence=0.99),
    )
    executor = FakeExecutor(
        call("read_file", {"path": "hello.txt"}),
        call("finish", {"summary": "The file says hello world"}, id="final"),
    )
    events = []

    async def emit(event):
        events.append(event)

    agent = Agent(settings, router, executor)
    result = await agent.run("Tell me what hello.txt contains", emit, deny)
    assert result.status == "completed"
    assert "hello world" in router.states[1].observation
    assert router.states[1].last_action == "read_file"
    messages, schemas, selected = executor.requests[1]
    assert selected == "finish"
    assert [tool["function"]["name"] for tool in schemas] == ["finish"]
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call"
    assert len(agent.history) == 1
    assert events[-1].kind == "done"


async def test_fallback_exposes_full_catalog(settings):
    router = FakeRouter(RouteDecision.fallback_to_planner("low_confidence"))
    executor = FakeExecutor(Completion(content="Hello"))
    result = await Agent(settings, router, executor).run("hi", ignore, deny)
    _, schemas, selected = executor.requests[0]
    assert selected is None
    assert {schema["function"]["name"] for schema in schemas} == {
        "list_files",
        "read_file",
        "read_files",
        "search_files",
        "write_file",
        "edit_file",
        "run_command",
        "finish",
        "delegate_tasks",
        "skill_load",
        "skill_search",
        "skill_read",
        "memory_search",
        "memory_save",
        "memory_forget",
    }
    assert result.message == "Hello"


async def test_skill_discovery_and_loading_survive_refresh_with_their_scope(settings):
    agent = Agent(settings, FakeRouter(), FakeExecutor())
    agent.skills.activate("code-review")
    agent.skills.begin_task("Debug a failing test")
    found = await agent.registry.execute("skill_search", {"query": "debug tests"})
    assert found.ok and "debug-tests" in found.content
    loaded = await agent.registry.execute("skill_load", {"name": "debug-tests"})
    assert loaded.ok
    invalid = await agent.registry.execute("skill_load", {"name": "missing-skill"})
    assert not invalid.ok
    assert "Unknown skill" in invalid.content
    agent.refresh_skills()
    assert set(agent.skills.active) == {"code-review", "debug-tests"}
    assert agent.skills.pinned == {"code-review"}
    agent.skills.begin_task("Translate newsletter into Spanish")
    assert set(agent.skills.active) == {"code-review"}
    found = await agent.registry.execute("skill_search", {"query": "debug tests"})
    assert found.ok
    agent.clear()
    assert not agent.skills.pinned
    assert not agent.skills.active


async def test_multiple_files_are_read_in_one_agent_step(settings, tmp_path):
    for name in ("one.py", "two.py", "three.py"):
        (tmp_path / name).write_text(f"Contents of {name}")
    router = FakeRouter(RouteDecision(tool="read_files"), RouteDecision(tool="finish"))
    executor = FakeExecutor(
        call(
            "read_files", {"files": [{"path": name} for name in ("one.py", "two.py", "three.py")]}
        ),
        call("finish", {"summary": "Three files inspected"}),
    )
    result = await Agent(settings, router, executor).run("Inspect these files", ignore, deny)
    assert result.status == "completed"
    assert result.steps == 2
    assert len(executor.requests) == len(router.states) == 2
    output = json.loads(json.loads(executor.requests[1][0][-1]["content"])["output"])
    assert [file["path"] for file in output["files"]] == ["one.py", "two.py", "three.py"]
    assert all(file["ok"] for file in output["files"])


async def test_whole_repository_is_read_in_one_agent_step(settings, tmp_path):
    (tmp_path / "src").mkdir()
    paths = [f"src/file-{i:02}.py" for i in range(12)] + ["README.md"]
    for name in paths:
        (tmp_path / name).write_text(f"Contents of {name}")
    router = FakeRouter(RouteDecision(tool="read_files"), RouteDecision(tool="finish"))
    executor = FakeExecutor(
        call("read_files", {}), call("finish", {"summary": "Repository inspected"})
    )
    events = []

    async def emit(event):
        events.append(event)

    result = await Agent(settings, router, executor).run("Read the repository", emit, deny)
    assert result.status == "completed" and result.steps == 2
    assert [event.text for event in events if event.kind == "tool_start"] == [
        "read_files",
        "finish",
    ]
    output = json.loads(json.loads(executor.requests[1][0][-1]["content"])["output"])
    assert {file["path"] for file in output["files"]} == set(paths)
    assert not output["truncated"]
    assert "read_files" in router.catalogs[0]
    assert "read_files" not in router.catalogs[1]


@pytest.mark.parametrize("first_arguments", [{}, {"path": "missing"}])
async def test_read_files_executes_at_most_once_even_with_different_arguments(
    settings, tmp_path, first_arguments
):
    (tmp_path / "file.py").write_text("source")
    router = FakeRouter(
        RouteDecision(tool="read_files"),
        RouteDecision(tool="read_files"),
        RouteDecision(tool="read_file"),
        RouteDecision(tool="finish"),
    )
    executor = FakeExecutor(
        call("read_files", first_arguments),
        call("read_files", {"files": [{"path": "file.py"}]}),
        call("read_file", {"path": "file.py"}),
        call("finish", {"summary": "Inspected"}),
    )
    events = []

    async def emit(event):
        events.append(event)

    result = await Agent(settings, router, executor).run("Inspect repository", emit, deny)
    assert result.status == "completed"
    assert [event.text for event in events if event.kind == "tool_start"] == [
        "read_files",
        "read_file",
        "finish",
    ]
    _, schemas, selected = executor.requests[1]
    assert selected is None
    assert "read_files" not in {schema["function"]["name"] for schema in schemas}
    assert "read_files already executed this task" in executor.requests[2][0][-1]["content"]
    assert "1: source" in executor.requests[3][0][-1]["content"]


async def test_repository_read_is_available_again_on_next_task(settings, tmp_path):
    file = tmp_path / "file.py"
    file.write_text("before")
    router = FakeRouter(*[RouteDecision(tool=name) for name in ["read_files", "finish"] * 2])
    executor = FakeExecutor(
        call("read_files", {}),
        call("finish", {"summary": "Inspected before"}),
        call("read_files", {}),
        call("finish", {"summary": "Inspected after"}),
    )
    agent = Agent(settings, router, executor)
    assert (await agent.run("Read repository", ignore, deny)).status == "completed"
    file.write_text("after")
    assert (await agent.run("Read it again", ignore, deny)).status == "completed"
    assert "1: before" in executor.requests[1][0][-1]["content"]
    assert "1: after" in executor.requests[3][0][-1]["content"]
    assert "read_files" in router.catalogs[2]


async def test_agent_omits_previous_task_skill_instructions_after_topic_change(settings):
    router = FakeRouter(RouteDecision(tool="finish"), RouteDecision(tool="finish"))
    executor = FakeExecutor(
        call("finish", {"summary": "No test failures reported"}),
        call("finish", {"summary": "Hola"}),
    )
    agent = Agent(settings, router, executor)
    result = await agent.run("Use $debug-tests to inspect failures", ignore, deny)
    assert result.status == "completed"
    assert "Active skill: debug-tests" in executor.requests[0][0][0]["content"]
    result = await agent.run("Translate hello into Spanish", ignore, deny)
    assert result.status == "completed"
    assert "Active skill: debug-tests" not in executor.requests[1][0][0]["content"]
    assert not agent.skills.active
    assert len(agent.history) == 2


def test_context_always_keeps_pinned_skill_instructions(settings):
    from agent_tui.openrouter import ExecutorError

    agent = Agent(settings, FakeRouter(), FakeExecutor())
    agent.skills.activate("code-review")
    agent.skills.begin_task("Translate a newsletter")
    messages = agent._context("Translate a newsletter", [])
    assert "Active skill: code-review" in messages[0]["content"]
    agent.context.settings = replace(settings, context_chars=300)
    with pytest.raises(ExecutorError, match="active skills"):
        agent._context("Translate a newsletter", [])


@pytest.mark.parametrize(
    "completion",
    [
        call("write_file", {"path": "changed", "content": "bad"}),
        Completion(calls=[ToolCall("a", "read_file", "not json")]),
        Completion(calls=[ToolCall("a", "read_file", '{"path":2}')]),
        Completion(calls=[ToolCall("a", "read_file", "{}"), ToolCall("b", "write_file", "{}")]),
    ],
)
async def test_invalid_or_mismatched_calls_have_no_effect(settings, tmp_path, completion):
    router = FakeRouter(RouteDecision(tool="read_file"), RouteDecision(tool="finish"))
    executor = FakeExecutor(completion, call("finish", {"summary": "Stopped"}))
    result = await Agent(settings, router, executor).run("Read a file", ignore, allow)
    assert result.status == "completed"
    assert not (tmp_path / "changed").exists()
    messages = executor.requests[1][0]
    errors = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
    assert errors and all(error["ok"] is False for error in errors)


async def test_denial_is_returned_to_model_without_writing(settings, tmp_path):
    router = FakeRouter(RouteDecision(tool="write_file"), RouteDecision(tool="finish"))
    executor = FakeExecutor(
        call("write_file", {"path": "file", "content": "new"}),
        call("finish", {"summary": "Action denied"}),
    )
    previews = []

    async def review(name, preview):
        previews.append(preview)
        return False

    result = await Agent(settings, router, executor).run("Create a file", ignore, review)
    assert result.status == "completed"
    assert not (tmp_path / "file").exists()
    assert "+new" in previews[0]
    assert "User denied" in executor.requests[1][0][-1]["content"]


async def test_file_change_during_approval_invalidates_diff(settings, tmp_path):
    file = tmp_path / "file"
    file.write_text("before")
    router = FakeRouter(RouteDecision(tool="write_file"), RouteDecision(tool="finish"))
    executor = FakeExecutor(
        call("write_file", {"path": "file", "content": "replacement"}),
        call("finish", {"summary": "Conflict"}),
    )

    async def review(*_):
        file.write_text("edited by user")
        return True

    await Agent(settings, router, executor).run("Change file", ignore, review)
    assert file.read_text() == "edited by user"
    assert "changed while approval" in executor.requests[1][0][-1]["content"]


async def test_repeated_actions_and_step_budget_stop_loop(settings):
    router = FakeRouter(*(RouteDecision(tool="list_files") for _ in range(4)))
    executor = FakeExecutor(*(call("list_files", {}) for _ in range(4)))
    result = await Agent(settings, router, executor).run("Loop", ignore, allow)
    assert result.status == "limit"
    assert "Repeated identical action blocked" in executor.requests[3][0][-1]["content"]


async def test_cancel_during_approval_keeps_tool_history_valid(settings, tmp_path):
    router = FakeRouter(RouteDecision(tool="write_file"))
    executor = FakeExecutor(call("write_file", {"path": "file", "content": "x"}))
    entered = asyncio.Event()

    async def review(*_):
        entered.set()
        await asyncio.Future()

    agent = Agent(settings, router, executor)
    task = asyncio.create_task(agent.run("Create a file", ignore, review))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    result = await task
    assert result.status == "cancelled"
    assert not agent.running
    assert not (tmp_path / "file").exists()
    assert agent.history[0][-2]["role"] == "tool"
    assert agent.history[0][-2]["tool_call_id"] == "call"


def test_context_pruning_keeps_tool_exchanges_intact(settings):
    agent = Agent(replace(settings, context_chars=2700), FakeRouter(), FakeExecutor())
    exchanges = []
    for i in range(6):
        exchange = [
            call("read_file", {"path": "file"}, str(i)).as_message(),
            {"role": "tool", "tool_call_id": str(i), "content": "x" * 450},
        ]
        exchanges.append(exchange)
    messages = agent._context("Original goal", exchanges)
    assert any(m.get("content") == "Original goal" for m in messages)
    for i, message in enumerate(messages):
        if message["role"] == "tool":
            assert messages[i - 1]["tool_calls"][0]["id"] == message["tool_call_id"]
    assert len(messages) < 14


async def test_obvious_read_skips_router_and_first_executor_call(settings, tmp_path):
    (tmp_path / "hello.txt").write_text("fast hello")
    router = FakeRouter(RouteDecision(tool="finish", confidence=0.99))
    executor = FakeExecutor(call("finish", {"summary": "fast hello"}))
    events = []

    async def emit(event):
        events.append(event)

    result = await Agent(settings, router, executor).run("Read hello.txt", emit, deny)

    assert result.status == "completed"
    assert len(router.states) == 1
    assert len(executor.requests) == 1
    assert router.states[0].last_action == "read_file"
    assert "fast hello" in router.states[0].observation
    assert any(event.kind == "extension" and "Fast path" in event.text for event in events)


async def test_direct_jev_does_not_call_planner(settings):
    class CountingPlanner:
        def __init__(self):
            self.calls = 0

        async def plan(self, state, tools):
            self.calls += 1
            raise AssertionError("planner should not run for direct Jev routing")

    planner = CountingPlanner()
    router = FakeRouter(RouteDecision(tool="finish", confidence=0.99))
    executor = FakeExecutor(call("finish", {"summary": "Done"}))
    direct = replace(settings, routing="jev", planning=True)

    result = await Agent(direct, router, executor, planner=planner).run(
        "Explain the architecture", ignore, deny
    )

    assert result.status == "completed"
    assert planner.calls == 0
    assert len(router.states) == 1
    assert len(executor.requests) == 1


async def test_edit_schedules_related_test_without_router_or_executor(settings, tmp_path):
    source = tmp_path / "src" / "agent_tui"
    source.mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir()
    (source / "sample.py").write_text("value = 1\n")
    (tests / "test_sample.py").write_text("def test_sample():\n    assert True\n")

    router = FakeRouter(
        RouteDecision(tool="edit_file", confidence=0.99),
        RouteDecision(tool="finish", confidence=0.99),
    )
    executor = FakeExecutor(
        call(
            "edit_file",
            {"path": "src/agent_tui/sample.py", "old_text": "value = 1", "new_text": "value = 2"},
        ),
        call("finish", {"summary": "Verified"}),
    )

    approvals = []

    async def approve(name, preview):
        approvals.append((name, preview))
        return True

    agent = Agent(settings, router, executor)
    # Avoid actually launching pytest; preserve the execution boundary while making the
    # deterministic verification cheap and observable.
    original_execute = agent.registry.execute

    async def execute(name, arguments):
        if name == "run_command":
            from agent_tui.tools import ToolResult

            return ToolResult(True, "1 passed")
        return await original_execute(name, arguments)

    agent.registry.execute = execute
    result = await agent.run("Change sample value", ignore, approve)

    assert result.status == "completed"
    assert len(router.states) == 2
    assert len(executor.requests) == 2
    assert any(name == "run_command" for name, _ in approvals)
    assert "run_command" in router.states[-1].recent_actions[-1].tool
