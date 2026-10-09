from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace

import pytest
from harness_router import RouteDecision

from agent_tui.agent import Agent
from agent_tui.models import Completion, ToolCall
from agent_tui.planner import Planner
from agent_tui.tools import ToolError


def call(name, arguments):
    return Completion(calls=[ToolCall(name, name, json.dumps(arguments))])


async def ignore(*_):
    pass


async def deny(*_):
    return False


class Router:
    def __init__(self, *names):
        self.names = iter(names)
        self.closed = False

    async def route(self, *_):
        return RouteDecision(tool=next(self.names))

    async def aclose(self):
        self.closed = True


class Executor:
    def __init__(self, *completions):
        self.completions = iter(completions)
        self.requests = []
        self.closed = False

    async def complete(self, messages, schemas, selected, on_token):
        self.requests.append(copy.deepcopy((messages, schemas, selected)))
        return next(self.completions)

    async def aclose(self):
        self.closed = True


def parent_agent(settings, tasks, factory):
    executor = Executor(
        call("delegate_tasks", {"tasks": tasks}),
        call("finish", {"summary": "Delegated results reviewed"}),
    )
    return Agent(settings, Router("delegate_tasks", "finish"), executor, subagent_factory=factory)


def delegated_output(agent):
    reply = json.loads(agent.executor.requests[1][0][-1]["content"])
    return reply["ok"], reply["output"]["results"]


async def test_delegation_runs_concurrently_with_isolated_contexts(settings, tmp_path):
    entered = 0
    together = asyncio.Event()
    children = []

    class Reader(Executor):
        async def complete(self, messages, schemas, selected, on_token):
            nonlocal entered
            if selected == "read_file":
                entered += 1
                if entered == 2:
                    together.set()
                await asyncio.wait_for(together.wait(), 2)
            return await super().complete(messages, schemas, selected, on_token)

    def factory():
        index = len(children)
        child = Agent(
            settings,
            Router("read_file", "finish"),
            Reader(
                call("read_file", {"path": f"{index}.py"}),
                call("finish", {"summary": f"Inspected {index}.py"}),
            ),
            allow_delegation=False,
        )
        children.append(child)
        return child

    for index in range(2):
        (tmp_path / f"{index}.py").write_text(f"source {index}")
    tasks = [
        {"name": f"reader-{i}", "prompt": f"Read {i}.py", "context": f"unique-context-{i}"}
        for i in range(2)
    ]
    parent = parent_agent(settings, tasks, factory)
    events = []

    async def emit(event):
        events.append(event)

    result = await parent.run("Inspect independent files", emit, deny)
    assert result.status == "completed" and result.steps == 2
    ok, results = delegated_output(parent)
    assert ok and [result["name"] for result in results] == [task["name"] for task in tasks]
    assert all(result["status"] == "completed" and result["steps"] == 2 for result in results)
    for i, child in enumerate(children):
        assert child.context is not parent.context
        assert child.registry is not parent.registry
        assert "delegate_tasks" not in child.registry.specs
        assert child.router.closed and child.executor.closed
        history = json.dumps(child.history)
        assert f"unique-context-{i}" in history
        assert f"unique-context-{1 - i}" not in history
        assert f"source {i}" in history
        assert "not alone in the workspace" in child.executor.requests[0][0][0]["content"]
    assert len([e for e in events if e.kind == "subagent" and e.data["event"] == "done"]) == 2
    assert [e.text for e in events if e.kind == "tool_start"] == ["delegate_tasks", "finish"]
    assert parent.subagents.emit is None


@pytest.mark.parametrize("mode", ["review", "read_only", "auto_approve"])
async def test_children_preserve_policy_and_serialize_approvals(settings, tmp_path, mode):
    settings = replace(settings, read_only=mode == "read_only", auto_approve=mode == "auto_approve")
    children = []

    def factory():
        index = len(children)
        child = Agent(
            settings,
            Router("write_file", "finish"),
            Executor(
                call("write_file", {"path": f"{index}.py", "content": f"worker {index}"}),
                call("finish", {"summary": "Reported the write result"}),
            ),
            allow_delegation=False,
        )
        children.append(child)
        return child

    active = maximum = 0
    reviews = []

    async def review(name, preview):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        reviews.append((name, preview))
        try:
            await asyncio.sleep(0.01)
            return name.startswith("writer-0")
        finally:
            active -= 1

    parent = parent_agent(
        settings,
        [{"name": f"writer-{i}", "prompt": f"Own {i}.py; write worker {i}."} for i in range(2)],
        factory,
    )
    result = await parent.run("Create independent files", ignore, review)
    assert result.status == "completed"
    assert (tmp_path / "0.py").exists() == (mode != "read_only")
    assert (tmp_path / "1.py").exists() == (mode == "auto_approve")
    if mode == "review":
        assert maximum == 1 and len(reviews) == 2
        assert {name for name, _ in reviews} == {"writer-0 / write_file", "writer-1 / write_file"}
        assert "User denied" in json.dumps(children[1].history)
    else:
        assert not reviews
    if mode == "read_only":
        assert all("write_file" not in child.registry.specs for child in children)


async def test_stopping_parent_cancels_and_closes_all_children(settings):
    entered = asyncio.Event()
    started = 0
    children = []

    class BlockingExecutor(Executor):
        async def complete(self, messages, schemas, selected, on_token):
            nonlocal started
            started += 1
            if started == 2:
                entered.set()
            await asyncio.Future()

    def factory():
        child = Agent(settings, Router("finish"), BlockingExecutor(), allow_delegation=False)
        children.append(child)
        return child

    parent = parent_agent(
        settings,
        [{"name": f"worker-{i}", "prompt": "Wait for a response"} for i in range(2)],
        factory,
    )
    task = asyncio.create_task(parent.run("Delegate slow work", ignore, deny))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    result = await asyncio.wait_for(task, 2)
    assert result.status == "cancelled"
    assert not parent.running and parent.subagents.emit is None
    assert all(
        not child.running and child.executor.closed and child.router.closed for child in children
    )
    assert "Execution cancelled" in json.dumps(parent.history)


async def test_one_child_failure_does_not_discard_other_results(settings):
    created = 0

    def factory():
        nonlocal created
        created += 1
        if created == 1:
            raise RuntimeError("Failed to initialize " + settings.api_key)
        return Agent(
            settings,
            Router("finish"),
            Executor(call("finish", {"summary": "Useful independent finding"})),
            allow_delegation=False,
        )

    parent = parent_agent(
        settings,
        [
            {"name": "broken", "prompt": "First task"},
            {"name": "working", "prompt": "Second task"},
        ],
        factory,
    )
    assert (await parent.run("Delegate work", ignore, deny)).status == "completed"
    ok, results = delegated_output(parent)
    assert not ok
    assert [result["status"] for result in results] == ["error", "completed"]
    assert settings.api_key not in json.dumps(results)
    assert "[REDACTED]" in results[0]["summary"]
    assert results[1]["summary"] == "Useful independent finding"


async def test_delegated_results_stay_valid_json_when_truncated(settings):
    settings = replace(settings, max_output_chars=700)

    def factory():
        return Agent(
            settings,
            Router("finish"),
            Executor(call("finish", {"summary": (settings.api_key + 'Źródło "🙂"') * 100})),
            allow_delegation=False,
        )

    parent = parent_agent(
        settings, [{"name": f"worker-{i}", "prompt": "Report findings"} for i in range(3)], factory
    )
    assert (await parent.run("Summarize results", ignore, deny)).status == "completed"
    ok, results = delegated_output(parent)
    assert ok and len(results) == 3
    assert all(result["truncated"] for result in results)
    output = json.loads(parent.executor.requests[1][0][-1]["content"])["output"]
    encoded = json.dumps(output, ensure_ascii=False)
    assert len(encoded) <= 700 and settings.api_key not in encoded
    assert json.loads(parent.executor.requests[1][0][-1]["content"])["truncated"]


def test_delegation_schema_enforces_batch_limits(settings):
    registry = Agent(settings, Router(), Executor()).registry
    for tasks in ([], [{"name": "x", "prompt": "task"}] * 4, [{"name": "x"}]):
        with pytest.raises(ToolError):
            registry.validate("delegate_tasks", json.dumps({"tasks": tasks}))


async def test_delegation_requires_active_run_and_distinct_names(settings):
    parent = parent_agent(
        settings,
        [
            {"name": "same", "prompt": "task"},
            {"name": "same", "prompt": "other"},
        ],
        lambda: pytest.fail("Invalid delegation must not start a child"),
    )
    outside = await parent.registry.execute(
        "delegate_tasks",
        {
            "tasks": [{"name": "worker", "prompt": "task"}],
        },
    )
    assert not outside.ok and "active task" in outside.content
    assert (await parent.run("Delegate invalid work", ignore, deny)).status == "completed"
    assert "distinct name" in parent.executor.requests[1][0][-1]["content"]


async def test_default_factory_inherits_settings_skills_and_middleware(settings):
    parent = Agent(settings, Router(), Executor(), planner=Planner(Executor()))
    parent.skills.activate("code-review")
    parent.middleware.add(ignore)
    child = parent._new_subagent()
    try:
        assert child.settings is parent.settings
        assert child.router is not parent.router and child.executor is not parent.executor
        assert child.planner is not parent.planner
        assert child.subagents is None and "delegate_tasks" not in child.registry.specs
        assert "code-review" in child.skills.pinned
        assert child.middleware.handlers == [ignore]
        assert child.middleware is not parent.middleware
    finally:
        await child.aclose()
    assert child.executor.client.is_closed
