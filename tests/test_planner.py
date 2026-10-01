from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace

import httpx
import pytest
from harness_router import HarnessState

from agent_tui.agent import Agent
from agent_tui.models import Completion, ToolCall
from agent_tui.planner import Plan, Planner, PlanningError, PredictedAction, parse_plan
from agent_tui.routing import HarnessDecisionLayer
from agent_tui.tools import ToolRegistry


def plan_completion(**overrides):
    raw = {
        "summary": "Inspect and explain",
        "steps": ["Read file", "Report"],
        "paths": [
            [
                {"tool": "read_file", "outcome": "Source inspected", "value": 0.3},
                {"tool": "finish", "outcome": "Accurate answer", "value": 1.0},
            ],
            [{"tool": "finish", "outcome": "Uninformed answer", "value": 0.0}],
        ],
    }
    raw.update(overrides)
    return Completion(
        calls=[ToolCall("plan", "submit_plan", json.dumps(raw))], model="free/model", tokens=50
    )


async def test_planner_uses_same_executor_with_a_validated_plan(settings):
    requests = []

    class Executor:
        async def complete(self, messages, schemas, selected, on_token):
            requests.append((messages, schemas, selected))
            return plan_completion()

    state = HarnessState(goal="Explain file", observation="Not read yet")
    plan = await Planner(Executor()).plan(state, ToolRegistry(settings).descriptors())
    assert plan.paths[0][0].tool == "read_file"
    assert plan.tokens == 50
    assert requests[0][2] == "submit_plan"
    assert json.loads(requests[0][0][-1]["content"])["state"]["observation"] == "Not read yet"


@pytest.mark.parametrize(
    "paths",
    [
        [[{"tool": "delete_everything", "outcome": "x", "value": 0}]],
        [[{"tool": "read_file", "outcome": "x", "value": float("nan")}]],
        [[{"tool": "read_file", "outcome": "x", "value": 0}]] * 2,
        [
            [
                {"tool": "finish", "outcome": "x", "value": 0},
                {"tool": "read_file", "outcome": "x", "value": 1},
            ]
        ],
        [],
    ],
)
def test_invalid_predictions_rejected(settings, paths):
    with pytest.raises(PlanningError):
        parse_plan(plan_completion(paths=paths), ToolRegistry(settings).descriptors(), 3)


async def test_real_mcts_looks_ahead_without_executing_tools(settings, tmp_path):
    settings = replace(settings, api_key="", mcts_simulations=128, mcts_depth=3)
    plan = Plan(
        "Compare approaches",
        ("Inspect before answering",),
        (
            (
                PredictedAction("read_file", "Inspected evidence", 0.1),
                PredictedAction("write_file", "Correct file written", 0.8),
                PredictedAction("finish", "Task done", 1.0),
            ),
            (PredictedAction("finish", "Stopped without completing the task", 0.2),),
        ),
    )
    state = HarnessState(goal="Fix a file", observation="Need to inspect first")
    original = copy.deepcopy(state)
    layer = HarnessDecisionLayer(settings)
    result = await layer.route_mcts(state, ToolRegistry(settings).descriptors(), plan)
    assert result.decision.tool == "read_file"
    assert result.principal_variation == ("read_file", "write_file", "finish")
    assert sum(result.root_visits.values()) == 128
    assert result.policy_evaluations == 0
    assert state == original
    assert list(tmp_path.iterdir()) == []


async def test_mcts_uses_at_most_one_real_jev_prior(settings):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "read_file",
                        "confidence": 0.99,
                        "probabilities": {"read_file": 0.99, "finish": 0.01},
                    }
                }
            },
        )

    tools = ToolRegistry(settings).descriptors()
    plan = parse_plan(plan_completion(), tools, 3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        layer = HarnessDecisionLayer(settings, client)
        search = await layer.route_mcts(HarnessState(goal="Explain"), tools, plan)
    assert search.decision.tool == "read_file"
    assert search.policy_evaluations == 1
    assert len(requests) == 1


async def test_agent_replans_with_actual_results_and_only_executes_first_tool(settings, tmp_path):
    (tmp_path / "file").write_text("verified fact")
    planned_states, events = [], []
    tools = ToolRegistry(settings).descriptors()

    class TestPlanner:
        async def plan(self, state, catalog):
            planned_states.append(copy.deepcopy(state))
            if state.last_action:
                return Plan(
                    "Finish",
                    ("Report observed evidence",),
                    ((PredictedAction("finish", "Answer", 1.0),),),
                )
            return parse_plan(plan_completion(), tools, 3)

    class Executor:
        async def complete(self, messages, schemas, selected, on_token):
            if selected == "read_file":
                return Completion(calls=[ToolCall("read", "read_file", '{"path":"file"}')])
            assert "verified fact" in messages[-1]["content"]
            return Completion(calls=[ToolCall("done", "finish", '{"summary":"verified fact"}')])

    async def emit(event):
        events.append(event)

    async def deny(*_):
        return False

    config = replace(settings, api_key="", routing="mcts")
    agent = Agent(config, HarnessDecisionLayer(config), Executor(), planner=TestPlanner())
    result = await agent.run("Explain file", emit, deny)
    assert result.status == "completed"
    assert len(planned_states) == 2
    assert "verified fact" in planned_states[1].observation
    assert [event.text for event in events if event.kind == "tool_start"] == ["read_file", "finish"]
    assert [event.kind for event in events].count("mcts") == 2


@pytest.mark.parametrize("planner_fails", [False, True])
async def test_jev_planning_and_routing_overlap_without_duplicate_requests(settings, planner_fails):
    planning_started, routing_started = asyncio.Event(), asyncio.Event()
    requests, events = [], []

    class TestPlanner:
        async def plan(self, state, tools):
            planning_started.set()
            await asyncio.wait_for(routing_started.wait(), 1)
            if planner_fails:
                raise PlanningError("Unavailable plan")
            return Plan("Answer", ("Report",), ((PredictedAction("finish", "Done", 1),),))

    class Router:
        async def route(self, state, tools):
            from harness_router import RouteDecision

            requests.append(state.observation)
            routing_started.set()
            await asyncio.wait_for(planning_started.wait(), 1)
            return RouteDecision(tool="finish")

    class Executor:
        async def complete(self, messages, schemas, selected, on_token):
            assert selected == "finish"
            assert any("Current tentative plan" in m.get("content", "") for m in messages) == (
                not planner_fails
            )
            return Completion(calls=[ToolCall("done", "finish", '{"summary":"Done"}')])

    async def emit(event):
        events.append(event)

    async def deny(*_):
        return False

    agent = Agent(replace(settings, routing="jev"), Router(), Executor(), planner=TestPlanner())
    result = await agent.run("Answer", emit, deny)
    assert result.status == "completed"
    assert len(requests) == 1
    assert any(event.kind == "warning" for event in events) == planner_fails


@pytest.mark.parametrize("failure", ["cancel", "planner", "router"])
async def test_parallel_decision_requests_are_cleaned_up_on_cancel_or_failure(settings, failure):
    entered = {name: asyncio.Event() for name in ("planner", "router")}
    finished = set()

    async def request(name):
        try:
            entered[name].set()
            await asyncio.gather(*(event.wait() for event in entered.values()))
            if failure == name:
                raise RuntimeError(f"{name} failed")
            await asyncio.Future()
        finally:
            finished.add(name)

    class TestPlanner:
        async def plan(self, *_):
            return await request("planner")

    class Router:
        async def route(self, *_):
            return await request("router")

    class Executor:
        async def complete(self, *_):
            pytest.fail("A cancelled or failed decision must not reach the executor")

    async def ignore(*_):
        pass

    agent = Agent(replace(settings, routing="jev"), Router(), Executor(), planner=TestPlanner())
    task = asyncio.create_task(agent.run("Wait", ignore, ignore))
    if failure == "cancel":
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 1)
        task.cancel()
    result = await asyncio.wait_for(task, 2)
    assert result.status == ("cancelled" if failure == "cancel" else "error")
    if failure != "cancel":
        assert result.message == f"{failure} failed"
    assert finished == {"planner", "router"}
    assert not agent.running
