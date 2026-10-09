from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest
from harness_router import RouteDecision
from textual.widgets import Input, Markdown, Static

from agent_tui.agent import Agent
from agent_tui.models import AgentEvent, Completion, ToolCall
from agent_tui.tui import AgentApp, ApprovalScreen


async def test_offline_demo_completes_planning_mcts_and_narrow_layout(settings):
    app = AgentApp(replace(settings, demo=True))
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert not app._busy
        assert app.agent.history
        assert any(role == "PLAN" for role, _ in app.transcript)
        assert any(role == "MCTS" for role, _ in app.transcript)
        assert "Completed" in str(app.query_one("#phase", Static).render())
        assert not app.query_one("#sidebar").display
        await pilot.press("ctrl+o")
        assert app.query_one("#sidebar").display
        await pilot.resize_terminal(80, 24)
        assert not app.query_one("#sidebar").display
        await pilot.resize_terminal(120, 42)
        assert app.query_one("#sidebar").display
        await pilot.press("ctrl+o")
        assert not app.query_one("#sidebar").display
        await pilot.press("ctrl+n")
        assert not app.agent.history
        assert not app.transcript
        await pilot.press("ctrl+c")


async def test_code_mode_demo_renders_each_nested_result_without_planning(settings):
    app = AgentApp(replace(settings, demo=True, code_mode=True))
    async with app.run_test(size=(120, 42)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert not app._busy
        assert "Completed" in str(app.query_one("#phase", Static).render())
        assert "Code Mode" in str(app.query_one("#routing-mode", Static).render())
        assert "Jev" in str(app.query_one("#routing-mode", Static).render())
        assert "offline/demo" in str(app.query_one("#router-model", Static).render())
        assert "finish" in str(app.query_one("#next-tool", Static).render())
        assert not any(role in {"PLAN", "MCTS"} for role, _ in app.transcript)
        assert [role for role, _ in app.transcript if role.startswith("TOOL ")] == [
            "TOOL list_files",
            "TOOL execute_code",
            "TOOL finish",
        ]
        assert len(app._tool_cards) == 3
        assert all("done" in str(card.title) for card, _, _ in app._tool_cards.values())


async def test_failed_code_is_expanded_with_its_own_result(settings):
    app = AgentApp(replace(settings, code_mode=True))
    async with app.run_test():
        assert settings.router_model in str(app.query_one("#router-model", Static).render())
        for name, call_id in [("execute_code", 1), ("read_file", 2)]:
            await app._event(
                AgentEvent("tool_start", name, step=1, data={"call_id": call_id, "arguments": {}})
            )
        await app._event(
            AgentEvent(
                "tool_result",
                "read failed",
                step=1,
                data={"tool": "read_file", "call_id": 2, "ok": False},
            )
        )
        await app._event(
            AgentEvent(
                "tool_result",
                "program stopped",
                step=1,
                data={"tool": "execute_code", "call_id": 1, "ok": False},
            )
        )
        outer, outer_body, _ = app._tool_cards[(1, 1)]
        inner, inner_body, _ = app._tool_cards[(1, 2)]
        assert not outer.collapsed and not inner.collapsed
        assert "program stopped" in str(outer_body.render())
        assert "read failed" in str(inner_body.render())


async def test_missing_key_is_reported_without_starting_a_run(settings):
    app = AgentApp(replace(settings, api_key=""))
    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "Hello"
        await pilot.press("enter")
        assert not app._busy
        assert not app.agent.history
        assert "OPENROUTER_API_KEY" in app.transcript[-1][1]


async def test_approval_modal_denies_a_write(settings, tmp_path):
    class Router:
        async def route(self, state, tools):
            return RouteDecision(tool="write_file" if not state.last_action else "finish")

    class Executor:
        async def complete(self, messages, schemas, selected, on_token):
            args = (
                '{"path":"file","content":"should not be written"}'
                if selected == "write_file"
                else '{"summary":"Denied"}'
            )
            return Completion(calls=[ToolCall("call", selected, args)])

    agent = Agent(settings, Router(), Executor())
    app = AgentApp(settings, agent=agent)
    async with app.run_test(size=(120, 42)) as pilot:
        await app.submit_goal("Write a file")
        for _ in range(50):
            if isinstance(app.screen, ApprovalScreen):
                break
            await pilot.pause(0.01)
        assert isinstance(app.screen, ApprovalScreen)
        await pilot.pause()  # The modal is pushed before its first layout is complete.
        await pilot.click("#deny")
        await app.workers.wait_for_complete()
        assert not (tmp_path / "file").exists()
        assert not app._busy


async def test_stop_cancels_in_flight_executor_and_restores_input(settings):
    entered = asyncio.Event()

    class Router:
        async def route(self, state, tools):
            return RouteDecision(tool="finish")

    class Executor:
        async def complete(self, *_):
            entered.set()
            await asyncio.Future()

    app = AgentApp(settings, agent=Agent(settings, Router(), Executor()))
    async with app.run_test() as pilot:
        await app.submit_goal("Wait")
        await asyncio.wait_for(entered.wait(), 2)
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        assert not app._busy
        assert not app.query_one("#prompt", Input).disabled
        assert any(role == "CANCELLED" for role, _ in app.transcript)


@pytest.mark.parametrize("terminal", ["usage", "cancelled", "error"])
async def test_stream_rendering_is_batched_and_flushes_last_tokens(settings, monkeypatch, terminal):
    now = [100.0]
    monkeypatch.setattr("agent_tui.tui.monotonic", lambda: now[0])
    app = AgentApp(settings)
    async with app.run_test():
        await app._event(AgentEvent("token", "First ", step=1))
        assert app._stream_card is not None
        markdown = app._stream_card.query_one(Markdown)
        update = markdown.update
        rendered = []

        def record(text):
            rendered.append(text)
            return update(text)

        monkeypatch.setattr(markdown, "update", record)
        for _ in range(200):
            await app._event(AgentEvent("token", "word ", step=1))
        assert rendered == []
        now[0] += 0.06
        await app._event(AgentEvent("token", "after interval ", step=1))
        assert len(rendered) == 1
        await app._event(AgentEvent("token", "last", step=1))
        assert len(rendered) == 1
        if terminal == "usage":
            await app._event(AgentEvent("usage", step=1, data={"tokens": 201}))
            assert app.transcript[-1] == ("ASSISTANT", app._stream_text)
        else:
            await app._event(AgentEvent("done", "Stopped", step=1, data={"status": terminal}))
        assert rendered[-1] == "First " + "word " * 200 + "after interval last"
        assert len(rendered) == 2


async def test_extension_commands_are_usable_without_api_credentials(settings):
    app = AgentApp(replace(settings, api_key=""))
    async with app.run_test() as pilot:
        for command in ("/skills", "/skill code-review", "/mcp", "/hooks", "/plugins", "/context"):
            await app.submit_goal(command)
        assert "code-review" in app.agent.skills.active
        assert any("superpowers" in text for _, text in app.transcript)
        await app.submit_goal("/memory set testing Use pytest")
        await app.submit_goal("/memory search pytest")
        assert "Use pytest" in app.transcript[-1][1]
        await app.submit_goal("/new")
        assert not app.agent.skills.active
        assert app.agent.memory.search("pytest")
        await app.submit_goal("/memory forget testing")
        assert not app.agent.memory.search("pytest")
        await pilot.pause()


async def test_plugin_install_before_first_task_initializes_metrics_and_can_cancel(settings):
    app = AgentApp(replace(settings, api_key=""))
    entered = asyncio.Event()

    async def install(name):
        entered.set()
        await asyncio.Future()

    app.agent.plugins.install = install
    async with app.run_test() as pilot:
        await app.submit_goal("/plugin install superpowers")
        await asyncio.wait_for(entered.wait(), 2)
        app._update_elapsed()
        await pilot.pause(0.3)  # Also exercise the periodic metrics callback.
        assert "0/" in str(app.query_one("#metrics", Static).render())
        assert app._step == 0
        await pilot.press("escape")
        await app.workers.wait_for_complete()
        assert not app._busy
        assert not app.query_one("#prompt", Input).disabled


async def test_subagent_events_keep_parent_progress_and_stream_separate(settings):
    app = AgentApp(settings)
    async with app.run_test():
        app._step = 7
        app._stream_text = "Parent response"

        async def child_event(kind, text="", details=None):
            await app._event(
                AgentEvent(
                    "subagent",
                    text,
                    1,
                    {
                        "id": "subagent-1",
                        "name": "reviewer",
                        "event": kind,
                        "details": details or {},
                    },
                )
            )

        await child_event("start", "Review the source")
        await child_event("tool_start", "read_file")
        await child_event("usage", details={"tokens": 12})
        await child_event("done", "Found the problem", {"status": "completed"})
        assert app._step == 7 and app._stream_text == "Parent response"
        assert app._calls == 1 and app._tokens == 12
        assert not app._tool_cards
        assert "completed" in app._subagent_cards["subagent-1"][0].title
        assert "Completed" not in str(app.query_one("#phase", Static).render())
        assert app.transcript[-1] == ("SUBAGENT reviewer / completed", "Found the problem")
        await app.action_new_chat()
        assert not app._subagent_cards


async def test_delegated_write_uses_approval_modal_and_renders_result(settings, tmp_path):
    class Router:
        def __init__(self, first):
            self.first = first

        async def route(self, state, tools):
            return RouteDecision(tool=self.first if not state.last_action else "finish")

    class Executor:
        async def complete(self, messages, schemas, selected, on_token):
            if selected == "delegate_tasks":
                args = {"tasks": [{"name": "writer", "prompt": "Own delegated.py; create it"}]}
            elif selected == "write_file":
                args = {"path": "delegated.py", "content": "source"}
            else:
                args = {"summary": "Reviewed the denied action"}
            return Completion(calls=[ToolCall("call", selected, json.dumps(args))])

    def factory():
        return Agent(settings, Router("write_file"), Executor(), allow_delegation=False)

    agent = Agent(settings, Router("delegate_tasks"), Executor(), subagent_factory=factory)
    app = AgentApp(settings, agent=agent)
    async with app.run_test(size=(120, 42)) as pilot:
        await app.submit_goal("Delegate the file creation")
        for _ in range(100):
            if isinstance(app.screen, ApprovalScreen):
                break
            await pilot.pause(0.01)
        assert isinstance(app.screen, ApprovalScreen)
        await pilot.pause()
        await pilot.click("#deny")
        await app.workers.wait_for_complete()
        assert not (tmp_path / "delegated.py").exists()
        assert not app._busy
        assert any(role == "SUBAGENT writer / completed" for role, _ in app.transcript)
        assert len(app._subagent_cards) == 1
        assert "Completed" in str(app.query_one("#phase", Static).render())
