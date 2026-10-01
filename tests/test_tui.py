from __future__ import annotations

import asyncio
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
        await pilot.resize_terminal(80, 24)
        assert not app.query_one("#sidebar").display
        await pilot.press("ctrl+n")
        assert not app.agent.history
        assert not app.transcript
        await pilot.press("ctrl+c")


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
