from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from harness_router import RouteDecision

from agent_tui.agent import Agent
from agent_tui.extensions import ExtensionConfig, HookConfig
from agent_tui.middleware import HookError, MiddlewareManager
from agent_tui.models import Completion, ToolCall

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "hooks"


async def allow(*_):
    return True


async def ignore(*_):
    pass


async def test_example_hooks_transform_block_and_log(settings):
    manager = MiddlewareManager(
        settings,
        (
            HookConfig("before_tool", (sys.executable, str(EXAMPLES / "protect_files.py"))),
            HookConfig("before_tool", (sys.executable, str(EXAMPLES / "final_newline.py"))),
            HookConfig("after_run", (sys.executable, str(EXAMPLES / "audit_log.py"))),
        ),
    )
    await manager.authorize(allow)
    data = await manager.dispatch(
        "before_tool",
        {
            "tool": "write_file",
            "arguments": {"path": "hello.py", "content": "hello"},
        },
    )
    assert data["arguments"]["content"] == "hello\n"
    with pytest.raises(HookError, match="protected"):
        await manager.dispatch(
            "before_tool",
            {
                "tool": "write_file",
                "arguments": {"path": "uv.lock", "content": "hello"},
            },
        )
    await manager.dispatch("after_run", {"status": "completed", "goal": settings.api_key})
    audit = (settings.workspace / ".agent-tui" / "hooks.jsonl").read_text()
    assert json.loads(audit)["status"] == "completed"
    assert settings.api_key not in audit and "goal" not in audit


async def test_hook_output_timeout_and_denial_fail_closed(settings):
    for program, match in [
        ("print('bad json')", "invalid JSON"),
        ("print('x'*65000)", "64 KB"),
        ("import time; time.sleep(10)", "timed out"),
        ("raise SystemExit(2)", "exited 2"),
    ]:
        manager = MiddlewareManager(
            settings, (HookConfig("before_tool", (sys.executable, "-c", program), timeout=0.2),)
        )
        await manager.authorize(allow)
        with pytest.raises(HookError, match=match):
            await manager.dispatch("before_tool", {})

    async def deny(*_):
        return False

    with pytest.raises(HookError, match="denied"):
        await manager.authorize(deny)
    assert not manager.commands_enabled
    manager = MiddlewareManager(replace(settings, read_only=True), manager.hooks)
    await manager.authorize(allow)
    assert not manager.commands_enabled


async def test_middleware_changes_are_validated_and_approved_before_execution(settings):
    events, previews = [], []
    manager = MiddlewareManager(settings)

    async def middleware(event, payload):
        events.append(event)
        if event == "before_tool" and payload["tool"] == "write_file":
            return {"arguments": {"path": "changed.txt", "content": "transformed"}}

    manager.add(middleware)

    class Router:
        async def route(self, state, tools):
            return RouteDecision(tool="write_file" if not state.last_action else "finish")

    class Executor:
        async def complete(self, messages, schemas, selected, on_token):
            args = (
                {"path": "original.txt", "content": "original"}
                if selected == "write_file"
                else {
                    "summary": "Done",
                }
            )
            return Completion(calls=[ToolCall("call", selected, json.dumps(args))])

    async def approve(name, preview):
        previews.append(preview)
        return True

    agent = Agent(settings, Router(), Executor(), middleware=manager)
    result = await agent.run("Write", ignore, approve)
    assert result.status == "completed"
    assert "changed.txt" in previews[0] and "+transformed" in previews[0]
    assert (settings.workspace / "changed.txt").read_text() == "transformed"
    assert not (settings.workspace / "original.txt").exists()
    assert events[0] == "before_run" and events[-1] == "after_run"
    assert events.count("after_tool") == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
async def test_hook_cancel_kills_process(settings):
    pidfile = settings.workspace / "hook.pid"
    program = "import os,pathlib,time; pathlib.Path('hook.pid').write_text(str(os.getpid())); "
    program += "time.sleep(60)"
    manager = MiddlewareManager(
        settings, (HookConfig("before_run", (sys.executable, "-c", program)),)
    )
    await manager.authorize(allow)
    task = asyncio.create_task(manager.dispatch("before_run", {}))
    for _ in range(200):
        if pidfile.exists():
            break
        await asyncio.sleep(0.01)
    assert pidfile.exists()
    pid = int(pidfile.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_invalid_config_rejected_before_startup(settings):
    path = settings.workspace / "extensions.toml"
    path.write_text('[[hooks]]\nevent="before_tool"\ncommand="sh bad"\n')
    with pytest.raises(ValueError, match="configuration"):
        ExtensionConfig.load(replace(settings, extensions_path=path))
