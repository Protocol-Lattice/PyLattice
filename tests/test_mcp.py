from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from agent_tui.extensions import MCPServerConfig
from agent_tui.mcp import MCPManager, tool_name
from agent_tui.tools import ToolRegistry

EXAMPLE_SERVER = Path(__file__).resolve().parents[1] / "examples" / "mcp_server.py"


async def allow(*_):
    return True


async def ignore(*_):
    pass


@pytest.mark.parametrize("is_error", [False, True])
async def test_mcp_keeps_text_and_structured_data_in_separate_fields(settings, is_error):
    class Session:
        async def call_tool(self, name, arguments):
            return CallToolResult(
                isError=is_error,
                content=[
                    TextContent(type="text", text="Remote explanation"),
                    ImageContent(type="image", data="omitted-binary-data", mimeType="image/png"),
                ],
                structuredContent={"items": [{"id": 7}], "count": 1},
            )

    manager = MCPManager(settings, ())
    handler = manager._handler(Session(), MCPServerConfig("example", command="unused"), "tool")
    reply = (await handler({})).as_dict()
    assert reply["ok"] is not is_error
    assert reply["output"] == {
        "content": [
            {"type": "text", "text": "Remote explanation"},
            {"type": "image", "omitted": True},
        ],
        "structured_content": {"items": [{"id": 7}], "count": 1},
    }
    if is_error:
        assert reply["error"]["message"] == "Remote explanation"
        assert reply["error"]["code"] == "mcp_error"
    else:
        assert reply["error"] is None


async def test_empty_mcp_response_has_the_same_output_shape(settings):
    class Session:
        async def call_tool(self, name, arguments):
            return CallToolResult(isError=False, content=[], structuredContent=None)

    manager = MCPManager(settings, ())
    handler = manager._handler(Session(), MCPServerConfig("example", command="unused"), "tool")
    assert (await handler({})).as_dict() == {
        "ok": True,
        "output": {"content": [], "structured_content": None},
        "error": None,
        "truncated": False,
    }


@pytest.mark.parametrize("read_only", [False, True])
async def test_real_stdio_server_is_discovered_called_and_closed(settings, read_only):
    settings = replace(settings, read_only=read_only)
    manager = MCPManager(
        settings,
        (
            MCPServerConfig(
                "example",
                command=sys.executable,
                args=(str(EXAMPLE_SERVER),),
                read_only_tools=("add",),
                timeout=5,
            ),
        ),
    )
    registry = ToolRegistry(settings)
    try:
        await manager.connect(registry, allow, ignore)
        assert manager.status["example"] == "connected (1 tools)"
        assert not registry.requires_approval("mcp__example__add")
        result = await registry.execute("mcp__example__add", {"a": 17, "b": 25})
        assert result.ok and "42" in result.content
        invalid = await registry.execute("mcp__example__add", {"a": "wrong", "b": 1})
        assert not invalid.ok
    finally:
        await manager.aclose()
    assert "mcp__example__add" not in registry.specs


async def test_mcp_server_denial_and_default_risk(settings):
    manager = MCPManager(
        settings,
        (
            MCPServerConfig(
                "example",
                command=sys.executable,
                args=(str(EXAMPLE_SERVER),),
                timeout=5,
            ),
        ),
    )
    registry = ToolRegistry(settings)

    async def deny(*_):
        return False

    await manager.connect(registry, deny, ignore)
    assert manager.status["example"] == "denied"
    assert "mcp__example__add" not in registry.specs
    await manager.aclose()
    try:
        await manager.connect(registry, allow, ignore)
        assert registry.requires_approval("mcp__example__add")
    finally:
        await manager.aclose()


async def test_mcp_initialization_timeout_is_recoverable(settings):
    manager = MCPManager(
        settings,
        (
            MCPServerConfig(
                "stalled",
                command=sys.executable,
                args=("-c", "import time; time.sleep(30)"),
                timeout=0.1,
            ),
        ),
    )
    events = []

    async def emit(event):
        events.append(event)

    try:
        async with asyncio.timeout(5):
            await manager.connect(ToolRegistry(settings), allow, emit)
        assert manager.status["stalled"] == "connection failed"
        assert any(e.kind == "warning" for e in events)
    finally:
        await manager.aclose()


def test_names_are_bounded_and_do_not_collide_after_sanitization():
    names = [tool_name("server", name) for name in ("a.b", "a/b", "a_b", "x" * 200)]
    assert len(set(names)) == 4
    assert all(len(name) <= 64 for name in names)


async def test_http_transport_headers_and_error_results(settings, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    import agent_tui.mcp as module

    seen = []

    @asynccontextmanager
    async def transport(url, **kwargs):
        seen.append((url, kwargs))
        yield (None, None, None)

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def initialize(self):
            pass

        async def list_tools(self, *, params=None):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="remote",
                        description="Remote tool",
                        inputSchema={"type": "object"},
                    )
                ],
                nextCursor=None,
            )

        async def call_tool(self, name, args):
            return SimpleNamespace(isError=True, content=[], structuredContent={"error": "failed"})

    monkeypatch.setenv("EXAMPLE_MCP_TOKEN", "token-value")
    monkeypatch.setattr(module, "streamablehttp_client", transport)
    monkeypatch.setattr(module, "ClientSession", Session)
    registry = ToolRegistry(settings)
    manager = MCPManager(
        settings,
        (
            MCPServerConfig(
                "http",
                transport="http",
                url="https://example.com/mcp",
                headers={"Authorization": "Bearer ${EXAMPLE_MCP_TOKEN}"},
            ),
        ),
    )
    try:
        await manager.connect(registry, allow, ignore)
        result = await registry.execute("mcp__http__remote", {})
        assert not result.ok
        assert result.as_dict()["output"]["structured_content"] == {"error": "failed"}
        assert result.as_dict()["error"]["code"] == "mcp_error"
        assert result.as_dict()["error"]["message"] == "failed"
        assert seen[0][1]["headers"]["Authorization"] == "Bearer token-value"
    finally:
        await manager.aclose()
