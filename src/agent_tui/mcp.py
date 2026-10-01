"""MCP SDK clients with per-run sessions, namespaced tools, and explicit risk policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from harness_router import RiskLevel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .config import Settings
from .extensions import MCPServerConfig, expand_environment, safe_environment
from .models import AgentEvent, Approval, EventSink
from .tools import ToolError, ToolRegistry, ToolResult, ToolSpec


def tool_name(server: str, name: str) -> str:
    raw = f"mcp__{server}__{name}"
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if clean != raw or len(clean) > 64:
        clean = clean[:51] + "_" + hashlib.sha256(raw.encode()).hexdigest()[:12]
    return clean


class MCPManager:
    def __init__(self, settings: Settings, servers: tuple[MCPServerConfig, ...]) -> None:
        self.settings = settings
        self.servers = tuple(server for server in servers if server.enabled)
        self.status = {server.name: "not connected" for server in self.servers}
        self._stack: AsyncExitStack | None = None
        self._registered: list[str] = []
        self._registry: ToolRegistry | None = None

    async def connect(self, registry: ToolRegistry, approve: Approval, emit: EventSink) -> None:
        self._stack = AsyncExitStack()
        self._registry = registry
        if self.settings.demo:
            return
        for server in self.servers:
            if self.settings.read_only and not server.read_only_tools:
                self.status[server.name] = "disabled in read-only mode"
                continue
            preview = (
                f"Connect MCP server {server.name} via {server.transport}.\n"
                + (
                    f"Command: {json.dumps([server.command, *server.args])}\n"
                    f"Directory: {self.settings.workspace}\n"
                    "The server runs with your OS permissions."
                    if server.transport == "stdio"
                    else f"URL: {server.url}"
                )
                + "\nTool calls use the normal approval policy."
            )
            await emit(AgentEvent("approval", f"MCP {server.name}"))
            if not self.settings.auto_approve and not await approve(
                f"mcp:{server.name}", self.settings.redact(preview)
            ):
                self.status[server.name] = "denied"
                continue
            stack = AsyncExitStack()
            added: list[str] = []
            try:
                if server.transport == "stdio":
                    # Keep server stderr out of Textual's terminal and out of model context.
                    errlog = stack.enter_context(open(os.devnull, "w"))  # noqa: SIM115
                    parameters = StdioServerParameters(
                        command=server.command,
                        args=list(server.args),
                        cwd=self.settings.workspace,
                        env={**safe_environment(), **expand_environment(server.env)},
                    )
                    read, write = await stack.enter_async_context(stdio_client(parameters, errlog))
                else:
                    read, write, _ = await stack.enter_async_context(
                        streamablehttp_client(
                            server.url,
                            headers=expand_environment(server.headers),
                            timeout=server.timeout,
                            sse_read_timeout=server.timeout,
                        )
                    )
                session = await stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        read_timeout_seconds=timedelta(seconds=server.timeout),
                    )
                )
                await session.initialize()
                cursor = None
                seen: set[str] = set()
                while True:
                    listing = await session.list_tools(cursor=cursor)
                    for tool in listing.tools:
                        name = tool_name(server.name, tool.name)
                        readonly = tool.name in server.read_only_tools
                        if self.settings.read_only and not readonly:
                            continue
                        spec = ToolSpec(
                            name,
                            f"MCP {server.name}: {tool.description or tool.name}",
                            tool.inputSchema,
                            "mcp",
                            RiskLevel.LOW if readonly else RiskLevel.HIGH,
                        )
                        registry.register(spec, self._handler(session, server, tool.name))
                        added.append(name)
                        if len(added) > 256:
                            raise ToolError("MCP server exceeds the 256-tool limit")
                    cursor = listing.nextCursor
                    if not cursor:
                        break
                    if cursor in seen or len(seen) >= 100:
                        raise ToolError("MCP tools/list pagination did not terminate")
                    seen.add(cursor)
                # SDK cancel scopes must exit in reverse order in this same task.
                self._stack.push_async_callback(stack.aclose)
                self._registered.extend(added)
                self.status[server.name] = f"connected ({len(added)} tools)"
                await emit(AgentEvent("extension", f"MCP {server.name}: {len(added)} tools"))
            except BaseException as exc:
                for name in added:
                    registry.unregister(name)
                await stack.aclose()
                if not isinstance(exc, Exception):
                    raise
                self.status[server.name] = "connection failed"
                await emit(
                    AgentEvent(
                        "warning",
                        self.settings.redact(
                            f"MCP {server.name} unavailable: {str(exc) or type(exc).__name__}"
                        ),
                    )
                )

    def _handler(self, session: ClientSession, server: MCPServerConfig, name: str):
        async def execute(arguments: dict[str, Any]) -> ToolResult:
            try:
                async with asyncio.timeout(server.timeout):
                    result = await session.call_tool(name, arguments)
                content = []
                for block in result.content:
                    if block.type == "text":
                        content.append(block.text)
                    else:
                        content.append(f"[MCP {block.type} content omitted by this text interface]")
                if result.structuredContent is not None:
                    content.append(json.dumps(result.structuredContent, ensure_ascii=False))
                return ToolResult(not result.isError, "\n".join(content) or "(empty MCP result)")
            except TimeoutError:
                return ToolResult(
                    False,
                    f"MCP {server.name}/{name} timed out; "
                    "the remote action may have partially completed",
                )
            except Exception as exc:
                return ToolResult(False, f"MCP {server.name}/{name} failed: {exc}")

        return execute

    async def aclose(self) -> None:
        try:
            if self._stack:
                await self._stack.aclose()
        finally:
            if self._registry:
                for name in self._registered:
                    self._registry.unregister(name)
            for name, status in self.status.items():
                if status.startswith("connected"):
                    self.status[name] = "disconnected (reconnects next run)"
            self._registered.clear()
            self._stack = None
