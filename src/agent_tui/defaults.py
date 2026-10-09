"""Built-in factories and small adapters for replaceable harness components."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic_monty import AsyncMonty

from .codemode import CODE_SPEC, EXECUTION_SECONDS, adapt_tool_calls, code_mode_prompt, execute_code
from .context import ContextManager
from .extension_tools import register_extensions
from .extensions import ExtensionConfig
from .mcp import MCPManager
from .memory import MemoryStore
from .middleware import MiddlewareManager
from .mods import ModContext
from .plugins import PluginManager
from .skills import SkillManager
from .subagents import DELEGATE_SPEC, SubagentManager
from .tools import ToolRegistry, ToolResult


class DefaultResponses:
    @staticmethod
    def as_dict(result: ToolResult) -> dict[str, Any]:
        return result.as_dict()

    def message(self, call, result) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(self.as_dict(result), ensure_ascii=False, allow_nan=False),
        }


class DefaultCodeRuntime:
    spec = CODE_SPEC
    prompt = staticmethod(code_mode_prompt)
    adapt = staticmethod(adapt_tool_calls)

    def __init__(self, responses: Any) -> None:
        self.responses = responses
        self._pool: AsyncMonty | None = None
        self._pool_lock = asyncio.Lock()

    async def execute(self, code, settings, run_tool):
        # Reuse Monty workers across independent programs, without sharing variables.
        # Checkout creates a fresh isolated session for every execute_code call.
        async with self._pool_lock:
            if self._pool is None:
                pool = AsyncMonty(max_processes=1, request_timeout=EXECUTION_SECONDS + 2)
                await pool.__aenter__()
                self._pool = pool
            return await execute_code(
                code, settings, run_tool, format_result=self.responses.as_dict, pool=self._pool
            )

    async def aclose(self) -> None:
        async with self._pool_lock:
            pool, self._pool = self._pool, None
            if pool is not None:
                await pool.__aexit__(None, None, None)


class DefaultToolBindings:
    def register(self, registry, skills, memory, subagents) -> None:
        register_extensions(registry, skills, memory)
        if subagents is not None:
            registry.register(DELEGATE_SPEC, subagents.execute)

    def refresh(self, registry, skills, memory) -> None:
        for name in (
            "skill_load",
            "skill_search",
            "skill_read",
            "memory_search",
            "memory_save",
            "memory_forget",
        ):
            registry.unregister(name)
        register_extensions(registry, skills, memory)


def agent(context: ModContext):
    from .agent import Agent

    return Agent(context.settings, runtime=context.runtime)


def app(context: ModContext):
    from .tui import AgentApp

    return AgentApp(
        context.settings,
        initial_prompt=context.runtime.initial_prompt,
        agent=context.get("agent"),
    )


def router(context: ModContext):
    from .demo import DemoRouter
    from .routing import HarnessDecisionLayer

    return DemoRouter() if context.settings.demo else HarnessDecisionLayer(context.settings)


def executor(context: ModContext):
    from .demo import DemoExecutor
    from .openrouter import OpenRouterExecutor

    return (
        DemoExecutor(context.settings.workspace)
        if context.settings.demo
        else OpenRouterExecutor(context.settings)
    )


def planner(context: ModContext):
    from .demo import DemoPlanner
    from .planner import Planner

    if not context.settings.planning or context.settings.code_mode:
        return None
    return (
        DemoPlanner()
        if context.settings.demo
        else Planner(context.get("executor"), context.settings.mcts_depth)
    )


def prompts(context: ModContext):
    from .prompts import DefaultPrompts

    return DefaultPrompts()


def tool_policy(context: ModContext):
    from .policy import DefaultToolPolicy

    return DefaultToolPolicy()


def loop(context: ModContext):
    from .loop import DefaultLoop

    return DefaultLoop()


def subagents(context: ModContext):
    if not context.runtime.allow_delegation:
        return None
    return SubagentManager(
        context.settings,
        context.runtime.child_factory or (lambda: context.runtime.fork().get("agent")),
    )


FACTORIES = {
    "agent": agent,
    "app": app,
    "router": router,
    "executor": executor,
    "planner": planner,
    "tools": lambda context: ToolRegistry(context.settings),
    "context": lambda context: ContextManager(context.settings),
    "memory": lambda context: MemoryStore(context.settings),
    "plugins": lambda context: PluginManager(context.settings),
    "skills": lambda context: SkillManager(context.settings, context.get("plugins").skill_roots()),
    "extensions": lambda context: ExtensionConfig.load(context.settings),
    "middleware": lambda context: MiddlewareManager(
        context.settings, context.get("extensions").hooks
    ),
    "mcp": lambda context: MCPManager(context.settings, context.get("extensions").servers),
    "subagents": subagents,
    "prompts": prompts,
    "code_runtime": lambda context: DefaultCodeRuntime(context.get("responses")),
    "tool_policy": tool_policy,
    "tool_bindings": lambda context: DefaultToolBindings(),
    "responses": lambda context: DefaultResponses(),
    "loop": loop,
}
