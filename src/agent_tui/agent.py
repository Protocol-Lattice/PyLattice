"""A composed agent whose services and orchestration are replaceable harness mods."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from harness_router import HarnessState, RouteDecision, ToolDescriptor

from .config import Settings
from .contracts import Executor, Router, TaskPlanner
from .middleware import MiddlewareManager
from .models import AgentEvent, Approval, EventSink, RunResult, ToolCall
from .mods import ModRuntime
from .openrouter import ExecutorError
from .planner import Plan, PlanningError
from .policy import RunTools
from .tools import ToolRegistry, ToolResult


class Agent:
    def __init__(
        self,
        settings: Settings,
        router: Router | None = None,
        executor: Executor | None = None,
        registry: ToolRegistry | None = None,
        planner: TaskPlanner | None = None,
        middleware: MiddlewareManager | None = None,
        *,
        subagent_factory: Callable[[], Agent] | None = None,
        allow_delegation: bool = True,
        runtime: ModRuntime | None = None,
    ) -> None:
        self.settings = settings
        if runtime is None:
            instances: dict[str, Any] = {
                name: value
                for name, value in {
                    "router": router,
                    "executor": executor,
                    "tools": registry,
                    "middleware": middleware,
                }.items()
                if value is not None
            }
            # Preserve the existing constructor's explicitly supplied provider behavior.
            if router is not None or executor is not None or planner is not None:
                instances["planner"] = planner
            runtime = ModRuntime(settings, instances=instances, allow_delegation=allow_delegation)
        elif any(value is not None for value in (router, executor, registry, planner, middleware)):
            raise ValueError("Use mod factories or constructor instances, not both")
        self.mods = runtime
        self.running = False
        self.mods.child_factory = subagent_factory or self._new_subagent
        self.plugins = runtime.get("plugins")
        self.skills = runtime.get("skills")
        self.context = runtime.get("context")
        self.memory = runtime.get("memory")
        self.registry = runtime.get("tools")
        self.middleware = runtime.get("middleware")
        self.mcp = runtime.get("mcp")
        self.responses = runtime.get("responses")
        self.prompts = runtime.get("prompts")
        self.code_runtime = runtime.get("code_runtime")
        self.tool_policy = runtime.get("tool_policy")
        self.tool_bindings = runtime.get("tool_bindings")
        self.loop = runtime.get("loop")
        self.router = runtime.get("router")
        self.executor = runtime.get("executor")
        self.planner = runtime.get("planner")
        self.subagents = runtime.get("subagents") if runtime.allow_delegation else None
        self.tool_bindings.register(self.registry, self.skills, self.memory, self.subagents)
        if hasattr(self.registry, "bind_context") and hasattr(self.context, "cache_file"):
            self.registry.bind_context(self.context)

    @property
    def requires_api_key(self) -> bool:
        return bool(getattr(self.executor, "requires_api_key", False))

    def _new_subagent(self) -> Agent:
        child = self.mods.fork().get("agent")
        if hasattr(self.middleware, "handlers") and hasattr(
            getattr(child, "middleware", None), "handlers"
        ):
            child.middleware.handlers = list(self.middleware.handlers)
        if hasattr(child, "skills"):
            for name in self.skills.active:
                if name in child.skills.skills:
                    child.skills.activate(name, persistent=name in self.skills.pinned)
        return child

    async def aclose(self) -> None:
        await self.mods.aclose()

    def refresh_skills(self) -> None:
        if self.running:
            raise RuntimeError("Stop the current run before changing plugins")
        active = list(self.skills.active)
        pinned = set(self.skills.pinned)
        task_goal = self.skills.task_goal
        self.skills = self.mods.rebuild("skills")
        self.skills.task_goal = task_goal
        for name in active:
            if name in self.skills.skills:
                self.skills.activate(name, persistent=name in pinned)
        self.tool_bindings.refresh(self.registry, self.skills, self.memory)

    @property
    def history(self) -> list[list[dict[str, Any]]]:
        return self.context.history

    def clear(self) -> None:
        if self.running:
            raise RuntimeError("Stop the current run before clearing the conversation")
        self.context.clear()
        self.skills.clear()

    def _context(
        self,
        goal: str,
        exchanges: list[list[dict[str, Any]]],
        plan: Plan | None = None,
        schemas: list[dict[str, Any]] | None = None,
        extra_context: str = "",
        selected: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.prompts.build(self, goal, exchanges, plan, schemas, extra_context, selected)

    async def _plan(
        self, state: HarnessState, tools: Sequence[ToolDescriptor], emit: EventSink, step: int
    ) -> Plan | None:
        assert self.planner is not None
        await emit(AgentEvent("planning", "Planning from the latest observation", step))
        try:
            plan = await self.planner.plan(state, tools)
        except (PlanningError, ExecutorError) as exc:
            await emit(AgentEvent("warning", f"Planner unavailable; using Jev: {exc}", step))
            return None
        await emit(AgentEvent("plan", plan.summary, step, plan.as_dict()))
        await emit(
            AgentEvent(
                "usage",
                step=step,
                data={"model": plan.model, "tokens": plan.tokens, "source": "planner"},
            )
        )
        return plan

    async def _plan_and_route(
        self, state: HarnessState, tools: Sequence[ToolDescriptor], emit: EventSink, step: int
    ) -> tuple[Plan | None, RouteDecision]:
        # Jev depends on the observation, not on the planner's predictions.
        # MCTS still waits for a fresh plan because its search uses those paths.
        planning = asyncio.create_task(self._plan(state, tools, emit, step))
        routing = asyncio.create_task(self.router.route(state, tools))
        try:
            plan, decision = await asyncio.gather(planning, routing)
            return plan, decision
        except BaseException:
            # A failed or cancelled run must not leave either request in flight.
            for task in (planning, routing):
                if not task.done():
                    task.cancel()
            await asyncio.gather(planning, routing, return_exceptions=True)
            raise

    async def _execute_tool(
        self,
        call: ToolCall,
        run_tools: RunTools,
        emit: EventSink,
        approve: Approval,
        step: int,
        exchange: list[dict[str, Any]] | None = None,
        allowed: set[str] | None = None,
    ) -> tuple[ToolResult, dict[str, Any]]:
        return await self.tool_policy.execute(
            self, call, run_tools, emit, approve, step, exchange, allowed
        )

    async def run(self, goal: str, emit: EventSink, approve: Approval) -> RunResult:
        return await self.loop.run(self, goal, emit, approve)

    def _tool_message(self, call: ToolCall, result: ToolResult) -> dict[str, Any]:
        return self.responses.message(call, result)
