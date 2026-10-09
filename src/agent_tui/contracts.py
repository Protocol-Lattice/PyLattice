"""Public provider contracts; data models are shared across built-in and custom mods."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from harness_router import HarnessState, MCTSResult, RouteDecision, ToolDescriptor

from .config import Settings
from .models import Approval, Completion, EventSink, RunResult, TokenSink, ToolCall
from .planner import Plan
from .tools import ToolRegistry, ToolResult, ToolSpec

if TYPE_CHECKING:
    from .agent import Agent
    from .policy import RunTools


class Router(Protocol):
    """Routing required in every mode; MCTS is an additional capability."""

    async def route(
        self, state: HarnessState, tools: Sequence[ToolDescriptor], /
    ) -> RouteDecision: ...


class DecisionLayer(Router, Protocol):
    async def route_mcts(
        self,
        state: HarnessState,
        tools: Sequence[ToolDescriptor],
        plan: Plan,
        /,
    ) -> MCTSResult: ...


class TaskPlanner(Protocol):
    async def plan(self, state: HarnessState, tools: Sequence[ToolDescriptor], /) -> Plan: ...


class Executor(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        selected: str | None,
        on_token: TokenSink,
        /,
    ) -> Completion: ...


class Prompts(Protocol):
    def build(
        self,
        agent: Agent,
        goal: str,
        exchanges: list[list[dict[str, Any]]],
        plan: Plan | None = None,
        schemas: list[dict[str, Any]] | None = None,
        extra_context: str = "",
        selected: str | None = None,
    ) -> list[dict[str, Any]]: ...


class CodeRuntime(Protocol):
    spec: ToolSpec

    def prompt(self, registry: ToolRegistry, selected: str | None = None) -> str: ...

    def adapt(self, completion: Completion, registry: ToolRegistry) -> Completion: ...

    async def execute(
        self,
        code: str,
        settings: Settings,
        run_tool: Callable[[str, dict[str, Any]], Awaitable[ToolResult]],
    ) -> ToolResult: ...


class Responses(Protocol):
    def as_dict(self, result: ToolResult) -> dict[str, Any]: ...

    def message(self, call: ToolCall, result: ToolResult) -> dict[str, Any]: ...


class ToolPolicy(Protocol):
    async def execute(
        self,
        agent: Agent,
        call: ToolCall,
        run_tools: RunTools,
        emit: EventSink,
        approve: Approval,
        step: int,
        exchange: list[dict[str, Any]] | None = None,
        allowed: set[str] | None = None,
    ) -> tuple[ToolResult, dict[str, Any]]: ...


class AgentLoop(Protocol):
    async def run(
        self, agent: Agent, goal: str, emit: EventSink, approve: Approval
    ) -> RunResult: ...
