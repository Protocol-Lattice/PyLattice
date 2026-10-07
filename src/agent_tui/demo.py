"""Deterministic offline providers. They still exercise the real agent loop and file tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harness_router import HarnessState, MCTSConfig, MCTSToolRouter, RouteDecision, ToolDescriptor

from .models import Completion, TokenSink, ToolCall
from .planner import Plan, PlanEnvironment, PredictedAction


class DemoRouter:
    async def route(self, state: HarnessState, tools: Sequence[ToolDescriptor]) -> RouteDecision:
        await asyncio.sleep(0.25)
        first = (
            "execute_code" if any(tool.name == "execute_code" for tool in tools) else "list_files"
        )
        tool = first if not state.last_action else "finish"
        return RouteDecision(tool=tool, confidence=1.0)

    async def aclose(self) -> None:
        pass

    async def route_mcts(self, state, tools, plan):
        return await MCTSToolRouter(
            PlanEnvironment(state, plan, tools),
            config=MCTSConfig(simulations=64, max_depth=3),
        ).search(state)


class DemoPlanner:
    async def plan(self, state, tools):
        if state.last_action:
            return Plan(
                "Report the observed files",
                ("Summarize the real tool result",),
                ((PredictedAction("finish", "Summarize the observed listing", 1.0),),),
                model="offline/demo",
            )
        return Plan(
            "Inspect the workspace, then summarize",
            ("List workspace files", "Report findings"),
            (
                (
                    PredictedAction("list_files", "Discover workspace files", 0.6),
                    PredictedAction("finish", "Explain the observed files", 1.0),
                ),
                (PredictedAction("finish", "Reply without inspecting files", 0.1),),
            ),
            model="offline/demo",
        )


class DemoExecutor:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def complete(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        selected: str | None,
        on_token: TokenSink,
    ) -> Completion:
        code_mode = any(s["function"]["name"] == "execute_code" for s in schemas)
        has_result = messages[-1].get("role") == "tool"
        if code_mode and not has_result:
            return Completion(
                calls=[
                    ToolCall(
                        "demo_code",
                        "execute_code",
                        json.dumps(
                            {
                                "code": 'result = await call_tool("list_files", '
                                '{"max_entries": 12})\n'
                                'result["output"]',
                            }
                        ),
                    )
                ],
                model="offline/demo",
            )
        if selected == "list_files":
            for chunk in ["I’ll inspect ", "the workspace ", "using the local file tool."]:
                await on_token(chunk)
                await asyncio.sleep(0.08)
            return Completion(
                "I’ll inspect the workspace using the local file tool.",
                [ToolCall("demo_list", "list_files", '{"max_entries":12}')],
                "offline/demo",
                0,
            )
        result = json.loads(messages[-1]["content"])
        output = result["output"]
        summary = (
            f"Offline demonstration complete in `{self.workspace.name}`.\n\n"
            f"The real `list_files` tool returned:\n\n```json\n{output}\n```\n\n"
            "Model responses were simulated; no API was contacted. "
            "Restart without `--demo` and set `OPENROUTER_API_KEY` to execute your tasks."
        )
        return Completion(
            calls=[ToolCall("demo_finish", "finish", json.dumps({"summary": summary}))],
            model="offline/demo",
        )

    async def aclose(self) -> None:
        pass
