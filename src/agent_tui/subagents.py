"""Bounded delegated runs with independent agent state and shared workspace policy."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .config import Settings
from .models import AgentEvent, Approval, EventSink, RunResult
from .tools import ToolError, ToolResult, ToolSpec, object_schema

if TYPE_CHECKING:
    from .agent import Agent


DELEGATE_SPEC = ToolSpec(
    "delegate_tasks",
    "Delegate 1–3 independent tasks to subagents concurrently in the same workspace. "
    "Give each a clear prompt, relevant context and exclusive file ownership for edits. "
    "Waits for all results. Subagents inherit permissions and cannot delegate further.",
    object_schema(
        {
            "tasks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": object_schema(
                    {
                        "name": {"type": "string", "minLength": 1, "maxLength": 60},
                        "prompt": {"type": "string", "minLength": 1, "maxLength": 12000},
                        "context": {"type": "string", "maxLength": 16000},
                    },
                    ["name", "prompt"],
                ),
            },
        },
        ["tasks"],
    ),
    "delegate",
)


class SubagentManager:
    def __init__(self, settings: Settings, factory: Callable[[], Agent]) -> None:
        self.settings = settings
        self.factory = factory
        self.emit: EventSink | None = None
        self.approve: Approval | None = None
        self.parent_goal = ""
        self.sequence = 0
        self.approval_lock = asyncio.Lock()

    def begin_run(self, goal: str, emit: EventSink, approve: Approval) -> None:
        self.parent_goal, self.emit, self.approve = goal, emit, approve
        self.sequence = 0

    def end_run(self) -> None:
        self.parent_goal = ""
        self.emit = self.approve = None

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        if self.emit is None or self.approve is None:
            raise ToolError("Subagents can only be delegated during an active task")
        tasks = arguments["tasks"]
        if len({task["name"].strip() for task in tasks}) != len(tasks):
            raise ToolError("Give each subagent a distinct name")
        if any(not task["name"].strip() or not task["prompt"].strip() for task in tasks):
            raise ToolError("Subagent names and prompts must not be blank")
        workers = []
        for task in tasks:
            self.sequence += 1
            workers.append(asyncio.create_task(self._run(f"subagent-{self.sequence}", task)))
        try:
            results = await asyncio.gather(*workers)
        finally:
            # Parent cancellation must finish every child's cleanup before returning.
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        text = json.dumps({"results": results}, ensure_ascii=False)
        while len(text) > self.settings.max_output_chars:
            candidates = [result for result in results if result["summary"]]
            if not candidates:
                return ToolResult(False, "Subagent metadata exceeds the output limit; increase it.")
            largest = max(candidates, key=lambda result: len(result["summary"]))
            largest["summary"] = largest["summary"][: len(largest["summary"]) // 2]
            largest["truncated"] = True
            text = json.dumps({"results": results}, ensure_ascii=False)
        return ToolResult(
            all(result["status"] == "completed" for result in results),
            text,
            truncated=any(result.get("truncated", False) for result in results),
        )

    async def _run(self, identifier: str, task: dict[str, str]) -> dict[str, Any]:
        assert self.emit is not None and self.approve is not None
        emit, approve = self.emit, self.approve
        name = self.settings.redact(task["name"])

        async def forward(event: AgentEvent) -> None:
            # Child streaming and terminal messages must not enter the parent's stream.
            if event.kind not in {"token", "done"}:
                await emit(
                    AgentEvent(
                        "subagent",
                        self.settings.redact(event.text),
                        event.step,
                        {
                            "id": identifier,
                            "name": name,
                            "event": event.kind,
                            "details": event.data,
                        },
                    )
                )

        async def review(tool: str, preview: str) -> bool:
            async with self.approval_lock:
                return await approve(f"{name} / {tool}", preview)

        await emit(
            AgentEvent(
                "subagent",
                self.settings.redact(task["prompt"]),
                data={
                    "id": identifier,
                    "name": name,
                    "event": "start",
                },
            )
        )
        child = None
        result = RunResult("error", "Subagent did not start", 0)
        try:
            child = self.factory()
            # Enforce the same boundary even when a custom factory is supplied.
            child.registry.unregister("delegate_tasks")
            child.subagents = None
            goal = f"Parent task: {self.parent_goal}\n\nYour assigned task: {task['prompt']}" + (
                f"\n\nContext supplied by the parent:\n{task['context']}"
                if task.get("context")
                else ""
            )
            result = await child.run(self.settings.redact(goal), forward, review)
        except asyncio.CancelledError:
            result = RunResult("cancelled", "Subagent stopped with the parent task", 0)
            raise
        except Exception as exc:
            result = RunResult("error", self.settings.redact(str(exc)), 0)
        finally:
            if child is not None:
                try:
                    await child.aclose()
                except Exception as exc:
                    result = RunResult("error", f"Subagent cleanup failed: {exc}", result.steps)
            await emit(
                AgentEvent(
                    "subagent",
                    self.settings.redact(result.message),
                    result.steps,
                    {
                        "id": identifier,
                        "name": name,
                        "event": "done",
                        "details": {"status": result.status},
                    },
                )
            )
        return {
            "id": identifier,
            "name": name,
            "status": result.status,
            "summary": self.settings.redact(result.message),
            "steps": result.steps,
        }
