"""Offline example: replace providers and extend prompts/tools without editing the agent."""

from __future__ import annotations

import json

from harness_router import RouteDecision

from agent_tui.models import Completion, ToolCall
from agent_tui.mods import ModContext
from agent_tui.tools import ToolResult, ToolSpec, object_schema


class WorkspaceRouter:
    async def route(self, state, tools):
        available = {tool.name for tool in tools}
        first = "execute_code" if "execute_code" in available else "list_files"
        return RouteDecision(tool="finish" if state.last_action else first, confidence=1.0)


class WorkspaceExecutor:
    async def complete(self, messages, schemas, selected, on_token):
        if selected == "execute_code":
            arguments = {
                "code": 'label = await call_tool("workspace_label", {})\n'
                'files = await call_tool("list_files", {"max_entries": 12})\n'
                '{"label": label["output"], "listing": files["output"]}'
            }
        elif selected == "list_files":
            arguments = {"max_entries": 12}
        else:
            selected = "finish"
            replies = [message for message in messages if message["role"] == "tool"]
            summary = "Custom harness connected."
            if replies:
                output = json.loads(replies[-1]["content"])["output"]
                summary = (
                    "Custom harness observed:\n\n```json\n"
                    + json.dumps(output, ensure_ascii=False, indent=2)
                    + "\n```"
                )
            arguments = {"summary": summary}
        return Completion(
            calls=[ToolCall(f"mod_{selected}", selected, json.dumps(arguments))],
            model="example/offline",
        )


def build_router(context: ModContext):
    return WorkspaceRouter()


def build_executor(context: ModContext):
    return WorkspaceExecutor()


def build_prompts(context: ModContext):
    prompts = context.default()
    prompts.system_prompt += "\n" + context.options["instructions"]
    return prompts


def build_tools(context: ModContext):
    registry = context.default()

    async def label(arguments):
        return ToolResult(True, json.dumps(context.options["label"]))

    registry.register(
        ToolSpec("workspace_label", "The user's workspace label", object_schema({})), label
    )
    return registry
