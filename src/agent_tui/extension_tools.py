"""Expose skills and memory through the same validated registry as local and MCP tools."""

from __future__ import annotations

import json
import sqlite3

from harness_router import RiskLevel

from .memory import MemoryStore
from .skills import SkillManager
from .tools import ToolRegistry, ToolResult, ToolSpec, object_schema


def register_extensions(registry: ToolRegistry, skills: SkillManager, memory: MemoryStore) -> None:
    async def load_skill(args):
        return ToolResult(True, skills.activate(args["name"], persistent=False))

    registry.register(
        ToolSpec(
            "skill_load",
            "Activate a skill's instructions for this task and related follow-ups.",
            # activate() validates the exact name without sending every installed name.
            object_schema({"name": {"type": "string", "minLength": 1}}, ["name"]),
            "context",
        ),
        load_skill,
    )

    async def search_skills(args):
        return ToolResult(True, skills.catalog(args["query"]))

    registry.register(
        ToolSpec(
            "skill_search",
            "Find relevant installed skills by keywords before activating one.",
            object_schema(
                {"query": {"type": "string", "minLength": 1, "maxLength": 1000}}, ["query"]
            ),
            "context",
        ),
        search_skills,
    )

    async def read_skill(args):
        return ToolResult(True, skills.read_resource(args["name"], args["path"]))

    registry.register(
        ToolSpec(
            "skill_read",
            "Read a reference or script inside an active skill, without executing it.",
            object_schema(
                {"name": {"type": "string"}, "path": {"type": "string"}}, ["name", "path"]
            ),
            "context",
        ),
        read_skill,
    )
    if not memory.enabled:
        return

    def handler(action):
        async def execute(args):
            try:
                return ToolResult(True, json.dumps(action(args), ensure_ascii=False))
            except sqlite3.Error as exc:
                return ToolResult(False, f"Memory database unavailable: {exc}")

        return execute

    registry.register(
        ToolSpec(
            "memory_search",
            "Search saved workspace notes and past run outcomes.",
            object_schema({"query": {"type": "string", "maxLength": 1000}}),
            "memory",
        ),
        handler(lambda args: memory.search(args.get("query", ""))),
    )
    registry.register(
        ToolSpec(
            "memory_save",
            "Save or update a durable workspace note. Never store credentials.",
            object_schema(
                {
                    "key": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 100,
                        "pattern": r"^[\w.:-]+$",
                    },
                    "content": {"type": "string", "minLength": 1, "maxLength": 4000},
                },
                ["key", "content"],
            ),
            "memory",
            RiskLevel.MEDIUM,
        ),
        handler(lambda args: {"saved": memory.put(args["key"], args["content"])}),
    )
    registry.register(
        ToolSpec(
            "memory_forget",
            "Delete a saved workspace memory by its exact key.",
            object_schema({"key": {"type": "string", "minLength": 1}}, ["key"]),
            "memory",
            RiskLevel.MEDIUM,
        ),
        handler(lambda args: {"deleted": memory.forget(args["key"])}),
    )
