from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from agent_tui.context import ContextManager
from agent_tui.memory import MemoryStore
from agent_tui.openrouter import ExecutorError
from agent_tui.skills import Skill, SkillManager
from agent_tui.tools import ToolError


def test_context_budget_includes_schemas_and_preserves_latest_tool_pair(settings):
    manager = ContextManager(replace(settings, context_chars=1200))
    manager.remember_turn([{"role": "user", "content": "old" * 500}])
    pair = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call", "content": "result" * 500},
    ]
    schemas = [{"name": "x", "description": "schema" * 30}]
    messages = manager.build(
        "System", "Current task", [pair], schemas=schemas, references=["Memory" * 100]
    )
    assert len(json.dumps(messages, ensure_ascii=False)) + len(json.dumps(schemas)) <= 1200
    assert messages[-2]["tool_calls"][0]["id"] == messages[-1]["tool_call_id"]
    assert pair[-1]["content"] == "result" * 500
    assert manager.stats.dropped_turns == 1
    assert manager.stats.truncated_results
    with pytest.raises(ExecutorError, match="context limit"):
        manager.build("System" * 1000, "Current task", [])


@pytest.mark.parametrize("budget", [900, 2000, 12000, 800000])
def test_context_size_is_exact_with_unicode_escapes_and_empty_groups(settings, budget):
    manager = ContextManager(replace(settings, context_chars=budget))
    manager.remember_turn([])
    manager.remember_turn([{"role": "user", "content": 'Old "task"\n' * 100}])
    exchanges = [[]]
    for index in range(24):
        exchanges.append(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": str(index),
                            "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"x"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": str(index), "content": 'Źródło "🙂"\n\\' * 80},
            ]
        )
    original = copy.deepcopy(exchanges)
    history = copy.deepcopy(manager.history)
    schemas = [{"name": "read_file", "description": "Źródło 🙂"}]
    messages = manager.build(
        "Rules", 'Read "x"\n🙂', exchanges, schemas=schemas, references=["", "Memory\n" * 60]
    )
    actual = len(json.dumps(messages, ensure_ascii=False)) + len(
        json.dumps(schemas, ensure_ascii=False)
    )
    assert manager.stats.chars == actual <= budget
    assert messages[-1]["tool_call_id"] == "23"
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            assert messages[index - 1]["tool_calls"][0]["id"] == message["tool_call_id"]
    assert exchanges == original
    assert manager.history == history


def test_compact_and_clear_do_not_leave_orphan_tool_messages(settings):
    manager = ContextManager(settings)
    manager.remember_turn(
        [
            {"role": "user", "content": "Task"},
            {"role": "tool", "content": "x" * 4000},
            {"role": "assistant", "content": "Outcome"},
        ]
    )
    assert manager.compact() > 3000
    assert [m["role"] for m in manager.history[0]] == ["user", "assistant"]
    manager.clear()
    assert not manager.history


def test_history_selects_relevant_turns_and_keeps_latest_for_followups(settings):
    manager = ContextManager(settings)
    for task, outcome in (
        ("Inspect cache.py", "Found cache configuration"),
        ("Translate newsletter", "Translated text"),
        ("Fix eviction policy", "Changed cache TTL"),
        ("Change theme", "Updated colors"),
        ("Deploy website", "Deployment is pending approval"),
    ):
        manager.remember_turn(
            [{"role": "user", "content": task}, {"role": "assistant", "content": outcome}]
        )
    original = copy.deepcopy(manager.history)
    messages = manager.build("Rules", "Update cache.py eviction", [])
    tasks = [m["content"] for m in messages if m["role"] == "user"]
    assert tasks == [
        "Inspect cache.py",
        "Fix eviction policy",
        "Deploy website",
        "Update cache.py eviction",
    ]
    assert manager.stats.dropped_turns == 2
    assert manager.stats.chars == len(json.dumps(messages, ensure_ascii=False))
    followup = manager.build("Rules", "Continue", [])
    assert [m["content"] for m in followup if m["role"] == "user"] == ["Deploy website", "Continue"]
    assert manager.history == original


def test_large_history_uses_excerpts_without_altering_saved_turn_or_current_tools(settings):
    manager = ContextManager(settings)
    manager.remember_turn(
        [
            {"role": "user", "content": "Inspect cache.py"},
            {"role": "assistant", "tool_calls": [{"id": "old", "function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": "old", "content": "Old source\n" * 10000},
            {"role": "assistant", "content": "Cache invalidation needs fixing."},
        ]
    )
    current = [
        {"role": "assistant", "tool_calls": [{"id": "new", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "new", "content": "Latest verified source"},
    ]
    original = copy.deepcopy(manager.history)
    messages = manager.build("Rules", "Fix cache invalidation", [current])
    assert "Cache invalidation needs fixing." in json.dumps(messages)
    assert "Old source" not in json.dumps(messages)
    assert messages[-2:] == current
    assert manager.stats.summarized_turns == 1
    assert manager.stats.chars < 2000
    assert manager.history == original


def test_history_excerpt_does_not_present_user_text_as_an_assistant_outcome(settings):
    manager = ContextManager(settings)
    manager.remember_turn([{"role": "user", "content": "Unfinished task " * 2000}])
    messages = manager.build("Rules", "Continue", [])
    summaries = [m["content"] for m in messages if m["role"] == "assistant"]
    assert len(summaries) == 1
    assert summaries[0].endswith("No final answer recorded.")


def test_memory_persists_searches_redacts_and_forgets(settings):
    store = MemoryStore(settings)
    assert not store.path.exists()
    assert store.search() == []
    store.put("preferences", "Use pytest and " + settings.api_key)
    store.record_run("Investigate pytest", "completed", "Fixed fixture")
    other = MemoryStore(settings)
    assert len(other.search("pytest")) == 2
    assert settings.api_key not in json.dumps(other.search())
    assert "past observations" in other.context("pytest")
    assert other.forget("preferences") and not other.forget("preferences")
    assert len(store.search("pytest")) == 1
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_memory_readonly_disabled_and_symlink_policy(settings, tmp_path):
    MemoryStore(settings).put("key", "original")
    readonly = MemoryStore(replace(settings, read_only=True))
    assert readonly.search()[0]["content"] == "original"
    with pytest.raises(ToolError, match="read-only"):
        readonly.put("key", "new")
    disabled = MemoryStore(replace(settings, memory_enabled=False))
    assert not disabled.search()
    with pytest.raises(ToolError, match="disabled"):
        disabled.put("key", "new")
    readonly.path.unlink()
    readonly.path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ToolError, match="symlink"):
        readonly.search()


def test_skill_metadata_mentions_resources_and_invalid_files(settings, tmp_path):
    root = tmp_path / ".agent-tui" / "skills"
    skill = root / "custom"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: custom\ndescription: >\n  Custom workflow\n---\nFollow these steps.\n"
    )
    (skill / "reference.md").write_text("Reference")
    invalid = root / "invalid"
    invalid.mkdir()
    (invalid / "SKILL.md").write_text("broken")
    manager = SkillManager(settings)
    assert "Custom workflow" in manager.catalog()
    assert "Follow these steps" not in manager.catalog()
    assert len(manager.warnings) == 1
    manager.select_mentions("Try $custom")
    assert "Follow these steps" in manager.instructions()
    assert manager.read_resource("custom", "reference.md") == "Reference"
    with pytest.raises(ToolError, match="inside"):
        manager.read_resource("custom", "../invalid/SKILL.md")


def test_plugin_skills_are_namespaced(settings, tmp_path):
    root = tmp_path / "plugin" / "skills"
    skill = root / "brainstorming"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: brainstorming\ndescription: Design things\n---\nThink first.\n"
    )
    manager = SkillManager(settings, [("superpowers", root)])
    manager.select_mentions("Use $superpowers:brainstorming")
    assert "superpowers:brainstorming" in manager.active


def test_large_real_world_skill_is_supported_within_explicit_limit(settings, tmp_path):
    root = tmp_path / ".agent-tui" / "skills" / "long-workflow"
    root.mkdir(parents=True)
    text = "---\nname: long-workflow\ndescription: A long workflow\n---\n" + "Step.\n" * 10000
    (root / "SKILL.md").write_text(text)
    skills = SkillManager(settings)
    assert "long-workflow" in skills.skills
    assert not skills.warnings


def test_skill_catalog_ranks_matches_and_bounds_descriptions(settings, tmp_path):
    skills = SkillManager(settings)
    skills.skills = {
        f"unrelated-{i:02d}": Skill(
            f"unrelated-{i:02d}", "Other workflow " * 100, tmp_path, "Instructions"
        )
        for i in range(20)
    }
    skills.skills["z-postgres"] = Skill(
        "z-postgres", "PostgreSQL schema migrations", tmp_path, "Database instructions"
    )
    catalog = skills.catalog("Plan PostgreSQL migrations")
    assert catalog.splitlines()[1].startswith("z-postgres:")
    assert "Showing 6 of 21" in catalog
    assert "skill_search" in catalog
    assert "Database instructions" not in catalog
    assert len(catalog) < 2000


@pytest.mark.parametrize("followup", ["Continue", "Kontynuuj", "Fix another failing test"])
def test_task_skills_expire_on_topic_change_but_manual_skills_stay(settings, followup):
    skills = SkillManager(settings)
    skills.activate("code-review")
    skills.begin_task("Debug a failing test")
    skills.activate("debug-tests", persistent=False)
    skills.begin_task(followup)
    assert "debug-tests" in skills.active
    skills.begin_task("Translate newsletter into Spanish")
    assert set(skills.active) == {"code-review"}
    assert skills.pinned == {"code-review"}
    skills.deactivate("code-review")
    assert not skills.pinned
    skills.activate("debug-tests")
    skills.clear()
    assert not skills.active
    assert not skills.pinned
    assert skills.task_goal == ""


def test_explicit_skill_mention_is_loaded_for_the_current_task(settings):
    skills = SkillManager(settings)
    skills.begin_task("Use $debug-tests to investigate")
    assert set(skills.active) == {"debug-tests"}
    assert not skills.pinned
    skills.begin_task("Translate newsletter into Spanish")
    assert not skills.active


def test_short_continuations_preserve_the_topic_used_for_skill_selection(settings):
    skills = SkillManager(settings)
    skills.begin_task("Investigate parser")
    skills.activate("debug-tests", persistent=False)
    skills.begin_task("Continue")
    assert skills.task_goal == "Investigate parser"
    skills.begin_task("Optimize parser")
    assert "debug-tests" in skills.active
