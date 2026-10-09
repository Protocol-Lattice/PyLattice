"""Repository Code Mode context caching and batched refactoring regression tests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace

import pytest

from agent_tui.context import ContextManager
from agent_tui.tools import ToolError, ToolRegistry


def bound_registry(settings):
    registry = ToolRegistry(settings)
    context = ContextManager(settings)
    registry.bind_context(context)
    return registry, context


async def test_context_collect_reads_matching_files_and_reuses_verified_source(settings, tmp_path):
    (tmp_path / "pkg").mkdir()
    first = tmp_path / "pkg" / "handler.py"
    first.write_text("def handler():\n    return 1\n")
    (tmp_path / "pkg" / "other.py").write_text("def unrelated(): pass\n")
    registry, context = bound_registry(settings)

    result = await registry.execute("context_collect", {"query": "handler", "max_files": 3})
    assert result.ok, result.content
    data = json.loads(result.content)
    assert [item["path"] for item in data["files"]] == ["pkg/handler.py"]
    assert data["files"][0]["content"] == first.read_text()
    assert data["files"][0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert "pkg/handler.py" in context.workspace_sources
    assert "def handler()" in context.workspace_context("Refactor handler")

    # Cache is invalidated when someone else edits the workspace, even after a read.
    first.write_text("def handler():\n    return a_different_value\n")
    assert context.cached_file("pkg/handler.py") is None
    assert "def handler()" not in context.workspace_context("Refactor handler")


async def test_collection_accepts_explicit_paths_and_stays_bounded(settings, tmp_path):
    (tmp_path / "src").mkdir()
    for i in range(4):
        (tmp_path / "src" / f"unit{i}.py").write_text("line\n" * 300)
    registry, context = bound_registry(settings)
    result = await registry.execute(
        "context_collect",
        {"paths": ["src/unit0.py", "src/unit1.py", "src/unit2.py"], "max_chars": 600},
    )
    assert result.ok
    body = json.loads(result.content)
    assert len(body["files"]) == 3
    assert body["truncated"]
    assert any(f["excerpt"] for f in body["files"])
    assert len(context.workspace_sources) == 3


async def test_patchset_edits_from_cache_and_creates_files_in_one_operation(settings, tmp_path):
    src = tmp_path / "source.py"
    src.write_text("VALUE = 1\n")
    src.chmod(0o755)
    registry, context = bound_registry(settings)
    collected = await registry.execute("context_collect", {"paths": ["source.py"]})
    digest = json.loads(collected.content)["files"][0]["sha256"]
    changes = {
        "changes": [
            {
                "action": "edit", "path": "source.py",
                "old_text": "VALUE = 1", "new_text": "VALUE = 2",
                "expected_sha256": digest,
            },
            {
                "action": "create", "path": "pkg/new_module.py",
                "content": "from source import VALUE\n",
            },
        ]
    }

    preview = registry.preview("apply_patchset", changes)
    assert "VALUE = 2" in preview and "new_module.py" in preview
    result = await registry.execute("apply_patchset", changes)
    assert result.ok, result.content
    assert json.loads(result.content)["count"] == 2
    assert src.read_text() == "VALUE = 2\n"
    assert (src.stat().st_mode & 0o777) == 0o755
    assert (tmp_path / "pkg" / "new_module.py").read_text() == "from source import VALUE\n"
    assert "VALUE = 2" in context.workspace_context("VALUE")
    assert context.cached_file("pkg/new_module.py") is not None


async def test_patchset_rejects_stale_context_without_partial_writes(settings, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("ORIGINAL = True\n")
    registry, context = bound_registry(settings)
    await registry.execute("context_collect", {"paths": ["source.py"]})
    source.write_text("ORIGINAL = False\nEXTRA = 1\n")
    changes = {
        "changes": [
            {"action": "create", "path": "new.py", "content": "new\n"},
            {
                "action": "edit", "path": "source.py",
                "old_text": "ORIGINAL = True", "new_text": "ORIGINAL = False",
            },
        ]
    }
    result = await registry.execute("apply_patchset", changes)
    assert not result.ok and result.error.code == "stale_context"
    assert not (tmp_path / "new.py").exists()
    assert source.read_text().startswith("ORIGINAL = False")
    assert context.cached_file("source.py") is None


async def test_patchset_rejects_duplicate_or_existing_creates(settings, tmp_path):
    (tmp_path / "existing.py").write_text("original")
    registry, _ = bound_registry(settings)
    result = await registry.execute(
        "apply_patchset",
        {"changes": [{"action": "create", "path": "existing.py", "content": "replacement"}]},
    )
    assert not result.ok and result.error.code == "file_exists"
    assert (tmp_path / "existing.py").read_text() == "original"
    result = await registry.execute(
        "apply_patchset",
        {"changes": [
            {"action": "create", "path": "dup.py", "content": "one"},
            {"action": "create", "path": "dup.py", "content": "two"},
        ]},
    )
    assert not result.ok
    assert not (tmp_path / "dup.py").exists()


async def test_patchset_rolls_back_on_mid_commit_failure(settings, tmp_path, monkeypatch):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("first = 1\n")
    second.write_text("second = 1\n")
    registry, _ = bound_registry(settings)
    await registry.execute("context_collect", {"paths": ["first.py", "second.py"]})
    real_replace = os.replace
    writes = 0

    def fail_second_replace(src, dst):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated mid-commit failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    result = await registry.execute(
        "apply_patchset",
        {"changes": [
            {"action": "edit", "path": "first.py", "old_text": "first = 1", "new_text": "first = 2"},
            {"action": "edit", "path": "second.py", "old_text": "second = 1", "new_text": "second = 2"},
        ]},
    )
    assert not result.ok
    assert first.read_text() == "first = 1\n"
    assert second.read_text() == "second = 1\n"


async def test_context_tools_respect_workspace_policy_and_readonly(readonly_settings, tmp_path):
    registry, _ = bound_registry(readonly_settings)
    assert "context_collect" in registry.specs
    assert "apply_patchset" not in registry.specs
    with pytest.raises(ToolError, match="not available"):
        registry.validate("apply_patchset", '{"changes": []}')
    denied = await registry.execute("context_collect", {"paths": ["../outside.py"]})
    assert not denied.ok


def test_workspace_cache_is_bounded_and_clear_resets_files(settings, tmp_path):
    registry, context = bound_registry(replace(settings, context_chars=2000))
    for i in range(72):
        path = tmp_path / f"cached_{i}.py"
        path.write_text(f"cached_{i} = {i}\n")
        registry.context.cache_file(path.name, path.read_text())
    assert len(context.workspace_sources) == 64
    assert "cached_0.py" not in context.workspace_sources
    context.clear()
    assert context.workspace_sources == {}
    assert context.workspace_context("cached") == ""
