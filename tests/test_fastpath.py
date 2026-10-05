from __future__ import annotations

import sys

from agent_tui.fastpath import FastPathResolver


def test_resolves_obvious_read_without_model(settings):
    (settings.workspace / "README.md").write_text("hello")
    resolver = FastPathResolver(settings.workspace)

    action = resolver.resolve("Read README.md", {"read_file", "finish"})

    assert action is not None
    assert action.tool == "read_file"
    assert action.arguments == {"path": "README.md"}


def test_resolves_literal_search_but_not_ambiguous_find(settings):
    resolver = FastPathResolver(settings.workspace)

    action = resolver.resolve('find "ExecutorError"', {"search_files"})
    assert action is not None
    assert action.arguments == {"query": "ExecutorError"}

    assert resolver.resolve("find the bug", {"search_files"}) is None


def test_resolves_tests_with_uv_when_available(settings):
    (settings.workspace / "uv.lock").write_text("")
    resolver = FastPathResolver(settings.workspace)

    action = resolver.resolve("run tests", {"run_command"})

    assert action is not None
    assert action.arguments["argv"] == ["uv", "run", "pytest"]


def test_resolves_tests_with_python_without_uv(settings):
    resolver = FastPathResolver(settings.workspace)

    action = resolver.resolve("pytest", {"run_command"})

    assert action is not None
    assert action.arguments["argv"] == [sys.executable, "-m", "pytest"]


def test_related_test_is_selected_after_source_edit(settings):
    test_dir = settings.workspace / "tests"
    test_dir.mkdir()
    (test_dir / "test_openrouter.py").write_text("def test_x(): pass")
    resolver = FastPathResolver(settings.workspace)

    action = resolver.verification_for(
        "src/agent_tui/openrouter.py", {"run_command", "read_file"}
    )

    assert action is not None
    assert action.tool == "run_command"
    assert action.arguments["argv"][-1] == "tests/test_openrouter.py"
