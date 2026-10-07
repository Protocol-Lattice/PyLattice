from __future__ import annotations

import pytest

from agent_tui.cli import main
from agent_tui.config import Settings


def test_fast_mode_overrides_environment_routing(tmp_path, monkeypatch):
    captured = []

    class App:
        def __init__(self, settings, initial_prompt=None):
            captured.append(settings)

        def run(self):
            pass

    monkeypatch.setattr("agent_tui.tui.AgentApp", App)
    monkeypatch.setenv("AGENT_TUI_ROUTING", "mcts")
    monkeypatch.setattr("sys.argv", ["agent-tui", "--workspace", str(tmp_path), "--fast"])
    main()
    assert captured[0].routing == "jev"
    assert not captured[0].code_mode
    assert not captured[0].planning
    assert not captured[0].auto_approve


@pytest.mark.parametrize("args", [["--routing", "mcts"], ["--route-mcts"]])
def test_fast_mode_rejects_conflicting_mcts_options(monkeypatch, capsys, args):
    monkeypatch.setattr("sys.argv", ["agent-tui", "--fast", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--fast cannot be combined" in capsys.readouterr().err


def test_smaller_default_context_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_TUI_CONTEXT_CHARS", raising=False)
    assert Settings(workspace=tmp_path).context_chars == 64000
    assert Settings.from_env(tmp_path).context_chars == 64000
    monkeypatch.setenv("AGENT_TUI_CONTEXT_CHARS", "800000")
    assert Settings.from_env(tmp_path).context_chars == 800000
    assert Settings.from_env(tmp_path, context_chars=32000).context_chars == 32000


@pytest.mark.parametrize(
    "args, env, enabled",
    [
        ([], None, True),
        (["--code-mode"], "false", True),
        (["--no-code-mode"], "true", False),
        ([], "false", False),
        (["--routing", "jev"], "true", False),
        (["--route-mcts"], "true", False),
    ],
)
def test_code_mode_selection(tmp_path, monkeypatch, args, env, enabled):
    captured = []

    class App:
        def __init__(self, settings, initial_prompt=None):
            captured.append(settings)

        def run(self):
            pass

    monkeypatch.setattr("agent_tui.tui.AgentApp", App)
    monkeypatch.delenv("AGENT_TUI_CODE_MODE", raising=False)
    if env is not None:
        monkeypatch.setenv("AGENT_TUI_CODE_MODE", env)
    monkeypatch.setattr("sys.argv", ["agent-tui", "--workspace", str(tmp_path), *args])
    main()
    assert captured[0].code_mode is enabled


@pytest.mark.parametrize("args", [["--fast"], ["--routing", "jev"], ["--no-planner"]])
def test_explicit_code_mode_rejects_conflicting_routing_options(monkeypatch, capsys, args):
    monkeypatch.setattr("sys.argv", ["agent-tui", "--code-mode", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--code-mode cannot be combined" in capsys.readouterr().err
