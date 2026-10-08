from __future__ import annotations

import pytest

from agent_tui.cli import main
from agent_tui.config import Settings


def test_fast_mode_overrides_environment_routing(tmp_path, monkeypatch):
    captured = []

    class App:
        def __init__(self, settings, initial_prompt=None, agent=None):
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
        def __init__(self, settings, initial_prompt=None, agent=None):
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


def test_mod_selection_from_environment_and_cli(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "mods.toml"
    manifest.write_text('version = 1\n[mods]\nexecutor = "unknown:factory"\n')
    monkeypatch.setenv("AGENT_TUI_MODS", str(manifest))
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-tui",
            "--workspace",
            str(tmp_path),
            "--list-mods",
            "--mod",
            "executor=default",
        ],
    )
    main()
    assert "executor = default" in capsys.readouterr().out
    assert Settings.from_env(tmp_path).mods_path == manifest


def test_list_mods_does_not_import_factories(tmp_path, monkeypatch, capsys):
    module = tmp_path / "custom.py"
    module.write_text('raise RuntimeError("must not be imported")\n')
    manifest = tmp_path / "mods.toml"
    manifest.write_text('version = 1\n[mods]\napp = "custom.py:build"\n')
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-tui",
            "--workspace",
            str(tmp_path),
            "--mods",
            "mods.toml",
            "--list-mods",
        ],
    )
    main()
    output = capsys.readouterr().out
    assert "app = custom.py:build" in output
    assert "responses = default" in output


def test_cli_runs_custom_app_and_cleans_up(tmp_path, monkeypatch):
    module = tmp_path / "app.py"
    module.write_text(
        "class App:\n"
        "    def __init__(self, ctx): self.ctx = ctx\n"
        "    def run(self):\n"
        "        path = self.ctx.settings.workspace / 'ran'\n"
        "        path.write_text(self.ctx.runtime.initial_prompt)\n"
        "    async def aclose(self):\n"
        "        (self.ctx.settings.workspace / 'closed').touch()\n"
        "def build(ctx): return App(ctx)\n"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-tui",
            "--workspace",
            str(tmp_path),
            "--mod",
            "app=app.py:build",
            "--prompt",
            "from CLI",
        ],
    )
    main()
    assert (tmp_path / "ran").read_text() == "from CLI"
    assert (tmp_path / "closed").exists()


def test_cli_reports_invalid_mod_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-tui",
            "--workspace",
            str(tmp_path),
            "--mod",
            "missing=default",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Unknown mod slot 'missing'" in capsys.readouterr().err
