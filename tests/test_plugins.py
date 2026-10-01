from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_tui.plugins import PluginManager
from agent_tui.skills import SkillManager
from agent_tui.tools import ToolError


def write_plugin(checkout: Path) -> None:
    skill = checkout / "skills" / "using-superpowers"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: using-superpowers\ndescription: Use skills\n---\nLoad relevant skills.\n"
    )


async def test_install_enable_disable_and_uninstall_marketplace_plugin(settings, monkeypatch):
    manager = PluginManager(settings)
    calls = []

    async def git(*args):
        calls.append(args)
        if args[0] == "clone":
            write_plugin(Path(args[-1]))
            return ""
        return "a" * 40

    monkeypatch.setattr(manager, "_git", git)
    assert "superpowers" in manager.catalog()
    assert await manager.install("superpowers") == "a" * 40
    assert "https://github.com/obra/superpowers.git" in calls[0]
    reopened = PluginManager(settings)
    assert reopened.bootstrap_skills() == ["superpowers:using-superpowers"]
    assert "superpowers:using-superpowers" in SkillManager(settings, reopened.skill_roots()).skills
    reopened.enable("superpowers", False)
    assert not reopened.skill_roots() and not reopened.bootstrap_skills()
    reopened.enable("superpowers", True)
    assert reopened.skill_roots()
    reopened.uninstall("superpowers")
    assert not manager.installed()
    assert not (manager.packages / "superpowers").exists()


async def test_cancelled_install_cleans_staging_and_never_enables_plugin(settings, monkeypatch):
    manager = PluginManager(settings)
    entered = asyncio.Event()

    async def git(*args):
        write_plugin(Path(args[-1]))
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(manager, "_git", git)
    task = asyncio.create_task(manager.install("superpowers"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not manager.installed()
    assert not list(manager.packages.iterdir())


async def test_plugin_write_policy_and_path_validation(settings, tmp_path):
    manager = PluginManager(replace(settings, read_only=True))
    with pytest.raises(ToolError, match="read-only"):
        await manager.install("superpowers")
    manager = PluginManager(settings)
    with pytest.raises(ToolError, match="name"):
        await manager.install("../escape")
    manager.root.mkdir()
    (manager.root / "marketplace.json").write_text(
        json.dumps(
            [
                {
                    "name": "unsafe",
                    "repository": "https://github.com/owner/repo",
                    "skills_path": "../escape",
                },
            ]
        )
    )
    with pytest.raises(ToolError, match="inside"):
        manager.catalog()


async def test_invalid_plugin_does_not_persist_install(settings, monkeypatch):
    manager = PluginManager(settings)

    async def git(*args):
        Path(args[-1]).mkdir()
        return ""

    monkeypatch.setattr(manager, "_git", git)
    with pytest.raises(ToolError, match="skills"):
        await manager.install("superpowers")
    assert not manager.installed()
    assert not list(manager.packages.iterdir())
