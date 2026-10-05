from __future__ import annotations

from pathlib import Path

from agent_tui.mods import ModManager


def write_mod(root: Path, name: str, source: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(source)


async def test_mods_compose_first_loaded_first_and_last(settings):
    root = settings.workspace / ".agent-tui" / "mods"
    write_mod(
        root,
        "01_outer.py",
        """
EVENTS = ("tool",)

async def mod(event, payload, next):
    result = await next(payload)
    return "outer(" + result + ")"
""",
    )
    write_mod(
        root,
        "02_inner.py",
        """
EVENTS = ("tool",)

async def mod(event, payload, next):
    result = await next(payload)
    return "inner(" + result + ")"
""",
    )

    manager = ModManager(settings)

    async def terminal(payload):
        return "core"

    assert await manager.invoke("tool", {"tool": "read_file"}, terminal) == "outer(inner(core))"


async def test_mod_can_rewrite_or_replace_builtin_behavior(settings):
    root = settings.workspace / ".agent-tui" / "mods"
    write_mod(
        root,
        "rewrite.py",
        """
EVENTS = ("model",)

async def mod(event, payload, next):
    payload["selected"] = "read_file"
    return await next(payload)
""",
    )

    manager = ModManager(settings)

    async def terminal(payload):
        return payload["selected"]

    assert await manager.invoke("model", {"selected": None}, terminal) == "read_file"

    (root / "rewrite.py").unlink()
    write_mod(
        root,
        "replace.py",
        """
EVENTS = ("permission",)

async def mod(event, payload, next):
    return False
""",
    )
    manager.reload()

    called = False

    async def permission(payload):
        nonlocal called
        called = True
        return True

    assert await manager.invoke("permission", {"tool": "run_command"}, permission) is False
    assert not called


async def test_plugin_mod_root_is_loaded(settings):
    root = settings.workspace / "plugin-mods"
    write_mod(
        root,
        "status.py",
        """
EVENTS = ("render",)

async def mod(event, payload, next):
    return await next(payload)
""",
    )

    manager = ModManager(settings, [("example", root)])
    assert manager.names() == ["example:status"]
