from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace

import pytest
from harness_router import RiskLevel
from jsonschema.exceptions import SchemaError

from agent_tui.tools import (
    ToolError,
    ToolFailure,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    object_schema,
)


@pytest.mark.parametrize(
    "path",
    ["../outside", "/etc/passwd", ".git/config", ".env", ".env.local", ".ssh/id_rsa", "cert.key"],
)
async def test_protected_paths_cannot_be_read_or_written(settings, path):
    tools = ToolRegistry(settings)
    for name, args in [
        ("read_file", {"path": path}),
        ("write_file", {"path": path, "content": "changed"}),
    ]:
        result = await tools.execute(name, args)
        assert not result.ok


async def test_symlink_escape_and_aliases_are_rejected(settings, tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    (tmp_path / "link").symlink_to(outside)
    tools = ToolRegistry(settings)
    assert not (await tools.execute("read_file", {"path": "link"})).ok
    prefixed = await tools.execute("read_file", {"path": f"{tmp_path.name}/link"})
    assert not prefixed.ok and " Use 'link'." not in prefixed.content
    assert not (await tools.execute("write_file", {"path": "link", "content": "changed"})).ok
    assert outside.read_text() == "secret"
    listing = json.loads((await tools.execute("list_files", {})).content)
    assert "link" not in listing["files"]


@pytest.mark.parametrize(
    "path,exists", [("README.md", True), ("src/module.py", True), ("missing.py", False)]
)
async def test_missing_workspace_prefixed_path_suggests_only_existing_files(
    settings, tmp_path, path, exists
):
    if exists:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("Requested contents")
    tools = ToolRegistry(settings)
    result = await tools.execute("read_file", {"path": f"{tmp_path.name}/{path}"})
    assert not result.ok
    assert "Paths are relative to" in result.content
    assert (f"Use {path!r}." in result.content) == exists
    assert "Requested contents" not in result.content
    if exists:
        corrected = await tools.execute("read_file", {"path": path})
        assert corrected.ok and "Requested contents" in corrected.content


async def test_workspace_named_subdirectory_is_read_literally(settings, tmp_path):
    (tmp_path / "README.md").write_text("Root contents")
    directory = tmp_path / tmp_path.name
    directory.mkdir()
    (directory / "README.md").write_text("Nested contents")
    path = f"{tmp_path.name}/README.md"
    result = await ToolRegistry(settings).execute("read_file", {"path": path})
    assert result.ok
    output = json.loads(result.content)
    assert output["path"] == path
    assert output["content"] == "1: Nested contents"


async def test_listing_search_and_read_ranges(settings, tmp_path):
    (tmp_path / "code.py").write_text("first\nneedle\nlast\n")
    (tmp_path / ".env").write_text("needle=secret")
    (tmp_path / ".env.example").write_text("KEY=")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x").write_text("needle")
    tools = ToolRegistry(settings)
    listing = json.loads((await tools.execute("list_files", {})).content)
    assert listing["files"] == [".env.example", "code.py"]
    search = json.loads((await tools.execute("search_files", {"query": "needle"})).content)
    assert search["matches"] == [{"path": "code.py", "line": 2, "text": "needle"}]
    read = json.loads(
        (
            await tools.execute(
                "read_file",
                {
                    "path": "code.py",
                    "start_line": 2,
                    "end_line": 2,
                },
            )
        ).content
    )
    assert read["content"] == "2: needle"
    assert read["has_more"]


async def test_unique_edit_and_atomic_write_preserve_mode(settings, tmp_path):
    file = tmp_path / "script.py"
    file.write_text("print('old')\n")
    file.chmod(0o755)
    tools = ToolRegistry(settings)
    args = {"path": "script.py", "old_text": "old", "new_text": "new"}
    assert "-print('old')" in tools.preview("edit_file", args)
    assert (await tools.execute("edit_file", args)).ok
    assert file.read_text() == "print('new')\n"
    assert file.stat().st_mode & 0o777 == 0o755
    assert not (await tools.execute("edit_file", args)).ok
    assert (await tools.execute("write_file", {"path": "dir/new.txt", "content": "hi"})).ok
    assert (tmp_path / "dir/new.txt").read_text() == "hi"


async def test_ambiguous_edit_is_rejected(settings, tmp_path):
    (tmp_path / "file").write_text("same same")
    result = await ToolRegistry(settings).execute(
        "edit_file",
        {
            "path": "file",
            "old_text": "same",
            "new_text": "new",
        },
    )
    assert not result.ok
    assert (tmp_path / "file").read_text() == "same same"


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_file", {"path": 12}),
        ("read_file", {"path": "x", "unknown": 1}),
        ("run_command", {"argv": []}),
        ("read_file", {"path": "x", "start_line": True}),
        ("read_files", {}),
        ("finish", {"summary": ""}),
        ("unknown", {}),
    ],
)
def test_schema_validation(settings, name, args):
    with pytest.raises(ToolError):
        ToolRegistry(settings).validate(name, json.dumps(args))


async def test_read_only_removes_and_denies_mutating_tools(readonly_settings):
    tools = ToolRegistry(readonly_settings)
    assert set(tools.specs) == {"list_files", "read_file", "search_files", "finish"}
    result = await tools.execute("write_file", {"path": "file", "content": "x"})
    assert not result.ok


async def test_binary_and_large_files_are_rejected(settings, tmp_path):
    (tmp_path / "binary").write_bytes(b"a\0b")
    (tmp_path / "large").write_bytes(b"x" * 1_000_001)
    tools = ToolRegistry(settings)
    for path in ("binary", "large"):
        assert not (await tools.execute("read_file", {"path": path})).ok


async def test_commands_are_argv_only_with_secret_environment_removed(settings, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-inherit")
    result = await ToolRegistry(settings).execute(
        "run_command",
        {
            "argv": [
                sys.executable,
                "-c",
                "import os,sys; print(os.getenv('OPENROUTER_API_KEY')); print(sys.argv[1])",
                "$(touch should-not-exist)",
            ]
        },
    )
    assert result.ok
    output = json.loads(result.content)
    assert output["output"] == "None\n$(touch should-not-exist)\n"
    assert not (settings.workspace / "should-not-exist").exists()


async def test_command_error_timeout_and_bounded_output(settings):
    tools = ToolRegistry(replace(settings, command_timeout=0.15, max_output_chars=1200))
    failed = await tools.execute(
        "run_command", {"argv": [sys.executable, "-c", "raise SystemExit(3)"]}
    )
    assert not failed.ok
    assert json.loads(failed.content)["exit_code"] == 3
    assert failed.as_dict()["error"]["code"] == "command_failed"
    assert failed.as_dict()["output"]["exit_code"] == 3
    timeout = await tools.execute(
        "run_command",
        {
            "argv": [
                sys.executable,
                "-c",
                "import time; print('start', flush=True); time.sleep(20)",
            ],
        },
    )
    assert not timeout.ok
    assert '"timed_out": true' in timeout.content
    assert timeout.as_dict()["error"]["code"] == "command_timeout"
    output = await tools.execute(
        "run_command", {"argv": [sys.executable, "-c", "print('x'*1000000)"]}
    )
    assert len(output.content) < 1300
    bounded = json.loads(output.content)
    assert bounded["exit_code"] == 0 and bounded["truncated"]
    assert isinstance(bounded["output"], str)
    assert output.as_dict()["truncated"]


@pytest.mark.parametrize(
    "content,expected",
    [
        ('{"files":["one.py"],"count":1}', {"files": ["one.py"], "count": 1}),
        ('[1, "two", null]', [1, "two", None]),
        ('"plain JSON string"', "plain JSON string"),
        ("true", True),
        ("0", 0),
        ("null", None),
        ("plain text\nsecond line", "plain text\nsecond line"),
        ("", ""),
        ("NaN", "NaN"),
        ('{"value":Infinity}', '{"value":Infinity}'),
    ],
)
def test_tool_reply_has_native_output_and_a_stable_envelope(content, expected):
    reply = ToolResult(True, content).as_dict()
    assert reply == {"ok": True, "output": expected, "error": None, "truncated": False}
    assert json.loads(json.dumps(reply, allow_nan=False)) == reply


def test_error_reply_keeps_partial_output_and_separates_diagnostics():
    error = ToolFailure("command_failed", "Command exited with status 3.", "Inspect output.")
    reply = ToolResult(False, '{"exit_code":3,"output":"traceback"}', error).as_dict()
    assert reply == {
        "ok": False,
        "output": {"exit_code": 3, "output": "traceback"},
        "error": {
            "code": "command_failed",
            "message": "Command exited with status 3.",
            "retry_hint": "Inspect output.",
        },
        "truncated": False,
    }
    plain = ToolResult.from_error(ToolError("Bad input", code="invalid_arguments")).as_dict()
    assert plain["output"] is None and plain["error"]["code"] == "invalid_arguments"
    assert plain["error"]["message"] == "Bad input" and plain["error"]["retry_hint"] is None


def test_plain_text_truncation_and_error_metadata_are_redacted(settings):
    tools = ToolRegistry(replace(settings, max_output_chars=200))
    source = settings.api_key + "x" * 1000
    success = tools.sanitize(ToolResult(True, source)).as_dict()
    assert success["ok"] and success["truncated"] and isinstance(success["output"], str)
    error = ToolResult.from_error(
        ToolError(source, code="example_error", retry_hint=settings.api_key)
    )
    failed = tools.sanitize(error).as_dict()
    assert not failed["ok"] and failed["truncated"]
    assert len(failed["error"]["message"]) <= 200
    assert failed["error"]["retry_hint"] == "[REDACTED]"
    assert settings.api_key not in json.dumps([success, failed])


@pytest.mark.parametrize(
    "name,field,payload",
    [
        (
            "list_files",
            "files",
            {"files": [f"src/path-{i}.py" for i in range(100)], "truncated": False},
        ),
        (
            "search_files",
            "matches",
            {
                "matches": [
                    {"path": "source.py", "line": i, "text": 'quoted "text" 🙂'}
                    for i in range(1, 100)
                ],
                "truncated": False,
                "skipped": 2,
            },
        ),
        (
            "read_file",
            "content",
            {
                "path": "source.py",
                "content": 'quoted "text" and slashes \\\\ 🙂\n' * 100,
                "total_lines": 100,
                "has_more": False,
            },
        ),
    ],
)
def test_bounded_tool_results_keep_json_shapes_and_whole_list_entries(
    settings, name, field, payload
):
    tools = ToolRegistry(replace(settings, max_output_chars=500))
    result = tools.sanitize(ToolResult(True, json.dumps(payload, ensure_ascii=False)), name)
    assert result.ok and len(result.content) <= 500
    bounded = json.loads(result.content)
    assert bounded["truncated"]
    assert bounded[field] and len(bounded[field]) < len(payload[field])
    assert bounded[field] == payload[field][: len(bounded[field])]
    if name == "read_file":
        assert bounded["has_more"] and bounded["total_lines"] == 100
        assert bounded["path"] == "source.py"
    if name == "search_files":
        assert bounded["skipped"] == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
async def test_cancel_kills_running_process(settings, tmp_path):
    pidfile = tmp_path / "pid"
    task = asyncio.create_task(
        ToolRegistry(settings).execute(
            "run_command",
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "import os,time,pathlib; "
                    "pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(60)",
                ],
            },
        )
    )
    for _ in range(200):
        if pidfile.exists():
            break
        await asyncio.sleep(0.01)
    assert pidfile.exists()
    pid = int(pidfile.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def test_credentials_are_redacted_in_results(settings, tmp_path):
    (tmp_path / "log.txt").write_text(f"token={settings.api_key}")
    result = await ToolRegistry(settings).execute("read_file", {"path": "log.txt"})
    assert settings.api_key not in result.content
    assert "[REDACTED]" in result.content


async def test_builtin_tools_can_be_removed(settings):
    tools = ToolRegistry(settings)
    tools.unregister("run_command")
    tools.unregister("run_command")
    assert "run_command" not in tools.specs
    assert "run_command" not in {item.name for item in tools.descriptors()}
    assert tools.schemas("run_command") == []
    result = await tools.execute("run_command", {"argv": ["never-run"]})
    assert not result.ok and result.error.code == "unavailable_tool"


async def test_replaced_builtin_can_change_schema_and_keeps_validation(settings):
    tools = ToolRegistry(settings)
    calls = []

    async def custom(arguments):
        calls.append(arguments)
        return ToolResult(True, settings.api_key)

    spec = ToolSpec(
        "run_command",
        "Send a named job to a custom runner",
        object_schema({"job": {"type": "string"}}, ["job"]),
        risk=RiskLevel.HIGH,
    )
    tools.replace(spec, custom)
    assert tools.requires_approval("run_command")
    assert json.loads(tools.preview("run_command", {"job": "check"})) == {"job": "check"}
    invalid = await tools.execute("run_command", {"argv": ["not-accepted"]})
    assert not invalid.ok and not calls
    result = await tools.execute("run_command", {"job": "check"})
    assert calls == [{"job": "check"}]
    assert result.ok and result.content == "[REDACTED]"


async def test_replacement_respects_readonly_and_invalid_schema_is_atomic(readonly_settings):
    tools = ToolRegistry(readonly_settings)
    old = tools.specs["read_file"]

    async def handler(arguments):
        pytest.fail("A disallowed replacement must not execute")

    with pytest.raises(SchemaError):
        tools.replace(replace(old, parameters={"type": "invalid-type"}), handler)
    assert tools.specs["read_file"] is old
    tools.replace(replace(old, risk=RiskLevel.HIGH), handler)
    assert "read_file" not in tools.specs
    assert not (await tools.execute("read_file", {"path": "x"})).ok
