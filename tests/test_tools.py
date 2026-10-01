from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from dataclasses import replace

import pytest

from agent_tui.tools import ToolError, ToolRegistry


@pytest.mark.parametrize(
    "path",
    ["../outside", "/etc/passwd", ".git/config", ".env", ".env.local", ".ssh/id_rsa", "cert.key"],
)
async def test_protected_paths_cannot_be_read_or_written(settings, path):
    tools = ToolRegistry(settings)
    for name, args in [
        ("read_file", {"path": path}),
        ("read_files", {"files": [{"path": path}]}),
        ("read_files", {"path": path}),
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
    assert not (await tools.execute("read_files", {"files": [{"path": "link"}]})).ok
    assert not (await tools.execute("read_files", {"path": "link"})).ok
    assert not (await tools.execute("write_file", {"path": "link", "content": "changed"})).ok
    assert outside.read_text() == "secret"
    listing = json.loads((await tools.execute("list_files", {})).content)
    assert "link" not in listing["files"]


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
        ("read_files", {"files": []}),
        ("read_files", {"path": ".", "files": [{"path": "x"}]}),
        ("read_files", {"path": 12}),
        ("read_files", {"unknown": True}),
        ("read_files", {"files": [{"path": "x"}, {"path": "x"}]}),
        ("read_files", {"files": [{"path": 12}]}),
        ("finish", {"summary": ""}),
        ("unknown", {}),
    ],
)
def test_schema_validation(settings, name, args):
    with pytest.raises(ToolError):
        ToolRegistry(settings).validate(name, json.dumps(args))


async def test_read_only_removes_and_denies_mutating_tools(readonly_settings):
    tools = ToolRegistry(readonly_settings)
    assert set(tools.specs) == {"list_files", "read_file", "read_files", "search_files", "finish"}
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
    output = await tools.execute(
        "run_command", {"argv": [sys.executable, "-c", "print('x'*1000000)"]}
    )
    assert len(output.content) < 1300


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


async def test_batch_reads_preserve_ranges_errors_and_redaction(readonly_settings, tmp_path):
    (tmp_path / "one").write_text("first\nsecond\nthird")
    (tmp_path / "two").write_text(readonly_settings.api_key)
    result = await ToolRegistry(readonly_settings).execute(
        "read_files",
        {
            "files": [
                {"path": "one", "start_line": 2, "end_line": 2},
                {"path": "missing"},
                {"path": "two"},
            ]
        },
    )
    assert not result.ok
    assert readonly_settings.api_key not in result.content
    files = json.loads(result.content)["files"]
    assert [file["path"] for file in files] == ["one", "missing", "two"]
    assert files[0]["ok"] and files[0]["content"] == "2: second"
    assert not files[1]["ok"] and files[1]["error"]
    assert files[2]["ok"] and "[REDACTED]" in files[2]["content"]


async def test_batch_output_keeps_all_files_in_valid_json(settings, tmp_path):
    paths = [f"file-{i}" for i in range(8)]
    for path in paths:
        (tmp_path / path).write_text('Źródło "🙂"\n' * 200)
    result = await ToolRegistry(replace(settings, max_output_chars=2400)).execute(
        "read_files", {"files": [{"path": path} for path in paths]}
    )
    assert result.ok and len(result.content) <= 2400
    files = json.loads(result.content)["files"]
    assert [file["path"] for file in files] == paths
    assert all(file["content"] and file["truncated"] and file["has_more"] for file in files)


async def test_batch_reads_run_concurrently_with_bounded_workers(settings, monkeypatch):
    tools = ToolRegistry(settings)
    barrier = threading.Barrier(4, timeout=2)
    lock = threading.Lock()
    active = maximum = 0

    def read(name, args):
        nonlocal active, maximum
        assert name == "read_file"
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            barrier.wait()
            return {"path": args["path"], "content": "Read", "total_lines": 1, "has_more": False}
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(tools, "_execute_file_tool", read)
    result = await tools.execute("read_files", {"files": [{"path": str(i)} for i in range(16)]})
    assert result.ok
    assert maximum == 4
    assert [file["path"] for file in json.loads(result.content)["files"]] == list(
        map(str, range(16))
    )


@pytest.mark.parametrize("arguments", [{}, {"path": "."}])
async def test_repository_read_visits_all_files_once(
    readonly_settings, tmp_path, monkeypatch, arguments
):
    paths = [f"src/nested/file-{i:02}.py" for i in range(12)] + ["README.md", ".env.example"]
    for name in paths:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("line\n" * 300 if name == "README.md" else f"Contents of {name}")
    for name in (
        ".env",
        "cert.key",
        ".git/config",
        "node_modules/dep.js",
        ".venv/dep.py",
        ".pytest_cache/data",
        ".ruff_cache/data",
        ".codebase-memory/graph",
    ):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("excluded")
    (tmp_path / "alias.py").symlink_to(tmp_path / paths[0])
    (tmp_path / "alias-dir").symlink_to(tmp_path / "src", target_is_directory=True)
    tools = ToolRegistry(readonly_settings)
    read_paths = []
    original = tools._read_text

    def read(path):
        read_paths.append(str(path.relative_to(tmp_path)))
        return original(path)

    monkeypatch.setattr(tools, "_read_text", read)
    result = await tools.execute("read_files", arguments)
    assert result.ok
    output = json.loads(result.content)
    assert sorted(read_paths) == sorted(paths)
    assert output["total_files"] == len(paths)
    assert output["failed_files"] == output["omitted_files"] == 0
    assert not output["truncated"]
    files = {file["path"]: file for file in output["files"]}
    assert set(files) == set(paths)
    assert files["README.md"]["content"].endswith("300: line")
    assert all(file["ok"] and not file["has_more"] for file in files.values())


async def test_repository_read_subtree_and_empty_directory(settings, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/file.py").write_text("source")
    (tmp_path / "outside.py").write_text("not selected")
    (tmp_path / "empty").mkdir()
    tools = ToolRegistry(settings)
    result = await tools.execute("read_files", {"path": "src"})
    assert result.ok
    assert [file["path"] for file in json.loads(result.content)["files"]] == ["src/file.py"]
    result = await tools.execute("read_files", {"path": "empty"})
    assert result.ok
    output = json.loads(result.content)
    assert output["files"] == [] and output["total_files"] == 0
    assert not output["truncated"]
    for path in ("missing", "outside.py"):
        assert not (await tools.execute("read_files", {"path": path})).ok


async def test_repository_read_reports_unsupported_files(settings, tmp_path):
    (tmp_path / "valid.py").write_text("source")
    (tmp_path / "binary").write_bytes(b"a\0b")
    (tmp_path / "large").write_bytes(b"x" * 1_000_001)
    result = await ToolRegistry(settings).execute("read_files", {})
    assert not result.ok
    output = json.loads(result.content)
    assert output["total_files"] == 3 and output["failed_files"] == 2
    files = {file["path"]: file for file in output["files"]}
    assert files["valid.py"]["content"] == "1: source"
    assert "Binary" in files["binary"]["error"]
    assert "1 MB" in files["large"]["error"]


async def test_repository_read_bounds_metadata_without_skipping_reads(
    settings, tmp_path, monkeypatch
):
    paths = [f"file-{i:03}-{'x' * 80}" for i in range(50)]
    for name in paths:
        (tmp_path / name).write_text("source " + settings.api_key)
    (tmp_path / paths[-1]).write_bytes(b"\0")
    tools = ToolRegistry(replace(settings, max_output_chars=1200))
    read_paths = []
    original = tools._read_text

    def read(path):
        read_paths.append(path.name)
        return original(path)

    monkeypatch.setattr(tools, "_read_text", read)
    result = await tools.execute("read_files", {})
    assert not result.ok  # The failed file is accounted for even if its entry is omitted.
    assert len(result.content) <= 1200
    assert settings.api_key not in result.content
    output = json.loads(result.content)
    assert sorted(read_paths) == paths
    assert 0 < len(output["files"]) < len(paths)
    assert output["total_files"] == len(paths)
    assert output["omitted_files"] == len(paths) - len(output["files"])
    assert output["failed_files"] == 1 and output["truncated"]
