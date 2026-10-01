"""Validated local tools. File paths are restricted to the selected workspace."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import heapq
import json
import os
import signal
import stat
import tempfile
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness_router import RiskLevel, ToolDescriptor
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .config import Settings

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".agent-tui",
    "dist",
    ".codebase-memory",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
MAX_FILE_BYTES = 1_000_000


class ToolError(Exception):
    """An actionable error safe to feed back to the executor."""


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


PATH = {"type": "string", "minLength": 1, "description": "Workspace-relative path"}
READ_FILE_PARAMETERS = object_schema(
    {
        "path": PATH,
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
    },
    ["path"],
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    category: str = "inspect"
    risk: RiskLevel = RiskLevel.LOW

    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            self.name, self.description, self.category, self.risk, self.parameters
        )

    def api_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


SPECS = [
    ToolSpec(
        "list_files",
        "List workspace files to discover project structure.",
        object_schema(
            {
                "path": PATH,
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 500},
            }
        ),
    ),
    ToolSpec(
        "read_file",
        "Read a UTF-8 file with line numbers; use ranges for large files.",
        READ_FILE_PARAMETERS,
    ),
    ToolSpec(
        "read_files",
        "Read the whole repository recursively in one call with {} or path='.'. "
        "Alternatively, supply files for known paths or line ranges. Available once per task; "
        "reuse the result, then use read_file for changed or omitted content. "
        "Protected files, symlinks and dependency/cache directories are excluded. "
        "Results share the output limit and report truncation.",
        object_schema(
            {
                "path": {**PATH, "description": "Directory to read recursively; defaults to '.'"},
                "files": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10000,
                    "uniqueItems": True,
                    "items": READ_FILE_PARAMETERS,
                },
            },
        )
        | {"not": {"required": ["path", "files"]}},
    ),
    ToolSpec(
        "search_files",
        "Find literal text in workspace files and return matching lines.",
        object_schema(
            {
                "query": {"type": "string", "minLength": 1},
                "path": PATH,
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            ["query"],
        ),
    ),
    ToolSpec(
        "write_file",
        "Create or replace a UTF-8 file. Read existing files before replacing them.",
        object_schema({"path": PATH, "content": {"type": "string"}}, ["path", "content"]),
        "edit",
        RiskLevel.MEDIUM,
    ),
    ToolSpec(
        "edit_file",
        "Replace one exact, unique text block in an existing UTF-8 file.",
        object_schema(
            {
                "path": PATH,
                "old_text": {"type": "string", "minLength": 1},
                "new_text": {"type": "string"},
            },
            ["path", "old_text", "new_text"],
        ),
        "edit",
        RiskLevel.MEDIUM,
    ),
    ToolSpec(
        "run_command",
        "Run a program with argv in the workspace, e.g. ['python', '-m', 'pytest']. "
        "No shell expansion. Commands have the user's OS permissions and require approval.",
        object_schema(
            {
                "argv": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 128,
                    "items": {"type": "string", "maxLength": 16000},
                },
                "cwd": PATH,
            },
            ["argv"],
        ),
        "execute",
        RiskLevel.HIGH,
    ),
    ToolSpec(
        "finish",
        "Answer the user when the task is complete, or explain a blocker needing user input.",
        object_schema({"summary": {"type": "string", "minLength": 1}}, ["summary"]),
        "respond",
    ),
]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str


class ToolRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.workspace
        self.specs = {
            spec.name: spec
            for spec in SPECS
            if not settings.read_only or spec.risk == RiskLevel.LOW
        }
        self.handlers: dict[str, Callable[[dict[str, Any]], Awaitable[ToolResult]]] = {}

    def register(
        self, spec: ToolSpec, handler: Callable[[dict[str, Any]], Awaitable[ToolResult]]
    ) -> None:
        if spec.name in self.specs:
            raise ToolError(f"Tool already registered: {spec.name}")
        Draft202012Validator.check_schema(spec.parameters)
        if self.settings.read_only and spec.risk != RiskLevel.LOW:
            return
        self.specs[spec.name] = spec
        self.handlers[spec.name] = handler

    def unregister(self, name: str) -> None:
        if name in self.handlers:
            self.handlers.pop(name)
            self.specs.pop(name)

    def descriptors(self) -> list[ToolDescriptor]:
        return [spec.descriptor() for spec in self.specs.values()]

    def schemas(self, selected: str | None = None) -> list[dict[str, Any]]:
        return [
            spec.api_schema()
            for spec in self.specs.values()
            if selected is None or selected == spec.name
        ]

    def validate(self, name: str, raw_arguments: str) -> dict[str, Any]:
        if name not in self.specs:
            raise ToolError(f"Tool is not available: {name}")
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            Draft202012Validator(self.specs[name].parameters).validate(arguments)
        except (ValueError, ValidationError) as exc:
            message = exc.message if isinstance(exc, ValidationError) else str(exc)
            raise ToolError(f"Invalid arguments for {name}: {message[:400]}") from None
        return arguments

    def requires_approval(self, name: str) -> bool:
        return self.specs[name].risk != RiskLevel.LOW and not self.settings.auto_approve

    @staticmethod
    def _protected(parts: tuple[str, ...]) -> bool:
        return any(
            part in {".git", ".ssh", ".aws", ".agent-tui"}
            or (part.startswith(".env") and part not in {".env.example", ".env.sample"})
            or part.endswith((".pem", ".key", ".p12", ".pfx"))
            for part in parts
        )

    def resolve(self, value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # abspath normalizes .. without dereferencing symlinks.
        candidate = Path(os.path.abspath(candidate))
        if not candidate.is_relative_to(self.root):
            raise ToolError("Path must stay inside the workspace")
        relative = candidate.relative_to(self.root)
        if self._protected(relative.parts):
            raise ToolError(
                "Credential files and internal metadata are not available to file tools"
            )
        cursor = self.root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ToolError("Symlink paths are not available to file tools")
        return candidate

    def _read_text(self, path: Path) -> str:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ToolError("Expected a regular text file")
        if info.st_size > MAX_FILE_BYTES:
            raise ToolError("File exceeds the 1 MB text-file limit")
        text = path.read_text(encoding="utf-8")
        if "\0" in text:
            raise ToolError("Binary files are not supported")
        return text

    def _files(self, path: Path) -> Iterator[Path]:
        if path.is_file():
            yield path
            return
        if not path.is_dir():
            raise ToolError("Directory does not exist")
        visited = 0
        for directory, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in SKIP_DIRS
                and not self._protected((d,))
                and not (Path(directory) / d).is_symlink()
            )
            for name in sorted(files):
                visited += 1
                if visited > 10000:
                    raise ToolError("Search exceeded 10,000 files; use a narrower path")
                item = Path(directory) / name
                if not self._protected((name,)) and not item.is_symlink():
                    yield item

    def _replacement(self, name: str, arguments: dict[str, Any]) -> tuple[Path, str, str]:
        path = self.resolve(arguments["path"])
        old = self._read_text(path) if path.exists() else ""
        if name == "edit_file":
            if not path.exists() or old.count(arguments["old_text"]) != 1:
                raise ToolError("old_text must match exactly once; read the file and try again")
            new = old.replace(arguments["old_text"], arguments["new_text"], 1)
        else:
            new = arguments["content"]
        if len(new.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError("New content exceeds the 1 MB text-file limit")
        return path, old, new

    def preview(self, name: str, arguments: dict[str, Any]) -> str:
        if name in {"write_file", "edit_file"}:
            path, old, new = self._replacement(name, arguments)
            relative = str(path.relative_to(self.root))
            diff = "".join(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
            return f"{relative}\n\n{diff or '(no content change)'}"
        if name == "run_command":
            cwd = self.resolve(arguments.get("cwd", "."))
            return (
                f"Directory: {cwd}\n"
                f"Timeout: {self.settings.command_timeout:g}s\n"
                f"Arguments: {json.dumps(arguments['argv'], ensure_ascii=False, indent=2)}\n\n"
                "This program runs with your OS permissions, outside a sandbox."
            )
        return json.dumps(arguments, ensure_ascii=False, indent=2)

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            # Revalidate at the execution boundary, including read-only policy.
            self.validate(name, json.dumps(arguments))
            if name in self.handlers:
                result = await self.handlers[name](arguments)
                return self.sanitize(result)
            if name == "run_command":
                output = await self._run_command(**arguments)
            elif name == "read_files":
                return await self._read_files(**arguments)
            elif name in {"list_files", "read_file", "search_files"}:
                output = await asyncio.to_thread(self._execute_file_tool, name, arguments)
            else:
                output = self._execute_file_tool(name, arguments)
            ok = not isinstance(output, dict) or output.get("exit_code", 0) == 0
            text = json.dumps(output, ensure_ascii=False)
        except (ToolError, OSError, UnicodeError, ValueError) as exc:
            ok, text = False, f"Tool error: {exc}"
        return self.sanitize(ToolResult(ok, text))

    def sanitize(self, result: ToolResult) -> ToolResult:
        text = self.settings.redact(result.content)
        limit = self.settings.max_output_chars
        if len(text) > limit:
            text = text[:limit] + "\n[output truncated; request a narrower range]"
        return ToolResult(result.ok, text)

    def _repository_files(self, path: str) -> list[dict[str, Any]]:
        directory = self.resolve(path)
        if not directory.is_dir():
            raise ToolError("Expected a directory; use files for individual file paths")
        return [
            {"path": str(item.relative_to(self.root)), "end_line": MAX_FILE_BYTES}
            for item in self._files(directory)
        ]

    async def _read_files(
        self, files: list[dict[str, Any]] | None = None, path: str = "."
    ) -> ToolResult:
        if files is None:
            files = await asyncio.to_thread(self._repository_files, path)
        content_limit = self.settings.max_output_chars

        async def read(arguments: dict[str, Any]) -> dict[str, Any]:
            try:
                output = await asyncio.to_thread(self._execute_file_tool, "read_file", arguments)
                # Redact before any clipping so a shortened credential cannot leak a prefix.
                result = {
                    "ok": True,
                    **{
                        key: self.settings.redact(value) if isinstance(value, str) else value
                        for key, value in output.items()
                    },
                }
                if len(result["content"]) > content_limit:
                    result["content"] = result["content"][:content_limit]
                    result["truncated"] = result["has_more"] = True
                return result
            except (ToolError, OSError, UnicodeError, ValueError) as exc:
                return {
                    "path": self.settings.redact(arguments["path"]),
                    "ok": False,
                    "error": self.settings.redact(str(exc))[:400],
                }

        # A fixed worker pool avoids creating a task for every file in a large repository.
        pending = iter(enumerate(files))
        results: list[dict[str, Any]] = [{} for _ in files]
        content_sizes: list[tuple[int, int]] = []
        retained_chars = 0

        async def worker() -> None:
            nonlocal retained_chars
            for index, arguments in pending:
                results[index] = await read(arguments)
                size = len(results[index].get("content", ""))
                if size:
                    heapq.heappush(content_sizes, (-size, index))
                    retained_chars += size
                # Keep retained content bounded without truncating files whose combined
                # content fits. Shrink the largest entries first to share the budget.
                while retained_chars > content_limit:
                    negative_size, largest_index = heapq.heappop(content_sizes)
                    size = -negative_size
                    kept = max(size // 2, size - (retained_chars - content_limit))
                    largest = results[largest_index]
                    largest["content"] = largest["content"][:kept]
                    largest["truncated"] = largest["has_more"] = True
                    retained_chars -= size - kept
                    if kept:
                        heapq.heappush(content_sizes, (-kept, largest_index))

        await asyncio.gather(*(worker() for _ in range(min(4, len(files)))))
        total_files = len(results)
        failed_files = sum(not result["ok"] for result in results)
        truncated = any(result.get("has_more", False) for result in results)

        def serialize() -> str:
            return self.settings.redact(
                json.dumps(
                    {
                        "files": results,
                        "total_files": total_files,
                        "failed_files": failed_files,
                        "omitted_files": total_files - len(results),
                        "truncated": truncated or len(results) < total_files,
                    },
                    ensure_ascii=False,
                )
            )

        # If even metadata cannot fit, retain a bounded prefix and report exactly how many
        # entries were omitted. Every discovered file has still been read once above.
        full_results = results
        results = [
            {**result, "content": "", "truncated": True, "has_more": True}
            if result.get("content")
            else result
            for result in full_results
        ]
        metadata_fits = len(serialize()) <= self.settings.max_output_chars
        results = full_results
        if not metadata_fits:
            low, high = 0, len(results)
            while low < high:
                mid = (low + high + 1) // 2
                results = full_results[:mid]
                if len(serialize()) <= self.settings.max_output_chars:
                    low = mid
                else:
                    high = mid - 1
            results = full_results[:low]

        text = serialize()
        while len(text) > self.settings.max_output_chars:
            candidates = [result for result in results if result.get("content")]
            if not candidates:
                return self.sanitize(
                    ToolResult(False, "Repository summary exceeds the output limit; increase it.")
                )
            # Keep every file's metadata and valid JSON instead of clipping the last files away.
            largest = max(candidates, key=lambda result: len(result["content"]))
            largest["content"] = largest["content"][: len(largest["content"]) // 2]
            largest["truncated"] = True
            largest["has_more"] = True
            truncated = True
            text = serialize()
        return ToolResult(failed_files == 0, text)

    def _execute_file_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "finish":
            return args["summary"]
        if name in {"write_file", "edit_file"}:
            path, _, content = self._replacement(name, args)
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent, delete=False
                ) as temp:
                    temp_name = temp.name
                    temp.write(content)
                os.chmod(temp_name, mode)
                os.replace(temp_name, path)
            finally:
                if temp_name and os.path.exists(temp_name):
                    os.unlink(temp_name)
            return {"path": str(path.relative_to(self.root)), "bytes": len(content.encode("utf-8"))}
        path = self.resolve(args.get("path", "."))
        if name == "list_files":
            entries = []
            limit = args.get("max_entries", 200)
            for item in self._files(path):
                if len(entries) == limit:
                    return {"files": entries, "truncated": True}
                entries.append(str(item.relative_to(self.root)))
            return {"files": entries, "truncated": False}
        if name == "read_file":
            lines = self._read_text(path).splitlines()
            start = args.get("start_line", 1)
            end = args.get("end_line", start + 249)
            if end < start:
                raise ToolError("end_line must be at least start_line")
            return {
                "path": str(path.relative_to(self.root)),
                "total_lines": len(lines),
                "content": "\n".join(
                    f"{n}: {lines[n - 1]}" for n in range(start, min(end, len(lines)) + 1)
                ),
                "has_more": end < len(lines),
            }
        if name == "search_files":
            matches, skipped = [], 0
            scanned_bytes = 0
            for item in self._files(path):
                try:
                    scanned_bytes += item.stat().st_size
                    if scanned_bytes > 20_000_000:
                        raise ToolError("Search exceeds 20 MB; use a narrower path")
                    lines = self._read_text(item).splitlines()
                except (OSError, UnicodeError, ToolError):
                    if scanned_bytes > 20_000_000:
                        return {
                            "matches": matches,
                            "truncated": True,
                            "skipped": skipped,
                            "note": "20 MB scan limit reached; use a narrower path",
                        }
                    skipped += 1
                    continue
                for number, line in enumerate(lines, 1):
                    if args["query"] in line:
                        if len(matches) == args.get("max_matches", 50):
                            return {"matches": matches, "truncated": True, "skipped": skipped}
                        matches.append(
                            {
                                "path": str(item.relative_to(self.root)),
                                "line": number,
                                "text": line[:1000],
                            }
                        )
            return {"matches": matches, "truncated": False, "skipped": skipped}
        raise ToolError(f"Unknown tool: {name}")

    async def _run_command(self, argv: list[str], cwd: str = ".") -> dict[str, Any]:
        directory = self.resolve(cwd)
        if not directory.is_dir():
            raise ToolError("Command directory does not exist")
        if not argv[0] or any("\0" in arg for arg in argv):
            raise ToolError("Invalid command arguments")
        # Do not pass model credentials to subprocesses. This is not an OS sandbox.
        env = {
            key: value
            for key, value in os.environ.items()
            if not any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
        }
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=directory,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        output = bytearray()
        truncated = False

        async def drain() -> None:
            nonlocal truncated
            assert process.stdout is not None
            while chunk := await process.stdout.read(8192):
                remaining = self.settings.max_output_chars - len(output)
                output.extend(chunk[: max(remaining, 0)])
                truncated |= len(chunk) > remaining
            await process.wait()

        task = asyncio.create_task(drain())

        async def stop() -> None:
            # Kill the whole process group, including children holding the pipe open.
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.returncode is None:
                    process.kill()
            await process.wait()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(task), self.settings.command_timeout)
        except TimeoutError:
            timed_out = True
            await stop()
        except asyncio.CancelledError:
            await stop()
            raise
        return {
            "exit_code": process.returncode if not timed_out else -1,
            "output": output.decode("utf-8", errors="replace"),
            "truncated": truncated,
            "timed_out": timed_out,
        }
