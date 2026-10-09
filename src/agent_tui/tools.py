"""Validated local tools. File paths are restricted to the selected workspace."""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import hashlib
import json
import os
import signal
import stat
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from harness_router import RiskLevel, ToolDescriptor
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .config import Settings
from .relevance import keywords

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

    def __init__(
        self, message: str, *, code: str = "tool_error", retry_hint: str | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_hint = retry_hint


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def bounded_json(payload: dict[str, Any], field: str, limit: int) -> str:
    """Shorten one text/list field without cutting JSON escapes or list entries."""
    payload = {**payload, "truncated": True}
    value = payload[field]
    payload[field] = value[:0]
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > limit:
        # Even the metadata cannot fit. Preserve an explicit truncation signal whenever
        # the configured budget permits it, rather than returning malformed JSON.
        return '{"truncated":true}' if limit >= 18 else "0"
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        payload[field] = value[:middle]
        candidate = json.dumps(payload, ensure_ascii=False)
        if len(candidate) <= limit:
            low, text = middle, candidate
        else:
            high = middle - 1
    return text


PATH = {
    "type": "string",
    "minLength": 1,
    "description": "Workspace-relative path, e.g. README.md; omit the workspace folder prefix.",
}
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
        "Read a UTF-8 file with line numbers; use ranges for large files. When changes are "
        "requested, follow the read with edit_file or write_file and verify the result.",
        READ_FILE_PARAMETERS,
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
        "context_collect",
        "Find and read the most relevant workspace files in one batch; cache their exact "
        "source in Context Manager for later Code Mode steps and repository refactors. "
        "Accept a query, explicit paths, or both. Each result includes a source SHA-256. "
        "Safe read-only operation; repeat only when the source changes.",
        object_schema(
            {
                "query": {"type": "string"},
                "paths": {
                    "type": "array", "maxItems": 24,
                    "items": PATH, "uniqueItems": True,
                },
                "path": PATH,
                "max_files": {"type": "integer", "minimum": 1, "maximum": 24},
                "max_chars": {"type": "integer", "minimum": 500, "maximum": 14000},
            }
        ),
    ),
    ToolSpec(
        "apply_patchset",
        "Refactor several previously inspected files and/or create new files in one "
        "preflight-validated change set and one approval. For edits, use an exact unique "
        "old_text and optional expected_sha256 from context_collect. Create actions "
        "never overwrite existing files. Always inspect and verify affected files.",
        object_schema(
            {
                "changes": {
                    "type": "array", "minItems": 1, "maxItems": 24,
                    "items": object_schema(
                        {
                            "action": {"enum": ["edit", "create"]},
                            "path": PATH,
                            "old_text": {"type": "string", "minLength": 1},
                            "new_text": {"type": "string"},
                            "content": {"type": "string"},
                            "expected_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                        ["action", "path"],
                    ) | {
                        "oneOf": [
                            {
                                "properties": {"action": {"const": "edit"}},
                                "required": ["old_text", "new_text"],
                                "not": {"required": ["content"]},
                            },
                            {
                                "properties": {"action": {"const": "create"}},
                                "required": ["content"],
                                "not": {
                                    "anyOf": [
                                        {"required": ["old_text"]},
                                        {"required": ["new_text"]},
                                        {"required": ["expected_sha256"]},
                                    ]
                                },
                            },
                        ]
                    },
                },
            },
            ["changes"],
        ),
        "edit",
        RiskLevel.MEDIUM,
    ),
    ToolSpec(
        "write_file",
        "Create a UTF-8 file, or replace one in full only when the task explicitly requests it. "
        "Use small edit_file patches for refactoring and other changes to existing files. "
        "Read existing files before replacing them.",
        object_schema({"path": PATH, "content": {"type": "string"}}, ["path", "content"]),
        "edit",
        RiskLevel.MEDIUM,
    ),
    ToolSpec(
        "edit_file",
        "Apply a small patch by replacing one exact, unique text block in an existing UTF-8 "
        "file. Prefer this tool for refactoring and other changes to existing files; "
        "read the relevant content first.",
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
class ToolFailure:
    code: str
    message: str
    retry_hint: str | None = None


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    error: ToolFailure | None = None
    truncated: bool = False

    @classmethod
    def from_error(cls, error: Exception) -> ToolResult:
        return cls(
            False,
            str(error),
            ToolFailure(
                error.code if isinstance(error, ToolError) else "tool_error",
                str(error),
                error.retry_hint if isinstance(error, ToolError) else None,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """One model-facing envelope for both chat replies and the sandbox bridge."""
        try:
            output = json.loads(self.content, parse_constant=_reject_nonfinite)
        except (ValueError, RecursionError):
            output = self.content
        error = self.error
        if not self.ok:
            if error is None:
                message = (
                    output.get("error", "Tool failed; inspect output for details.")
                    if isinstance(output, dict)
                    else self.content
                )
                error = ToolFailure("tool_error", str(message))
            if isinstance(output, str):
                output = None
        return {
            "ok": self.ok,
            "output": output,
            "error": asdict(error) if not self.ok and error else None,
            "truncated": self.truncated
            or isinstance(output, dict)
            and output.get("truncated") is True,
        }


@dataclass(frozen=True)
class _IndexedSource:
    """Metadata-keyed search excerpt; large source bodies remain on disk."""

    size: int
    mtime_ns: int
    folded_prefix: str
    memory_bytes: int


class ToolRegistry:
    MAX_SEARCH_INDEX_BYTES = 8 * 1024 * 1024

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.workspace
        self.specs = {
            spec.name: spec
            for spec in SPECS
            if not settings.read_only or spec.risk == RiskLevel.LOW
        }
        self.handlers: dict[str, Callable[[dict[str, Any]], Awaitable[ToolResult]]] = {}
        self.context = None
        self._search_index: OrderedDict[str, _IndexedSource] = OrderedDict()
        self._search_index_bytes = 0
        self._index_lock = threading.RLock()

    def clear_index(self) -> None:
        with self._index_lock:
            self._search_index.clear()
            self._search_index_bytes = 0

    def _indexed_text(
        self, path: Path, *, size: int, mtime_ns: int
    ) -> tuple[str, str | None, bool]:
        """Only read unchanged files once across repeated repository searches.

        A hit is validated using the same filesystem metadata used by the source
        context cache. Return full source on misses so selected files need no reread.
        """
        key = str(path.relative_to(self.root))
        with self._index_lock:
            cached = self._search_index.get(key)
            if cached and cached.size == size and cached.mtime_ns == mtime_ns:
                self._search_index.move_to_end(key)
                return cached.folded_prefix, None, True
            if cached:
                self._search_index_bytes -= cached.memory_bytes
                del self._search_index[key]
        content = self._read_text(path)
        updated = path.stat()
        if updated.st_size != size or updated.st_mtime_ns != mtime_ns:
            raise ToolError("File changed while indexing; collect again", code="file_changed")
        prefix = content[:100_000].casefold()
        memory_bytes = len(prefix.encode("utf-8"))
        if memory_bytes <= self.MAX_SEARCH_INDEX_BYTES:
            with self._index_lock:
                self._search_index[key] = _IndexedSource(
                    size, mtime_ns, prefix, memory_bytes
                )
                self._search_index_bytes += memory_bytes
                while self._search_index_bytes > self.MAX_SEARCH_INDEX_BYTES:
                    _, evicted = self._search_index.popitem(last=False)
                    self._search_index_bytes -= evicted.memory_bytes
        return prefix, content, False

    def bind_context(self, manager: Any) -> None:
        """Attach the current agent's workspace cache without sharing it across agents."""
        self.context = manager

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
        self.handlers.pop(name, None)
        self.specs.pop(name, None)

    def replace(
        self, spec: ToolSpec, handler: Callable[[dict[str, Any]], Awaitable[ToolResult]]
    ) -> None:
        """Replace a tool, including a built-in, without exposing a disallowed risk level."""
        Draft202012Validator.check_schema(spec.parameters)
        self.unregister(spec.name)
        self.register(spec, handler)

    def descriptors(self) -> list[ToolDescriptor]:
        return [spec.descriptor() for spec in self.specs.values()]

    def schemas(self, selected: str | None = None) -> list[dict[str, Any]]:
        return [
            spec.api_schema()
            for spec in self.specs.values()
            if selected is None or selected == spec.name
        ]

    def validate(self, name: str, raw_arguments: str | dict[str, Any]) -> dict[str, Any]:
        if name not in self.specs:
            raise ToolError(f"Tool is not available: {name}", code="unavailable_tool")
        try:
            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments, parse_constant=_reject_nonfinite)
                # Some OpenAI-compatible providers double-encode the argument object.
                # Unwrap at most once and still validate against the exact tool schema.
                if isinstance(arguments, str):
                    arguments = json.loads(arguments, parse_constant=_reject_nonfinite)
            elif isinstance(raw_arguments, dict):
                # Enforce JSON-only values at the boundary, including finite numbers.
                arguments = json.loads(
                    json.dumps(raw_arguments, ensure_ascii=False, allow_nan=False),
                    parse_constant=_reject_nonfinite,
                )
            else:
                raise ValueError("Tool arguments must be a JSON object or encoded object")
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            Draft202012Validator(self.specs[name].parameters).validate(arguments)
        except (TypeError, ValueError, OverflowError, RecursionError, ValidationError) as exc:
            message = exc.message if isinstance(exc, ValidationError) else str(exc)
            raise ToolError(
                f"Invalid arguments for {name}: {message[:400]}",
                code="invalid_arguments",
                retry_hint="Use the current tool schema to correct the arguments.",
            ) from None
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
        try:
            info = path.stat()
        except FileNotFoundError:
            relative = path.relative_to(self.root)
            message = (
                f"File not found: {str(relative)!r}. Paths are relative to {str(self.root)!r}."
            )
            if relative.parts and relative.parts[0] == self.root.name:
                suggested = Path(*relative.parts[1:])
                try:
                    corrected = self.resolve(str(suggested))
                except ToolError:
                    pass
                else:
                    if corrected.is_file():
                        message += f" Use {str(suggested)!r}."
            raise ToolError(message) from None
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

    def _collect_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Scan metadata once; cache search text and retain chosen source in context.

        Subsequent searches only stat unchanged files, then read the few winners.
        Full sources stay with Context Manager and excerpts are bounded for the model.
        """
        query = args.get("query", "").strip()
        terms = keywords(query)
        max_files = args.get("max_files", 8)
        max_chars = min(
            args.get("max_chars", 10000), max(500, self.settings.max_output_chars - 2500)
        )
        root = self.resolve(args.get("path", "."))
        requested = args.get("paths") or []
        ranked: list[tuple[int, str, Path, str | None]] = []
        skipped = 0
        scanned_bytes = 0
        disk_reads = 0
        index_hits = 0
        scan_limit_reached = False

        if requested:
            candidates = [self.resolve(value) for value in requested]
            if any(not item.is_file() for item in candidates):
                raise ToolError("context_collect paths must be existing regular files")
        else:
            candidates = self._files(root)

        for item in candidates:
            try:
                info = item.stat()
                if info.st_size > MAX_FILE_BYTES:
                    skipped += 1
                    continue
                if not requested and scanned_bytes + info.st_size > 20_000_000:
                    scan_limit_reached = True
                    break
                scanned_bytes += info.st_size
                relative = str(item.relative_to(self.root))
                content: str | None = None
                if requested:
                    # Preserve explicit path order, even if a scan query was supplied.
                    score = len(requested) - requested.index(
                        str(item.relative_to(self.root))
                    ) if relative in requested else 1
                elif not terms:
                    # Listing a project should not read the source of every file.
                    score = (
                        3 if item.name in {"README.md", "pyproject.toml", "go.mod"}
                        else 1 if item.suffix in {".py", ".go", ".ts", ".js", ".rs"}
                        else 0
                    )
                else:
                    folded, content, hit = self._indexed_text(
                        item, size=info.st_size, mtime_ns=info.st_mtime_ns
                    )
                    disk_reads += not hit
                    index_hits += hit
                    path_hits = len(terms & keywords(relative))
                    body_hits = sum(term in folded for term in terms)
                    score = 6 * path_hits + body_hits
                    if score == 0:
                        continue
                ranked.append((score, relative, item, content))
            except (OSError, UnicodeError, ToolError):
                skipped += 1
                continue

        if not requested:
            ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        chosen = ranked[:max_files]
        # Leave headroom for JSON keys, paths and digests under output sanitization.
        remaining = max_chars
        files: list[dict[str, Any]] = []
        for position, (_, path, source_path, content) in enumerate(chosen):
            try:
                if content is None:
                    cached = (
                        self.context.cached_file(path)
                        if self.context is not None and hasattr(self.context, "cached_file")
                        else None
                    )
                    if cached is not None:
                        content = cached.content
                    else:
                        content = self._read_text(source_path)
                        disk_reads += 1
            except (OSError, UnicodeError, ToolError):
                skipped += 1
                continue

            slots = len(chosen) - position
            allowance = max(0, (remaining // slots) - 180)
            lower = content.casefold()
            positions = [lower.find(term) for term in terms]
            found = [position for position in positions if position >= 0]
            anchor = min(found) if found else 0
            offset = max(0, anchor - allowance // 3)
            if offset:
                offset = content.rfind("\n", 0, offset) + 1
            excerpt = content[offset : offset + allowance]
            remaining -= len(excerpt) + 180
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if self.context is not None and hasattr(self.context, "cache_file"):
                digest = self.context.cache_file(path, content)
            files.append(
                {
                    "path": path,
                    "sha256": digest,
                    "content": excerpt,
                    "start_line": content.count("\n", 0, offset) + 1,
                    "total_lines": len(content.splitlines()),
                    "excerpt": offset > 0 or len(excerpt) < len(content),
                }
            )
        return {
            "files": files,
            "scanned_bytes": scanned_bytes,
            "disk_reads": disk_reads,
            "index_hits": index_hits,
            "skipped": skipped,
            "truncated": scan_limit_reached or len(ranked) > max_files
            or any(item["excerpt"] for item in files),
        }

    def _patchset_preflight(self, args: dict[str, Any]) -> list[tuple[Path, str, str]]:
        prepared: list[tuple[Path, str, str]] = []
        seen: set[Path] = set()
        for item in args["changes"]:
            path = self.resolve(item["path"])
            if path in seen:
                raise ToolError("Each path may appear only once per patchset")
            seen.add(path)
            relative = str(path.relative_to(self.root))
            action = item["action"]
            if action == "create":
                if set(item) != {"action", "path", "content"}:
                    raise ToolError("Create needs action, path and content only")
                if path.exists():
                    raise ToolError(f"File already exists: {relative}", code="file_exists")
                old, new = "", item["content"]
            else:
                if not {"old_text", "new_text"}.issubset(item) or "content" in item:
                    raise ToolError("Edit requires old_text and new_text, not content")
                old = self._read_text(path)
                digest = hashlib.sha256(old.encode("utf-8")).hexdigest()
                if "expected_sha256" in item and item["expected_sha256"] != digest:
                    raise ToolError(
                        f"File changed since context collection: {relative}", code="file_changed"
                    )
                if self.context is not None and hasattr(self.context, "cached_file"):
                    cached = self.context.cached_file(relative)
                    if cached is None or cached.sha256 != digest:
                        raise ToolError(
                            f"Stale or missing context for {relative}; call context_collect first",
                            code="stale_context",
                        )
                if old.count(item["old_text"]) != 1:
                    raise ToolError(f"old_text must match exactly once in {relative}")
                new = old.replace(item["old_text"], item["new_text"], 1)
            if len(new.encode("utf-8")) > MAX_FILE_BYTES:
                raise ToolError(f"New file exceeds the 1 MB limit: {relative}")
            prepared.append((path, old, new))
        return prepared

    def _apply_patchset(self, args: dict[str, Any]) -> dict[str, Any]:
        """Preflight all files before writing and stage replacements on the same volume."""
        prepared = self._patchset_preflight(args)
        pending: list[tuple[Path, str, str, str, bool]] = []
        applied: list[tuple[Path, str, str, bool]] = []
        try:
            for (path, old, new), change in zip(prepared, args["changes"], strict=True):
                # Approval may have taken time; re-check source before staging.
                creating = change["action"] == "create"
                if creating and path.exists():
                    raise ToolError("File appeared since preflight", code="file_changed")
                if not creating and (not path.exists() or self._read_text(path) != old):
                    raise ToolError("File changed since preflight", code="file_changed")
                path.parent.mkdir(parents=True, exist_ok=True)
                mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=path.parent, delete=False
                ) as tmp:
                    tmp.write(new)
                    staged = tmp.name
                os.chmod(staged, mode)
                pending.append((path, old, new, staged, creating))
            for path, old, new, staged, creating in pending:
                # Reapply workspace path rules before committing staged data.
                if self.resolve(str(path.relative_to(self.root))) != path:
                    raise ToolError("Workspace path changed during patchset", code="file_changed")
                # Reject concurrent creates as well as concurrent edits.
                if creating and path.exists():
                    raise ToolError("File appeared during patchset", code="file_changed")
                if not creating and (not path.exists() or self._read_text(path) != old):
                    raise ToolError("File changed during patchset", code="file_changed")
                if creating:
                    # An exclusive hard link cannot clobber a concurrent new file.
                    os.link(staged, path)
                else:
                    os.replace(staged, path)
                applied.append((path, old, new, creating))
            for path, _, new, _ in applied:
                if self.context is not None and hasattr(self.context, "cache_file"):
                    self.context.cache_file(str(path.relative_to(self.root)), new)
        except Exception:
            # A commit can fail mid-way (e.g. disk full). Restore best-effort, but never
            # roll back a path another process already changed after our replacement.
            for path, old, new, creating in reversed(applied):
                if path.exists() and self._read_text(path) == new:
                    if not creating:
                        with tempfile.NamedTemporaryFile(
                            mode="w", encoding="utf-8", dir=path.parent, delete=False
                        ) as tmp:
                            tmp.write(old)
                            rollback = tmp.name
                        os.chmod(rollback, stat.S_IMODE(path.stat().st_mode))
                        os.replace(rollback, path)
                    else:
                        path.unlink()
                if self.context is not None and hasattr(self.context, "invalidate_file"):
                    self.context.invalidate_file(str(path.relative_to(self.root)))
            raise
        finally:
            for _, _, _, staged, _ in pending:
                if os.path.exists(staged):
                    os.unlink(staged)
        return {
            "changes": [
                {"path": str(path.relative_to(self.root)), "bytes": len(new.encode("utf-8"))}
                for path, _, new in prepared
            ],
            "count": len(prepared),
        }

    def preview(self, name: str, arguments: dict[str, Any]) -> str:
        if name in self.handlers:
            # A replacement may use a different schema and implementation from the built-in.
            return json.dumps(arguments, ensure_ascii=False, indent=2)
        if name == "apply_patchset":
            previews = []
            for path, old, new in self._patchset_preflight(arguments):
                relative = str(path.relative_to(self.root))
                diff = "".join(
                    difflib.unified_diff(
                        old.splitlines(keepends=True),
                        new.splitlines(keepends=True),
                        fromfile=f"a/{relative}",
                        tofile=f"b/{relative}",
                    )
                )
                previews.append(diff or f"{relative}: no change")
            return "\n".join(previews)
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
                if not isinstance(result, ToolResult):
                    raise ToolError(
                        f"Tool {name} returned {type(result).__name__}; expected ToolResult",
                        code="invalid_tool_response",
                        retry_hint="Fix the tool handler to return ToolResult(ok, content).",
                    )
                return self.sanitize(result, name)
            if name == "run_command":
                output = await self._run_command(**arguments)
            elif name in {
                "list_files",
                "read_file",
                "search_files",
                "context_collect",
            }:
                output = await asyncio.to_thread(self._execute_file_tool, name, arguments)
            else:
                output = self._execute_file_tool(name, arguments)
            ok = not isinstance(output, dict) or output.get("exit_code", 0) == 0
            text = json.dumps(output, ensure_ascii=False)
            error = None
            if name == "run_command" and not ok:
                error = ToolFailure(
                    "command_timeout" if output["timed_out"] else "command_failed",
                    "Command timed out."
                    if output["timed_out"]
                    else f"Command exited with status {output['exit_code']}.",
                    "Inspect command output and current state before retrying.",
                )
            result = ToolResult(ok, text, error)
        except (ToolError, OSError, UnicodeError, ValueError, TypeError) as exc:
            result = ToolResult.from_error(exc)
        return self.sanitize(result, name)

    def sanitize(self, result: ToolResult, name: str | None = None) -> ToolResult:
        if (
            not isinstance(result, ToolResult)
            or type(result.ok) is not bool
            or not isinstance(result.content, str)
            or (result.error is not None and not isinstance(result.error, ToolFailure))
            or type(result.truncated) is not bool
        ):
            result = ToolResult.from_error(
                ToolError(
                    "Tool handler returned an invalid ToolResult; expected text content "
                    "and a structured ToolFailure for errors",
                    code="invalid_tool_response",
                )
            )
        text = self.settings.redact(result.content)
        limit = self.settings.max_output_chars
        truncated = result.truncated or len(text) > limit
        if len(text) > limit:
            try:
                payload = json.loads(text)
            except ValueError:
                marker = "\n[output truncated; request a narrower range]"
                text = text[: max(0, limit - len(marker))] + marker[:limit]
            else:
                field = {
                    "list_files": "files",
                    "read_file": "content",
                    "search_files": "matches",
                    "context_collect": "files",
                    "apply_patchset": "changes",
                    "run_command": "output",
                }.get(name)
                if (
                    isinstance(payload, dict)
                    and field is not None
                    and isinstance(payload.get(field), (str, list))
                ):
                    if name == "read_file":
                        payload["has_more"] = True
                    text = bounded_json(payload, field, limit)
                else:
                    text = bounded_json({"output_excerpt": text}, "output_excerpt", limit)
        error = result.error
        if error is not None:
            message = self.settings.redact(error.message)
            hint = self.settings.redact(error.retry_hint) if error.retry_hint else None
            truncated |= len(message) > limit or bool(hint and len(hint) > limit)
            error = replace(
                error,
                code=self.settings.redact(error.code)[:100],
                message=message[:limit],
                retry_hint=hint[:limit] if hint else None,
            )
        return ToolResult(result.ok, text, error, truncated)

    def _execute_file_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name == "context_collect":
            return self._collect_context(args)
        if name == "apply_patchset":
            return self._apply_patchset(args)
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
            if self.context is not None and hasattr(self.context, "cache_file"):
                self.context.cache_file(str(path.relative_to(self.root)), content)
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
            source = self._read_text(path)
            if self.context is not None and hasattr(self.context, "cache_file"):
                self.context.cache_file(str(path.relative_to(self.root)), source)
            lines = source.splitlines()
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
