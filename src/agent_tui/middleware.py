"""Ordered async middleware and bounded JSON stdin/stdout command hooks."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import signal
from collections.abc import Awaitable, Callable
from typing import Any

from .config import Settings
from .extensions import HOOK_EVENTS, HookConfig, safe_environment
from .models import Approval
from .tools import ToolError

Middleware = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | None]]


class HookError(ToolError):
    pass


class MiddlewareManager:
    def __init__(self, settings: Settings, hooks: tuple[HookConfig, ...] = ()) -> None:
        self.settings = settings
        self.hooks = tuple(h for h in hooks if h.enabled)
        self.handlers: list[Middleware] = []
        self.commands_enabled = False

    def add(self, handler: Middleware) -> None:
        self.handlers.append(handler)

    async def authorize(self, approve: Approval) -> None:
        self.commands_enabled = False
        if not self.hooks or self.settings.demo or self.settings.read_only:
            return
        preview = "Command hooks for this run (execute with your OS permissions):\n" + "\n".join(
            f"{h.event}: {json.dumps(h.command)} (timeout {h.timeout:g}s)" for h in self.hooks
        )
        if not self.settings.auto_approve and not await approve(
            "hooks", self.settings.redact(preview)
        ):
            raise HookError("User denied command hooks; the run was not started")
        self.commands_enabled = True

    async def dispatch(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        if event not in HOOK_EVENTS:
            raise HookError(f"Unknown hook event: {event}")
        data = copy.deepcopy(payload)
        for handler in self.handlers:
            try:
                async with asyncio.timeout(self.settings.command_timeout):
                    patch = await handler(event, copy.deepcopy(data))
            except Exception as exc:
                raise HookError(f"Middleware {event} failed: {exc}") from exc
            self._merge(event, data, patch)
        if self.commands_enabled:
            for hook in self.hooks:
                if hook.event == event:
                    patch = await self._command(hook, data)
                    self._merge(event, data, patch)
        return data

    @staticmethod
    def _merge(event: str, data: dict, patch: Any) -> None:
        if patch is None:
            return
        if not isinstance(patch, dict):
            raise HookError(f"Hook {event} must return a JSON object or no output")
        if patch.get("block"):
            raise HookError(f"Blocked by {event} hook: {patch['block']}")
        # Deliberately narrow mutations: approval always sees the final arguments.
        allowed = {"block"}
        if event == "before_tool":
            allowed.add("arguments")
            if "arguments" in patch and not isinstance(patch["arguments"], dict):
                raise HookError("Hook arguments must be an object")
        if event == "before_model":
            allowed.add("context")
            if "context" in patch and not isinstance(patch["context"], str):
                raise HookError("Hook context must be a string")
        if set(patch) - allowed:
            raise HookError(
                f"Unsupported output fields for {event}: {sorted(set(patch) - allowed)}"
            )
        data.update({key: value for key, value in patch.items() if key != "block"})

    async def _command(self, hook: HookConfig, data: dict[str, Any]) -> dict | None:
        payload = self.settings.redact(
            json.dumps(
                {"event": hook.event, "workspace": str(self.settings.workspace), **data},
                ensure_ascii=False,
            )
        ).encode()
        process = await asyncio.create_subprocess_exec(
            *hook.command,
            cwd=self.settings.workspace,
            env=safe_environment(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )

        async def read(stream) -> bytes:
            output = bytearray()
            while chunk := await stream.read(8192):
                output.extend(chunk)
                if len(output) > 64000:
                    raise HookError(f"Hook {hook.event} output exceeded 64 KB")
            return bytes(output)

        async def write() -> None:
            stdin = process.stdin
            assert stdin is not None  # Created with stdin=PIPE above.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                stdin.write(payload)
                await stdin.drain()
            stdin.close()

        tasks = [
            asyncio.create_task(read(process.stdout)),
            asyncio.create_task(read(process.stderr)),
            asyncio.create_task(write()),
        ]
        try:
            async with asyncio.timeout(hook.timeout):
                stdout, stderr, _ = await asyncio.gather(*tasks)
                await process.wait()
            if process.returncode:
                raise HookError(
                    f"Hook {hook.event} exited {process.returncode}: "
                    + stderr.decode(errors="replace")[:1000]
                )
            return json.loads(stdout) if stdout.strip() else None
        except TimeoutError:
            raise HookError(f"Hook {hook.event} timed out after {hook.timeout:g}s") from None
        except (ValueError, UnicodeError) as exc:
            raise HookError(f"Hook {hook.event} returned invalid JSON: {exc}") from None
        finally:
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.returncode is None:
                    process.kill()
            await process.wait()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
