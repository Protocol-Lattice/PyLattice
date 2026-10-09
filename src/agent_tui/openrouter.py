"""OpenRouter streaming chat client; no tool side effects occur in this module."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Settings
from .models import Completion, TokenSink, ToolCall

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class ExecutorError(Exception):
    """A provider or protocol failure with a user-readable message."""


class _ResponseShapeError(ExecutorError):
    """An upstream streaming normalization failure; retry once without streaming."""


class _InterruptedResponseError(ExecutorError):
    """The provider ended generation before returning a complete, usable response."""


def response_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if (
                not isinstance(part, dict)
                or part.get("type") not in {"text", "output_text"}
                or not isinstance(part.get("text"), str)
            ):
                raise ExecutorError("Unsupported non-text content in executor response")
            parts.append(part["text"])
        return "".join(parts)
    raise ExecutorError("Expected text or text content blocks from the executor")


def provider_error(message: str, code: Any = None) -> ExecutorError:
    if "unsupported response content shape" in message.casefold():
        return _ResponseShapeError(message[:500])
    if str(code) in {"408", "429", "500", "502", "503", "504"}:
        return _InterruptedResponseError(message[:500])
    return ExecutorError(message[:500])


class _RetryableStatus(Exception):
    def __init__(self, status: int, delay: float) -> None:
        self.status, self.delay = status, delay


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Read SSE data records, including comments and multi-line data fields."""
    lines: list[str] = []
    size = 0
    async for line in response.aiter_lines():
        if not line:
            if lines:
                yield "\n".join(lines)
                lines, size = [], 0
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value)
            if size > 1_000_000:
                raise ExecutorError("Provider sent an oversized stream event")
            lines.append(value)
    if lines:
        yield "\n".join(lines)


class _Accumulator:
    def __init__(self) -> None:
        self.content = ""
        self.calls: dict[int, dict[str, str]] = {}
        self.model = ""
        self.tokens = 0
        self.finish_reason: str | None = None
        self.size = 0

    async def add(self, payload: dict[str, Any], on_token: TokenSink) -> None:
        if "error" in payload:
            error = payload["error"]
            message = (
                error.get("message", "Stream error") if isinstance(error, dict) else str(error)
            )
            raise provider_error(
                str(message), error.get("code") if isinstance(error, dict) else None
            )
        self.model = payload.get("model") or self.model
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            raise _ResponseShapeError("Invalid token usage in executor response")
        if isinstance(usage.get("total_tokens"), int):
            self.tokens = usage["total_tokens"]
        choices = payload.get("choices") or []
        if not isinstance(choices, list):
            raise _ResponseShapeError("Executor choices must be a list")
        if not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _ResponseShapeError("Executor choice must be an object")
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, dict):
            raise _ResponseShapeError("Executor message must be an object")
        content = response_text(delta.get("content"))
        self.content += content
        self.size += len(content)
        if content:
            await on_token(content)
        fragments = delta.get("tool_calls")
        if fragments is None and delta.get("function_call") is not None:
            # Some OpenAI-compatible endpoints still use the legacy single-call shape.
            fragments = [{"index": 0, "id": "legacy_call_0",
                          "function": delta["function_call"]}]
        if fragments is None:
            fragments = []
        if isinstance(fragments, dict):
            fragments = [fragments]
        if not isinstance(fragments, list):
            raise _ResponseShapeError("Executor tool_calls must be a list")
        for position, fragment in enumerate(fragments):
            if not isinstance(fragment, dict):
                raise _ResponseShapeError("Executor tool call must be an object")
            index = fragment.get("index", position)
            if type(index) is not int or not 0 <= index < 16:
                raise _ResponseShapeError("Invalid tool-call index in response")
            call = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            call_id = fragment.get("id")
            if call_id is not None:
                if not isinstance(call_id, str):
                    raise _ResponseShapeError("Invalid tool-call ID in response")
                call["id"] = call_id
            function = fragment.get("function") or {}
            if not isinstance(function, dict):
                raise _ResponseShapeError("Executor tool function must be an object")
            name = function.get("name")
            if name is not None:
                if not isinstance(name, str):
                    raise _ResponseShapeError("Invalid tool name in response")
                call["name"] += name
                self.size += len(name)
            arguments = function.get("arguments")
            if arguments is not None:
                if isinstance(arguments, str):
                    # Streaming fragments are partial JSON, concatenated in order.
                    call["arguments"] += arguments
                    self.size += len(arguments)
                elif isinstance(arguments, dict) and not call["arguments"]:
                    # Non-streaming providers may return a JSON object, not a JSON string.
                    encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
                    call["arguments"] = encoded
                    self.size += len(encoded)
                else:
                    raise _ResponseShapeError("Tool arguments must be JSON text or an object")
        if self.size > 1_000_000:
            raise ExecutorError("Provider response exceeds the 1 MB limit")

    def finish(self) -> Completion:
        if self.finish_reason in {"length", "max_tokens"}:
            raise ExecutorError(
                "Executor reached its token limit; no tool was executed. Retry a "
                "smaller task or increase max_tokens in Settings."
            )
        if self.finish_reason in {None, "error"}:
            raise _InterruptedResponseError(
                f"Provider generation interrupted ({self.finish_reason or 'stream ended'})"
            )
        if self.finish_reason not in {"stop", "tool_calls", "function_call"}:
            raise ExecutorError(
                f"Incomplete executor response ({self.finish_reason or 'stream ended'})"
            )
        calls = []
        for index in sorted(self.calls):
            call = self.calls[index]
            if not call["name"]:
                raise ExecutorError("Incomplete tool call from executor")
            # IDs are required for transcript pairing; synthesize one only when the
            # provider has returned a complete call without its optional-looking ID.
            calls.append(ToolCall(call["id"] or f"call_{index}", call["name"], call["arguments"]))
        if not self.content.strip() and not calls:
            raise ExecutorError("Executor returned an empty response")
        if len({call.id for call in calls}) != len(calls):
            raise ExecutorError("Executor returned duplicate tool-call IDs")
        return Completion(self.content, calls, self.model, self.tokens)


class OpenRouterExecutor:
    requires_api_key = True

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]],
        selected: str | None,
        on_token: TokenSink,
    ) -> Completion:
        if not self.settings.api_key:
            raise ExecutorError("Set OPENROUTER_API_KEY in your environment or workspace .env file")
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "tools": schemas,
            "tool_choice": (
                {"type": "function", "function": {"name": selected}} if selected else "auto"
            ),
            # The free router selects tool-capable models itself. Requiring optional
            # parameters such as parallel_tool_calls filters out its endpoints (404).
            # Single-call and selected-tool enforcement happen locally in Agent.run.
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.settings.max_tokens,
        }
        # Fail closed if a user prompt or tool result accidentally contains the credential.
        if self.settings.api_key in json.dumps(payload, ensure_ascii=False):
            raise ExecutorError(
                "Refusing a request containing the OpenRouter API key in its content"
            )
        emitted = False

        async def forward(text: str) -> None:
            nonlocal emitted
            emitted = True
            await on_token(text)

        for attempt in range(3):
            try:
                async with asyncio.timeout(self.settings.request_timeout):
                    return await self._request(payload, forward)
            except _ResponseShapeError as exc:
                if payload["stream"] and not emitted and attempt < 2:
                    payload["stream"] = False
                    payload.pop("stream_options", None)
                    continue
                raise ExecutorError(
                    "OpenRouter could not normalize the model response. "
                    "Retry or choose a different --model. " + self.settings.redact(str(exc))
                ) from None
            except _InterruptedResponseError as exc:
                # Tool calls are only dispatched after a complete response. Discard the
                # failed accumulator, preserving earlier executed actions in messages.
                # Once text is visible, stop rather than concatenate two responses.
                if not emitted and attempt < 2:
                    payload["stream"] = False
                    payload.pop("stream_options", None)
                    await asyncio.sleep(2**attempt)
                    continue
                raise ExecutorError(
                    f"OpenRouter generation failed after {attempt + 1} attempt(s). "
                    "No tool from the incomplete response was executed. "
                    "Retry or choose a different --model. " + self.settings.redact(str(exc))
                ) from None
            except _RetryableStatus as exc:
                if attempt == 2:
                    raise ExecutorError(
                        f"OpenRouter HTTP {exc.status} after 3 attempts. "
                        "Free-model capacity may be limited; try again later."
                    ) from None
                await asyncio.sleep(max(exc.delay, 2**attempt))
            except (TimeoutError, httpx.TimeoutException):
                raise ExecutorError("OpenRouter request timed out; no tool was executed") from None
            except httpx.HTTPError:
                raise ExecutorError(
                    "Could not reach OpenRouter; check your network connection"
                ) from None
            except (ValueError, TypeError, KeyError, AttributeError):
                raise ExecutorError("OpenRouter returned a malformed response") from None
            except ExecutorError as exc:
                raise ExecutorError(self.settings.redact(str(exc))) from None
        raise AssertionError("unreachable")

    async def _request(self, payload: dict[str, Any], on_token: TokenSink) -> Completion:
        async with self.client.stream(
            "POST",
            CHAT_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "X-Title": "Python Agent TUI",
            },
        ) as response:
            if response.status_code in {429, 502, 503, 504}:
                try:
                    delay = min(10.0, max(0.0, float(response.headers.get("retry-after", "0"))))
                except ValueError:
                    delay = 0.0
                raise _RetryableStatus(response.status_code, delay)
            if response.is_error:
                hints = {
                    401: "check OPENROUTER_API_KEY",
                    402: "check account credits or limits",
                    403: "check API-key permissions",
                    400: "request parameters were rejected",
                    404: "no compatible endpoint is available for this model",
                }
                # Include the API's diagnostic message, not metadata, headers or raw bodies.
                await response.aread()
                detail = hints.get(response.status_code, "request failed")
                try:
                    error = response.json().get("error", {})
                    if isinstance(error, dict) and isinstance(error.get("message"), str):
                        detail += ". " + self.settings.redact(error["message"])[:500]
                except (ValueError, AttributeError):
                    pass
                raise provider_error(f"OpenRouter HTTP {response.status_code}: {detail}")
            accumulator = _Accumulator()
            if "application/json" in response.headers.get("content-type", ""):
                await response.aread()
                await accumulator.add(response.json(), on_token)
            else:
                async for data in sse_data(response):
                    if data.strip() == "[DONE]":
                        break
                    await accumulator.add(json.loads(data), on_token)
            return accumulator.finish()

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()
