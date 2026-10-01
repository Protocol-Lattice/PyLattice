from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_tui.openrouter import ExecutorError, OpenRouterExecutor
from agent_tui.tools import ToolRegistry


def stream_response(events):
    text = ": heartbeat\n\n" + "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(
        200, text=text + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
    )


async def ignore_token(text):
    pass


async def test_streamed_arguments_and_forced_tool_request(settings):
    requests, tokens = [], []

    def handler(request):
        requests.append(json.loads(request.content))
        return stream_response(
            [
                {"model": "a-real-free-model", "choices": [{"delta": {"content": "Inspecting."}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"pa',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": 'th":"README.md"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {"usage": {"total_tokens": 42}, "choices": []},
            ]
        )

    async def on_token(text):
        tokens.append(text)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        executor = OpenRouterExecutor(settings, client)
        response = await executor.complete(
            [{"role": "user", "content": "Inspect the readme"}],
            ToolRegistry(settings).schemas("read_file"),
            "read_file",
            on_token,
        )
    assert response.calls[0].arguments == '{"path":"README.md"}'
    assert response.calls[0].name == "read_file"
    assert response.model == "a-real-free-model"
    assert response.tokens == 42
    assert tokens == ["Inspecting."]
    body = requests[0]
    assert body["model"] == "openrouter/free"
    assert body["tool_choice"]["function"]["name"] == "read_file"
    assert len(body["tools"]) == 1
    assert "provider" not in body
    assert "parallel_tool_calls" not in body


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
async def test_partial_or_filtered_responses_cannot_execute(settings, finish_reason):
    response = stream_response(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call1",
                                    "function": {"name": "write_file", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": finish_reason,
                    }
                ]
            }
        ]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(ExecutorError):
            await OpenRouterExecutor(settings, client).complete([], [], "write_file", ignore_token)


async def test_fallback_uses_auto_and_retries_429(settings, monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(json.loads(request.content))
        if len(attempts) < 3:
            return httpx.Response(429, headers={"retry-after": "0"})
        return stream_response(
            [{"choices": [{"delta": {"content": "Done"}, "finish_reason": "stop"}]}]
        )

    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await OpenRouterExecutor(settings, client).complete([], [], None, ignore_token)
    assert result.content == "Done"
    assert len(attempts) == 3
    assert attempts[0]["tool_choice"] == "auto"


async def test_auth_failure_is_actionable_without_response_secrets(settings):
    response = httpx.Response(401, text=settings.api_key)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(ExecutorError, match="check OPENROUTER_API_KEY") as error:
            await OpenRouterExecutor(settings, client).complete([], [], None, ignore_token)
    assert settings.api_key not in str(error.value)


async def test_stream_error_redacts_credentials(settings):
    response = stream_response([{"error": {"message": f"rejected {settings.api_key}"}}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(ExecutorError) as error:
            await OpenRouterExecutor(settings, client).complete([], [], None, ignore_token)
    assert settings.api_key not in str(error.value)


async def test_credentials_in_prompt_never_sent(settings):
    def handler(_):
        pytest.fail("Credentials must be blocked before network access")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ExecutorError, match="API key"):
            await OpenRouterExecutor(settings, client).complete(
                [{"role": "user", "content": settings.api_key}],
                [],
                None,
                ignore_token,
            )


async def test_malformed_json_fails_cleanly(settings):
    response = httpx.Response(
        200, text="data: not-json\n\n", headers={"content-type": "text/event-stream"}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(ExecutorError, match="malformed response"):
            await OpenRouterExecutor(settings, client).complete([], [], None, ignore_token)


@pytest.mark.parametrize("streaming", [False, True])
async def test_text_content_blocks_and_null_terminal_delta(settings, streaming):
    message = {
        "content": [{"type": "text", "text": "Hello "}, {"type": "output_text", "text": "world"}]
    }
    response = (
        stream_response(
            [
                {"choices": [{"delta": message}]},
                {"choices": [{"delta": None, "finish_reason": "stop"}]},
            ]
        )
        if streaming
        else httpx.Response(
            200,
            json={
                "choices": [{"message": message, "finish_reason": "stop"}],
            },
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        result = await OpenRouterExecutor(settings, client).complete([], [], None, ignore_token)
    assert result.content == "Hello world"


async def test_provider_content_shape_error_retries_nonstreaming_once(settings):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return stream_response(
                [
                    {
                        "error": {
                            "message": "unsupported response content shape from OpenRouter",
                        }
                    }
                ]
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "ok",
                                    "function": {
                                        "name": "finish",
                                        "arguments": '{"summary":"Done"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await OpenRouterExecutor(settings, client).complete([], [], "finish", ignore_token)
    assert result.calls[0].name == "finish"
    assert len(requests) == 2
    assert requests[0]["stream"] is True and requests[1]["stream"] is False
    assert "stream_options" not in requests[1]


async def test_shape_error_after_tokens_is_not_retried(settings):
    requests = []

    def handler(request):
        requests.append(request)
        return stream_response(
            [
                {"choices": [{"delta": {"content": "Partial"}}]},
                {"error": {"message": "unsupported response content shape from OpenRouter"}},
            ]
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ExecutorError, match="could not normalize"):
            await OpenRouterExecutor(settings, client).complete([], [], None, ignore_token)
    assert len(requests) == 1
