from __future__ import annotations

import json
from dataclasses import replace

import httpx
from harness_router import HarnessState

from agent_tui.routing import HarnessDecisionLayer
from agent_tui.tools import ToolRegistry


async def test_real_router_redecides_after_observations_change(settings):
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        choice = "read_file" if len(payloads) == 1 else "finish"
        return httpx.Response(
            200,
            json={
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.99,
                        "probabilities": {choice: 0.99},
                    }
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        layer = HarnessDecisionLayer(settings, client)
        tools = ToolRegistry(settings).descriptors()
        first = await layer.route(HarnessState(goal="Read file", observation="Not read yet"), tools)
        second = await layer.route(
            HarnessState(goal="Read file", observation="Read succeeded"), tools
        )
    assert first.tool == "read_file"
    assert second.tool == "finish"
    assert len(payloads) == 2
    assert payloads[0]["model"] == "typesafe/jev-1.13"
    assert "read_file" in payloads[0]["questions"]["route"]["criteria"]


async def test_provider_failure_is_visible_and_falls_back(settings):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(402, json={"error": {"message": "Insufficient credits"}})
        )
    ) as client:
        layer = HarnessDecisionLayer(settings, client)
        result = await layer.route(HarnessState(goal="hi"), ToolRegistry(settings).descriptors())
    assert result.fallback
    assert "402" in result.fallback_reason
    assert "Insufficient credits" in result.fallback_reason


async def test_missing_key_fallback(settings):
    layer = HarnessDecisionLayer(replace(settings, api_key=""))
    result = await layer.route(HarnessState(goal="hi"), [])
    assert result.fallback_reason == "missing_openrouter_api_key"
    await layer.aclose()


async def test_exact_state_route_is_cached_but_observation_change_is_not(settings):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "read_file",
                        "confidence": 0.99,
                        "probabilities": {"read_file": 0.99},
                    }
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        layer = HarnessDecisionLayer(settings, client)
        tools = ToolRegistry(settings).descriptors()
        state = HarnessState(goal="Read file", observation="Not read yet")
        first = await layer.route(state, tools)
        second = await layer.route(state, tools)
        third = await layer.route(
            HarnessState(goal="Read file", observation="Read succeeded"), tools
        )

    assert first.tool == second.tool == third.tool == "read_file"
    assert calls == 2
