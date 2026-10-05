from __future__ import annotations

import json

import httpx

from agent_tui.code_actions import (
    CodeAction,
    CodeActionCandidate,
    CodebaseActionRouter,
)
from agent_tui.routing import HarnessDecisionLayer


async def test_jev_picks_best_file_action_candidate(settings):
    seen = {}

    def handler(request):
        payload = json.loads(request.content)
        seen.update(payload)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "candidate_001",
                        "confidence": 0.97,
                        "probabilities": {"candidate_001": 0.97},
                    }
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        layer = HarnessDecisionLayer(settings, client)
        router = CodebaseActionRouter(layer)
        candidates = [
            CodeActionCandidate(
                action=CodeAction.READ,
                path="tests/test_openrouter.py",
                summary="Inspect expected malformed-response behavior.",
                evidence=("Existing tests cover OpenRouter response handling.",),
            ),
            CodeActionCandidate(
                action=CodeAction.PATCH,
                path="src/agent_tui/openrouter.py",
                summary="Patch the response parser where malformed content is handled.",
                evidence=(
                    "The failing path is inside OpenRouter response normalization.",
                    "The implementation raises malformed-response errors here.",
                ),
            ),
            CodeActionCandidate(
                action=CodeAction.WRITE,
                path="src/agent_tui/json_repair.py",
                summary="Add a new JSON repair helper.",
                evidence=("This would be a speculative new abstraction.",),
            ),
        ]

        selected = await router.choose(
            "Fix malformed OpenRouter JSON handling",
            candidates,
            observation="The current request fails while parsing the provider response.",
        )

    assert selected.index == 1
    assert selected.candidate.action is CodeAction.PATCH
    assert selected.candidate.path == "src/agent_tui/openrouter.py"
    criteria = seen["questions"]["route"]["criteria"]
    assert "candidate_000" in criteria
    assert "candidate_001" in criteria
    assert "src/agent_tui/openrouter.py" in criteria


async def test_codebase_action_router_rejects_empty_candidates(settings):
    layer = HarnessDecisionLayer(settings)
    router = CodebaseActionRouter(layer)

    try:
        await router.choose("Do something", [])
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
