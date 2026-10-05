"""JEV-backed ranking of concrete codebase next actions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from harness_router import HarnessState, RiskLevel, ToolDescriptor

from .routing import HarnessDecisionLayer


class CodeAction(StrEnum):
    READ = "read_file"
    PATCH = "edit_file"
    WRITE = "write_file"
    TEST = "run_tests"
    SEARCH = "search_files"


@dataclass(frozen=True)
class CodeActionCandidate:
    action: CodeAction
    path: str | None
    summary: str
    evidence: tuple[str, ...] = ()

    def description(self) -> str:
        target = self.path or "<workspace>"
        evidence = "; ".join(self.evidence[:4]) or "no extra evidence"
        return (
            f"Next action={self.action.value}; target={target}; "
            f"summary={self.summary}; evidence={evidence}"
        )


@dataclass(frozen=True)
class CodeActionSelection:
    candidate: CodeActionCandidate
    index: int


class CodebaseActionRouter:
    """Use JEV to choose one concrete (action, file) pair as the next step."""

    def __init__(self, decision_layer: HarnessDecisionLayer) -> None:
        self.decision_layer = decision_layer

    async def choose(
        self,
        task: str,
        candidates: Sequence[CodeActionCandidate],
        *,
        observation: str = "",
    ) -> CodeActionSelection:
        if not candidates:
            raise ValueError("At least one code action candidate is required")

        descriptors = [
            ToolDescriptor(
                f"candidate_{index:03d}",
                candidate.description(),
                "codebase_next_action",
                self._risk(candidate.action),
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
            for index, candidate in enumerate(candidates)
        ]

        state = HarnessState(
            goal=task,
            observation=observation or "Choose the best concrete next codebase action.",
            constraints=[
                "Choose exactly one candidate.",
                "Prefer direct evidence over speculative new files.",
                "Prefer reading/searching when there is not enough evidence to patch safely.",
                "Prefer verification when the implementation already appears complete.",
            ],
        )
        decision = await self.decision_layer.route(state, descriptors)
        if decision.fallback or not decision.tool:
            reason = decision.fallback_reason or "JEV did not select a candidate"
            raise RuntimeError(f"Could not choose next codebase action: {reason}")

        try:
            index = int(decision.tool.removeprefix("candidate_"))
            candidate = candidates[index]
        except (ValueError, IndexError):
            raise RuntimeError(f"JEV selected an unknown candidate: {decision.tool}") from None

        return CodeActionSelection(candidate=candidate, index=index)

    @staticmethod
    def _risk(action: CodeAction) -> RiskLevel:
        if action in {CodeAction.READ, CodeAction.SEARCH}:
            return RiskLevel.LOW
        if action in {CodeAction.PATCH, CodeAction.WRITE}:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH


def candidate_from_file(
    path: str | Path,
    *,
    action: CodeAction,
    summary: str,
    evidence: Sequence[str] = (),
) -> CodeActionCandidate:
    """Small convenience helper for callers building candidates from repo scans."""

    return CodeActionCandidate(
        action=action,
        path=str(path),
        summary=summary,
        evidence=tuple(evidence),
    )
