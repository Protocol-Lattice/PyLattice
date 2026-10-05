"""Adapter around the real Harness Router library and its Decisions API provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence

import httpx
from harness_router import (
    ChoiceDecision,
    HarnessState,
    JevToolRouter,
    MCTSConfig,
    MCTSResult,
    MCTSToolRouter,
    OpenRouterJevProvider,
    RouteDecision,
    RoutingConfig,
    ToolDescriptor,
)
from harness_router.errors import ProviderError

from .config import Settings
from .planner import Plan, PlanEnvironment


class _ObservableProvider(OpenRouterJevProvider):
    last_error: str = ""

    async def choose(self, **kwargs) -> ChoiceDecision:
        self.last_error = ""
        try:
            return await super().choose(**kwargs)
        except ProviderError as exc:
            self.last_error = str(exc)
            raise


class HarnessDecisionLayer:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.provider = (
            _ObservableProvider(
                api_key=settings.api_key,
                model=settings.router_model,
                timeout_seconds=settings.router_timeout,
                client=client,
            )
            if settings.api_key
            else None
        )
        self._route_cache: OrderedDict[str, RouteDecision] = OrderedDict()
        self._route_cache_size = 128
        self.router = (
            JevToolRouter(
                self.provider,
                RoutingConfig(
                    # This upstream cache deliberately ignores changing observations.
                    # An execution loop must make its next choice from the actual latest result.
                    obvious_cache_size=0,
                    state_field_limit=6000,
                    description_limit=256,
                    history_limit=6,
                    constraint_limit=4,
                ),
            )
            if self.provider
            else None
        )

    async def route(self, state: HarnessState, tools: Sequence[ToolDescriptor]) -> RouteDecision:
        if not self.router:
            return RouteDecision.fallback_to_planner("missing_openrouter_api_key")
        cache_key = self._cache_key(state, tools)
        cached = self._route_cache.get(cache_key)
        if cached is not None:
            self._route_cache.move_to_end(cache_key)
            return cached
        try:
            async with asyncio.timeout(self.settings.router_timeout + 1):
                decision = await self.router.route(state, tools)
                if decision.fallback_reason == "provider_error" and self.provider:
                    detail = self.settings.redact(self.provider.last_error)[:500]
                    return RouteDecision.fallback_to_planner(f"provider_error: {detail}")
                if not decision.fallback:
                    self._route_cache[cache_key] = decision
                    self._route_cache.move_to_end(cache_key)
                    while len(self._route_cache) > self._route_cache_size:
                        self._route_cache.popitem(last=False)
                return decision
        except TimeoutError:
            return RouteDecision.fallback_to_planner("routing_timeout")

    @staticmethod
    def _cache_key(state: HarnessState, tools: Sequence[ToolDescriptor]) -> str:
        payload = {
            "state": state.compact(),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "risk": tool.risk.value,
                }
                for tool in tools
            ],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    async def aclose(self) -> None:
        if self.provider:
            await self.provider.aclose()

    async def route_mcts(
        self,
        state: HarnessState,
        tools: Sequence[ToolDescriptor],
        plan: Plan,
    ) -> MCTSResult:
        """Search the planner's predicted paths; execute only the returned first tool later."""
        if self.provider:
            self.provider.last_error = ""
        search = MCTSToolRouter(
            PlanEnvironment(state, plan, tools),
            policy_router=self.router,
            config=MCTSConfig(
                simulations=self.settings.mcts_simulations,
                max_depth=self.settings.mcts_depth,
                max_policy_evaluations=1,
            ),
        )
        async with asyncio.timeout(self.settings.router_timeout + 2):
            return await search.search(state)
