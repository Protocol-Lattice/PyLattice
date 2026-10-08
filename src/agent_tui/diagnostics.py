"""Small, explicit checks for the configured router and executor mods."""

from __future__ import annotations

from harness_router import HarnessState

from .config import Settings
from .mods import HarnessMods, ModRuntime


async def check_connection(settings: Settings, mods: HarnessMods | None = None) -> bool:
    async def ignore_token(text: str) -> None:
        pass

    try:
        async with ModRuntime(settings, mods, allow_delegation=False) as runtime:
            executor = runtime.get("executor")
            if getattr(executor, "requires_api_key", False) and not settings.api_key:
                print("Missing OPENROUTER_API_KEY. Set it in the environment or workspace .env.")
                return False
            router = runtime.get("router")
            registry = runtime.get("tools")
            print(f"Router: {type(router).__name__}", flush=True)
            try:
                decision = await router.route(
                    HarnessState(goal="User says hello. No workspace actions are needed."),
                    [
                        tool
                        for tool in registry.descriptors()
                        if tool.name in {"finish", "read_file"}
                    ],
                )
                if decision.fallback:
                    print(f"  Fallback: {settings.redact(decision.fallback_reason)}", flush=True)
                else:
                    print(f"  OK: {decision.tool} ({decision.confidence:.0%})", flush=True)
            except Exception as exc:
                print(f"  Fallback: {settings.redact(str(exc))}", flush=True)
            print(f"Executor: {type(executor).__name__}", flush=True)
            completion = await executor.complete(
                [{"role": "user", "content": 'Call finish with summary "Connection works".'}],
                registry.schemas("finish"),
                "finish",
                ignore_token,
            )
            if len(completion.calls) != 1 or completion.calls[0].name != "finish":
                print("  ERROR: provider did not return the requested tool call", flush=True)
                return False
            registry.validate("finish", completion.calls[0].arguments)
            print(f"  OK: tool calling works via {completion.model}", flush=True)
            return True
    except Exception as exc:
        print(f"  ERROR: {settings.redact(str(exc))}", flush=True)
        return False
