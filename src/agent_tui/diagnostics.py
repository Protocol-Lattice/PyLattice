"""Small, explicit live checks for the two independent OpenRouter API endpoints."""

from __future__ import annotations

from harness_router import OpenRouterJevProvider

from .config import Settings
from .openrouter import OpenRouterExecutor
from .tools import ToolRegistry


async def check_connection(settings: Settings) -> bool:
    if not settings.api_key:
        print("Missing OPENROUTER_API_KEY. Set it in the environment or workspace .env.")
        return False
    print(f"Decision model: {settings.router_model}", flush=True)
    async with OpenRouterJevProvider(
        api_key=settings.api_key, model=settings.router_model
    ) as router:
        try:
            decision = await router.choose(
                state="User says hello. No workspace actions are needed.",
                instructions="Choose the appropriate next action.",
                criteria={"finish": "Reply to the greeting", "read_file": "Read a project file"},
            )
            print(f"  OK: {decision.choice} ({decision.confidence:.0%})", flush=True)
        except Exception as exc:
            print(f"  Fallback: {settings.redact(str(exc))}", flush=True)
    print(f"Executor: {settings.model}", flush=True)
    executor = OpenRouterExecutor(settings)

    async def ignore_token(text: str) -> None:
        pass

    try:
        completion = await executor.complete(
            [{"role": "user", "content": 'Call finish with summary "Connection works".'}],
            ToolRegistry(settings).schemas("finish"),
            "finish",
            ignore_token,
        )
        if len(completion.calls) != 1 or completion.calls[0].name != "finish":
            print("  ERROR: provider did not return the requested tool call", flush=True)
            return False
        ToolRegistry(settings).validate("finish", completion.calls[0].arguments)
        print(f"  OK: tool calling works via {completion.model}", flush=True)
        return True
    except Exception as exc:
        print(f"  ERROR: {settings.redact(str(exc))}", flush=True)
        return False
    finally:
        await executor.aclose()
