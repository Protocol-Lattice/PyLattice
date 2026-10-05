"""Example PyLattice mod: add a guardrail around command execution."""

EVENTS = ("tool",)


async def mod(event, payload, next):
    if payload.get("tool") == "run_command":
        argv = payload.get("arguments", {}).get("argv", [])
        text = " ".join(argv)
        if "production" in text or "prod" in text:
            raise RuntimeError("production commands are blocked by the example mod")

    result = await next(payload)

    # Code here runs after the built-in tool (and all later-loaded mods).
    return result
