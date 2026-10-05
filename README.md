# Python Agent TUI

A Python execution layer in the terminal that uses **Harness Router** to decide the next tool, and OpenRouter's [`openrouter/free`](https://openrouter.ai/docs/guides/routing/routers/free-router) for tool arguments and responses.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for dependency management)

## Quick start

```bash
# Install dependencies
uv sync

# Copy the environment template
cp .env.example .env

# Set your API key (or export it in your shell)
export OPENROUTER_API_KEY=your_key_here

# Run the agent
uv run agent-tui
```

Requests use a 64,000-character context budget, task-relevant history, and a short
skill catalog. Use `--context-chars N` or `AGENT_TUI_CONTEXT_CHARS` to change the limit.
`/context` shows usage; `/skill NAME` pins a workflow across tasks.

## Demo

To run a local demo without credentials or network requests:

```bash
uv run agent-tui --demo
```

## Extensions and memory

The agent supports `SKILL.md` skills, a plugin marketplace (including Superpowers), MCP servers, middleware with hooks, context management, and persistent SQLite memory for the workspace.

```text
/plugins
/plugin install superpowers
/skills
/skill superpowers:brainstorming
/mcp
/hooks
/context
/memory
```

To start the example MCP server and hooks, run from the project directory:

```bash
uv run agent-tui --extensions examples/extensions.toml
```

Configuration, commands, and example descriptions: [extensions documentation](docs/extensions.md).

## Project structure

- `src/agent_tui/` – Main TUI agent source code
- `src/agent_tui/openrouter.py` – OpenRouter integration
- `src/agent_tui/routing.py` – Harness Router next-tool decision logic
- `src/agent_tui/tools.py` – Tools available to the agent
- `src/agent_tui/__main__.py` – Entry point

## License

This project is open source. See the LICENSE file for details.


## JEV codebase action routing

PyLattice can use the existing JEV decision layer to compare concrete codebase actions,
not only generic tools. Build a short list such as `read_file:tests/test_x.py`,
`edit_file:src/x.py`, `write_file:src/new.py`, or `run_tests`, then let JEV pick
the single best next step:

```python
from agent_tui.code_actions import CodeAction, CodeActionCandidate, CodebaseActionRouter
from agent_tui.routing import HarnessDecisionLayer

layer = HarnessDecisionLayer(settings)
router = CodebaseActionRouter(layer)

selection = await router.choose(
    "Fix malformed OpenRouter JSON handling",
    [
        CodeActionCandidate(
            CodeAction.READ,
            "tests/test_openrouter.py",
            "Inspect expected behavior before editing.",
        ),
        CodeActionCandidate(
            CodeAction.PATCH,
            "src/agent_tui/openrouter.py",
            "Patch the response parser where the failure occurs.",
            ("The stack trace points at response parsing.",),
        ),
    ],
)

print(selection.candidate.action, selection.candidate.path)
```

Keep the candidate list small and evidence-rich: cheap repository search should narrow
the codebase first, then JEV decides among the strongest concrete next actions.


## Python mods

PyLattice mods are trusted Python functions that can change built-in behavior, inspired by
Claude Code mods. A mod can run before and after an operation, rewrite its payload, replace
the operation entirely by not calling `next`, or wrap it by doing work on both sides of
`await next(payload)`.

Supported events:

- `model` — rewrite model messages, schemas, or the selected tool; replace model execution.
- `tool` — rewrite, block, retry, wrap, or replace a tool execution.
- `permission` — approve or deny a permission request programmatically.
- `render` — rewrite, suppress, or replace an `AgentEvent` before the TUI receives it.

Workspace mods live in `.agent-tui/mods/*.py`. Mods may also ship inside installed plugins
under their configured `mods_path` (default: `mods`). They load in filename/plugin order.
The first loaded mod sees the event first and receives the final result last.

```python
# .agent-tui/mods/01-protect-production.py
EVENTS = ("tool",)

async def mod(event, payload, next):
    if payload["tool"] == "run_command":
        command = " ".join(payload["arguments"].get("argv", []))
        if "production" in command:
            raise RuntimeError("production commands are blocked")

    result = await next(payload)
    return result
```

Calling `next` is optional:

```python
EVENTS = ("permission",)

async def mod(event, payload, next):
    if payload["tool"] == "run_command":
        return False          # replace the built-in permission flow
    return await next(payload)
```

Mods execute in-process with the same OS permissions as PyLattice. They are not sandboxed.
Only install or enable mods from code you trust.


## Fast execution path

PyLattice avoids model calls when the next operation and its arguments are already
unambiguous.

Examples that can execute through the zero-LLM path:

```text
Read README.md
list files
find "ExecutorError"
run tests
```

The fast path still goes through normal validation, mods, hooks, permission checks, tool
execution, history, and observation handling; it only skips unnecessary routing/model calls.

Direct `jev` routing no longer starts the planner in parallel. The planner is reserved for
`mcts`, where its predicted paths are actually used.

After a successful Python `edit_file` or `write_file`, PyLattice looks for a matching
`tests/test_<module>.py`. When found, that focused test is scheduled as a deterministic
next action, avoiding another router + executor round trip.

JEV decisions also have a small exact-state LRU cache. Its key includes the compact harness
state and tool catalog, including the current observation, so a tool result invalidates the
previous routing decision instead of reusing stale state.
