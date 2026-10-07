# PyLattice — Python Agent TUI

A terminal coding agent with **Code Mode**: the model writes a Python program that calls tools, passes results between them, and returns the useful output in one run.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for dependency management)

## Quick Start

```bash
# Install dependencies
uv sync

# Copy the environment template
cp .env.example .env

# Set your API key
export OPENROUTER_API_KEY=your_key_here

# Run the agent
uv run agent-tui
```

## Features

### Code Mode
Enabled by default (`--code-mode` or `AGENT_TUI_CODE_MODE=true`). The model receives `execute_code` and `finish`, plus a catalog of available workspace, skill, memory, delegation, and MCP tools. A program can make dependent tool calls without another model request between them.

```python
listing = await call_tool("list_files", {"path": "src"})
results = []
for path in listing["output"]["files"][:3]:
    result = await call_tool("read_file", {"path": path})
    results.append(result["output"])
results
```

### Routing Options

- **Planner + Harness Router MCTS**: `uv run agent-tui --no-code-mode`
- **Fast routing**: `uv run agent-tui --fast` (Jev routing without the planner)
- **Demo mode**: `uv run agent-tui --demo` (offline Code Mode demo)
- **Offline routed demo**: `uv run agent-tui --demo --no-code-mode`

Explicit flags like `--routing`, `--route-mcts`, `--no-planner`, or `--fast` select the routed loop. `--max-steps` bounds model turns; a Code Mode turn can contain several tool actions.

### Extensions & Memory
The agent supports `SKILL.md` skills, a plugin marketplace (including Superpowers), MCP servers, middleware with hooks, context management, and persistent SQLite memory.

See `[extensions documentation](docs/extensions.md)` for configuration, commands, and example descriptions.

## Project Structure

- `src/agent_tui/` — Main TUI agent source code
- `src/agent_tui/openrouter.py` — OpenRouter integration
- `src/agent_tui/codemode.py` — Sandboxed Python execution and tool bridge
- `src/agent_tui/routing.py` — Harness Router next-tool decision logic
- `src/agent_tui/tools.py` — Tools available to the agent
- `src/agent_tui/__main__.py` — Entry point

## License

This project is open source. See the [LICENSE](LICENSE) file for details.
