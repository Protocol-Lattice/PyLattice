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
