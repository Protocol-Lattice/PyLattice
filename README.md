# PyLattice — Python Agent TUI

A terminal-based coding agent that uses **Code Mode** to write Python programs that call tools, pass results between them, and return useful output in a single run.

Routes executor requests through OpenRouter's [free router](https://openrouter.ai/docs/guides/routing/routers/free-router) by default. **Harness Router** uses the configured decision model to select each Code Mode turn or individual tools in the routed loop.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for dependency management)

---

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

Each request uses a 64,000-character context budget, task-relevant history, and a short skill catalog. Configure with:

- `--context-chars N` or `AGENT_TUI_CONTEXT_CHARS` environment variable
- `/context` — Check current usage
- `/skill NAME` — Pin a workflow across tasks

---

## Features

### Code Mode

Enabled by default via `--code-mode` flag or `AGENT_TUI_CODE_MODE=true`. The decision model (`--router-model` or `AGENT_TUI_ROUTER_MODEL`) chooses `execute_code` or `finish` once per turn using the latest result. The executor then writes the selected program or final answer, with a catalog of workspace, skill, memory, delegation, and MCP tools. On a routing fallback, the executor chooses between both tools. A program can make dependent tool calls without requiring another model request between them.

**Example:**

```python
listing = await call_tool("list_files", {"path": "src"})
results = []
for path in listing["output"]["files"][:3]:
    result = await call_tool("read_file", {"path": path})
    results.append(result["output"])
results
```

**Behavior:**

- Send one `execute_code` tool call with a raw Python `code` string, without Markdown fences or prose. Use Python `True`, `False`, and `None` inside that string.
- Use top-level `await call_tool("name", {"argument": value})` for every tool; names and arguments must match the current catalog. Tools are not Python functions, and no `asyncio.run` wrapper is needed.
- Chat tool replies and `call_tool` share `{"ok": true, "output": ..., "error": null, "truncated": false}`. `output` is native JSON data or plain text; do not decode it a second time. `after_tool` hooks receive these same fields.
- On failure, `error` contains `code`, `message`, and a nullable `retry_hint`. Plain errors have `output: null`; failed commands and other partial results retain structured output. In a program, a failed tool still raises and blocks later calls; its structured error and partial output are recorded in the tool log.
- MCP output keeps `content` as typed blocks and `structured_content` as a JSON value or `null`. Text and structured data are not concatenated. Unsupported media blocks are represented by `{ "type": "image", "omitted": true }` (with the actual media type).
- `truncated` is always present in the reply, including for plain text and delegated results. Check it before relying on returned data. Existing built-in output fields remain available.
- `read_file` text is in `result["output"]["content"]` and includes display line numbers. Remove those prefixes from exact edit patches; check `has_more` and `truncated` before using excerpts. Large built-in results retain valid JSON and their output shapes when the metadata fits the output limit.
- The final expression and printed output are returned to the model
- Intermediate results stay inside the program; variables don't persist between programs
- Syntax and types are checked before execution using [Monty's type checker](https://pydantic.dev/docs/monty/concepts/type-checking/). Missing awaits, undefined names, invalid bridge arguments and unsupported imports are rejected before tools run.
- Errors include a `phase`, tool log and `retry_hint`. Fix validation errors in a new program. After runtime failures, inspect current state before retrying: successful actions are not rolled back, and a failed tool blocks later calls in that program.
- Press **Esc** to cancel a running program

**Execution:**

Programs run in a [Monty](https://github.com/pydantic/monty) sandbox with no host filesystem or network access. All tool calls retain schema validation, middleware, workspace restrictions, and normal approvals.

| Flag | Effect |
|------|--------|
| `--read-only` | Disables mutating tools |
| `--yes` | Approves actions automatically |

**Resource Limits:**

- 32 tool calls maximum
- 5 seconds computation time (excluding tools/approvals)
- 64 MiB memory limit
- Bounded output

---

### Routing Options

```bash
uv run agent-tui --no-code-mode          # Planner + Harness Router MCTS
uv run agent-tui --fast                  # Jev routing without the planner
uv run agent-tui --demo                  # Offline Code Mode demo
uv run agent-tui --demo --no-code-mode   # Offline routed demo
```

Use these flags to control routing behavior:

- `--routing`, `--route-mcts`, `--no-planner`, or `--fast` — Select the routed loop
- `--max-steps` — Bound model turns (a Code Mode turn can contain several tool actions)

The activity panel shows every nested call, and failed actions expand to show their error. Incomplete tool calls are never executed; interrupted provider responses are retried up to three attempts when no response text has been displayed.

---

## Demo

Run a local demo without credentials or network requests:

```bash
uv run agent-tui --demo
```

---

## Extensions & Memory

The agent supports:

- **SKILL.md files** — Workflow definitions
- **Plugin marketplace** — Including Superpowers
- **MCP servers** — Model Context Protocol integration
- **Middleware** — With pre/post hooks
- **Context management** — View and modify conversation context
- **Persistent SQLite memory** — Long-term storage

Interactive commands:

```text
/plugins                    List installed plugins
/plugin install superpowers Install from marketplace
/skills                     List available skills
/skill NAME                 Run or pin a skill
/mcp                        Manage MCP servers
/hooks                      View middleware hooks
/context                    Manage conversation context
/memory                     View and edit memory
```

Start example MCP servers and hooks:

```bash
uv run agent-tui --extensions examples/extensions.toml
```

See [extensions documentation](docs/extensions.md) for full configuration details.

---

## Harness Mods

Replace any agent component through Python factories: the app, agent, run loop, providers,
prompts, code runtime, tools, policies, responses, context, memory, skills, plugins,
middleware, MCP, and delegation. Components can extend a built-in or replace it completely.
Child agents use the same mods with fresh instances.

```bash
uv run agent-tui --mods examples/mods.toml   # Working offline customization example
uv run agent-tui --mods my-mods.toml --mod executor=my_package:build_executor
uv run agent-tui --list-mods                # Inspect selection without importing mods
```

See [the mod guide](docs/mods.md) for every slot, factory contracts, configuration, and
lifecycle handling. Mods are explicitly selected trusted Python code; no manifest is
required to use the defaults.

---

## Project Structure

```
.
├── src/agent_tui/           # Main TUI agent source code
│   ├── __main__.py          # Entry point
│   ├── cli.py               # Command-line interface
│   ├── mods.py              # Mod loading, composition and lifecycle
│   ├── defaults.py          # Built-in component factories
│   ├── contracts.py         # Public component protocols
│   ├── loop.py              # Replaceable default orchestration
│   ├── prompts.py           # Replaceable prompt composition
│   ├── policy.py            # Replaceable tool execution policy
│   ├── codemode.py          # Sandboxed Python execution and tool bridge
│   ├── routing.py           # Harness Router next-tool decision logic
│   ├── tools.py             # Tools available to the agent
│   ├── tui.py               # Terminal UI components
│   ├── memory.py            # Persistent SQLite memory
│   ├── skills.py            # SKILL.md workflow handling
│   ├── plugins.py           # Plugin marketplace integration
│   ├── mcp.py               # Model Context Protocol support
│   ├── middleware.py        # Hook system for extensions
│   └── openrouter.py        # OpenRouter API integration
├── examples/                # Example configurations and scripts
└── tests/                   # Test suite
```

---

## Website

The product website is in `website/dist/`. Its pink and plum palette comes from
`src/agent_tui/app.tcss` and the `jev-pink` theme in `tui.py`. It is a static site
with no JavaScript dependencies or build step.

Preview it locally with Python 3:

```bash
python3 -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

Open `http://127.0.0.1:4173`. From `website/`, `npm start` runs the same preview
and `npm run check` checks the JavaScript syntax. Any static host can serve the
contents of `website/dist/`.

For Vercel, deploy the static files directly. The included `vercel.json` files
select the **Other** framework preset, skip installation and building, and set
the output directory for either project root:

| Vercel Root Directory | Output Directory |
| --- | --- |
| Repository root (default) | `website/dist` |
| `website` | `dist` |

Do not use `npm start` or `npm run start` as the Vercel Build Command: it launches
a local preview server and keeps running. The configuration overrides that build
command on the next deployment. Commit and push the configuration, then redeploy.
See Vercel's [static-site build settings](https://vercel.com/docs/builds/configure-a-build#skip-build-step).

The interactive terminal uses simulated local examples; it does not execute
agent tools or contact a model. Setup tabs provide normal and offline-demo
commands. The site uses system font fallbacks when Google Fonts is unavailable.

---

## License

This project is open source. See the [LICENSE](LICENSE) file for details.
