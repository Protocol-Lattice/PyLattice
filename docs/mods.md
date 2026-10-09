# Harness mods

Every agent service, the orchestration loop, the agent itself, and the terminal app have
replaceable factory slots. Select mods at startup through a TOML manifest, CLI overrides,
or Python. Unspecified slots use built-ins; existing commands work without a manifest.

New to mods? Follow the [website tutorial](../website/dist/docs/harness-mods.html)
to create prompt and tool factories, wire a manifest, and verify an offline run.
Preview the website as described in the [README](../README.md#website), then open
`/docs/harness-mods.html`.

## Run the example

From the project root:

```bash
uv run agent-tui --mods examples/mods.toml --prompt "Inspect this workspace"
uv run agent-tui --mods examples/mods.toml --no-code-mode
uv run agent-tui --mods examples/mods.toml --check
uv run agent-tui --mods examples/mods.toml --list-mods
```

This example uses offline router/executor mods, adds a `workspace_label` tool, and extends
the system prompt. It exercises the real loop and tools without an API key. The providers
only demonstrate workspace inspection; they are not general-purpose language models.

## Configuration

```toml
version = 1

[mods]
executor = "my_package.providers:build_executor"
prompts = "local_mods.py:build_prompts"
planner = "none"

[options.executor]
model = "my-local-model"

[options.prompts]
instructions = "Include the checks performed in the final answer."
```

- `--mods PATH` selects a manifest; `AGENT_TUI_MODS` supplies the default path.
- `--mod SLOT=FACTORY` overrides one slot and can be repeated. The last value wins.
- `--mod executor=default` restores a built-in factory and retains that slot's options.
- `none` disables `planner` or `subagents`. Other slots require a replacement.
- `--list-mods` prints the effective targets without importing custom code.
- Relative manifest paths resolve from `--workspace`. Python file targets resolve from
  the manifest directory, or from the workspace when there is no manifest.
- Module targets must be importable in the active Python environment. A file target can
  point at a package's `__init__.py` to use relative imports within that package.
- Options are private to each factory invocation and are not automatically applied to
  built-ins. A custom factory reads them through `context.options`.

Mod files execute trusted Python with the application's permissions. They are distinct
from sandboxed Code Mode programs. Manifests are loaded only when explicitly selected;
the agent does not automatically import `.agent-tui/mods.toml` or discover workspace code.
Restart the app to change factory selection or reload Python modules.

## Factories and dependencies

A factory is a synchronous callable that accepts one `ModContext` and returns a component.
Components need the methods used by their consumers; inheritance is optional. Asynchronous
work belongs in component methods. The loader rejects unknown slots, unsupported manifest
versions, invalid import targets, asynchronous factories, missing entry points, and
dependency cycles before the component is used.

```python
from agent_tui.mods import ModContext

def build_prompts(context: ModContext):
    prompts = context.default()
    prompts.system_prompt += "\n" + context.options["instructions"]
    return prompts

def build_executor(context: ModContext):
    return MyExecutor(model=context.options["model"])
```

`context.settings` provides the current settings. `context.get("slot")` resolves another
configured component, creating it once for that runtime. `context.default()` builds the
built-in implementation of the current slot with its configured dependencies; it does not
recursively invoke the replacement factory. This supports subclasses, wrappers, and small
customizations as well as complete replacements. `context.runtime.initial_prompt` supplies
the CLI prompt to a replacement app. `context.runtime.allow_delegation` distinguishes the
root runtime from delegated children.

Factories must return fresh stateful instances. Each child uses the same factory definitions
and options with a fresh runtime, history, registry, and providers. The default agent also
copies active skill selections and middleware callbacks into the child's fresh managers.
Delegated runtimes return `None` for `subagents`, retaining the default no-nested-delegation
behavior. A replacement `subagents` component can define another delegation strategy.

## Slots and contracts

The table describes the interfaces used by the built-in consumers. The runtime checks the
main callable entry points; it does not type-check every argument, attribute, or result.
Keep the consumer's contract, or replace the consumer together with the component.

| Slot | Built-in / interface |
| --- | --- |
| `app` | `tui.AgentApp`: synchronous `run()`. The CLI supplies settings and initial prompt to its factory. |
| `agent` | `agent.Agent`: async `run(goal, emit, approve) -> RunResult`, async `aclose()`. The built-in TUI additionally uses the agent's service attributes, `running`, `history`, `requires_api_key`, `clear()`, and `refresh_skills()`. |
| `loop` | `loop.DefaultLoop`: async `run(agent, goal, emit, approve) -> RunResult`. Owns routing, model turns, events, history, and run cleanup. |
| `router` | `routing.HarnessDecisionLayer`: async `route(state, tools) -> RouteDecision`; also `route_mcts(state, tools, plan)` when a plan is used with MCTS. |
| `executor` | `openrouter.OpenRouterExecutor`: async `complete(messages, schemas, selected, on_token) -> Completion`. Set `requires_api_key = True` only when using the OpenRouter key; other providers handle their own credentials. |
| `planner` | `planner.Planner`: async `plan(state, tools) -> Plan`. Defaults to `None` in Code Mode or when planning is disabled. |
| `prompts` | `prompts.DefaultPrompts`: `build(agent, goal, exchanges, plan, schemas, extra_context, selected) -> messages`. The built-in has a mutable `system_prompt` string. |
| `code_runtime` | `defaults.DefaultCodeRuntime`: `spec: ToolSpec`, `prompt(registry, selected)`, `adapt(completion, registry)`, async `execute(code, settings, run_tool) -> ToolResult`. |
| `tools` | `tools.ToolRegistry`: registration, schemas, descriptors, validation, execution, approval previews, and sanitization; `specs` exposes the tool catalog. |
| `tool_policy` | `policy.DefaultToolPolicy`: async `execute(agent, call, run_tools, emit, approve, step, exchange, allowed) -> (ToolResult, arguments)`. Used for top-level and nested Code Mode tool calls. |
| `tool_bindings` | `defaults.DefaultToolBindings`: `register(registry, skills, memory, subagents)` and `refresh(registry, skills, memory)`. Binds skill, memory and delegation tools to their configured services. |
| `responses` | `defaults.DefaultResponses`: `as_dict(result) -> dict`, `message(call, result) -> chat message`. Used by model replies, tool events/hooks and the Code Mode bridge. |
| `context` | `context.ContextManager`: `build`, `clear`, `remember_turn`, `compact`; `history` and `stats.as_dict()`. |
| `memory` | `memory.MemoryStore`: `context`, `record_run`, `search`, `put`, `forget`; `enabled` for the TUI. |
| `skills` | `skills.SkillManager`: activation, task selection, catalog, instructions and resource reads; `skills`, `active`, `pinned`, `task_goal`, and `warnings`. Plugin refresh rebuilds this configured factory. |
| `plugins` | `plugins.PluginManager`: `skill_roots`, `bootstrap_skills`, `installed`, `catalog`; `install`, `enable`, `uninstall` for TUI plugin commands. |
| `extensions` | `extensions.ExtensionConfig`: `servers` and `hooks` consumed by the default MCP and middleware factories. |
| `middleware` | `middleware.MiddlewareManager`: async `authorize(approve)`, `dispatch(event, payload) -> dict`; `commands_enabled`, `hooks`, and `handlers` for built-in lifecycle/UI behavior. |
| `mcp` | `mcp.MCPManager`: async `connect(registry, approve, emit)`, async `aclose()`; `status` for the TUI. |
| `subagents` | `subagents.SubagentManager`: `begin_run(goal, emit, approve)`, `end_run()`, async `execute(arguments) -> ToolResult`. |

Typed provider, prompt, loop, policy, response and Code Mode protocols are in
`src/agent_tui/contracts.py`. Shared values are in `models.py`, `tools.py`, and `planner.py`.
The built-in service classes define the detailed signatures for their remaining methods.

The default loop reserves `execute_code` (with a Python `code` argument) and `finish`
(with a `summary` argument). Replace `loop` and `prompts` too if changing that protocol.
Default prompts and Monty's bridge type declarations describe the four standard response
fields: `ok`, `output`, `error`, and `truncated`. Keep those fields when extending responses;
replace the corresponding prompt/runtime consumers when changing their meaning or layout.
The default TUI consumes `AgentEvent` values; a wholly different agent interface should
be paired with a replacement `app`.

## Replace individual tools

A `tools` factory can start with `registry = context.default()`, then:

```python
registry.register(spec, handler)  # Add a new tool; duplicates are rejected.
registry.replace(spec, handler)   # Replace a tool, including a built-in.
registry.unregister("run_command")  # Remove a built-in or extension tool.
```

Handlers are async callables receiving validated argument dictionaries and returning
`ToolResult`. The built-in registry still enforces read-only risk filtering, validation,
sanitization and output limits on replacements. Label a tool's risk accurately. To replace
tools added later by skill/memory/delegation binding, customize `tool_bindings`; MCP tools
are installed by `mcp.connect()` for each run. Replacing `tool_policy` changes the policy
boundary itself, including approvals, so a custom policy must implement the desired rules.

## Python composition and cleanup

```python
from agent_tui.mods import HarnessMods, ModRuntime, ModSpec

mods = HarnessMods({
    "executor": ModSpec(build_executor, {"model": "local"}),
    "planner": "none",
})

async def run_task(settings, emit, approve):
    async with ModRuntime(settings, mods) as runtime:
        agent = runtime.get("agent")
        return await agent.run("Review this project", emit, approve)
```

`create_agent(settings, mods)` and `create_app(settings, initial_prompt, mods)` are convenience
constructors. The default agent/app shut down their runtime. For custom agents/apps or to
guarantee cleanup when construction fails, use `ModRuntime` as above. The CLI always closes
its runtime, including when app construction or execution fails.

Any component can provide `aclose()` for resources it owns. The runtime attempts all closers
once per distinct instance in reverse construction order and reports grouped failures.
It also retains rebuilt components for cleanup; closing the runtime twice is harmless.
Components should not close dependencies obtained through `context.get()` themselves.
The default loop closes MCP connections after each run, so MCP implementations must support
closing more than once and reconnecting on a later run. Custom agents that own a runtime
can delegate their `aclose()` to it; recursive runtime closure is guarded.

The existing `Agent(settings, router, executor, ...)` constructor remains supported for
direct instance injection. Use factories for replacements that must propagate to children;
legacy injected instances are local to the parent.
