from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from harness_router import RouteDecision

from agent_tui.defaults import FACTORIES, DefaultResponses
from agent_tui.diagnostics import check_connection
from agent_tui.models import Completion, RunResult, ToolCall
from agent_tui.mods import (
    CONTRACTS,
    HarnessMods,
    ModError,
    ModRuntime,
    ModSpec,
    create_agent,
    create_app,
)
from agent_tui.tools import ToolResult


async def ignore(*args):
    pass


async def deny(*args):
    return False


class Router:
    async def route(self, state, tools):
        return RouteDecision(tool="execute_code" if not state.last_action else "finish")


class Executor:
    def __init__(self):
        self.messages = []
        self.closed = False

    async def complete(self, messages, schemas, selected, on_token):
        self.messages.append(messages)
        arguments = (
            {
                "code": 'reply = await call_tool("read_file", {"path": "missing.txt"})\n'
                'reply["output"]'
            }
            if selected == "execute_code"
            else {"summary": "Custom result"}
        )
        return Completion(calls=[ToolCall("mod", selected, json.dumps(arguments))], model="custom")

    async def aclose(self):
        self.closed = True


async def test_every_slot_is_resolved_through_its_factory(settings):
    seen = Counter()

    def build(context):
        seen[context.name] += 1
        return context.default()

    assert FACTORIES.keys() == CONTRACTS.keys()
    async with ModRuntime(
        replace(settings, demo=True), HarnessMods(dict.fromkeys(CONTRACTS, build))
    ) as runtime:
        app = runtime.get("app")
        assert app.agent is runtime.get("agent")
        assert set(seen) == set(CONTRACTS)
        assert set(seen.values()) == {1}
        assert runtime.get("planner").__class__.__name__ == "DemoPlanner"


async def test_mods_participate_in_real_code_mode_and_share_response_format(settings):
    used = []

    def tools(context):
        registry = context.default()

        async def read(arguments):
            used.append("read")
            return ToolResult(True, '{"content":"custom file"}')

        registry.replace(registry.specs["read_file"], read)
        return registry

    def prompts(context):
        component = context.default()
        component.system_prompt += "\nCustom prompt instructions"
        return component

    class Responses(DefaultResponses):
        def as_dict(self, result):
            return {**super().as_dict(result), "source": "mod"}

    class AsyncWrapper:
        def __init__(self, component, method, name):
            self.component, self.method, self.name = component, method, name

        def __getattr__(self, name):
            original = getattr(self.component, name)
            if name != self.method:
                return original

            async def call(*args, **kwargs):
                used.append(self.name)
                return await original(*args, **kwargs)

            return call

    def wrap(context):
        method = "run" if context.name == "loop" else "execute"
        return AsyncWrapper(context.default(), method, context.name)

    mods = HarnessMods(
        {
            "router": lambda ctx: Router(),
            "executor": lambda ctx: Executor(),
            "tools": tools,
            "prompts": prompts,
            "responses": lambda ctx: Responses(),
            "code_runtime": wrap,
            "tool_policy": wrap,
            "loop": wrap,
        }
    )
    async with ModRuntime(replace(settings, code_mode=True), mods) as runtime:
        agent = runtime.get("agent")
        events = []

        async def emit(event):
            events.append(event)

        result = await agent.run("Read a custom file", emit, deny)
        assert result.status == "completed", result.message
        assert used == ["loop", "tool_policy", "code_runtime", "tool_policy", "read", "tool_policy"]
        assert "Custom prompt instructions" in agent.executor.messages[0][0]["content"]
        reply = json.loads(agent.executor.messages[1][-1]["content"])
        assert reply["source"] == "mod"
        assert reply["output"]["output"] == {"content": "custom file"}
        results = [event for event in events if event.kind == "tool_result"]
        assert all(event.data["source"] == "mod" for event in results)
        assert not (settings.workspace / "missing.txt").exists()
    assert agent.executor.closed


async def test_child_agents_use_same_mods_with_fresh_instances(settings):
    counts = Counter()

    def executor(context):
        counts["executor"] += 1
        return Executor()

    def skills(context):
        counts["skills"] += 1
        return context.default()

    mods = HarnessMods({"executor": executor, "router": lambda ctx: Router(), "skills": skills})
    async with ModRuntime(replace(settings, code_mode=True), mods) as runtime:
        parent = runtime.get("agent")
        parent.skills.activate("code-review")
        parent.middleware.add(ignore)
        child = parent._new_subagent()
        try:
            assert child.mods.mods is mods
            assert child.executor is not parent.executor
            assert child.context is not parent.context
            assert child.skills is not parent.skills
            assert child.registry is not parent.registry
            assert child.history == []
            assert child.subagents is None
            assert child.mods.get("subagents") is None
            assert "delegate_tasks" not in child.registry.specs
            assert "code-review" in child.skills.pinned
            assert child.middleware.handlers == [ignore]
            assert counts == {"executor": 2, "skills": 2}
        finally:
            await child.aclose()
        assert child.executor.closed and not parent.executor.closed


async def test_custom_loop_can_replace_orchestration(settings):
    class Loop:
        async def run(self, agent, goal, emit, approve):
            return RunResult("completed", f"Different loop: {goal}", 0)

    agent = create_agent(replace(settings, demo=True), HarnessMods({"loop": lambda ctx: Loop()}))
    try:
        result = await agent.run("hello", ignore, deny)
        assert result.message == "Different loop: hello"
        assert agent.history == []
    finally:
        await agent.aclose()


async def test_whole_agent_and_app_can_be_replaced_without_other_components(settings):
    closed = []

    class Agent:
        async def run(self, goal, emit, approve):
            return RunResult("completed", goal, 0)

        async def aclose(self):
            closed.append("agent")

    def app(context):
        agent = context.get("agent")
        return SimpleNamespace(run=lambda: agent, prompt=context.runtime.initial_prompt)

    mods = HarnessMods({"agent": lambda ctx: Agent(), "app": app})
    async with ModRuntime(settings, mods, initial_prompt="hello") as runtime:
        assert runtime.get("app").prompt == "hello"
        agent = runtime.get("app").run()
        assert (await agent.run("hi", ignore, deny)).message == "hi"
    assert closed == ["agent"]
    assert create_app(settings, "prompt", mods).prompt == "prompt"


async def test_manifest_loads_relative_python_factory_and_isolates_options(settings, tmp_path):
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / "mods.py").write_text(
        "from types import SimpleNamespace\n"
        "def build(ctx):\n"
        "    ctx.options['nested']['values'].append('local')\n"
        "    values = ctx.options['nested']['values']\n"
        "    return SimpleNamespace(complete=lambda: None, values=values)\n"
    )
    (directory / "mods.toml").write_text(
        'version = 1\n[mods]\nexecutor = "mods.py:build"\n'
        '[options.executor.nested]\nvalues = ["original"]\n'
    )
    settings = replace(settings, mods_path=Path("config/mods.toml"))
    mods = HarnessMods.load(settings)
    async with ModRuntime(settings, mods) as runtime:
        instance = runtime.get("executor")
        assert instance.values == ["original", "local"]
        assert runtime.get("executor") is instance
        async with runtime.fork() as child:
            assert child.get("executor").values == ["original", "local"]
    assert mods.specs["executor"].options == {"nested": {"values": ["original"]}}


async def test_importable_package_and_package_file_support_relative_imports(
    settings, tmp_path, monkeypatch
):
    package = tmp_path / "my_agent_mod"
    package.mkdir()
    (package / "part.py").write_text(
        "from types import SimpleNamespace\n"
        "def build(ctx): return SimpleNamespace(complete=lambda: None)\n"
    )
    (package / "__init__.py").write_text("from .part import build\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        for target in ("my_agent_mod:build", "my_agent_mod/__init__.py:build"):
            async with ModRuntime(
                settings, HarnessMods({"executor": target}, base=tmp_path)
            ) as runtime:
                assert callable(runtime.get("executor").complete)
    finally:
        sys.modules.pop("my_agent_mod", None)
        sys.modules.pop("my_agent_mod.part", None)


def test_manifest_is_explicit_and_overrides_are_ordered(settings, tmp_path):
    hidden = tmp_path / ".agent-tui"
    hidden.mkdir()
    (hidden / "mods.toml").write_text("not TOML")
    assert HarnessMods.load(settings).describe()["executor"] == "default"
    manifest = tmp_path / "mods.toml"
    manifest.write_text(
        'version = 1\n[mods]\nexecutor = "unknown:build"\n[options.executor]\nkey = 7\n'
    )
    mods = HarnessMods.load(
        replace(settings, mods_path=manifest, mod_overrides=("executor=none", "executor=default"))
    )
    assert mods.describe()["executor"] == "default"
    assert mods.specs["executor"].options == {"key": 7}


@pytest.mark.parametrize(
    "document, error",
    [
        ("version = 2", "version must be 1"),
        ("version = true", "version must be 1"),
        ('version = 1\nextra = "x"', "allowed manifest fields"),
        ('version = 1\nmods = "x"', "must be tables"),
        ("version = 1\n[mods]\nexecutor = 3", "must be a string"),
        ('version = 1\n[mods]\nunknown = "default"', "Unknown mod slot"),
        ("version = 1\n[options]\nexecutor = 3", "must be a mapping"),
        ('version = 1\n[mods]\nexecutor = "none"', "cannot be disabled"),
        ('version = 1\n[mods]\nexecutor = "missing_separator"', "needs module:factory"),
    ],
)
def test_invalid_manifests_report_the_problem(settings, tmp_path, document, error):
    manifest = tmp_path / "mods.toml"
    manifest.write_text(document)
    with pytest.raises(ModError, match=error):
        HarnessMods.load(replace(settings, mods_path=manifest))


@pytest.mark.parametrize("override", ["executor", "=module:build", "executor="])
def test_invalid_overrides_are_rejected(settings, override):
    with pytest.raises(ModError, match="--mod must be"):
        HarnessMods.load(replace(settings, mod_overrides=(override,)))


async def test_optional_mods_and_cycles(settings):
    async with ModRuntime(
        settings, HarnessMods({"planner": "none", "subagents": "none"})
    ) as runtime:
        assert runtime.get("planner") is None and runtime.get("subagents") is None
    mods = HarnessMods(
        {"router": lambda ctx: ctx.get("executor"), "executor": lambda ctx: ctx.get("router")}
    )
    async with ModRuntime(settings, mods) as runtime:
        with pytest.raises(ModError, match="router -> executor -> router"):
            runtime.get("router")


async def test_factory_options_can_extend_defaults_and_resolve_dependencies(settings):
    class Build:
        def __call__(self, context):
            result = context.default()
            result.system_prompt += context.options["suffix"]
            result.memory = context.get("memory")
            return result

    mods = HarnessMods({"prompts": ModSpec(Build(), {"suffix": "custom"})})
    assert mods.describe()["prompts"].endswith("Build")
    async with ModRuntime(settings, mods) as runtime:
        prompts = runtime.get("prompts")
        assert prompts.system_prompt.endswith("custom")
        assert prompts.memory is runtime.get("memory")


async def test_replacement_agent_works_with_default_delegation(settings):
    closed = []

    class Child:
        async def run(self, goal, emit, approve):
            return RunResult("completed", "custom child result", 1)

        async def aclose(self):
            closed.append(True)

    def agent(context):
        return context.default() if context.runtime.allow_delegation else Child()

    async with ModRuntime(replace(settings, demo=True), HarnessMods({"agent": agent})) as runtime:
        parent = runtime.get("agent")
        parent.subagents.begin_run("parent task", ignore, deny)
        result = await parent.subagents.execute({"tasks": [{"name": "worker", "prompt": "task"}]})
        assert result.ok, result.content
        assert json.loads(result.content)["results"][0]["summary"] == "custom child result"
        assert closed == [True]


@pytest.mark.parametrize(
    "factory, error",
    [
        (lambda: None, "must accept one ModContext"),
        (lambda ctx: object(), "missing required methods: complete"),
        (lambda ctx: None, "missing required methods"),
        ("agent_tui.config:MOD_DOES_NOT_EXIST", "Cannot load mod 'executor'"),
        ("agent_tui.mods:MOD_API_VERSION", "is not callable"),
    ],
)
async def test_invalid_factories_fail_before_a_run(settings, factory, error):
    async with ModRuntime(settings, HarnessMods({"executor": factory})) as runtime:
        with pytest.raises(ModError, match=error):
            runtime.get("executor")


async def test_async_factories_are_rejected_without_leaking_coroutines(settings):
    async def build(context):
        return Executor()

    async with ModRuntime(settings, HarnessMods({"executor": build})) as runtime:
        with pytest.raises(ModError, match="must be synchronous"):
            runtime.get("executor")


async def test_construction_errors_are_redacted(settings, tmp_path):
    module = tmp_path / "broken.py"
    module.write_text(f"raise RuntimeError({settings.api_key!r})\n")
    async with ModRuntime(settings, HarnessMods({"executor": f"{module}:build"})) as runtime:
        with pytest.raises(ModError) as error:
            runtime.get("executor")
        assert settings.api_key not in str(error.value)
        assert "[REDACTED]" in str(error.value)


async def test_cleanup_is_reverse_order_unique_and_continues_after_failure(settings):
    closed = []

    class Component:
        def __init__(self, name):
            self.name = name

        async def route(self, *args):
            pass

        async def complete(self, *args):
            pass

        async def aclose(self):
            closed.append(self.name)
            if self.name == "router":
                raise ValueError("close failure")

    def router(context):
        context.get("executor")
        return Component("router")

    runtime = ModRuntime(
        settings,
        HarnessMods(
            {
                "router": router,
                "executor": lambda ctx: Component("executor"),
                "extensions": lambda ctx: ctx.get("executor"),
            }
        ),
    )
    runtime.get("extensions")
    runtime.get("router")
    with pytest.raises(ExceptionGroup, match="Harness mod cleanup failed") as error:
        await runtime.aclose()
    assert str(error.value.exceptions[0]) == "close failure"
    assert closed == ["router", "executor"]
    await runtime.aclose()
    assert closed == ["router", "executor"]
    with pytest.raises(ModError, match="runtime is closed"):
        runtime.get("router")


async def test_cleanup_handles_partial_construction_and_rebuilt_components(settings):
    created = []

    def build(context):
        instance = Executor()
        created.append(instance)
        return instance

    def broken(context):
        context.get("executor")
        raise ValueError("broken router")

    with pytest.raises(ModError, match="broken router"):
        async with ModRuntime(
            settings, HarnessMods({"executor": build, "router": broken})
        ) as runtime:
            runtime.get("executor")
            runtime.rebuild("executor")
            runtime.get("router")
    assert len(created) == 2 and all(instance.closed for instance in created)


async def test_parent_cleanup_includes_failed_child_construction(settings):
    executor = Executor()

    def broken(context):
        context.get("executor")
        raise ValueError("broken child")

    async with ModRuntime(
        settings,
        HarnessMods(
            {
                "executor": lambda ctx: executor,
                "agent": broken,
            }
        ),
    ) as runtime:
        with pytest.raises(ModError, match="broken child"):
            runtime.fork().get("agent")
    assert executor.closed


async def test_refresh_skills_uses_configured_factory(settings):
    created = []

    def skills(context):
        instance = context.default()
        created.append(instance)
        return instance

    agent = create_agent(replace(settings, demo=True), HarnessMods({"skills": skills}))
    try:
        agent.skills.activate("code-review")
        agent.refresh_skills()
        assert len(created) == 2 and agent.skills is created[1]
        assert "code-review" in agent.skills.pinned
        result = await agent.registry.execute("skill_search", {"query": "code-review"})
        assert result.ok and "code-review" in result.content
    finally:
        await agent.aclose()


@pytest.mark.parametrize("code_mode", [False, True])
async def test_documented_example_runs_without_credentials(settings, code_mode):
    manifest = Path(__file__).resolve().parents[1] / "examples" / "mods.toml"
    settings = replace(settings, api_key="", code_mode=code_mode, mods_path=manifest)
    agent = create_agent(settings)
    try:
        assert not agent.requires_api_key
        result = await agent.run("Inspect this workspace", ignore, deny)
        assert result.status == "completed", result.message
        assert "Custom harness observed" in result.message
        if code_mode:
            assert "My modded workspace" in result.message
    finally:
        await agent.aclose()
    assert await check_connection(settings)


async def test_diagnostics_honors_mods_and_closes_clients(settings, capsys):
    executor = Executor()
    mods = HarnessMods({"executor": lambda ctx: executor, "router": lambda ctx: Router()})
    assert await check_connection(replace(settings, api_key=""), mods)
    assert executor.closed
    assert "tool calling works via custom" in capsys.readouterr().out


async def test_default_tui_runs_custom_providers_without_api_key(settings):
    manifest = Path(__file__).resolve().parents[1] / "examples" / "mods.toml"
    app = create_app(replace(settings, api_key="", code_mode=True, mods_path=manifest))
    async with app.run_test():
        await app.submit_goal("Inspect this workspace")
        await app.workers.wait_for_complete()
        assert any("Custom harness observed" in text for _, text in app.transcript)
        assert not any(role == "SETUP" for role, _ in app.transcript)
