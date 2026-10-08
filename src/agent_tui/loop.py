"""The default agent loop. Mods can replace orchestration without replacing services."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from harness_router import ActionSummary, HarnessState, RouteDecision

from .models import AgentEvent, Approval, EventSink, RunResult, ToolCall
from .planner import PlanningError
from .policy import RunTools
from .tools import ToolError, ToolResult


class DefaultLoop:
    @staticmethod
    async def run(agent, goal: str, emit: EventSink, approve: Approval) -> RunResult:
        if agent.running:
            raise RuntimeError("An agent run is already active")
        if not goal.strip():
            raise ValueError("Enter a task first")
        agent.running = True
        exchanges: list[list[dict[str, Any]]] = []
        run_tools = RunTools()
        if agent.subagents:
            agent.subagents.begin_run(goal, emit, approve)
        state = HarnessState(
            goal=agent.settings.redact(goal),
            observation=f"Workspace: {agent.settings.workspace}. No tools executed this turn yet.",
            constraints=[
                "Use finish only when the task is complete or requires a user answer. "
                "Reading alone does not complete a modification task: apply the needed "
                "edit or write and verify the result.",
                "Use the latest tool result; avoid repeating failures. "
                "Use list_files and search_files to find relevant files, then read_file "
                "to inspect their contents. Reuse results when the content has not changed.",
                "Read-only tools only."
                if agent.settings.read_only
                else "Writes and commands may require user approval.",
            ],
        )
        if agent.history:
            state.observation += (
                " Previous turn: " + json.dumps(agent.history[-1], ensure_ascii=False)[-3500:]
            )
        step = 0
        result = RunResult("error", "Run did not complete", step)

        async def run_code_tool(arguments: dict[str, Any]) -> ToolResult:
            async def run_tool(name: str, args: dict[str, Any]) -> ToolResult:
                result, _ = await agent._execute_tool(
                    ToolCall("code", name, json.dumps(args)), run_tools, emit, approve, step
                )
                return result

            return await agent.code_runtime.execute(arguments["code"], agent.settings, run_tool)

        try:
            if agent.settings.code_mode:
                agent.registry.register(agent.code_runtime.spec, run_code_tool)
            for warning in agent.skills.warnings:
                await emit(AgentEvent("warning", agent.settings.redact(warning)))
            agent.skills.begin_task(goal)
            for name in agent.plugins.bootstrap_skills():
                if name not in agent.skills.active:
                    agent.skills.activate(name, persistent=False)
            await agent.middleware.authorize(approve)
            await agent.middleware.dispatch("before_run", {"goal": goal})
            await agent.mcp.connect(agent.registry, approve, emit)
            memory_context = agent.memory.context(goal)
            state.constraints.append(
                "Active skills: "
                + (", ".join(agent.skills.active) or "none")
                + (". " + memory_context[:2500] if memory_context else "")
            )
            for step in range(1, agent.settings.max_steps + 1):
                if agent.settings.code_mode:
                    tools = [
                        tool
                        for tool in agent.registry.descriptors()
                        if tool.name in {"execute_code", "finish"}
                    ]
                    plan, mode = None, "jev"
                    await emit(AgentEvent("routing", "Choosing a Code Mode action", step))
                    decision = await agent.router.route(state, tools)
                else:
                    tools = agent.registry.descriptors()
                    plan = None
                    decision = None
                    if agent.planner:
                        if agent.settings.routing == "jev":
                            plan, decision = await agent._plan_and_route(state, tools, emit, step)
                        else:
                            plan = await agent._plan(state, tools, emit, step)
                    await emit(AgentEvent("routing", "Choosing the next tool", step))
                    mode = "jev"
                    if agent.settings.routing == "mcts" and plan:
                        try:
                            search = await agent.router.route_mcts(state, tools, plan)
                            decision = search.decision
                            mode = "mcts"
                            await emit(
                                AgentEvent(
                                    "mcts",
                                    " → ".join(search.principal_variation),
                                    step,
                                    {
                                        "visits": dict(search.root_visits),
                                        "values": dict(search.root_values),
                                        "simulations": search.simulations,
                                        "policy_evaluations": search.policy_evaluations,
                                    },
                                )
                            )
                        except (PlanningError, ValueError, TimeoutError) as exc:
                            await emit(
                                AgentEvent("warning", f"MCTS unavailable; using Jev: {exc}", step)
                            )
                            decision = await agent.router.route(state, tools)
                    elif decision is None:
                        decision = await agent.router.route(state, tools)
                selected = None if decision.fallback else decision.tool
                if selected not in {tool.name for tool in tools} and not decision.fallback:
                    decision = RouteDecision.fallback_to_planner("unavailable_tool")
                    selected = None
                await emit(
                    AgentEvent(
                        "route",
                        selected or "Executor fallback",
                        step,
                        {
                            "tool": selected,
                            "confidence": decision.confidence,
                            "fallback": decision.fallback,
                            "reason": decision.fallback_reason,
                            "mode": mode,
                        },
                    )
                )
                if agent.settings.code_mode and selected == "execute_code":
                    await emit(AgentEvent("code_mode", "Composing a tool program", step))

                async def on_token(text: str, step_number: int = step) -> None:
                    await emit(AgentEvent("token", agent.settings.redact(text), step_number))

                await emit(AgentEvent("generating", "Generating tool arguments", step))
                hook = await agent.middleware.dispatch(
                    "before_model", {"goal": goal, "step": step, "selected": selected}
                )
                schemas = agent.registry.schemas(selected)
                if agent.settings.code_mode:
                    schemas = [
                        schema
                        for schema in schemas
                        if schema["function"]["name"] in {"execute_code", "finish"}
                    ]
                messages = agent._context(
                    goal, exchanges, plan, schemas, hook.get("context", ""), selected
                )
                await emit(AgentEvent("context", step=step, data=agent.context.stats.as_dict()))
                completion = await agent.executor.complete(
                    messages,
                    schemas,
                    selected,
                    on_token,
                )
                await emit(
                    AgentEvent(
                        "usage",
                        step=step,
                        data={
                            "model": completion.model,
                            "tokens": completion.tokens,
                        },
                    )
                )
                if not completion.calls:
                    exchanges.append([completion.as_message()])
                    if selected in {None, "finish"} and completion.content.strip():
                        result = RunResult("completed", completion.content, step)
                        break
                    problem = (
                        f"The router selected {selected}. Return exactly one call to that tool."
                    )
                    exchanges.append([{"role": "user", "content": problem}])
                    state.observation = problem
                    await emit(AgentEvent("warning", problem, step))
                    continue

                if agent.settings.code_mode:
                    try:
                        adapted = agent.code_runtime.adapt(completion, agent.registry)
                    except ToolError as exc:
                        problem = agent.settings.redact(str(exc))
                        exchanges.append(
                            [
                                completion.as_message(),
                                *(
                                    agent._tool_message(call, ToolResult(False, problem))
                                    for call in completion.calls
                                ),
                            ]
                        )
                        state.observation = problem
                        await emit(AgentEvent("warning", problem, step))
                        continue
                    if adapted is not completion:
                        await emit(
                            AgentEvent(
                                "extension",
                                "Wrapped direct tool calls in a Code Mode program",
                                step,
                            )
                        )
                        completion = adapted

                exchange = [completion.as_message()]
                exchanges.append(exchange)
                if len(completion.calls) != 1:
                    problem = "Only one tool call per step is allowed. No tools were executed."
                    for call in completion.calls:
                        exchange.append(agent._tool_message(call, ToolResult(False, problem)))
                    state.observation = problem
                    await emit(AgentEvent("warning", problem, step))
                    continue
                call = completion.calls[0]
                allowed = {selected} if selected else None
                if agent.settings.code_mode and allowed is None:
                    allowed = {"execute_code", "finish"}
                tool_result, arguments = await agent._execute_tool(
                    call, run_tools, emit, approve, step, exchange, allowed
                )
                state.last_action = call.name
                state.observation = agent.settings.redact(
                    f"{call.name}({json.dumps(arguments, ensure_ascii=False)[:1500]})\n"
                    f"{'Success' if tool_result.ok else 'Error'}: {tool_result.content}"
                )[:6000]
                state.recent_actions.append(ActionSummary(call.name, state.observation[:2000]))
                if call.name == "finish" and tool_result.ok:
                    result = RunResult("completed", arguments["summary"], step)
                    break
            else:
                result = RunResult(
                    "limit",
                    f"Stopped at the {agent.settings.max_steps}-step limit. "
                    "Review the results and send a follow-up to continue.",
                    step,
                )
        except asyncio.CancelledError:
            result = RunResult(
                "cancelled", "Run stopped. Review any completed changes before continuing.", step
            )
        except Exception as exc:
            result = RunResult("error", agent.settings.redact(str(exc) or type(exc).__name__), step)
        finally:
            try:
                await agent.mcp.aclose()
                if result.status == "error":
                    await agent.middleware.dispatch("on_error", {"error": result.message})
                await agent.middleware.dispatch(
                    "after_run",
                    {
                        "goal": goal,
                        "status": result.status,
                        "message": result.message,
                        "steps": result.steps,
                    },
                )
            except Exception as exc:
                await emit(AgentEvent("warning", agent.settings.redact(f"Run cleanup: {exc}")))
            finally:
                if agent.settings.code_mode:
                    agent.registry.unregister("execute_code")
                agent.middleware.commands_enabled = False
                if agent.subagents:
                    agent.subagents.end_run()
                agent.running = False
            terminal = {"role": "assistant", "content": agent.settings.redact(result.message)}
            turn = [
                {"role": "user", "content": goal},
                *(message for exchange in exchanges for message in exchange),
                terminal,
            ]
            agent.context.remember_turn(turn)
            try:
                agent.memory.record_run(goal, result.status, result.message)
            except Exception as exc:
                await emit(AgentEvent("warning", agent.settings.redact(f"Memory not saved: {exc}")))
        await emit(
            AgentEvent(
                "done",
                agent.settings.redact(result.message),
                result.steps,
                {
                    "status": result.status,
                },
            )
        )
        return result
