"""Textual interface for conversational tasks, route traces, and action approvals."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime
from time import monotonic

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, Collapsible, Footer, Input, Label, Markdown, RichLog, Static
from textual.worker import Worker

from .agent import Agent
from .config import Settings
from .demo import DemoExecutor, DemoPlanner, DemoRouter
from .models import AgentEvent
from .openrouter import OpenRouterExecutor
from .planner import Planner
from .routing import HarnessDecisionLayer
from .tools import ToolError


class MessageCard(Vertical):
    def __init__(self, role: str, content: str, *, flavor: str = "assistant") -> None:
        super().__init__(classes=f"message {flavor}")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(self.role, classes="message-role", markup=False)
        yield Markdown(self.content, classes="message-body")


class ApprovalScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("escape", "deny", "Deny"),
        Binding("ctrl+y", "approve", "Approve once"),
        Binding("ctrl+x", "app.stop", "Stop run"),
    ]

    def __init__(self, tool: str, preview: str) -> None:
        super().__init__()
        self.tool, self.preview = tool, preview

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label(f"APPROVAL  /  {self.tool}", id="approval-title")
            yield Static("Review this action. Approval applies to this call only.")
            with VerticalScroll(id="approval-preview"):
                yield Static(self.preview, id="approval-content", markup=False)
            with Horizontal(id="approval-buttons"):
                yield Button("Deny", id="deny")
                yield Button("Approve once", id="approve", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#deny", Button).focus()

    @on(Button.Pressed, "#deny")
    def action_deny(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#approve")
    def action_approve(self) -> None:
        self.dismiss(True)


class AgentApp(App):
    TITLE = "Agent TUI"
    SUB_TITLE = "Harness Router × OpenRouter"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("escape", "stop", "Stop"),
        Binding("ctrl+n", "new_chat", "New chat"),
        Binding("ctrl+s", "save", "Save transcript"),
        Binding("ctrl+l", "focus_prompt", "Focus prompt"),
        Binding("ctrl+o", "toggle_activity", "Activity"),
    ]

    def __init__(
        self, settings: Settings, initial_prompt: str | None = None, agent: Agent | None = None
    ) -> None:
        super().__init__()
        self.register_theme(
            Theme(
                name="jev-pink",
                primary="#8e3e60",
                secondary="#785767",
                accent="#a32f60",
                foreground="#35272f",
                background="#f6dbe4",
                surface="#f9e7ee",
                panel="#efd0dd",
                boost="#d398af",
                success="#35624d",
                warning="#805410",
                error="#a32948",
                dark=False,
            )
        )
        self.theme = "jev-pink"
        self.settings = settings
        self.initial_prompt = initial_prompt
        if agent:
            self.agent = agent
        else:
            router = DemoRouter() if settings.demo else HarnessDecisionLayer(settings)
            executor = (
                DemoExecutor(settings.workspace) if settings.demo else OpenRouterExecutor(settings)
            )
            planner = None
            if settings.planning:
                planner = DemoPlanner() if settings.demo else Planner(executor, settings.mcts_depth)
            self.agent = Agent(settings, router, executor, planner=planner)
        self._worker: Worker | None = None
        self._busy = False
        self._closed = False
        self._started = 0.0
        self._tokens = 0
        self._calls = 0
        self._step = 0
        self._run_number = 0
        self._stream_card: MessageCard | None = None
        self._stream_text = ""
        self._stream_rendered = ""
        self._stream_updated = 0.0
        self._tool_cards: dict[int, tuple[Collapsible, Static, str]] = {}
        self._subagent_cards: dict[str, tuple[Collapsible, Static, str]] = {}
        self.transcript: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="masthead"):
            yield Static("✦  Agent TUI", id="brand")
            mode = "OFFLINE DEMO" if self.settings.demo else "HARNESS ROUTER × OPENROUTER"
            yield Static(mode, id="stack-label")
        with Horizontal(id="workspace-bar"):
            yield Static(str(self.settings.workspace), id="workspace", markup=False)
            policy = (
                "READ ONLY"
                if self.settings.read_only
                else "AUTO APPROVE"
                if self.settings.auto_approve
                else "REVIEW ACTIONS"
            )
            yield Static(policy, id="policy")
        with Horizontal(id="body"):
            with Vertical(id="conversation-pane"), VerticalScroll(id="conversation"):
                yield MessageCard(
                    "READY WHEN YOU ARE",
                    "## A workspace. A goal. A next step.\n\n"
                    "Describe a task to inspect files, make changes, or run a check. "
                    "Follow each tool decision in the activity panel.\n\n"
                    "Try **“Explain this project”** or **“Find and fix the failing tests.”**\n\n"
                    "`/help` commands · `Ctrl+O` activity · `Esc` stop",
                    flavor="welcome",
                )
            with Vertical(id="sidebar"):
                yield Static("EXECUTION", classes="section-title")
                yield Static("● Idle", id="phase", markup=False)
                method = (
                    f"MCTS · {self.settings.mcts_simulations} simulations"
                    if self.settings.routing == "mcts"
                    else "Jev · direct routing"
                )
                yield Static(method, id="routing-mode", markup=False)
                yield Static("DECISION MODEL", classes="field-label")
                yield Static(
                    "offline/demo" if self.settings.demo else self.settings.router_model,
                    id="router-model",
                    markup=False,
                )
                yield Static("EXECUTOR", classes="field-label")
                yield Static(
                    "offline/demo" if self.settings.demo else self.settings.model,
                    id="executor-model",
                    markup=False,
                )
                yield Static("NEXT TOOL", classes="field-label")
                yield Static("Waiting for a task", id="next-tool", markup=False)
                yield Static("0 steps  ·  0 calls\n0 tokens  ·  0.0s", id="metrics", markup=False)
                yield Static("ACTIVITY", classes="section-title")
                yield RichLog(
                    id="activity", wrap=True, markup=False, highlight=False, max_lines=300
                )
        with Vertical(id="composer"):
            yield Static("Enter a task", id="composer-label")
            with Horizontal(id="input-row"):
                yield Input(placeholder="What should we work on?", id="prompt")
                yield Button("Run  ↵", id="run", variant="primary")
                yield Button("Stop", id="stop", variant="error", disabled=True)
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        self.set_interval(0.2, self._update_elapsed)
        if not self.settings.demo and not self.settings.api_key:
            await self._message(
                "SETUP",
                "Set `OPENROUTER_API_KEY` in your environment or `.env` "
                "and restart. Use `agent-tui --demo` to try the interface offline.",
                "notice",
            )
        if self.initial_prompt:
            self.call_after_refresh(self.submit_goal, self.initial_prompt)
        elif self.settings.demo:
            self.call_after_refresh(self.submit_goal, "Show me how the execution loop works")

    def on_resize(self, event: events.Resize) -> None:
        self.screen.set_class(event.size.width < 100, "compact")

    def action_toggle_activity(self) -> None:
        self.screen.toggle_class("show-activity")

    @on(Input.Submitted, "#prompt")
    async def on_prompt(self, event: Input.Submitted) -> None:
        await self.submit_goal(event.value)

    @on(Button.Pressed, "#run")
    async def on_run_button(self) -> None:
        await self.submit_goal(self.query_one("#prompt", Input).value)

    @on(Button.Pressed, "#stop")
    def action_stop(self) -> None:
        if self._worker and self._busy:
            self._worker.cancel()
            if isinstance(self.screen, ApprovalScreen):
                self.screen.dismiss(False)
            self.query_one("#phase", Static).update("● Stopping…")

    async def submit_goal(self, goal: str) -> None:
        goal = goal.strip()
        if not goal or self._busy:
            return
        self.query_one("#prompt", Input).value = ""
        if goal.startswith("/"):
            if goal.startswith("/plugin install "):
                self._busy = True
                self._step = self._tokens = self._calls = 0
                self._started = monotonic()
                self.query_one("#phase", Static).update("● Installing plugin")
                self.query_one("#run", Button).disabled = True
                self.query_one("#stop", Button).disabled = False
                self.query_one("#prompt", Input).disabled = True
                self._worker = self.run_worker(
                    self._install_plugin(goal.removeprefix("/plugin install ").strip()),
                    name="plugin-install",
                    exit_on_error=False,
                )
                return
            await self._command(goal)
            return
        if not self.settings.demo and not self.settings.api_key:
            await self._message(
                "SETUP",
                "Missing `OPENROUTER_API_KEY`. Set it and restart, or launch with `--demo`.",
                "notice",
            )
            return
        self._busy = True
        self._run_number += 1
        self._tokens = self._calls = 0
        self._step = 0
        self._started = monotonic()
        self._tool_cards.clear()
        self._subagent_cards.clear()
        self._stream_card = None
        self._stream_text = ""
        self._stream_rendered = ""
        self.query_one("#run", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        self.query_one("#prompt", Input).disabled = True
        self.query_one("#composer-label", Static).update("Working · Esc to stop")
        await self._message("YOU", goal, "user")
        self._worker = self.run_worker(self._run(goal), name="agent-run", exit_on_error=False)

    async def _run(self, goal: str) -> None:
        try:
            await self.agent.run(goal, self._event, self._approve)
        finally:
            self._busy = False
            self.query_one("#run", Button).disabled = False
            self.query_one("#stop", Button).disabled = True
            self.query_one("#prompt", Input).disabled = False
            self.query_one("#composer-label", Static).update("Enter a task or follow-up")
            self.query_one("#prompt", Input).focus()

    async def _approve(self, name: str, preview: str) -> bool:
        return bool(await self.push_screen_wait(ApprovalScreen(name, preview)))

    async def _message(self, role: str, text: str, flavor: str = "assistant") -> MessageCard:
        text = self.settings.redact(text)
        card = MessageCard(role, text, flavor=flavor)
        await self.query_one("#conversation", VerticalScroll).mount(card)
        self.transcript.append((role, text))
        self._scroll()
        return card

    def _scroll(self) -> None:
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    def _log(self, text: str, tone: str = "secondary") -> None:
        now = datetime.now().strftime("%H:%M:%S")
        color = getattr(self.current_theme, tone, None) or self.current_theme.foreground
        self.query_one("#activity", RichLog).write(
            Text(f"{now}  {self.settings.redact(text)}", style=color)
        )

    def _update_elapsed(self) -> None:
        if self._busy:
            self.query_one("#metrics", Static).update(
                f"{self._step}/{self.settings.max_steps} steps  ·  {self._calls} calls\n"
                f"{self._tokens:,} tokens  ·  {monotonic() - self._started:.1f}s"
            )

    async def _flush_stream(self, *, force: bool = False) -> None:
        if self._stream_text == self._stream_rendered:
            return
        if not force and self._stream_card and monotonic() - self._stream_updated < 0.05:
            return
        if not self._stream_card:
            self._stream_card = await self._message("ASSISTANT", "")
        await self._stream_card.query_one(Markdown).update(self._stream_text)
        self._stream_rendered = self._stream_text
        self._stream_updated = monotonic()
        self._scroll()

    async def _event(self, event: AgentEvent) -> None:
        if event.kind == "subagent":
            await self._subagent_event(event)
            return
        if event.kind == "token":
            # Parsing Markdown for every token blocks the executor's network stream.
            self._stream_text += event.text
            await self._flush_stream()
            return
        self._step = event.step
        phase = self.query_one("#phase", Static)
        if event.kind == "planning":
            phase.update("● Planning")
            self._log(f"Step {event.step} · planning")
        elif event.kind == "plan":
            steps = "\n".join(
                f"{index}. {step}" for index, step in enumerate(event.data["steps"], 1)
            )
            paths = "\n".join(
                " → ".join(action["tool"] for action in path) for path in event.data["paths"]
            )
            text = f"{event.text}\n\n{steps}\n\nPredicted paths (not executed):\n{paths}"
            card = Collapsible(
                Static(text, markup=False, classes="tool-content"),
                title=f"PLAN  /  step {event.step}",
                collapsed=event.step > 1,
                classes="plan-card",
            )
            await self.query_one("#conversation", VerticalScroll).mount(card)
            self.transcript.append(("PLAN", text))
            self._log(event.text, "accent")
            self._scroll()
        elif event.kind == "mcts":
            visits = ", ".join(f"{tool}: {count}" for tool, count in event.data["visits"].items())
            text = (
                f"{event.data['simulations']} local simulations · "
                f"{event.data['policy_evaluations']} Jev prior calls\n"
                f"Predicted path: {event.text}\nRoot visits: {visits}\n"
                "Only the first selected tool will execute."
            )
            card = Collapsible(
                Static(text, markup=False, classes="tool-content"),
                title=f"MCTS  /  {event.text}",
                collapsed=True,
                classes="tool-card",
            )
            await self.query_one("#conversation", VerticalScroll).mount(card)
            self.transcript.append(("MCTS", text))
            self._log(f"MCTS · {event.data['simulations']} simulations", "accent")
        elif event.kind == "routing":
            await self._flush_stream(force=True)
            phase.update("● Routing")
            self._stream_card, self._stream_text = None, ""
            self._stream_rendered = ""
            self._log(f"Step {event.step} · routing")
        elif event.kind == "route":
            if event.data["fallback"]:
                detail = f"Fallback · {event.data['reason']}"
                self._log(detail, "warning")
            else:
                label = "visit share" if event.data.get("mode") == "mcts" else "confidence"
                detail = f"{event.text}  ·  {event.data['confidence']:.0%} {label}"
                self._log(detail, "success")
            self.query_one("#next-tool", Static).update(detail)
        elif event.kind == "generating":
            phase.update("● Generating")
        elif event.kind == "usage":
            if event.data.get("source") != "planner":
                await self._flush_stream(force=True)
            self._tokens += event.data.get("tokens", 0)
            if event.data.get("model") and event.data.get("source") != "planner":
                self.query_one("#executor-model", Static).update(event.data["model"])
            if self._stream_text and event.data.get("source") != "planner":
                self.transcript.append(("ASSISTANT", self._stream_text))
        elif event.kind == "tool_start":
            phase.update("● Executing")
            self._calls += 1
            arguments = self.settings.redact(json.dumps(event.data["arguments"], indent=2))
            body = Static(arguments, markup=False, classes="tool-content")
            card = Collapsible(
                body, title=f"{event.step:02d}  {event.text}", collapsed=True, classes="tool-card"
            )
            await self.query_one("#conversation", VerticalScroll).mount(card)
            self._tool_cards[event.step] = (card, body, arguments)
            self._log(f"Calling {event.text}")
            self._scroll()
        elif event.kind == "approval":
            phase.update("● Awaiting approval")
            self._log(f"Review {event.text}", "warning")
        elif event.kind == "tool_result":
            status = "done" if event.data["ok"] else "error"
            if event.step in self._tool_cards:
                card, body, arguments = self._tool_cards[event.step]
                card.title += f"  ·  {status}"
                body.update(f"ARGUMENTS\n{arguments}\n\nRESULT\n{event.text}")
            self.transcript.append((f"TOOL {event.data['tool']}", event.text))
            self._log(
                f"{event.data['tool']} · {status}", "success" if event.data["ok"] else "error"
            )
        elif event.kind == "warning":
            self._log(event.text, "warning")
        elif event.kind == "context":
            self._log(f"Context {event.data['chars']:,}/{event.data['budget']:,} chars")
        elif event.kind == "extension":
            self._log(event.text, "accent")
        elif event.kind == "done":
            await self._flush_stream(force=True)
            status = event.data["status"]
            phase.update(f"● {status.capitalize()}")
            self._log(status.capitalize(), "success" if status == "completed" else "warning")
            if not self._stream_card or event.text != self._stream_text:
                await self._message(
                    "ASSISTANT" if status == "completed" else status.upper(),
                    event.text,
                    "assistant" if status == "completed" else "notice",
                )
        self._update_elapsed()

    async def _subagent_event(self, event: AgentEvent) -> None:
        identifier, name = event.data["id"], event.data["name"]
        kind = event.data["event"]
        details = event.data.get("details", {})
        phase = self.query_one("#phase", Static)
        if kind == "start":
            phase.update("● Delegating")
            body = Static(event.text, markup=False, classes="tool-content")
            card = Collapsible(
                body, title=f"SUBAGENT  /  {name}  ·  running", collapsed=True, classes="tool-card"
            )
            await self.query_one("#conversation", VerticalScroll).mount(card)
            self._subagent_cards[identifier] = (card, body, event.text)
            self._log(f"Subagent {name} started", "accent")
            self._scroll()
        elif kind == "usage":
            self._tokens += details.get("tokens", 0)
        elif kind == "tool_start":
            self._calls += 1
            phase.update("● Delegating")
            self._log(f"{name} · calling {event.text}")
        elif kind == "approval":
            phase.update("● Awaiting approval")
            self._log(f"{name} · review {event.text}", "warning")
        elif kind in {"warning", "extension"}:
            self._log(f"{name} · {event.text}", "warning")
        elif kind == "done":
            status = details["status"]
            if identifier in self._subagent_cards:
                card, body, prompt = self._subagent_cards[identifier]
                card.title = f"SUBAGENT  /  {name}  ·  {status}"
                body.update(f"TASK\n{prompt}\n\nRESULT\n{event.text}")
            self.transcript.append((f"SUBAGENT {name} / {status}", event.text))
            self._log(f"{name} · {status}", "success" if status == "completed" else "warning")
        self._update_elapsed()

    async def _command(self, command: str) -> None:
        if command in {"/clear", "/new"}:
            await self.action_new_chat()
        elif command == "/save":
            await self.action_save()
        elif command in {"/quit", "/exit"}:
            await self.action_quit()
        elif await self._extension_command(command):
            pass
        elif command == "/help":
            await self._message(
                "COMMANDS",
                "`/new` or `/clear` reset conversation\n\n"
                "`/save` export a Markdown transcript to `.agent-tui/`\n\n"
                "`/skills` list skills · `/skill NAME` activate · `/skill off NAME` unload\n\n"
                "`/mcp` server status · `/hooks` middleware and command hooks\n\n"
                "`/plugins` marketplace · `/plugin install superpowers`\n\n"
                "`/plugin enable NAME`, `/plugin disable NAME`, `/plugin uninstall NAME`\n\n"
                "`/context` context usage · `/compact` shorten conversation history\n\n"
                "`/memory` list saved notes and outcomes · `/memory search QUERY`\n\n"
                "`/memory set KEY TEXT` save a note · `/memory forget KEY` delete\n\n"
                "`/quit` close the application\n\n"
                "**Esc** stop · **Ctrl+N** new chat · **Ctrl+S** save\n\n"
                "Writes and commands require approval unless `--yes` is set. "
                "Use `--read-only` to omit those tools.",
                "notice",
            )
        else:
            await self._message("COMMAND", f"Unknown command: `{command}`. Use `/help`.", "notice")

    async def _extension_command(self, command: str) -> bool:
        name, _, argument = command.partition(" ")
        argument = argument.strip()
        try:
            if name in {"/plugins", "/marketplace"}:
                installed = self.agent.plugins.installed()
                text = "\n\n".join(
                    f"**{spec.name}** — {spec.description}\n\n"
                    f"{spec.repository}\n\n"
                    + (
                        "Installed · "
                        f"{'enabled' if installed[spec.name]['enabled'] else 'disabled'}"
                        f" · commit `{installed[spec.name]['commit'][:12]}`"
                        if spec.name in installed
                        else f"Install: `/plugin install {spec.name}`"
                    )
                    for spec in self.agent.plugins.catalog().values()
                )
            elif name == "/plugin":
                action, _, plugin = argument.partition(" ")
                if action in {"enable", "disable"} and plugin:
                    self.agent.plugins.enable(plugin, action == "enable")
                elif action == "uninstall" and plugin:
                    self.agent.plugins.uninstall(plugin)
                else:
                    raise ToolError("Use /plugin install|enable|disable|uninstall NAME")
                self.agent.refresh_skills()
                text = f"Plugin {plugin}: {action} complete."
            elif name == "/skills":
                text = (
                    "\n\n".join(
                        f"**{skill.name}** "
                        f"{'(active)' if skill.name in self.agent.skills.active else ''}"
                        f" — {skill.description}"
                        for skill in self.agent.skills.skills.values()
                    )
                    or "No skills found."
                )
                text += "\n\nActivate with `/skill NAME` or mention `$NAME` in a task."
            elif name == "/skill":
                if argument.startswith("off "):
                    skill = argument[4:].strip()
                    self.agent.skills.deactivate(skill)
                    text = f"Unloaded {skill}."
                else:
                    text = self.agent.skills.activate(argument)
            elif name == "/mcp":
                text = "\n\n".join(
                    f"**{key}**: {value}" for key, value in self.agent.mcp.status.items()
                )
                text = text or "No enabled MCP servers. Configure `.agent-tui/extensions.toml`."
            elif name == "/hooks":
                manager = self.agent.middleware
                text = f"Python middleware: {len(manager.handlers)}\n\n" + "\n\n".join(
                    f"**{hook.event}**: `{json.dumps(hook.command)}` · {hook.timeout:g}s"
                    for hook in manager.hooks
                )
                if not manager.hooks:
                    text += "No command hooks configured. See `examples/extensions.toml`."
                if self.settings.demo or self.settings.read_only:
                    text += "\n\nCommand hooks are disabled in demo and read-only modes."
            elif name == "/context":
                stats = self.agent.context.stats
                text = (
                    f"Last request: **{stats.chars:,}/{stats.budget:,} characters**, "
                    f"including {stats.tool_chars:,} characters of tool schemas.\n\n"
                    f"History: {len(self.agent.history)} turns. Active skills: "
                    f"{', '.join(self.agent.skills.active) or 'none'}.\n\n"
                    f"Omitted: {stats.dropped_turns} turns, {stats.dropped_exchanges} exchanges, "
                    f"{stats.dropped_references} references. "
                    f"Earlier turns shown as excerpts: {stats.summarized_turns}. "
                    f"Shortened tool results: {stats.truncated_results}."
                )
            elif name == "/compact":
                saved = self.agent.context.compact()
                text = f"Kept task/outcome summaries; removed {saved:,} characters of history."
            elif name == "/memory":
                if not self.agent.memory.enabled:
                    raise ToolError("Persistent memory is disabled (--no-memory or --demo)")
                action, _, rest = argument.partition(" ")
                if action == "set":
                    key, _, content = rest.partition(" ")
                    self.agent.memory.put(key, content)
                    text = f"Saved memory `{key}`."
                elif action == "forget" and rest:
                    text = "Memory deleted." if self.agent.memory.forget(rest) else "Key not found."
                elif action in {"", "search"}:
                    rows = self.agent.memory.search(rest if action else "")
                    text = (
                        "\n\n".join(
                            f"**{row['key']}** ({row['kind']})\n\n{row['content']}" for row in rows
                        )
                        or "No matching memories."
                    )
                else:
                    raise ToolError("Use /memory [search QUERY | set KEY TEXT | forget KEY]")
            else:
                return False
        except (ToolError, OSError, ValueError, sqlite3.Error) as exc:
            await self._message("EXTENSIONS", str(exc), "notice")
            return True
        await self._message(name[1:].upper(), text, "notice")
        return True

    async def _install_plugin(self, name: str) -> None:
        import asyncio

        try:
            await self._message("PLUGINS", f"Installing `{name}` from the marketplace…", "notice")
            commit = await self.agent.plugins.install(name)
            self.agent.refresh_skills()
            await self._message(
                "PLUGINS",
                f"Installed **{name}** at `{commit[:12]}`. Skills are ready in `/skills`. "
                + (
                    "Use `/skill superpowers:brainstorming` for brainstorming."
                    if name == "superpowers"
                    else ""
                ),
                "notice",
            )
            self.query_one("#phase", Static).update("● Plugin installed")
        except asyncio.CancelledError:
            self.query_one("#phase", Static).update("● Installation cancelled")
        except (ToolError, OSError, ValueError) as exc:
            await self._message("PLUGINS", str(exc), "notice")
            self.query_one("#phase", Static).update("● Installation failed")
        finally:
            self._busy = False
            self.query_one("#run", Button).disabled = False
            self.query_one("#stop", Button).disabled = True
            self.query_one("#prompt", Input).disabled = False
            self.query_one("#prompt", Input).focus()

    async def action_new_chat(self) -> None:
        if self._busy:
            self.notify("Stop the current run before starting a new chat", severity="warning")
            return
        self.agent.clear()
        self.transcript.clear()
        self._subagent_cards.clear()
        await self.query_one("#conversation", VerticalScroll).remove_children()
        self.query_one("#activity", RichLog).clear()
        self.query_one("#phase", Static).update("● Idle")
        self.query_one("#next-tool", Static).update("Waiting for a task")
        self.query_one("#metrics", Static).update("0 steps  ·  0 calls\n0 tokens  ·  0.0s")
        self.action_focus_prompt()

    async def action_save(self) -> None:
        if not self.transcript:
            self.notify("There is no transcript to save yet")
            return
        directory = self.settings.workspace / ".agent-tui"
        try:
            if directory.is_symlink():
                raise OSError("Transcript directory must not be a symlink")
            directory.mkdir(exist_ok=True, mode=0o700)
            filename = f"session-{datetime.now():%Y%m%d-%H%M%S-%f}.md"
            path = directory / filename
            text = "# Agent TUI session\n\n" + "\n\n".join(
                f"## {role}\n\n{content}" for role, content in self.transcript if content
            )
            with path.open("x", encoding="utf-8") as stream:
                stream.write(self.settings.redact(text))
            path.chmod(0o600)
            self.notify(f"Saved {path.relative_to(self.settings.workspace)}")
        except OSError as exc:
            self.notify(str(exc), severity="error")

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    async def _close_clients(self) -> None:
        if not self._closed:
            self._closed = True
            await self.agent.aclose()

    async def action_quit(self) -> None:
        self.action_stop()
        if self._worker:
            with contextlib.suppress(Exception):
                await self._worker.wait()
        await self._close_clients()
        self.exit()

    async def on_unmount(self) -> None:
        await self._close_clients()
