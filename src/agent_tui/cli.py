from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .config import DEFAULT_CONTEXT_CHARS, Settings


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Agentic Python TUI — Harness Router decisions, OpenRouter execution"
    )
    cli.add_argument("--version", action="version", version=f"agent-tui {__version__}")
    cli.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace (default: cwd)")
    cli.add_argument("--model", help="Executor model (default: openrouter/free)")
    cli.add_argument("--router-model", help="Decision model (default: typesafe/jev-1.13)")
    cli.add_argument(
        "--fast",
        action="store_true",
        help="Use Jev routing without a planner request (same as --routing jev --no-planner)",
    )
    cli.add_argument("--routing", choices=["jev", "mcts"], help="Decision method (default: mcts)")
    cli.add_argument(
        "--route-mcts",
        action="store_const",
        const="mcts",
        dest="routing",
        help="Use the planner and Harness Router Monte Carlo Tree Search",
    )
    cli.add_argument(
        "--no-planner",
        action="store_false",
        dest="planning",
        default=None,
        help="Skip the planning request (Jev routing only)",
    )
    cli.add_argument("--mcts-simulations", type=int, help="MCTS simulation budget (default: 64)")
    cli.add_argument("--mcts-depth", type=int, help="MCTS lookahead depth, 1–5 (default: 3)")
    cli.add_argument("--max-steps", type=int, help="Maximum agent steps per task (default: 24)")
    cli.add_argument("--command-timeout", type=float, help="Command time limit in seconds")
    cli.add_argument(
        "--context-chars",
        type=int,
        help=f"Context budget including tool schemas (default: {DEFAULT_CONTEXT_CHARS:,})",
    )
    cli.add_argument(
        "--extensions", type=Path, help="MCP/hook TOML (default: .agent-tui/extensions.toml)"
    )
    cli.add_argument("--skills-dir", action="append", type=Path, help="Additional skill directory")
    cli.add_argument(
        "--no-memory",
        dest="memory_enabled",
        action="store_false",
        help="Disable persistent workspace memory",
    )
    cli.add_argument("--read-only", action="store_true", help="Disable file writes and commands")
    cli.add_argument(
        "--yes",
        action="store_true",
        dest="auto_approve",
        help="Approve actions, MCP connections, and hooks (trusted workspaces only)",
    )
    cli.add_argument("--demo", action="store_true", help="Run an offline demonstration")
    cli.add_argument(
        "--check", action="store_true", help="Check live routing and executor connections"
    )
    cli.add_argument("--prompt", help="Start with this task instead of waiting for input")
    return cli


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if args.fast:
        if args.routing == "mcts":
            cli.error("--fast cannot be combined with --routing mcts or --route-mcts")
        args.routing = "jev"
        args.planning = False
    workspace = args.workspace.expanduser().resolve()
    load_dotenv(workspace / ".env", override=False)
    try:
        settings = Settings.from_env(
            workspace,
            model=args.model,
            router_model=args.router_model,
            max_steps=args.max_steps,
            command_timeout=args.command_timeout,
            read_only=args.read_only,
            auto_approve=args.auto_approve,
            demo=args.demo,
            planning=args.planning,
            routing=args.routing,
            mcts_simulations=args.mcts_simulations,
            mcts_depth=args.mcts_depth,
            context_chars=args.context_chars,
            extensions_path=args.extensions,
            skills_dirs=tuple(args.skills_dir or ()),
            memory_enabled=args.memory_enabled,
        )
    except ValueError as exc:
        cli.error(str(exc))
    # The TUI reports routing failures; keep upstream traceback logging off the terminal.
    logging.getLogger("harness_router").addHandler(logging.NullHandler())
    logging.getLogger("harness_router").propagate = False
    if args.check:
        from .diagnostics import check_connection

        raise SystemExit(0 if asyncio.run(check_connection(settings)) else 1)
    from .tui import AgentApp

    try:
        app = AgentApp(settings, initial_prompt=args.prompt)
    except ValueError as exc:
        cli.error(str(exc))
    app.run()
