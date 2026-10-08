from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from . import __version__
from .config import DEFAULT_CONTEXT_CHARS, Settings
from .mods import HarnessMods, ModRuntime


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Agentic Python TUI — Code Mode with OpenRouter execution"
    )
    cli.add_argument("--version", action="version", version=f"agent-tui {__version__}")
    cli.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace (default: cwd)")
    cli.add_argument("--model", help="Executor model (default: openrouter/free)")
    cli.add_argument("--router-model", help="Decision model (default: typesafe/jev-1.13)")
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument(
        "--code-mode",
        dest="code_mode",
        action="store_true",
        default=None,
        help="Use the decision model to choose Python tool programs or finish (default)",
    )
    modes.add_argument(
        "--no-code-mode",
        dest="code_mode",
        action="store_false",
        help="Use the previous per-tool Harness Router loop",
    )
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
    cli.add_argument("--max-steps", type=int, help="Maximum agent steps per task (default: 1024)")
    cli.add_argument("--command-timeout", type=float, help="Command time limit in seconds")
    cli.add_argument(
        "--context-chars",
        type=int,
        help=f"Context budget including tool schemas (default: {DEFAULT_CONTEXT_CHARS:,})",
    )
    cli.add_argument(
        "--extensions", type=Path, help="MCP/hook TOML (default: .agent-tui/extensions.toml)"
    )
    cli.add_argument(
        "--mods", type=Path, help="Harness mods TOML (imports trusted Python factories)"
    )
    cli.add_argument(
        "--mod",
        action="append",
        default=[],
        metavar="SLOT=FACTORY",
        help="Override a mod with module:factory, path.py:factory, default, or none (repeatable)",
    )
    cli.add_argument(
        "--list-mods",
        action="store_true",
        help="Print configured mod slots without importing custom code",
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
    if args.code_mode and (args.fast or args.routing or args.planning is False):
        cli.error("--code-mode cannot be combined with per-tool routing or planning options")
    if args.fast:
        if args.routing == "mcts":
            cli.error("--fast cannot be combined with --routing mcts or --route-mcts")
        args.routing = "jev"
        args.planning = False
    if args.fast or args.routing or args.planning is False:
        args.code_mode = False
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
            code_mode=args.code_mode,
            planning=args.planning,
            routing=args.routing,
            mcts_simulations=args.mcts_simulations,
            mcts_depth=args.mcts_depth,
            context_chars=args.context_chars,
            extensions_path=args.extensions,
            mods_path=args.mods,
            mod_overrides=tuple(args.mod),
            skills_dirs=tuple(args.skills_dir or ()),
            memory_enabled=args.memory_enabled,
        )
        mods = HarnessMods.load(settings)
    except ValueError as exc:
        cli.error(str(exc))
    # The TUI reports routing failures; keep upstream traceback logging off the terminal.
    logging.getLogger("harness_router").addHandler(logging.NullHandler())
    logging.getLogger("harness_router").propagate = False
    if args.list_mods:
        for name, target in mods.describe().items():
            print(f"{name} = {target}")
        return
    if args.check:
        from .diagnostics import check_connection

        raise SystemExit(0 if asyncio.run(check_connection(settings, mods)) else 1)

    runtime = ModRuntime(settings, mods, initial_prompt=args.prompt)
    try:
        app = runtime.get("app")
        app.run()
    except ValueError as exc:
        cli.error(settings.redact(str(exc)))
    finally:
        asyncio.run(runtime.aclose())
