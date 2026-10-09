"""Reproducible cold-versus-warm Code Mode repository scan microbenchmark.

Run after installing PyLattice: python benchmarks/benchmark_repository_context.py.
This measures filesystem context collection only; not LLM or router latency.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

from agent_tui.config import Settings
from agent_tui.context import ContextManager
from agent_tui.tools import ToolRegistry


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pylattice-context-bench-") as directory:
        root = Path(directory)
        source_dir = root / "src"
        source_dir.mkdir()
        for number in range(240):
            marker = "needle" if number == 175 else "other_symbol"
            body = (f"def helper_{number}():\n    return '{marker}'\n" * 200)
            (source_dir / f"module_{number:03d}.py").write_text(body, encoding="utf-8")

        registry = ToolRegistry(Settings(workspace=root, max_output_chars=16000))
        registry.bind_context(ContextManager(registry.settings))
        query = {"query": "needle", "path": "src", "max_files": 4}

        started = time.perf_counter()
        first = await registry.execute("context_collect", query)
        cold_ms = 1000 * (time.perf_counter() - started)
        if not first.ok:
            raise RuntimeError(first.content)
        cold = json.loads(first.content)

        elapsed: list[float] = []
        warm = {}
        for _ in range(7):
            started = time.perf_counter()
            result = await registry.execute("context_collect", query)
            elapsed.append(1000 * (time.perf_counter() - started))
            if not result.ok:
                raise RuntimeError(result.content)
            warm = json.loads(result.content)

        print(
            json.dumps(
                {
                    "files_in_repo": 240,
                    "source_bytes": sum(p.stat().st_size for p in source_dir.iterdir()),
                    "cold_ms": round(cold_ms, 3),
                    "warm_median_ms": round(statistics.median(elapsed), 3),
                    "cold_disk_reads": cold["disk_reads"],
                    "warm_disk_reads": warm["disk_reads"],
                    "warm_index_hits": warm["index_hits"],
                    "matches": [entry["path"] for entry in warm["files"]],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
