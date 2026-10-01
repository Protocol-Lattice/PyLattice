"""Example observer: store event metadata without prompts, arguments, or tool output."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

payload = json.load(sys.stdin)
directory = Path(payload["workspace"]) / ".agent-tui"
path = directory / "hooks.jsonl"
if directory.is_symlink() or path.is_symlink():
    raise SystemExit("Audit paths must not be symlinks")
directory.mkdir(mode=0o700, exist_ok=True)
record = {
    key: payload[key]
    for key in ("event", "tool", "ok", "status", "step", "steps")
    if key in payload
}
record["timestamp"] = datetime.now(UTC).isoformat()
path.touch(mode=0o600, exist_ok=True)
with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\n")
