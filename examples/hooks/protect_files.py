"""Example policy hook: block writes/edits to lockfiles and production.env."""

import json
import sys
from pathlib import PurePath

payload = json.load(sys.stdin)
if payload["event"] == "before_tool" and payload.get("tool") in {"write_file", "edit_file"}:
    name = PurePath(payload["arguments"]["path"]).name
    if name.endswith(".lock") or name == "production.env":
        print(json.dumps({"block": f"{name} is protected by the example file policy"}))
