"""Example transforming hook: add a final newline before write approval."""

import json
import sys

payload = json.load(sys.stdin)
if payload["event"] == "before_tool" and payload.get("tool") == "write_file":
    arguments = payload["arguments"]
    if arguments["content"] and not arguments["content"].endswith("\n"):
        print(json.dumps({"arguments": {**arguments, "content": arguments["content"] + "\n"}}))
