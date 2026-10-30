"""Summarise the agent's real usage from logs/runs.jsonl."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/runs.jsonl")
runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not runs:
    sys.exit(f"No runs in {path} yet.")

n = len(runs)
tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in runs]
print(f"Runs             : {n}  ({path})")
print(f"Errors           : {sum(bool(r.get('error')) for r in runs)}/{n}")
print(f"Guardrail fired  : {sum(r['guardrail'] for r in runs)}/{n}")
print(f"Average tokens   : {sum(tokens) / n:,.0f}  (max {max(tokens):,})")
print(f"Average seconds  : {sum(r['seconds'] for r in runs) / n:.1f}")
print(f"Average tools    : {sum(len(r['tool_calls']) for r in runs) / n:.1f}")
tool_errors = sum(c["error"] for r in runs for c in r["tool_calls"])
tool_total = sum(len(r["tool_calls"]) for r in runs)
print(f"Tool errors      : {tool_errors}/{tool_total}")