"""Run the agent on a few scenarios with the real LLM and check that it behaves as expected.
Each case costs real Groq tokens (roughly 2k-10k), so keep the list short."""
import json
import re
import sys
import time
from pathlib import Path

import agent

PAUSE_SECONDS = 20          # pause between cases to stay under the free tier's 8,000 tokens per minute


def check(case, answer, run):
    """Return a list of problems with this run; an empty list means the case passed."""
    called = [c["tool"] for c in run["tool_calls"] if not c["error"]]
    problems = []
    if run.get("error"):
        problems.append(f"run failed: {run['error']}")
    for tool in case["must_call"]:
        if tool not in called:
            problems.append(f"did not call {tool}")
    for tool in case["must_not_call"]:
        if tool in called:
            problems.append(f"should not call {tool}")
    if len(run["tool_calls"]) > case["max_tool_calls"]:
        problems.append(f"{len(run['tool_calls'])} tool calls, limit {case['max_tool_calls']}")
    links = re.findall(r"https?://[^\s)<>|\]]+", answer)
    if any(not link.startswith("https://himalayas.app") for link in links):
        problems.append("answer contains a link that is not from Himalayas")
    if case["credit_himalayas"] and "himalayas.app" not in answer:
        problems.append("answer does not credit Himalayas")
    return problems


def main():
    args = sys.argv[1:]                                      # e.g. "python eval_agent.py 2 cv_match"
    repeats = int(args[0]) if args and args[0].isdigit() else 1
    names = [a for a in args if not a.isdigit()]
    cases = [c for c in json.loads(Path("eval_cases.json").read_text(encoding="utf-8"))
             if not names or c["name"] in names]
    agent.LOG_PATH = Path("logs/eval_runs.jsonl")            # keep eval runs apart from real usage
    results = []
    for case in cases:
        for _ in range(repeats):
            print(f"\n=== {case['name']}: {case['question']}")
            answer = agent.ask(case["question"])
            run = dict(agent.RUN)
            problems = check(case, answer, run)
            results.append({"case": case["name"], "passed": not problems, "problems": problems,
                            "tokens": run["prompt_tokens"] + run["completion_tokens"],
                            "seconds": run["seconds"], "tool_calls": len(run["tool_calls"]),
                            "guardrail": run["guardrail"]})
            print("PASS" if not problems else "FAIL: " + "; ".join(problems))
            time.sleep(PAUSE_SECONDS)

    n = len(results)
    print("\n========== AGENT EVAL ==========")
    print(f"Passed           : {sum(r['passed'] for r in results)}/{n} = "
          f"{100 * sum(r['passed'] for r in results) / n:.0f}%")
    print(f"Average tokens   : {sum(r['tokens'] for r in results) / n:,.0f}")
    print(f"Average seconds  : {sum(r['seconds'] for r in results) / n:.1f}")
    print(f"Average tools    : {sum(r['tool_calls'] for r in results) / n:.1f}")
    print(f"Guardrail fired  : {sum(r['guardrail'] for r in results)}/{n}")
    for r in results:
        if not r["passed"]:
            print(f"  FAIL {r['case']}: {'; '.join(r['problems'])}")
    Path("logs/eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()