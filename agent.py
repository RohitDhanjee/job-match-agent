import inspect
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import BadRequestError, Groq, RateLimitError

import tools
from tools import get_job_details, read_cv, score_match, search_jobs

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"
FREE_TIER_TOKENS_PER_MINUTE = 8000      # Groq free-tier limit for this model, from its 429 error message

# The user's own choices, applied to every search by our code instead of being guessed by the LLM
DEFAULT_SETTINGS = {"seniority": None, "worldwide": False, "exclude_unpaid": True}
SETTINGS_PATH = Path("settings.json")
SETTINGS = {**DEFAULT_SETTINGS,
            **(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")) if SETTINGS_PATH.exists() else {})}
LOG_PATH = Path("logs/runs.jsonl")

SYSTEM_PROMPT = """You help junior candidates find remote jobs.
Use the search_jobs tool to find real listings; never invent jobs.
Search at most 2 times. If the results are weak, say so honestly and show the closest matches.
The user's experience level, location and pay filters are applied to every search automatically,
so do not put words like "junior", "remote" or "entry-level" in queries.
If the user asks for jobs that fit their CV or profile: call read_cv, search with skills from the CV,
then call score_match. Recommend only jobs with matched skills. Base "why it fits" only on
matched_skills, and mention missing_skills as gaps. Never claim a skill fits unless it is listed there.
When you list jobs, include each job's title, company and url.
End with: "Jobs from Himalayas: https://himalayas.app"."""

# The "menu" the LLM sees: name, what the tool does, and which arguments it takes
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_jobs",
        "description": "Search remote job listings on Himalayas. Use short keyword queries, not full sentences.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short job keywords, e.g. 'machine learning engineer' or 'python developer'.",
                },
            },
            "required": ["query"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "read_cv",
        "description": "Read the user's CV: education, skills, projects and experience.",
        "parameters": {"type": "object", "properties": {}},
    },
}, {
    "type": "function",
    "function": {
        "name": "score_match",
        "description": "Compare the user's CV with every job found so far. Returns jobs ranked by how many "
                       "of the job's skills the CV has, with matched and missing skills. "
                       "Call after read_cv and search_jobs.",
        "parameters": {"type": "object", "properties": {}},
    },
}, {
    "type": "function",
    "function": {
        "name": "get_job_details",
        "description": "Get the full description and requirements of one job returned by search_jobs.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The 'id' of a job from search_jobs results."},
            },
            "required": ["job_id"],
        },
    },
}]

# Only these functions may be called, whatever name the LLM asks for
TOOL_FUNCTIONS = {"search_jobs": search_jobs, "read_cv": read_cv, "get_job_details": get_job_details,
                  "score_match": score_match}


MAX_STEPS = 8      # safety limit: at most this many tool rounds per question
MAX_SEARCHES = 2   # enforced in code, because the model often ignores "search at most 2 times" in the prompt
MAX_TOTAL_SEARCHES = 4   # empty searches are free, but only up to this many searches in total


def run_tool(name, raw_args):
    """Run one tool safely: any problem becomes an error message the LLM can read and fix."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(TOOL_FUNCTIONS)}."}
    try:
        args = json.loads(raw_args or "{}")
        known = inspect.signature(TOOL_FUNCTIONS[name]).parameters
        args = {k: v for k, v in args.items() if k in known}   # drop junk arguments the model invents
        if name == "search_jobs":
            args.update(SETTINGS)                    # the user's filters always win over anything the model sent
        return TOOL_FUNCTIONS[name](**args)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid arguments for {name}: {e}. Check the tool's parameters and try again."}
    except Exception as e:                       # network errors, missing CV file, etc.
        return {"error": f"{name} failed: {e}"}


RUN = {}                  # stats for the question being answered right now
MAX_RATE_LIMIT_WAIT = 30  # seconds; per-minute limits clear quickly, per-day limits do not           # stats for the question being answered right now


def chat(**kwargs):
    """Call the LLM, wait and retry on rate limits, and add the tokens used to RUN."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(model=MODEL, **kwargs)
            break
        except RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 10))
            if attempt == 2 or wait > MAX_RATE_LIMIT_WAIT:
                raise                                # a long wait means the daily limit is used up: stop now
            print(f"  -> rate limit reached, waiting {wait:.0f}s")
            time.sleep(wait)
    RUN["llm_calls"] += 1
    RUN["prompt_tokens"] += response.usage.prompt_tokens
    RUN["completion_tokens"] += response.usage.completion_tokens
    return response


def ask(user_message):
    """Answer one question, then log what it cost: steps, tokens, time and whether a guardrail fired."""
    RUN.clear()
    RUN.update(time=datetime.now().isoformat(timespec="seconds"), question=user_message, settings=SETTINGS,
               tool_calls=[], llm_calls=0, prompt_tokens=0, completion_tokens=0, guardrail=False)
    start = time.time()
    try:
        RUN["answer"] = run_agent(user_message)
        return RUN["answer"]
    except RateLimitError as e:
        minutes = max(1, round(float(e.response.headers.get("retry-after", 60)) / 60))
        RUN["error"] = "rate_limit"
        RUN["answer"] = f"The free Groq limit is used up for now. Please try again in about {minutes} minutes."
        return RUN["answer"]
    except BaseException as e:                       # BaseException also catches Ctrl+C, so it is logged too
        RUN["error"] = repr(e)
        raise
    finally:
        RUN["seconds"] = round(time.time() - start, 1)
        if LOG_PATH:                                 # the web app sets LOG_PATH = None to keep no visitor data
            LOG_PATH.parent.mkdir(exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(RUN, ensure_ascii=False) + "\n")
        tokens = RUN["prompt_tokens"] + RUN["completion_tokens"]
        print(f"  [{len(RUN['tool_calls'])} tool calls | {RUN['llm_calls']} LLM calls | "
              f"{tokens:,} tokens | {RUN['seconds']}s]")
        if tokens > FREE_TIER_TOKENS_PER_MINUTE:
            print(f"  [warning: more than the free tier's {FREE_TIER_TOKENS_PER_MINUTE:,} tokens per minute]")


def run_agent(user_message):
    tools.JOB_CACHE.clear()                         # each question starts with no jobs from earlier questions
    tools.CV_READ = False
    if is_cv_question(user_message):
        return cv_workflow(user_message)            # fixed steps in code; the agent loop below is for open questions
    searches = 0                                    # searches that found jobs
    total_searches = 0                              # all searches, including empty ones
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for step in range(MAX_STEPS):
        response = chat(messages=messages, tools=TOOLS)
        message = response.choices[0].message
        if not message.tool_calls:
            if message.content and message.content.strip():
                if tools.CV_READ and not tools.JOB_CACHE and total_searches < MAX_TOTAL_SEARCHES:
                    return fallback_search(user_message, messages)   # CV read, but every search was empty
                return verified(message.content, messages)   # no more tools needed: this is the final answer
            # Empty reply: the model only "thought" and stopped. Nudge it and try again.
            print(f"  -> step {step + 1}: empty reply, asking again")
            messages.append({"role": "system",
                             "content": "Your last reply was empty. Either call a tool or write the final answer."})
            continue

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ],
        })

        # Our code runs each requested tool and hands the result back, then the loop asks the LLM again
        for call in message.tool_calls:
            print(f"  -> step {step + 1}: {call.function.name}({call.function.arguments})")
            tool_start = time.time()
            is_search = call.function.name == "search_jobs"
            if is_search and (searches >= MAX_SEARCHES or total_searches >= MAX_TOTAL_SEARCHES):
                result = {"error": "Search limit reached. Use the jobs already found: "
                                   "call score_match if you read the CV, otherwise write the final answer."}
            else:
                result = run_tool(call.function.name, call.function.arguments)
                if is_search:
                    total_searches += 1
                    searches += isinstance(result, list)   # only searches that found jobs use up this budget
            RUN["tool_calls"].append({"tool": call.function.name, "args": call.function.arguments,
                                      "error": isinstance(result, dict) and "error" in result,
                                      "seconds": round(time.time() - tool_start, 2)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

    return final_answer(user_message, messages)


def run_tool_by_code(name, args, messages):
    """Run a tool ourselves (not because the model asked), log it, and add its result to the conversation."""
    start = time.time()
    result = run_tool(name, json.dumps(args))
    RUN["tool_calls"].append({"tool": name, "args": json.dumps(args), "by_code": True,
                              "error": isinstance(result, dict) and "error" in result,
                              "seconds": round(time.time() - start, 2)})
    messages.append({"role": "tool", "tool_call_id": f"code_{name}", "content": json.dumps(result)})
    return result


CV_QUESTION = re.compile(r"\b(cv|resume|profile)\b|\b(fits?|suits?|match(es)?) me\b", re.I)


def is_cv_question(user_message):
    """Route by keywords: questions about the user's own CV get the fixed workflow."""
    return bool(CV_QUESTION.search(user_message))


def cv_workflow(user_message):
    """The steps for a CV question are always the same, so code runs them and the LLM only writes the answer."""
    messages = []
    if "error" in run_tool_by_code("read_cv", {}, messages):
        return "Please upload your CV (PDF) first, then ask again."
    for title in tools.cv_titles()[:2]:             # common titles for the CV's skills, e.g. "python developer"
        print(f"  -> workflow: searching '{title}'")
        run_tool_by_code("search_jobs", {"query": title}, messages)
    if tools.JOB_CACHE:
        run_tool_by_code("score_match", {}, messages)
    return final_answer(user_message, messages)


def fallback_search(user_message, messages):
    """The model gave up after empty searches: search the CV's most common job title ourselves."""
    titles = tools.cv_titles()
    if titles:
        print(f"  -> fallback: searching '{titles[0]}', a common title for the CV's skills")
        run_tool_by_code("search_jobs", {"query": titles[0]}, messages)
    return final_answer(user_message, messages)


def final_answer(user_message, messages):
    """Write the answer from the tool results collected so far, without offering any tools."""
    cv_was_read = any(m["role"] == "tool" and '"cv":' in m["content"] for m in messages)
    if cv_was_read and tools.JOB_CACHE and not any(c["tool"] == "score_match" for c in RUN["tool_calls"]):
        # The model never scored the jobs: do the CV comparison in code so the answer has evidence
        run_tool_by_code("score_match", {}, messages)
    tool_results = [m["content"] for m in messages if m["role"] == "tool"]
    # No mention of tools here: words like "tool" make this model try to call one again
    try:
        response = chat(messages=[
            {"role": "system", "content": "You are a career advisor. Write a short markdown answer for the "
                                          "candidate. Never output JSON or function calls."},
            {"role": "user", "content": f"Candidate request: {user_message}\n\n"
                                        "CV and job listings collected earlier:\n" + "\n".join(tool_results) +
                                        "\n\nPick the best matches from these listings only, with title, company, "
                                        "url and one line on why each fits. If matched_skills are given, base "
                                        "\"why it fits\" only on them and list missing_skills as gaps. "
                                        "End with: Jobs from Himalayas: https://himalayas.app"},
        ])
        return verified(response.choices[0].message.content, messages)
    except BadRequestError:
        # The model still tried to call a tool: list the jobs we found with our own code instead
        return jobs_list_answer(messages, "I could not finish comparing these jobs with your CV. Here are the jobs I found:")


def found_jobs(messages):
    """Every job the tools returned in this conversation, from search results and job details."""
    jobs = {}
    for m in messages:
        if m["role"] != "tool":
            continue
        data = json.loads(m["content"])
        for job in data if isinstance(data, list) else [data]:
            if isinstance(job, dict) and "url" in job:
                jobs[job["url"]] = job
    return list(jobs.values())


def jobs_list_answer(messages, note):
    """A plain list of the jobs we really found, written by our own code instead of the LLM."""
    lines = [f"- {j['title']} @ {j['company']}: {j['url']}" for j in found_jobs(messages)[:5]]
    return note + "\n" + ("\n".join(lines) or "- No matching jobs were found.") + \
        "\n\nJobs from Himalayas: https://himalayas.app"


def verified(answer, messages):
    """Guardrail: if the answer links to a job the tools never returned, show the real list instead."""
    def strip(url):
        return url.rstrip(".,;:*").split("?")[0]
    allowed = {strip(j["url"]) for j in found_jobs(messages)} | {"https://himalayas.app"}
    links = [strip(u) for u in re.findall(r"https?://[^\s)<>|\]]+", answer)]
    if all(link in allowed for link in links):
        return answer
    print("  -> guardrail: the answer linked to a job the tools never returned")
    RUN["guardrail"] = True
    return jobs_list_answer(messages, "I found these jobs (my written summary had a link I could not verify):")


if __name__ == "__main__":
    while True:
        user_message = input("\nYou (empty Enter = exit): ").strip()
        if not user_message:
            break
        print("\n" + ask(user_message))