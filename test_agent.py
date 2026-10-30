"""Tests for the agent's code guardrails. A fake LLM plays back scripted replies,
so these tests use no Groq tokens, need no internet and give the same result every run."""
import json
from types import SimpleNamespace as N

import httpx
import pytest
from groq import RateLimitError

import agent
import tools

CV_TEXT = "Skills: Python, React, Node.js, Verilog, FPGA"
FAKE_JOB = {
    "guid": "job-1",
    "title": "Junior Python Developer",
    "companyName": "Acme",
    "seniority": ["Entry-level"],
    "locationRestrictions": [],
    "description": "<p>We use Python, React and Docker.</p>",
    "applicationLink": "https://himalayas.app/companies/acme/jobs/junior-python-developer?utm_source=api",
}
REAL_URL = FAKE_JOB["applicationLink"]


class FakeLLM:
    """Stands in for the Groq client: returns scripted replies and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):                   # a plain text answer
            message = N(content=reply, tool_calls=None)
        else:                                        # a list of (tool name, arguments) pairs
            message = N(content="", tool_calls=[
                N(id=f"call_{i}", function=N(name=name, arguments=json.dumps(args) if isinstance(args, dict) else args))
                for i, (name, args) in enumerate(reply)
            ])
        return N(choices=[N(message=message)], usage=N(prompt_tokens=100, completion_tokens=10))


def rate_limit(seconds):
    response = httpx.Response(429, headers={"retry-after": str(seconds)},
                              request=httpx.Request("POST", "https://api.groq.com"))
    return RateLimitError("rate limit", response=response, body=None)


@pytest.fixture
def searches(monkeypatch, tmp_path):
    """Replace the network and the CV with fakes, and keep test logs out of logs/runs.jsonl."""
    calls = []

    def fake_search(query, seniority=None, worldwide=False, exclude_unpaid=False, limit=10):
        calls.append({"query": query, "seniority": seniority, "worldwide": worldwide})
        tools.JOB_CACHE[FAKE_JOB["guid"]] = FAKE_JOB
        return [{"id": FAKE_JOB["guid"], "title": FAKE_JOB["title"], "company": FAKE_JOB["companyName"],
                 "url": REAL_URL}]

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_jobs", fake_search)
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "read_cv", lambda: {"cv": CV_TEXT})
    monkeypatch.setattr(tools, "read_cv", lambda: {"cv": CV_TEXT})
    monkeypatch.setattr(agent, "SETTINGS", {"seniority": "Entry-level", "worldwide": True, "exclude_unpaid": True})
    monkeypatch.setattr(agent, "LOG_PATH", tmp_path / "runs.jsonl")
    return calls


def use_llm(monkeypatch, replies):
    fake = FakeLLM(replies)
    monkeypatch.setattr(agent, "client", N(chat=N(completions=fake)))
    return fake


def test_search_limit_is_enforced_in_code(monkeypatch, searches):
    use_llm(monkeypatch, [[("search_jobs", {"query": "a"})], [("search_jobs", {"query": "b"})],
                          [("search_jobs", {"query": "c"})], "Done."])
    agent.ask("find jobs")
    assert len(searches) == 2                  # the third search never reached the job site
    assert agent.RUN["tool_calls"][2]["error"] is True


def test_empty_searches_do_not_use_up_the_budget(monkeypatch, searches):
    empty = []
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_jobs",
                        lambda query, **kw: empty.append(query) or {"jobs": [], "note": "No jobs found."})
    use_llm(monkeypatch, [[("search_jobs", {"query": "verilog"})], [("search_jobs", {"query": "fpga"})],
                          [("search_jobs", {"query": "asic"})], "Nothing found."])
    agent.ask("find hardware jobs")
    assert empty == ["verilog", "fpga", "asic"]                 # all three ran: none of them found any jobs


def test_total_search_cap_stops_endless_empty_searches(monkeypatch, searches):
    empty = []
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_jobs",
                        lambda query, **kw: empty.append(query) or {"jobs": [], "note": "No jobs found."})
    use_llm(monkeypatch, [[("search_jobs", {"query": q})] for q in ["a", "b", "c", "d", "e"]] + ["Nothing found."])
    agent.ask("find hardware jobs")
    assert len(empty) == 4                                      # the fifth empty search was blocked


def test_code_searches_a_cv_title_when_the_model_gives_up(monkeypatch, searches):
    queries = []

    def search(query, **kw):                     # hardware titles find nothing, software titles find a job
        queries.append(query)
        if "developer" not in query:
            return {"jobs": [], "note": "No jobs found."}
        tools.JOB_CACHE[FAKE_JOB["guid"]] = FAKE_JOB
        return [{"id": FAKE_JOB["guid"], "title": FAKE_JOB["title"], "company": "Acme", "url": REAL_URL}]

    def read_cv():
        tools.CV_READ = True
        return {"cv": CV_TEXT}

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_jobs", search)
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "read_cv", read_cv)
    fake = use_llm(monkeypatch, [[("read_cv", {})], [("search_jobs", {"query": "fpga engineer"})],
                                 "Sorry, I found nothing.", f"Junior Python Developer at Acme: {REAL_URL}"])
    answer = agent.ask("find jobs that use my skills")
    assert queries == ["fpga engineer", "python developer"]     # the second search was made by our code
    assert "matched_skills" in fake.requests[-1]["messages"][-1]["content"]   # and the jobs were scored
    assert REAL_URL in answer


@pytest.mark.parametrize("question, is_cv", [
    ("Read my CV and find 3 jobs that fit me.", True),
    ("meri cv ke hisaab se jobs batao", True),
    ("Which jobs match me?", True),
    ("Find me python developer jobs.", False),
    ("What is the capital of France?", False),
])
def test_cv_questions_are_routed_to_the_workflow(question, is_cv):
    assert agent.is_cv_question(question) is is_cv


def test_cv_workflow_runs_fixed_steps_and_one_llm_call(monkeypatch, searches):
    fake = use_llm(monkeypatch, [f"Junior Python Developer at Acme: {REAL_URL}\n\nJobs from Himalayas: https://himalayas.app"])
    answer = agent.ask("Read my CV and find jobs that fit me.")
    assert [c["tool"] for c in agent.RUN["tool_calls"]] == ["read_cv", "search_jobs", "search_jobs", "score_match"]
    assert [s["query"] for s in searches] == ["python developer", "frontend developer"]   # titles from CV_TEXT
    assert len(fake.requests) == 1 and "tools" not in fake.requests[0]   # the LLM only wrote the answer
    assert REAL_URL in answer


def test_open_questions_still_use_the_agent(monkeypatch, searches):
    fake = use_llm(monkeypatch, [[("search_jobs", {"query": "python"})], "Done."])
    agent.ask("Find me python developer jobs.")
    assert "tools" in fake.requests[0]


def empty_search_note(monkeypatch):
    """Call the real search_jobs with a fake job site that has no jobs, and return its note."""
    response = N(raise_for_status=lambda: None, json=lambda: {"jobs": []})
    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: response)
    return tools.search_jobs("fpga engineer")["note"]


def test_empty_search_suggests_titles_from_the_cv(monkeypatch):
    monkeypatch.setattr(tools, "CV_READ", True)
    monkeypatch.setattr(tools, "read_cv", lambda: {"cv": "Skills: Python, React, Verilog, FPGA"})
    note = empty_search_note(monkeypatch)
    assert "'python developer'" in note and "'frontend developer'" in note


def test_empty_search_does_not_use_the_cv_unless_it_was_read(monkeypatch):
    monkeypatch.setattr(tools, "CV_READ", False)
    monkeypatch.setattr(tools, "read_cv", lambda: pytest.fail("the CV should not be read"))
    assert "python developer" not in empty_search_note(monkeypatch)


def test_user_settings_override_the_model(monkeypatch, searches):
    use_llm(monkeypatch, [[("search_jobs", {"query": "python", "seniority": "Senior", "worldwide": False})], "Done."])
    agent.ask("find jobs")
    assert searches[0]["seniority"] == "Entry-level"
    assert searches[0]["worldwide"] is True


def test_guardrail_replaces_an_invented_link(monkeypatch, searches):
    use_llm(monkeypatch, [[("search_jobs", {"query": "python"})], "Apply at https://fake-jobs.com/dream-job"])
    answer = agent.ask("find jobs")
    assert "fake-jobs.com" not in answer
    assert REAL_URL in answer
    assert agent.RUN["guardrail"] is True


def test_real_links_pass_the_guardrail(monkeypatch, searches):
    reply = f"Junior Python Developer at Acme: <{REAL_URL}>\n\nJobs from Himalayas: https://himalayas.app"
    use_llm(monkeypatch, [[("search_jobs", {"query": "python"})], reply])
    assert agent.ask("find jobs") == reply
    assert agent.RUN["guardrail"] is False


def test_junk_arguments_and_unknown_tools_do_not_crash(monkeypatch, searches):
    use_llm(monkeypatch, [[("read_cv", {"": ""})], [("delete_files", {})], "Done."])
    assert agent.ask("show me some jobs") == "Done."
    assert agent.RUN["tool_calls"][0]["error"] is False         # junk argument was dropped, the tool still ran
    assert agent.RUN["tool_calls"][1]["error"] is True          # unknown tool became an error message


def test_empty_reply_gets_a_second_chance(monkeypatch, searches):
    fake = use_llm(monkeypatch, ["", "Here you go."])
    assert agent.ask("hello") == "Here you go."
    assert "empty" in fake.requests[1]["messages"][-1]["content"]


def test_step_limit_scores_the_cv_in_code(monkeypatch, searches):
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    fake = use_llm(monkeypatch, [[("read_cv", {})], [("search_jobs", {"query": "python"})], "Summary."])
    agent.ask("find jobs that use my skills")
    final_prompt = fake.requests[-1]["messages"][-1]["content"]
    assert "matched_skills" in final_prompt                     # score_match ran even though the model never called it
    assert "tools" not in fake.requests[-1]                     # the final call offers no tools


def test_daily_rate_limit_stops_at_once(monkeypatch, searches):
    monkeypatch.setattr(agent.time, "sleep", lambda s: pytest.fail("should not wait for a daily limit"))
    use_llm(monkeypatch, [rate_limit(342)])
    assert "try again in about 6 minutes" in agent.ask("find jobs")
    assert agent.RUN["error"] == "rate_limit"


def test_short_rate_limit_waits_and_retries(monkeypatch, searches):
    waits = []
    monkeypatch.setattr(agent.time, "sleep", waits.append)
    use_llm(monkeypatch, [rate_limit(2), "Done."])
    assert agent.ask("find jobs") == "Done."
    assert waits == [2.0]


def test_every_question_is_logged(monkeypatch, searches):
    use_llm(monkeypatch, [[("search_jobs", {"query": "python"})], "Done."])
    agent.ask("find jobs")
    log = json.loads(agent.LOG_PATH.read_text(encoding="utf-8").splitlines()[-1])
    assert log["question"] == "find jobs"
    assert log["prompt_tokens"] == 200 and log["llm_calls"] == 2


def test_redact_removes_personal_data():
    text = tools.redact("Call +92 309 1234567, mail me@x.com, CNIC 42301-0997025-7, GPA 3.58")
    assert "[PHONE]" in text and "[EMAIL]" in text and "[CNIC]" in text
    assert "3.58" in text


def test_find_skills_matches_whole_words_only():
    assert tools.find_skills("We love JavaScript and C++") == {"javascript", "c++"}   # no "java" inside "javascript"
    assert "golang" not in tools.find_skills("Let's go above and beyond")

def make_pdf(text):
    """A tiny one-page PDF containing `text`, so CV upload can be tested without a real file."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def test_uploaded_cv_is_read_and_redacted(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "CV_PATH", tmp_path / "missing.pdf")
    monkeypatch.setattr(tools, "CV_UPLOAD", make_pdf("Skills: Python, React. CNIC 42301-0997025-7"))
    cv = tools.read_cv()["cv"]
    assert "Python" in cv and "[CNIC]" in cv and "42301" not in cv


def test_cv_question_without_a_cv_asks_for_an_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "CV_PATH", tmp_path / "missing.pdf")
    monkeypatch.setattr(tools, "CV_UPLOAD", None)
    monkeypatch.setattr(agent, "LOG_PATH", None)
    fake = use_llm(monkeypatch, [])
    assert "upload your CV" in agent.ask("Find jobs for my CV")
    assert fake.requests == []                                  # no LLM call, no tokens spent


def test_no_log_file_when_logging_is_off(monkeypatch, searches, tmp_path):
    monkeypatch.setattr(agent, "LOG_PATH", None)
    use_llm(monkeypatch, ["Paris."])
    agent.ask("What is the capital of France?")
    assert list(tmp_path.iterdir()) == []
