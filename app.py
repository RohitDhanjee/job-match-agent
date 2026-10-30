import os
import threading

import streamlit as st

st.set_page_config(page_title="Job Match Agent", page_icon="🧭")
st.title("🧭 Job Match Agent")
st.caption("Finds remote jobs on Himalayas and compares them with your CV. "
           "Ask in English or Roman Urdu, e.g. \"Read my CV and find jobs that fit me\" or \"react ki jobs chahiye\".")

try:
    for key, value in st.secrets.items():        # Streamlit Cloud secrets -> environment variables
        os.environ.setdefault(key, str(value))
except Exception:
    pass                                          # no secrets file locally; .env is used instead

import agent  # noqa: E402  (needs GROQ_API_KEY from the secrets above)
import tools  # noqa: E402

agent.LOG_PATH = None                             # keep no record of visitors' questions or CVs
MAX_QUESTIONS = 10                                # per visitor session, to protect the shared free Groq quota


@st.cache_resource
def agent_lock():
    """One question at a time: the agent keeps per-question state (jobs, CV) in module variables."""
    return threading.Lock()


with st.sidebar:
    st.header("Your settings")
    level = st.selectbox("Experience level", ["Entry-level", "Mid-level", "Senior", "Any"])
    worldwide = st.checkbox("Only jobs open to any country", value=True)
    exclude_unpaid = st.checkbox("Hide unpaid and volunteer roles", value=True)
    cv_file = st.file_uploader("Your CV (PDF)", type="pdf")
    st.caption("Your CV stays in memory for your question only and is never saved. "
               "Phone numbers, emails and ID numbers are removed before any text is sent to the AI model.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about jobs..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        asked = sum(m["role"] == "user" for m in st.session_state.messages)
        if asked > MAX_QUESTIONS:
            reply, run = f"This demo allows {MAX_QUESTIONS} questions per visit. Please come back later.", None
        else:
            with st.spinner("The agent is searching..."), agent_lock():
                agent.SETTINGS = {"seniority": None if level == "Any" else level,
                                  "worldwide": worldwide, "exclude_unpaid": exclude_unpaid}
                tools.CV_UPLOAD = cv_file.getvalue() if cv_file else None
                try:
                    reply = agent.ask(question)
                    run = dict(agent.RUN)
                finally:
                    tools.CV_UPLOAD = None        # never leave one visitor's CV for the next question
        st.markdown(reply)
        if run:
            with st.expander(f"How the agent worked: {len(run['tool_calls'])} tool calls, "
                             f"{run['prompt_tokens'] + run['completion_tokens']:,} tokens, {run['seconds']}s"):
                for i, call in enumerate(run["tool_calls"], 1):
                    who = "code" if call.get("by_code") else "LLM"
                    status = "error" if call["error"] else "ok"
                    st.markdown(f"{i}. `{call['tool']}` {call['args']} — chosen by **{who}**, {status}")
                if run["guardrail"]:
                    st.warning("The link guardrail replaced an answer that contained an unverified link.")

    st.session_state.messages.append({"role": "assistant", "content": reply})
