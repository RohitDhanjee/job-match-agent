import html
import io
import re
from pathlib import Path

import requests
from pypdf import PdfReader

HIMALAYAS_SEARCH_URL = "https://himalayas.app/jobs/api/search"
CV_PATH = Path("private/cv.pdf")
JOB_CACHE = {}      # job id -> full job, filled by search_jobs so get_job_details can look it up
CV_READ = False     # set by read_cv; the agent resets it for every question
CV_UPLOAD = None    # bytes of a CV PDF uploaded in the web app; when None, CV_PATH is used

# Common job titles for skills that usually have remote openings, used to suggest a better search
SKILL_TITLES = {
    "python": "python developer", "javascript": "javascript developer", "typescript": "typescript developer",
    "react": "frontend developer", "node.js": "backend developer", "flask": "python developer",
    "django": "python developer", "fastapi": "python developer", "java": "java developer",
    "flutter": "flutter developer", "php": "php developer", "laravel": "php developer",
    "wordpress": "wordpress developer", "machine learning": "machine learning engineer",
    "llm": "ai engineer", "nlp": "ai engineer", "sql": "data analyst", "c++": "c++ developer",
}

# Skills we look for in the CV and in job descriptions. Ambiguous words like "go" are written as "golang".
SKILLS = [
    "python", "java", "javascript", "typescript", "react", "node.js", "flask", "django", "fastapi",
    "sql", "mongodb", "postgresql", "mysql", "docker", "kubernetes", "aws", "azure", "gcp", "git", "linux",
    "c++", "c#", "golang", "rust", "verilog", "systemverilog", "vhdl", "fpga", "embedded", "firmware",
    "machine learning", "deep learning", "pytorch", "tensorflow", "nlp", "llm", "computer vision",
    "flutter", "firebase", "rest api", "graphql", "html", "css", "php", "laravel", "wordpress",
    "blockchain", "web3", "solidity", "pandas", "numpy", "scikit-learn", "data analysis", "excel",
    "power bi", "tableau", "figma", "arduino", "iot", "rtl",
]


def search_jobs(query, seniority=None, worldwide=False, exclude_unpaid=False, limit=10):
    """Search remote jobs on Himalayas and return a short summary of each one."""
    # The model often adds filler like "remote entry-level", which shrinks results (10 -> 2); filters do that job
    query = " ".join(re.sub(r"\b(remote|entry[- ]level|jobs?|worldwide|anywhere)\b", " ", query, flags=re.I).split())
    params = {"q": query}
    if seniority:                      # e.g. "Entry-level", "Mid-level", "Senior"
        params["seniority"] = seniority
    if worldwide:                      # only jobs open to candidates from any country
        params["worldwide"] = "true"
    response = requests.get(HIMALAYAS_SEARCH_URL, params=params, timeout=20)
    response.raise_for_status()
    jobs = response.json()["jobs"]
    if exclude_unpaid:
        jobs = [j for j in jobs if not re.search(r"\b(unpaid|volunteer)\b", j["title"] + " " + j["description"], re.I)]
    jobs = jobs[:limit]
    if not jobs:
        return {"jobs": [], "note": f"No jobs found for '{query}'. " + search_hint()}
    for j in jobs:
        JOB_CACHE[j["guid"]] = j
    return [
        {
            "id": j["guid"],
            "title": j["title"],
            "company": j["companyName"],
            "seniority": j["seniority"],
            "locations": j["locationRestrictions"] or ["Anywhere"],
            "url": j["applicationLink"],
        }
        for j in jobs
    ]


def cv_titles():
    """Common job titles for the CV's skills, best first."""
    cv_skills = find_skills(read_cv().get("cv", ""))
    return list(dict.fromkeys(SKILL_TITLES[s] for s in SKILLS if s in cv_skills and s in SKILL_TITLES))


def search_hint():
    """What to search next after an empty result: titles from the CV's skills, if this question used the CV."""
    titles = cv_titles() if CV_READ else []      # never leak CV details into a question that did not ask about the CV
    if not titles:
        return "Try a broader, more common job title."
    return "Titles that match the CV and usually have openings: " + ", ".join(f"'{t}'" for t in titles[:3]) + "."


def clean_html(text):
    """Strip HTML tags and entities, leaving plain text."""
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def redact(text):
    """Remove personal identifiers before the text is sent to an external LLM."""
    text = re.sub(r"\b\d{5}-\d{7}-\d\b", "[CNIC]", text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d\s-]{8,}\d", "[PHONE]", text)
    return text


def read_cv():
    """Return the candidate's CV as plain text, with personal identifiers removed."""
    global CV_READ
    if CV_UPLOAD is None and not CV_PATH.exists():
        return {"error": "No CV uploaded. Ask the user to upload their CV as a PDF."}
    CV_READ = True
    reader = PdfReader(io.BytesIO(CV_UPLOAD) if CV_UPLOAD is not None else CV_PATH)
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    return {"cv": redact(" ".join(text.split()))[:4000]}


def get_job_details(job_id):
    """Return the full description of a job found earlier by search_jobs."""
    job = JOB_CACHE.get(job_id)
    if job is None:
        return {"error": "Unknown job id. Call search_jobs first and use an id from its results."}
    return {
        "id": job_id,
        "title": job["title"],
        "company": job["companyName"],
        "seniority": job["seniority"],
        "locations": job["locationRestrictions"] or ["Anywhere"],
        "description": clean_html(job["description"])[:1500],
        "url": job["applicationLink"],
    }


def find_skills(text):
    """Return the known skills that appear in a text as whole words."""
    text = text.lower()
    return {s for s in SKILLS if re.search(r"(?<![\w+#.])" + re.escape(s) + r"(?![\w+#])", text)}


def score_match():
    """Rank every job found so far by how many of its skills the CV also has."""
    if not JOB_CACHE:
        return {"error": "No jobs found yet. Call search_jobs first."}
    cv_skills = find_skills(read_cv().get("cv", ""))
    results = []
    for job_id, job in JOB_CACHE.items():
        job_skills = find_skills(job["title"] + " " + clean_html(job["description"]))
        matched = job_skills & cv_skills
        results.append({
            "id": job_id,
            "title": job["title"],
            "company": job["companyName"],
            "matched_skills": sorted(matched),
            "missing_skills": sorted(job_skills - cv_skills),
            "score": f"{len(matched)}/{len(job_skills)}",
            "url": job["applicationLink"],
        })
    results.sort(key=lambda r: (len(r["matched_skills"]), -len(r["missing_skills"])), reverse=True)
    return results[:8]


if __name__ == "__main__":
    print(read_cv()["cv"][:300], "...\n")
    jobs = search_jobs("machine learning", seniority="Entry-level", worldwide=True)
    details = get_job_details(jobs[0]["id"])
    print(f"{details['title']} @ {details['company']}")
    print(details["description"][:300], "...")
    print(get_job_details("made-up-id"))
    print(f"\nCV skills: {sorted(find_skills(read_cv()['cv']))}")
    for r in score_match()[:3]:
        print(f"{r['score']}  {r['title']} @ {r['company']}  matched={r['matched_skills']}  missing={r['missing_skills']}")
    print("\nJobs from Himalayas: https://himalayas.app")