"""Phase 2 tests: prompt switching, grounding helpers, log redaction, session isolation,
CLI resume parsing. All offline — no API key, no network."""
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from careergpt.middleware import (answer_cites, extract_sources, pick_prompt,
                                  truncate_resume, write_tool_log)
from careergpt.prompts import BASE_PROMPT, RESUME_AWARE_PROMPT

TOOL = ToolMessage("[1] source=placement_blog_2025 similarity=0.87\nFinCo process details.",
                   tool_call_id="c1")
HUMAN = HumanMessage("question about FinCo")


# --------------------------------------------------------------------------- #
# 2.1 / 2.3: dynamic prompt selection
# --------------------------------------------------------------------------- #
def test_prompt_is_base_before_resume_loaded():
    assert pick_prompt({}) == BASE_PROMPT
    assert pick_prompt({"resume_loaded": False, "uploaded_resume_text": "SECRET"}) == BASE_PROMPT


def test_prompt_is_resume_aware_after_load():
    out = pick_prompt({"resume_loaded": True, "uploaded_resume_text": "I built a chatbot."})
    assert out.startswith(BASE_PROMPT)
    assert RESUME_AWARE_PROMPT.split("{resume}")[0] in out
    assert "I built a chatbot." in out


def test_prompt_does_not_contain_resume_before_load():
    out = pick_prompt({"uploaded_resume_text": "SECRET RESUME TEXT"})
    assert "SECRET" not in out  # stored but not visible until load_user_resume flips the flag


def test_long_resume_is_truncated():
    out = truncate_resume("x" * 20000, limit=1000)
    assert len(out) < 1200 and out.endswith("[resume truncated]")


# --------------------------------------------------------------------------- #
# 2.2: grounding helpers
# --------------------------------------------------------------------------- #
def test_extract_sources_from_tool_messages():
    assert extract_sources([HUMAN, TOOL]) == {"placement_blog_2025"}
    assert extract_sources([HUMAN]) == set()


def test_answer_cites_source_case_insensitive():
    srcs = {"placement_blog_2025"}
    assert answer_cites("FinCo process (source=placement_blog_2025)", srcs)
    assert answer_cites("see PLACEMENT_BLOG_2025 for details", srcs)
    assert not answer_cites("FinCo process — trust me.", srcs)


# --------------------------------------------------------------------------- #
# 2.3: log redaction (PII) — monkeypatched log path, no shared-state surprises
# --------------------------------------------------------------------------- #
def test_tool_log_redacts_long_args_and_sets_event(tmp_path, monkeypatch):
    from careergpt import middleware as mw
    log = tmp_path / "log.jsonl"
    monkeypatch.setattr(mw, "TOOL_LOG_PATH", str(log))

    mw.write_tool_log({"name": "load_user_resume",
                       "args": {"resume_text": "A" * 5000}}, "success", None, 0.123)
    mw.write_tool_log({"name": "search_placement_blogs",
                       "args": {"query": "analytics internships"}}, "success", None, 0.05)

    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event"] == "tool_call" and lines[0]["status"] == "success"
    assert lines[0]["args"]["resume_text"] == "<5000 chars>"
    assert "AAAA" not in log.read_text(encoding="utf-8")      # raw resume never logged
    assert lines[1]["args"]["query"] == "analytics internships"  # short args untouched
    assert {"ts", "event", "tool", "args", "status", "error", "latency_s"} <= set(lines[0])


# --------------------------------------------------------------------------- #
# 2.3: session isolation — a resume in session A must be invisible in session B
# --------------------------------------------------------------------------- #
def test_session_isolation():
    from eval.run_eval import build_fake_agent
    agent, _ = build_fake_agent()
    cfg_a = {"configurable": {"thread_id": "sess-A"}}
    cfg_b = {"configurable": {"thread_id": "sess-B"}}

    agent.invoke({"messages": [{"role": "user", "content": "hello"}],
                  "uploaded_resume_text": "SECRET RESUME"}, cfg_a)

    state_a = agent.get_state(cfg_a).values
    state_b = agent.get_state(cfg_b).values
    assert state_a.get("uploaded_resume_text") == "SECRET RESUME"  # kept for its own session
    assert "uploaded_resume_text" not in state_b                   # never leaks across threads
    assert "SECRET" not in json.dumps(state_b, default=str)


# --------------------------------------------------------------------------- #
# 2.4: CLI resume parsing
# --------------------------------------------------------------------------- #
def test_parse_resume_arg_same_line():
    import run_cli
    path, followup = run_cli.parse_resume_arg(
        r"/resume E:\Downloads\Resume (15).pdf give me feedback")
    assert path == r"E:\Downloads\Resume (15).pdf"
    assert followup == "give me feedback"


def test_parse_resume_arg_path_only():
    import run_cli
    path, followup = run_cli.parse_resume_arg(r"/resume C:\resumes\cv.pdf")
    assert path == r"C:\resumes\cv.pdf" and followup == ""


def test_resolve_path_fuzzy_and_missing(tmp_path):
    import run_cli
    (tmp_path / "Resume (1).pdf").write_bytes(b"%PDF")
    hit, corrected = run_cli.resolve_path(str(tmp_path / "Resume(1).pdf"))
    assert hit.endswith("Resume (1).pdf") and corrected is True
    miss, _ = run_cli.resolve_path(str(tmp_path / "definitely_missing.pdf"))
    assert miss is None
