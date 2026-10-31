"""Evaluation harness: run eval/questions.jsonl against the agent and score it.

Usage:
  python eval/run_eval.py                  # full run with Gemini (needs GOOGLE_API_KEY)
  python eval/run_eval.py --filter multi_part
  python eval/run_eval.py --limit 5
  python eval/run_eval.py --dry-run        # validate the question file only, no LLM
  python eval/run_eval.py --fake           # offline plumbing check with a scripted model

Each question gets a FRESH thread_id, so no conversation state leaks between questions.
Scoring per question:
  tools_ok    = every expected tool was called AND no forbidden tool was called
  terms_ok    = every must_mention substring appears in the answer (case-insensitive)
                AND no must_not_mention substring appears
  passed      = tools_ok AND terms_ok
Results: printed table + eval/results/<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow `python eval/run_eval.py` from anywhere

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

KNOWN_TOOLS = {"search_placement_blogs", "search_career_booklets", "search_resume_patterns",
               "search_institute_policies", "load_user_resume"}
REQUIRED_KEYS = {"id", "category", "query", "expected_tools"}


# --------------------------------------------------------------------------- #
# loading / validation
# --------------------------------------------------------------------------- #
def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    questions = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path.name}:{lineno}: invalid JSON ({e})")
        missing = REQUIRED_KEYS - q.keys()
        if missing:
            raise SystemExit(f"{path.name}:{lineno} ({q.get('id', '?')}): missing {sorted(missing)}")
        bad = (set(q.get("expected_tools", [])) | set(q.get("forbidden_tools", []))) - KNOWN_TOOLS
        if bad:
            raise SystemExit(f"{path.name}:{lineno} ({q['id']}): unknown tools {sorted(bad)}")
        questions.append(q)
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path.name}: duplicate question ids")
    return questions


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_answer(q: dict, tools_called: list[str], answer: str) -> dict:
    called = set(tools_called)
    required = set(q.get("expected_tools", []))
    forbidden = set(q.get("forbidden_tools", []))

    missing_tools = sorted(required - called)
    forbidden_called = sorted(forbidden & called)
    tools_ok = not missing_tools and not forbidden_called

    answer_l = answer.lower()
    missing_terms = [t for t in q.get("must_mention", []) if t.lower() not in answer_l]
    banned_terms = [t for t in q.get("must_not_mention", []) if t.lower() in answer_l]
    terms_ok = not missing_terms and not banned_terms

    return {
        "tools_ok": tools_ok,
        "missing_tools": missing_tools,
        "forbidden_called": forbidden_called,
        "terms_ok": terms_ok,
        "missing_terms": missing_terms,
        "banned_terms": banned_terms,
        "passed": tools_ok and terms_ok,
    }


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def _retry_delay(e: Exception) -> float | None:
    """Seconds to wait from a 429's RetryInfo, or None if this isn't a rate limit."""
    text = str(e)
    if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
        return None
    m = re.search(r"retry in ([\d.]+)s", text) or re.search(r"'retryDelay': '(\d+)s'", text)
    return min(float(m.group(1)) + 3.0, 70.0) if m else 30.0


def run_one(agent, q: dict, max_retries: int = 3) -> tuple[list[str], str]:
    """Send one question on a fresh thread; return (tools called in order, final answer).
    Retries on free-tier rate limits (429), waiting the time the API asks for."""
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    payload = {"messages": [{"role": "user", "content": q["query"]}]}
    tools_called: list[str] = []
    messages = []
    attempt = 0
    while True:
        try:
            tools_called = []
            for state in agent.stream(payload, cfg, stream_mode="values"):
                messages = state["messages"]
                # collect tool calls from all messages seen so far, in order
                seen = []
                for m in messages:
                    for tc in getattr(m, "tool_calls", None) or []:
                        seen.append(tc["name"])
                tools_called = seen
            break
        except Exception as e:
            wait = _retry_delay(e)
            attempt += 1
            if wait is None or attempt > max_retries:
                raise
            print(f"    (rate limited, waiting {wait:.0f}s, attempt {attempt}/{max_retries})")
            time.sleep(wait)
    answer = _text(messages[-1]) if messages else ""
    return tools_called, answer


def _text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)


def build_real_agent(model_name: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from careergpt.agent import build_agent
    return build_agent(ChatGoogleGenerativeAI(model=model_name, temperature=0)), model_name


def build_fake_agent():
    """Offline scripted model: call one search tool, then answer. Proves the harness
    plumbing (thread handling, tool capture, scoring) without any API key or cost."""
    from langchain_core.messages import AIMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from careergpt.agent import build_agent

    class ScriptedModel(BaseChatModel):
        step: int = 0

        @property
        def _llm_type(self) -> str:
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self.step == 0:
                self.step = 1
                msg = AIMessage(content="", tool_calls=[{
                    "name": "search_placement_blogs", "args": {"query": "eval"},
                    "id": "call_eval_1", "type": "tool_call"}])
            else:
                self.step = 0
                msg = AIMessage(content="ExampleCorp offers a Data Analyst Intern role (fake model).")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return build_agent(model=ScriptedModel()), "fake-scripted"


def main() -> None:
    ap = argparse.ArgumentParser(description="CareerGPT evaluation harness")
    ap.add_argument("--filter", help="run only one category")
    ap.add_argument("--limit", type=int, help="run at most N questions")
    ap.add_argument("--dry-run", action="store_true", help="validate questions.jsonl only")
    ap.add_argument("--fake", action="store_true", help="offline scripted model, no API key")
    ap.add_argument("--sleep", type=float, default=15.0,
                    help="seconds between questions (free tier: 5 req/min means ~15s pacing)")
    ap.add_argument("--model", default=None, help="Gemini model name (default: config.GEMINI_MODEL)")
    args = ap.parse_args()
    if args.model is None:
        from careergpt.config import GEMINI_MODEL
        args.model = GEMINI_MODEL

    questions = load_questions()
    if args.filter:
        questions = [q for q in questions if q["category"] == args.filter]
        if not questions:
            raise SystemExit(f"no questions with category '{args.filter}'")
    if args.limit:
        questions = questions[: args.limit]

    if args.dry_run:
        cats = {}
        for q in questions:
            cats[q["category"]] = cats.get(q["category"], 0) + 1
        print(f"OK: {len(questions)} questions valid")
        for c, n in sorted(cats.items()):
            print(f"  {c:16} {n}")
        return

    agent, model_name = build_fake_agent() if args.fake else build_real_agent(args.model)
    print(f"Running {len(questions)} questions against '{model_name}'...\n")

    records = []
    for i, q in enumerate(questions, 1):
        try:
            tools_called, answer = run_one(agent, q)
            s = score_answer(q, tools_called, answer)
        except Exception as e:  # e.g. free-tier 429 — record and keep going
            tools_called, answer, s = [], "", {
                "tools_ok": False, "missing_tools": sorted(set(q.get("expected_tools", []))),
                "forbidden_called": [], "terms_ok": False,
                "missing_terms": q.get("must_mention", []), "banned_terms": [],
                "passed": False, "error": f"{type(e).__name__}: {e}"}
        mark = "PASS" if s["passed"] else "FAIL"
        print(f"[{i:2}/{len(questions)}] {mark} {q['id']:16} tools={tools_called}")
        records.append({**q, "tools_called": tools_called, "answer": answer, **s})
        if args.sleep and not args.fake:
            time.sleep(args.sleep)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{ts}-{'fake' if args.fake else model_name.replace('/', '_')}.json"

    total = len(records)
    passed = sum(r["passed"] for r in records)
    by_cat = {}
    for r in records:
        c = by_cat.setdefault(r["category"], {"n": 0, "passed": 0})
        c["n"] += 1
        c["passed"] += r["passed"]
    tool_freq = {}
    for r in records:
        for t in set(r["tools_called"]):
            tool_freq[t] = tool_freq.get(t, 0) + 1

    summary = {"ts": ts, "model": model_name, "n": total, "passed": passed,
               "pass_rate": round(passed / total, 3) if total else 0.0,
               "by_category": by_cat, "tool_frequency": tool_freq, "questions": records}
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'category':16} {'passed':>8} {'n':>4}")
    for c, v in sorted(by_cat.items()):
        print(f"{c:16} {v['passed']:>8} {v['n']:>4}")
    print(f"\nOVERALL: {passed}/{total} = {summary['pass_rate']:.0%}")
    failures = [r for r in records if not r["passed"]]
    for r in failures:
        print(f"  FAIL {r['id']}: missing_tools={r['missing_tools']} "
              f"forbidden={r['forbidden_called']} missing_terms={r['missing_terms']} "
              f"banned={r['banned_terms']}")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
