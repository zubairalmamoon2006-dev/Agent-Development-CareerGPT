"""Offline tests for the evaluation harness — no LLM, no API key, no network."""
import json
from pathlib import Path

from eval.run_eval import KNOWN_TOOLS, load_questions, score_answer, run_one, build_fake_agent

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def test_questions_file_is_valid():
    questions = load_questions(EVAL_DIR / "questions.jsonl")
    assert len(questions) >= 30
    assert all(set(q["expected_tools"]) <= KNOWN_TOOLS for q in questions)


def test_questions_cover_all_categories():
    cats = {q["category"] for q in load_questions(EVAL_DIR / "questions.jsonl")}
    assert cats >= {"company_role", "timeline_process", "preparation", "project_por",
                    "policy", "multi_part", "negative", "resume_gate"}


def test_multi_part_questions_need_two_tools():
    qs = [q for q in load_questions(EVAL_DIR / "questions.jsonl") if q["category"] == "multi_part"]
    assert qs and all(len(q["expected_tools"]) >= 2 for q in qs)


def test_gold_file_ids_match_seed_conventions():
    for line in (EVAL_DIR / "gold_passages.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        assert g["collection"] in {"blogs", "booklets", "resumes", "policies"}
        assert g["gold_ids"]


def test_score_pass_and_fail():
    q = {"expected_tools": ["search_placement_blogs"], "must_mention": ["ExampleCorp"],
         "forbidden_tools": [], "must_not_mention": []}
    ok = score_answer(q, ["search_placement_blogs"], "ExampleCorp offers a data analyst role.")
    assert ok["passed"]
    bad_tool = score_answer(q, ["search_career_booklets"], "ExampleCorp ...")
    assert not bad_tool["passed"] and bad_tool["missing_tools"] == ["search_placement_blogs"]
    bad_term = score_answer(q, ["search_placement_blogs"], "Some unnamed company.")
    assert not bad_term["passed"] and bad_term["missing_terms"] == ["ExampleCorp"]


def test_score_negative_question():
    q = {"expected_tools": [], "forbidden_tools": ["search_placement_blogs"], "must_mention": [],
         "must_not_mention": ["ExampleCorp"]}
    clean = score_answer(q, [], "I cannot help with that.")
    assert clean["passed"]
    fabricated = score_answer(q, ["search_placement_blogs"], "ExampleCorp is coming.")
    assert not fabricated["passed"]


def test_fake_agent_end_to_end():
    agent, model_name = build_fake_agent()
    assert model_name == "fake-scripted"
    q = {"id": "smoke", "query": "anything", "expected_tools": ["search_placement_blogs"],
         "must_mention": ["ExampleCorp"]}
    tools_called, answer = run_one(agent, q)
    assert tools_called == ["search_placement_blogs"]
    assert "ExampleCorp" in answer
