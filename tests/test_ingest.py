"""Offline tests for ingest.py helpers — no model load, no data files needed."""
import json

from ingest import friendly_source, load_records, normalize


def test_normalize_drops_nulls_and_non_scalars():
    meta = {"year": 2025, "company": "FinCo", "note": None, "tags": ["a"], "score": 0.8}
    out = normalize(meta)
    assert out["year"] == 2025 and out["company"] == "FinCo" and out["score"] == 0.8
    assert "note" not in out and "tags" not in out
    assert out["source"]  # contract: every passage gets a citation label


def test_normalize_adds_source_for_booklet_records():
    uni = normalize({"source_type": "univ_intern_booklet", "edition": "2025-26",
                     "program_name": "OIST"})
    assert uni["source"] == "University Internship Booklet 2025-26"
    grad = normalize({"source_type": "grad_booklet",
                      "source_document": "GRAD_Booklet'26.pdf"})
    assert grad["source"] == "GRAD_Booklet'26.pdf"
    plain = normalize({"source_document": "placement_blog_2025.pdf"})
    assert plain["source"] == "placement_blog_2025.pdf"


def test_normalize_never_overwrites_existing_source():
    assert normalize({"source": "placement_blog_2025"})["source"] == "placement_blog_2025"


def test_normalize_strips_auto_collection_field():
    assert "collection" not in normalize({"collection": "booklets", "year": 2025})


def test_load_records_accepts_wrapped_and_bare_lists(tmp_path):
    rec = {"id": "x1", "document": "text", "metadata": {}}
    wrapped = tmp_path / "w.json"
    wrapped.write_text(json.dumps({"records": [rec]}), encoding="utf-8")
    bare = tmp_path / "b.json"
    bare.write_text(json.dumps([rec]), encoding="utf-8")
    assert load_records(str(wrapped)) == [rec]
    assert load_records(str(bare)) == [rec]


def test_load_records_rejects_missing_id(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"document": "no id"}]), encoding="utf-8")
    try:
        load_records(str(bad))
        raise AssertionError("expected AssertionError for missing id")
    except AssertionError as e:
        assert "missing id" in str(e)
