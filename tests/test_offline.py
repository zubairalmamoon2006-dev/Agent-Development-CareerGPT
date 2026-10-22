"""No LLM / API key needed: checks the retrieval layer and middleware wiring."""
import os, tempfile
os.environ["CHROMA_DIR"] = tempfile.mkdtemp()
os.environ["TOOL_LOG_PATH"] = os.path.join(tempfile.mkdtemp(), "log.jsonl")

from careergpt.store import search
from careergpt.tools import search_placement_blogs, ALL_TOOLS


def test_empty_collection_message():
    assert "empty" in search("blogs", "anything").lower()


def test_search_after_seed():
    import seed_dummy_data as s
    from careergpt.store import upsert_passages
    for k, rows in s.DUMMY.items():
        upsert_passages(k, [{"id": f"d{k}{i}", "text": t, "source": src, "dummy": True}
                            for i, (t, src) in enumerate(rows)])
    out = search_placement_blogs.invoke({"query": "analytics internship"})
    assert "source=" in out and "similarity=" in out


def test_five_tools_registered():
    assert len(ALL_TOOLS) == 5
