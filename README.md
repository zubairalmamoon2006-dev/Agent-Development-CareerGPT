# CareerGPT — agent layer

ReAct agent (LangChain `create_agent` -> LangGraph loop) with 5 tools, 4 ChromaDB collections,
tool-call logging and resume-aware dynamic prompting. Works today on dummy data.

## Run
    pip install -r requirements.txt
    cp .env.example .env        # add GOOGLE_API_KEY
    python seed_dummy_data.py   # placeholder data (python seed_dummy_data.py --reset removes it)
    pytest tests                # offline checks, no API key needed
    python run_cli.py

## Data contract (for the data collection / indexing team)
1. Collections (`careergpt/config.py`): `careergpt_blogs`, `careergpt_booklets`,
   `careergpt_resumes`, `careergpt_policies`.
2. Same embedding model at index and query time (`EMBED_MODEL`, default `all-MiniLM-L6-v2`).
   If you change it, re-index everything.
3. Cosine space (`hnsw:space=cosine`); easiest is to index through
   `careergpt.store.upsert_passages(key, passages)`.
4. Each passage: `id` (unique), `text`, `source` (e.g. `placement_blog_2025`) plus optional
   scalar metadata (`year`, `company`, `domain`). `collection` is added automatically.
5. Resumes must be anonymized BEFORE indexing.

## Layout
- `careergpt/store.py` Chroma access, only file touching the DB
- `careergpt/tools.py` 5 tools (docstrings = routing)
- `careergpt/middleware.py` state, dynamic prompt, tool logging
- `careergpt/agent.py` agent assembly
