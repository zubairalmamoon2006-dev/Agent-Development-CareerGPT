"""Single source of truth for settings shared with the data/indexing teammates."""
import os
from dotenv import load_dotenv

load_dotenv()

# Logical key -> actual ChromaDB collection name. Indexing team MUST use these.
COLLECTIONS = {
    "blogs": "careergpt_blogs",
    "booklets": "careergpt_booklets",
    "resumes": "careergpt_resumes",
    "policies": "careergpt_policies",
}

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
# Must be the SAME model used at indexing time and query time.
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TOOL_LOG_PATH = os.getenv("TOOL_LOG_PATH", "./logs/tool_calls.jsonl")
