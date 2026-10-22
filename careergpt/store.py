"""ChromaDB access layer. The agent only talks to this module, so swapping in the
real indexed data later needs no agent changes as long as the contract holds
(see README: collection names, embedding model, metadata fields)."""
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_DIR, COLLECTIONS, EMBED_MODEL, TOP_K


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


@lru_cache(maxsize=1)
def _embedder():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def get_collection(key: str):
    """key is one of COLLECTIONS: blogs | booklets | resumes | policies."""
    return _client().get_or_create_collection(
        name=COLLECTIONS[key],
        embedding_function=_embedder(),
        metadata={"hnsw:space": "cosine"},  # cosine similarity, as in the design doc
    )


def upsert_passages(key: str, passages: list[dict]) -> None:
    """Helper for the indexing team. Each passage:
    {"id": str, "text": str, "source": str, ...extra scalar metadata (year, company, domain)}
    """
    col = get_collection(key)
    col.upsert(
        ids=[p["id"] for p in passages],
        documents=[p["text"] for p in passages],
        metadatas=[
            {**{k: v for k, v in p.items() if k not in ("id", "text")}, "collection": key}
            for p in passages
        ],
    )


def search(key: str, query: str, k: int = TOP_K, where: dict | None = None) -> str:
    """Top-k cosine search in one collection, formatted for the LLM."""
    col = get_collection(key)
    if col.count() == 0:
        return f"No results: the '{key}' collection is empty (data not indexed yet)."
    res = col.query(query_texts=[query], n_results=min(k, col.count()), where=where)
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    if not docs:
        return f"No relevant passages found in '{key}'."
    blocks = []
    for i, (d, m, dist) in enumerate(zip(docs, metas, dists), 1):
        blocks.append(f"[{i}] source={m.get('source', 'unknown')} similarity={1 - dist:.2f}\n{d}")
    return "\n\n".join(blocks)
