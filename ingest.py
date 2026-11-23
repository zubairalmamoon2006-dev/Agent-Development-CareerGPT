"""Ingest JSON chunk exports from the data team into the CareerGPT collections.

Usage:
  python ingest.py path/to/cg_career_booklets_minilm_v1.json --collection booklets
  python ingest.py ... --replace-dummy      # remove dummy=True placeholder rows first
  python ingest.py ... --re-embed           # force re-embedding with our EMBED_MODEL

By default the script REUSES the embeddings shipped in the JSON, but only after
verifying (cosine on 3 samples) that they were produced by our EMBED_MODEL —
otherwise it re-embeds automatically. That keeps query-time and index-time
vectors in the same space, which is the hard requirement in the README contract.
"""
from __future__ import annotations

import argparse
import json

from careergpt.config import COLLECTIONS, EMBED_MODEL
from careergpt import store


def load_records(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) else data
    for r in records:
        assert r.get("id") and r.get("document"), f"record missing id/document: {r.get('id')}"
    return records


def friendly_source(meta: dict) -> str:
    """Readable citation label shown as source=... in tool output."""
    st = meta.get("source_type")
    if st == "univ_intern_booklet":
        return f"University Internship Booklet {meta.get('edition', '')}".strip()
    if st == "grad_booklet":
        return meta.get("source_document", "GRAD Booklet")
    return meta.get("source_document") or meta.get("source") or "career_booklets"


def normalize(meta: dict) -> dict:
    """Chroma metadata must be non-null scalars; add the `source` field we cite."""
    out = {k: v for k, v in meta.items()
           if v is not None and isinstance(v, (str, int, float, bool))}
    out.pop("collection", None)          # added automatically by upsert_passages
    if not out.get("source"):
        out["source"] = friendly_source(meta)
    return out


def embeddings_match_model(records: list[dict], sample: int = 3) -> bool:
    """True if the stored embeddings were made with our EMBED_MODEL."""
    if not all(r.get("embedding") for r in records):
        return False
    ef = store._embedder()
    for r in records[:sample]:
        ours = ef([r["document"]])[0]
        theirs = r["embedding"]
        dot = sum(a * b for a, b in zip(ours, theirs))
        na = sum(a * a for a in ours) ** 0.5
        nb = sum(b * b for b in theirs) ** 0.5
        if na == 0 or nb == 0 or dot / (na * nb) < 0.98:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a data-team JSON export")
    ap.add_argument("path", help="JSON file with records[{id, document, metadata, embedding}]")
    ap.add_argument("--collection", required=True, choices=sorted(COLLECTIONS),
                    help="target logical collection")
    ap.add_argument("--replace-dummy", action="store_true",
                    help="delete dummy=True placeholder rows before ingesting")
    ap.add_argument("--re-embed", action="store_true",
                    help="ignore shipped embeddings, embed locally with EMBED_MODEL")
    args = ap.parse_args()

    records = load_records(args.path)
    print(f"loaded {len(records)} records from {args.path}")

    if args.replace_dummy:
        n = store.delete_where(args.collection, {"dummy": True})
        print(f"removed {n} dummy placeholder rows from '{args.collection}'")

    use_shipped = (not args.re_embed) and embeddings_match_model(records)
    if use_shipped:
        print(f"shipped embeddings verified as {EMBED_MODEL} — reusing them")
    else:
        print(f"re-embedding locally with {EMBED_MODEL}")
        records = [{k: v for k, v in r.items() if k != "embedding"} for r in records]

    passages = [{"id": r["id"], "text": r["document"], **normalize(r.get("metadata") or {})}
                for r in records]
    embeddings = [r["embedding"] for r in records] if use_shipped else None
    store.upsert_passages(args.collection, passages, embeddings=embeddings)

    col = store.get_collection(args.collection)
    print(f"done. '{args.collection}' now holds {col.count()} passages "
          f"(collection name: {COLLECTIONS[args.collection]})")


if __name__ == "__main__":
    main()
