#!/usr/bin/env python3
"""Compute and cache sentence-embeddings for the scraped papers.

Embeds "title. abstract" for each paper with a small local model and saves the
matrix to a .npz file. Run once after scraping; search_cvpr.py --semantic then
reuses the cache (no recompute, no network).

    python build_embeddings.py                 # -> cvpr2026_embeddings.npz
    python build_embeddings.py --model all-mpnet-base-v2   # higher quality, slower
"""

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer

DATA = "cvpr2026_papers.json"
OUT = "cvpr2026_embeddings.npz"
# Small, fast, good general-purpose model (~90 MB). Swap for all-mpnet-base-v2
# (~420 MB) for higher quality at lower speed.
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def doc_text(p):
    # Title carries the most signal; abstract adds recall.
    return f"{p['title']}. {p.get('abstract', '')}".strip()


def main():
    ap = argparse.ArgumentParser(description="Build embedding cache for semantic search.")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model)
    texts = [doc_text(p) for p in papers]

    print(f"Embedding {len(texts)} papers ...")
    emb = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    emb = emb.astype(np.float32)

    np.savez(args.out, embeddings=emb, model=args.model,
             ids=np.array([p.get("arxiv_id") or p["title"] for p in papers]))
    print(f"Saved {emb.shape} embeddings to {args.out} (model: {args.model})")


if __name__ == "__main__":
    main()
