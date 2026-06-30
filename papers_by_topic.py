#!/usr/bin/env python3
"""Assign CVPR papers to curated topic clusters by embedding similarity.

Loads human-readable cluster labels, embeds them with the same sentence
transformer used for papers, assigns each paper to the best-matching cluster
(cosine similarity by default), and writes papers grouped by topic.

    python papers_by_topic.py
    python papers_by_topic.py --model all-mpnet-base-v2
"""

import argparse
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

DATA = "cvpr2026_papers.json"
EMB = "cvpr2026_embeddings.npz"
LABELS = "cluster_labels.json"
OUT = "cvpr2026_papers_by_topic.json"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_cluster_labels(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [(int(k), v) for k, v in raw.items() if not str(k).startswith("_")]


def assign_by_euclidean(paper_emb, cluster_emb):
    """Return index of nearest cluster for each paper row."""
    # the embedding model is normalizing embeddings, thus this is the same as cosine similarity
    diff = paper_emb[:, np.newaxis, :] - cluster_emb[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2).argmin(axis=1)


def main():
    ap = argparse.ArgumentParser(description="Group papers by curated topic clusters.")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--emb", default=EMB)
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument(
        "--model",
        default=None,
        help="sentence-transformer model (default: from embedding cache)",
    )
    ap.add_argument("--batch-size", type=int, default=64)    
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        papers = json.load(f)

    clusters = sorted(load_cluster_labels(args.labels))
    cluster_names = [name for _, name in clusters]

    try:
        cache = np.load(args.emb, allow_pickle=True)
    except FileNotFoundError:
        sys.exit(f"No embedding cache at {args.emb}. Run: python build_embeddings.py")

    paper_emb = cache["embeddings"]
    if paper_emb.shape[0] != len(papers):
        sys.exit(
            f"Cache has {paper_emb.shape[0]} rows but {len(papers)} papers; "
            "rebuild with build_embeddings.py"
        )

    model_name = args.model or str(cache["model"])
    print(f"Loading model {model_name} ...")
    model = SentenceTransformer(model_name)

    print(f"Embedding {len(cluster_names)} cluster labels ...")
    cluster_emb = model.encode(
        cluster_names,
        batch_size=args.batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"Assigning {len(papers)} papers to topics ...")
    assign = assign_by_euclidean(paper_emb, cluster_emb)

    by_topic = {name: [] for name in cluster_names}
    for i, paper in enumerate(papers):
        by_topic[cluster_names[assign[i]]].append(paper)

    for topic in by_topic:
        by_topic[topic].sort(key=lambda p: p["title"].lower())

    out_dict = dict(sorted(by_topic.items(), key=lambda kv: kv[0].lower()))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out} ({len(out_dict)} topics, {len(papers)} papers)")
    for topic, members in sorted(out_dict.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(members):4d}  {topic}")


if __name__ == "__main__":
    main()
