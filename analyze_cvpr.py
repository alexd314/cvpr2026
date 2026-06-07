#!/usr/bin/env python3
"""Analyze the scraped CVPR papers.

Sub-commands (run `python analyze_cvpr.py <cmd>`):
    topics     keyword & method frequency
    cooccur    top co-occurring keyword pairs (which topics cluster together)
    authors    most prolific authors
    abstracts  corpus stats over abstracts (length, distinctive terms)
    clusters   k-means themes over the sentence embeddings  (needs npz cache)
    dupes      near-duplicate / very-similar paper pairs    (needs npz cache)

Examples:
    python analyze_cvpr.py topics --top 30
    python analyze_cvpr.py cooccur --top 20
    python analyze_cvpr.py clusters -k 25
    python analyze_cvpr.py dupes --threshold 0.9
"""

import argparse
import collections
import itertools
import json
import math
import re
import sys

DATA = "cvpr2026_papers.json"
EMB = "cvpr2026_embeddings.npz"


def load(path=DATA):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_topics(papers, args):
    tags = collections.Counter(k for p in papers for k in p["keywords"])
    print(f"{len(tags)} distinct keywords across {len(papers)} papers\n")
    for k, n in tags.most_common(args.top):
        bar = "█" * round(40 * n / tags.most_common(1)[0][1])
        print(f"{n:5d}  {bar} {k}")


def cmd_cooccur(papers, args):
    pairs = collections.Counter()
    for p in papers:
        for a, b in itertools.combinations(sorted(set(p["keywords"])), 2):
            pairs[(a, b)] += 1
    print(f"Top {args.top} co-occurring keyword pairs:\n")
    for (a, b), n in pairs.most_common(args.top):
        print(f"{n:4d}  {a}  +  {b}")


def cmd_authors(papers, args):
    auth = collections.Counter(a for p in papers for a in p["authors"])
    print(f"{len(auth)} distinct authors\n")
    for a, n in auth.most_common(args.top):
        print(f"{n:4d}  {a}")


_WORD = re.compile(r"[a-z][a-z-]{2,}")
_STOP = set("the and for with that this from are can our their have has was were "
            "based using use used into via per which while when where then than "
            "such these those they them its also more most very much many both "
            "each other some any all not but get got new novel propose proposed "
            "method methods approach approaches model models results show shows "
            "paper present existing different first second three two one".split())


def cmd_abstracts(papers, args):
    lens = [len(p.get("abstract", "").split()) for p in papers if p.get("abstract")]
    lens.sort()
    n = len(lens)
    print(f"{n} abstracts | words: min {lens[0]}, median {lens[n//2]}, "
          f"max {lens[-1]}, mean {sum(lens)/n:.0f}\n")
    df = collections.Counter()
    for p in papers:
        words = {w for w in _WORD.findall(p.get("abstract", "").lower()) if w not in _STOP}
        df.update(words)
    print(f"Top {args.top} content words by document frequency:\n")
    for w, c in df.most_common(args.top):
        print(f"{c:5d}  {w}")


def _load_emb():
    try:
        import numpy as np
    except ImportError:
        sys.exit("Needs numpy: pip install numpy")
    try:
        cache = np.load(EMB, allow_pickle=True)
    except FileNotFoundError:
        sys.exit(f"No embedding cache at {EMB}. Run: python build_embeddings.py")
    return np, cache["embeddings"]


def cmd_clusters(papers, args):
    np, emb = _load_emb()
    k = args.k
    # Lightweight k-means (cosine == dot product on normalized vectors).
    rng_idx = [i * len(emb) // k for i in range(k)]  # deterministic seeding
    cent = emb[rng_idx].copy()
    for _ in range(args.iters):
        sims = emb @ cent.T
        assign = sims.argmax(axis=1)
        for j in range(k):
            members = emb[assign == j]
            if len(members):
                v = members.mean(axis=0)
                cent[j] = v / (np.linalg.norm(v) or 1.0)
    # Label each cluster by its most common keywords.
    print(f"{k} clusters over {len(papers)} papers:\n")
    for j in range(k):
        members = [papers[i] for i in range(len(papers)) if assign[i] == j]
        if not members:
            continue
        kw = collections.Counter(t for m in members for t in m["keywords"])
        top = ", ".join(t for t, _ in kw.most_common(5))
        print(f"[{len(members):4d}] {top}")


def cmd_dupes(papers, args):
    np, emb = _load_emb()
    th = args.threshold
    sims = emb @ emb.T
    n = len(emb)
    iu = np.triu_indices(n, k=1)
    hi = np.where(sims[iu] >= th)[0]
    pairs = sorted(((sims[iu[0][x], iu[1][x]], iu[0][x], iu[1][x]) for x in hi),
                   reverse=True)
    print(f"{len(pairs)} paper pairs with cosine >= {th}:\n")
    for s, i, j in pairs[:args.top]:
        print(f"{s:.3f}")
        print(f"   {papers[i]['title']}")
        print(f"   {papers[j]['title']}\n")


def main():
    ap = argparse.ArgumentParser(description="Analyze scraped CVPR papers.")
    ap.add_argument("--data", default=DATA)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("topics"); p.add_argument("--top", type=int, default=25)
    p = sub.add_parser("cooccur"); p.add_argument("--top", type=int, default=20)
    p = sub.add_parser("authors"); p.add_argument("--top", type=int, default=20)
    p = sub.add_parser("abstracts"); p.add_argument("--top", type=int, default=30)
    p = sub.add_parser("clusters")
    p.add_argument("-k", type=int, default=20); p.add_argument("--iters", type=int, default=15)
    p = sub.add_parser("dupes")
    p.add_argument("--threshold", type=float, default=0.9); p.add_argument("--top", type=int, default=25)

    args = ap.parse_args()
    papers = load(args.data)
    {"topics": cmd_topics, "cooccur": cmd_cooccur, "authors": cmd_authors,
     "abstracts": cmd_abstracts, "clusters": cmd_clusters, "dupes": cmd_dupes}[args.cmd](papers, args)


if __name__ == "__main__":
    main()
