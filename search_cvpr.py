#!/usr/bin/env python3
"""Search the scraped CVPR papers by keyword tag and/or title text.

Examples:
    python search_cvpr.py "gaussian splatting"          # tag OR title contains
    python search_cvpr.py CLIP diffusion                # matches EITHER term (OR)
    python search_cvpr.py CLIP diffusion --all          # must match BOTH (AND)
    python search_cvpr.py --tags                        # list all keyword tags
    python search_cvpr.py video --field title           # title text only
    python search_cvpr.py SAM --field tag               # exact-ish tag only
    python search_cvpr.py diffusion --json > hits.json  # machine-readable

Matching is case-insensitive substring by default. Use --regex for patterns.
"""

import argparse
import collections
import json
import math
import re
import sys

DATA = "cvpr2026_papers.json"

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _WORD.findall(text.lower())


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_matcher(term, regex):
    if regex:
        pat = re.compile(term, re.IGNORECASE)
        return lambda s: bool(pat.search(s))
    t = term.lower()
    return lambda s: t in s.lower()


def paper_matches(paper, matcher, field):
    if field in ("tag", "both") and any(matcher(k) for k in paper["keywords"]):
        return True
    if field in ("title", "both") and matcher(paper["title"]):
        return True
    if field == "abstract" and matcher(paper.get("abstract", "")):
        return True
    return False


def semantic_search(papers, query, top_n, emb_path):
    """Cosine-similarity search over cached sentence embeddings.

    Imports numpy / sentence-transformers lazily so the rest of the tool stays
    dependency-free. Requires running build_embeddings.py first.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Semantic search needs: pip install sentence-transformers numpy")
    try:
        cache = np.load(emb_path, allow_pickle=True)
    except FileNotFoundError:
        sys.exit(f"No embedding cache at {emb_path}. Run: python build_embeddings.py")

    emb = cache["embeddings"]
    if emb.shape[0] != len(papers):
        sys.exit(f"Cache has {emb.shape[0]} rows but {len(papers)} papers; "
                 "rebuild with build_embeddings.py after re-scraping.")
    model = SentenceTransformer(str(cache["model"]))
    qv = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = emb @ qv  # both normalized -> cosine similarity
    order = scores.argsort()[::-1][:top_n] if top_n else scores.argsort()[::-1]
    return [(float(scores[i]), papers[i]) for i in order]


def paper_tokens(paper, title_boost=3, kw_boost=2):
    """Bag of tokens for a paper; title and keywords are repeated to weight them."""
    toks = []
    toks += tokenize(paper["title"]) * title_boost
    for k in paper["keywords"]:
        toks += tokenize(k) * kw_boost
    toks += tokenize(paper.get("abstract", ""))
    return toks


def rank(papers, query, top_n):
    """TF-IDF cosine ranking over title+keywords+abstract. Pure Python, no deps."""
    docs = [collections.Counter(paper_tokens(p)) for p in papers]
    n = len(docs)
    df = collections.Counter()
    for d in docs:
        df.update(d.keys())
    idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def vec(counter):
        v = {t: (1 + math.log(f)) * idf.get(t, 0.0) for t, f in counter.items()}
        norm = math.sqrt(sum(w * w for w in v.values())) or 1.0
        return v, norm

    q_counter = collections.Counter(tokenize(query))
    if not q_counter:
        return []
    qv, qn = vec(q_counter)

    scored = []
    for paper, d in zip(papers, docs):
        dv, dn = vec(d)
        # dot product over the (small) query vocabulary
        dot = sum(w * dv.get(t, 0.0) for t, w in qv.items())
        if dot:
            scored.append((dot / (qn * dn), paper))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_n] if top_n else scored


def main():
    ap = argparse.ArgumentParser(description="Search scraped CVPR papers.")
    ap.add_argument("terms", nargs="*", help="search term(s)")
    ap.add_argument("--all", action="store_true", help="require ALL terms (AND); default is ANY (OR)")
    ap.add_argument("--field", choices=["both", "title", "tag", "abstract"], default="both",
                    help="where to search (default: both = title+tag)")
    ap.add_argument("--regex", action="store_true", help="treat terms as regex patterns")
    ap.add_argument("--rank", action="store_true",
                    help="ranked full-text search (TF-IDF over title+keywords+abstract)")
    ap.add_argument("--semantic", action="store_true",
                    help="embedding-based semantic search (needs build_embeddings.py first)")
    ap.add_argument("--emb", default="cvpr2026_embeddings.npz", help="embedding cache path")
    ap.add_argument("--top", type=int, default=20, help="max ranked results to show (default: 20)")
    ap.add_argument("--tags", action="store_true", help="list all keyword tags with counts and exit")
    ap.add_argument("--json", action="store_true", help="output matches as JSON")
    ap.add_argument("--data", default=DATA, help=f"data file (default: {DATA})")
    args = ap.parse_args()

    papers = load(args.data)

    if args.tags:
        c = collections.Counter(k for p in papers for k in p["keywords"])
        for k, n in c.most_common():
            print(f"{n:5d}  {k}")
        print(f"\n{len(c)} distinct tags across {len(papers)} papers", file=sys.stderr)
        return

    if not args.terms:
        ap.error("provide search term(s), or use --tags to list keywords")

    if args.rank or args.semantic:
        query = " ".join(args.terms)
        if args.semantic:
            results = semantic_search(papers, query, args.top, args.emb)
            mode = "semantic similarity"
        else:
            results = rank(papers, query, args.top)
            mode = "TF-IDF relevance"
        if args.json:
            json.dump([{**p, "score": round(s, 4)} for s, p in results],
                      sys.stdout, ensure_ascii=False, indent=2)
            return
        for score, p in results:
            print(f"{score:.3f}  {p['title']}")
            print(f"        {p['link']}")
            print(f"        [{', '.join(p['keywords'])}]")
            print()
        print(f"top {len(results)} by {mode} for {query!r}", file=sys.stderr)
        return

    matchers = [make_matcher(t, args.regex) for t in args.terms]
    combine = all if args.all else any
    hits = [p for p in papers
            if combine(paper_matches(p, m, args.field) for m in matchers)]

    if args.json:
        json.dump(hits, sys.stdout, ensure_ascii=False, indent=2)
        return

    for p in hits:
        print(p["title"])
        print(f"  {p['link']}")
        print(f"  [{', '.join(p['keywords'])}]")
        print()
    mode = "ALL" if args.all else "ANY"
    print(f"{len(hits)} / {len(papers)} papers match {mode} of {args.terms} "
          f"in {args.field}", file=sys.stderr)


if __name__ == "__main__":
    main()
