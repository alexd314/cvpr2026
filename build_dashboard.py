#!/usr/bin/env python3
"""Build interactive Plotly charts for the CVPR 2026 "State of the Conference" blog post.

Reads the local corpus (cvpr2026_papers.json) and cached sentence embeddings
(cvpr2026_embeddings.npz) and emits 5 standalone interactive HTML charts, each
self-contained (Plotly loaded from CDN) and styled to match the blog. They are
meant to be embedded as full-width <iframe>s in posts/cvpr2026.html.

Charts:
    1. map        The Map of CVPR 2026   (2D projection of all papers, colored by cluster)
    2. topics     Topic landscape        (treemap of clusters, c-TF-IDF labels)
    3. buzzwords  Buzzword adoption       (paper counts per hot term)
    4. network    Keyword co-occurrence   (force-directed keyword graph)
    5. naming     Naming & linguistic trends (title patterns / length / first words / acronyms)

Usage:
    python3 build_dashboard.py --out ../genadiy.vasserman.github.io/posts/cvpr2026/charts
    python3 build_dashboard.py --out charts --k 24 --projection auto

Reuses loaders/tokenizers from analyze_cvpr.py and search_cvpr.py.
"""

import argparse
import collections
import itertools
import os
import re
import sys

# Reuse existing project utilities (paper loader, embedding loader, stopwords, tokenizer).
from analyze_cvpr import load, _load_emb, _STOP
from search_cvpr import tokenize


# --------------------------------------------------------------------------------------
# Shared blog-matching style
# --------------------------------------------------------------------------------------
BG = "#ffffff"
PLOT_BG = "#ffffff"
TEXT = "#20242b"
MUTED = "#6a7079"
RULE = "#e7e7e4"
ACCENT = "#0b5fb0"
SANS = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif')


def base_layout(**over):
    # A title passed as a plain string would otherwise clobber the styled title
    # dict and fall back to a default position that clips at the top. Pull it out
    # and re-wrap it with explicit left-align / top placement.
    title = over.pop("title", None)
    lay = dict(
        paper_bgcolor=BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=SANS, size=14, color=TEXT),
        margin=dict(l=60, r=30, t=70, b=50),
        hoverlabel=dict(font=dict(family=SANS, size=13), bgcolor="#ffffff",
                        bordercolor=RULE),
        colorway=None,
    )
    lay.update(over)
    if title is not None:
        lay["title"] = title if isinstance(title, dict) else dict(
            text=title, font=dict(size=16, color=TEXT),
            x=0.02, xanchor="left", y=0.97, yanchor="top",
        )
    return lay


# --------------------------------------------------------------------------------------
# Buzzword definitions (edit freely). Each entry: label -> list of regex patterns.
# Matched as document frequency over (title + abstract), case-insensitive.
# --------------------------------------------------------------------------------------
BUZZWORDS = {
    "Diffusion models": [r"diffusion"],
    "3D Gaussian Splatting": [r"gaussian splat", r"\b3dgs\b", r"\bgaussian splatting\b"],
    "Transformer": [r"transformer"],
    "Foundation model": [r"foundation model"],
    "Vision-language / VLM": [r"vision.?language", r"\bvlm\b", r"\bvlms\b"],
    "Large language model / LLM": [r"\bllm\b", r"\bllms\b", r"large language model"],
    "NeRF / radiance fields": [r"\bnerf\b", r"neural radiance", r"radiance field"],
    "CLIP": [r"\bclip\b"],
    "Mamba / state-space": [r"\bmamba\b", r"state.?space model", r"\bssm\b"],
    "Segment Anything / SAM": [r"\bsam\b", r"segment anything"],
    "Self-supervised": [r"self.?supervised"],
    "Multimodal": [r"multi.?modal"],
    "Generative / GenAI": [r"generative"],
    "Open-vocabulary": [r"open.?vocabulary", r"open.?set"],
    "Zero-shot": [r"zero.?shot"],
    "Few-shot": [r"few.?shot"],
    "Point cloud": [r"point cloud"],
    "Knowledge distillation": [r"distillation", r"distill"],
    "Test-time adaptation": [r"test.?time"],
    "Neural rendering": [r"neural render"],
    "Autonomous driving": [r"autonomous driv", r"self.?driving"],
    "Video understanding": [r"\bvideo\b"],
    "Reinforcement learning": [r"reinforcement learning", r"\brl\b"],
    "World model": [r"world model"],
}

# Year-over-year trend chart: prior CVPR main tracks (scraped by scrape_cvf_years.py)
# plus the 2026 main subset, compared on a curated set of the most telling buzzwords.
TREND_YEARS = [2024, 2025, 2026]
TREND_TERMS = ["Multimodal", "Vision-language / VLM", "Large language model / LLM",
               "Diffusion models", "Video understanding", "Reinforcement learning",
               "3D Gaussian Splatting", "NeRF / radiance fields"]

# A paper "released code" if its abstract points to a public repo / project page.
CODE_RE = re.compile(r"github\.com|gitlab\.com|huggingface\.co|//[\w.-]*\.github\.io"
                     r"|project page|code (?:is )?available|we release", re.I)


def has_code(p):
    return bool(CODE_RE.search(p.get("abstract", "") or ""))


def buzz_share(papers, label):
    rx = [re.compile(p) for p in BUZZWORDS[label]]
    blobs = [(p.get("title", "") + " " + p.get("abstract", "")).lower() for p in papers]
    return 100.0 * sum(1 for b in blobs if any(r.search(b) for r in rx)) / max(1, len(papers))


FIRST_WORD_INTEREST = [
    "Towards", "Rethinking", "Learning", "Understanding", "Exploring", "Beyond",
    "Revisiting", "Bridging", "Unified", "Efficient", "Generalizable", "Adaptive",
    "Self", "Deep", "Robust", "Scalable",
]


# --------------------------------------------------------------------------------------
# Computation helpers
# --------------------------------------------------------------------------------------
def cluster_papers(np, emb, k, seed=42):
    """KMeans over the (normalized) embeddings. Returns integer labels."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(emb)


def ctfidf_labels(papers, labels, k, n_terms=3, topk=18):
    """Label each cluster with the terms most *distinctive* to it.

    Each cluster becomes one "class document" (titles + keywords of its members)
    and TF-IDF runs across the k class-documents. But because this corpus is so
    concentrated, generic terms (diffusion, image, transformer, language) top
    almost every cluster. So we add a second pass: a term is penalized by how many
    clusters it shows up in (cross-cluster IDF), and we de-duplicate overlapping
    n-grams (keep "gaussian splatting", drop bare "gaussian"/"splatting"). The
    result is a label that says what makes each cluster *different*.
    """
    import math
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    docs = [[] for _ in range(k)]
    for p, lab in zip(papers, labels):
        toks = tokenize(p["title"]) + [t for kw in p["keywords"] for t in tokenize(kw)]
        docs[lab].extend(toks)
    class_docs = [" ".join(d) for d in docs]
    stop = list(ENGLISH_STOP_WORDS | _STOP)
    vec = TfidfVectorizer(stop_words=stop, token_pattern=r"[a-z][a-z0-9-]{2,}",
                          ngram_range=(1, 2), max_features=6000)
    mat = vec.fit_transform(class_docs)
    terms = vec.get_feature_names_out()

    def good_term(t):
        # Drop degenerate repeated-word bigrams ("diffusion diffusion") that the
        # vectorizer manufactures from concatenating repeated title/keyword tokens.
        parts = t.split()
        return not (len(parts) == 2 and parts[0] == parts[1])

    # Candidate terms per cluster, and how many clusters each term tops (cdf).
    cand = []
    cdf = collections.Counter()
    for j in range(k):
        row = mat[j].toarray().ravel()
        idx = [i for i in row.argsort()[::-1][:topk] if row[i] > 0 and good_term(terms[i])]
        cand.append([(terms[i], float(row[i])) for i in idx])
        for t, _ in cand[j]:
            cdf[t] += 1

    labels_out = []
    for j in range(k):
        # Re-rank: distinctive = weight * cross-cluster IDF.
        scored = sorted(((w * math.log((k + 1) / cdf[t]), t) for t, w in cand[j]),
                        reverse=True)
        picked = []
        for _, t in scored:
            toks = set(t.split())
            # Skip terms that overlap a token with one already chosen (de-dupe
            # diffusion / stable diffusion / diffusion transformer redundancy).
            if any(toks & set(p.split()) for p in picked):
                continue
            picked.append(t)
            if len(picked) >= n_terms:
                break
        if not picked and cand[j]:
            picked = [cand[j][0][0]]
        labels_out.append(picked)
    return labels_out


def project_2d(np, emb, method, cache_path, recompute):
    """Project embeddings to 2D. Fallback chain umap -> tsne -> pca. Cached to npz."""
    if not recompute and os.path.exists(cache_path):
        c = np.load(cache_path, allow_pickle=True)
        print(f"  reusing cached projection ({str(c['method'])}) from {cache_path}")
        return c["coords"], str(c["method"])

    used = None
    coords = None
    order = {"auto": ["umap", "tsne", "pca"], "umap": ["umap", "tsne", "pca"],
             "tsne": ["tsne", "pca"], "pca": ["pca"]}[method]
    for m in order:
        try:
            if m == "umap":
                import umap
                print("  projecting with UMAP ...")
                coords = umap.UMAP(n_neighbors=15, min_dist=0.12, metric="cosine",
                                   random_state=42).fit_transform(emb)
            elif m == "tsne":
                from sklearn.manifold import TSNE
                print("  projecting with t-SNE (this can take ~1 min) ...")
                coords = TSNE(n_components=2, metric="cosine", init="pca",
                              perplexity=30, random_state=42).fit_transform(emb)
            else:
                from sklearn.decomposition import PCA
                print("  projecting with PCA ...")
                coords = PCA(n_components=2, random_state=42).fit_transform(emb)
            used = m
            break
        except ImportError:
            print(f"  {m} unavailable, trying next ...")
            continue
    if coords is None:
        sys.exit("No projection backend available (need umap-learn or scikit-learn).")
    coords = np.asarray(coords, dtype="float32")
    np.savez(cache_path, coords=coords, method=used)
    return coords, used


# Generic / boilerplate / fragment words that would otherwise dominate the concept
# network. The document-frequency band in extract_concepts catches the most ubiquitous
# ones ("image", "video") automatically; this list removes mid-frequency academic
# filler ("novel", "efficient") and hyphenated-concept fragments ("multi", "shot",
# "modal") that read as nonsense on their own.
_GENERIC = set(
    "image images video videos model models method methods network networks feature "
    "features learning training task tasks approach framework data dataset datasets "
    "performance result results representation representations information module modules "
    "problem state art quality scale real world multiple single various input output "
    "function design architecture experiment experiments analysis study end baseline "
    "benchmark benchmarks setting sample samples content towards efficient robust "
    "generalizable adaptive unified deep aware diverse enabling fine level cross recent "
    "work novel guided driven free aided based via using high low large small general "
    "specific multi modal shot test sub self semi inter intra pre post coarse long short "
    "open closed wise net nets".split())


def extract_concepts(papers, min_df=45, max_df_frac=0.30, cap=120):
    """Derive a per-paper set of concept terms from paper titles.

    CVF papers carry no keyword tags, so we mine the vocabulary ourselves. Titles (not
    abstracts) are the source: they are concept-dense and free of the "code available
    at https://github.com/..." boilerplate that pollutes abstracts. We keep single
    words whose document frequency sits in a mid band — frequent enough to connect
    papers, but below max_df_frac of the corpus so ubiquitous words don't form a
    hairball hub. The resulting co-occurrence edges (e.g. gaussian–splatting,
    object–detection) express the multi-word concepts on their own, so we stay with
    unigrams and avoid redundant uni/bigram triangles. Returns (term_sets, freq):
    term_sets[i] is paper i's concept set, freq is each term's document frequency.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = ENGLISH_STOP_WORDS | _STOP | _GENERIC
    per_paper, df = [], collections.Counter()
    for p in papers:
        toks = {t for t in tokenize(p["title"])
                if len(t) >= 3 and not t.isdigit() and t not in stop}
        per_paper.append(toks)
        for t in toks:
            df[t] += 1

    max_df = int(max_df_frac * len(papers))
    vocab = {t for t, c in df.items() if min_df <= c <= max_df}
    vocab = set(sorted(vocab, key=lambda t: df[t], reverse=True)[:cap])

    term_sets = [s & vocab for s in per_paper]
    freq = collections.Counter()
    for s in term_sets:
        freq.update(s)
    return term_sets, freq


def cooccur_pairs(term_sets):
    pairs = collections.Counter()
    for terms in term_sets:
        for a, b in itertools.combinations(sorted(terms), 2):
            pairs[(a, b)] += 1
    return pairs


# --------------------------------------------------------------------------------------
# Chart builders
# --------------------------------------------------------------------------------------
def _map_custom(papers, idx):
    custom = []
    for i in idx:
        p = papers[i]
        auth = ", ".join(p["authors"][:3]) + (" et al." if len(p["authors"]) > 3 else "")
        track = p.get("track", "")
        title = p["title"] if len(p["title"]) <= 90 else p["title"][:87] + "…"
        custom.append([title, auth, track])
    return custom


def fig_map(go, px, papers, coords, labels, cluster_labels, k):
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    hover = ("<b>%{customdata[0]}</b><br>%{customdata[1]}"
             "<br><i>%{customdata[2]}</i><extra></extra>")
    fig = go.Figure()
    # Topic-cluster traces (the default view).
    n_topic = 0
    for j in range(k):
        idx = [i for i in range(len(papers)) if labels[i] == j]
        if not idx:
            continue
        terms = cluster_labels[j]
        name = ", ".join(terms[:2]) if terms else f"cluster {j}"
        fig.add_trace(go.Scattergl(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=name[:40],
            marker=dict(size=5, color=palette[j % len(palette)], opacity=0.78,
                        line=dict(width=0)),
            customdata=_map_custom(papers, idx), hovertemplate=hover,
        ))
        n_topic += 1

    # Track overlay (hidden until toggled): main vs Findings.
    track_colors = {"main": "#0b5fb0", "findings": "#e07b00"}
    n_track = 0
    for tname in ("main", "findings"):
        idx = [i for i in range(len(papers)) if papers[i].get("track") == tname]
        if not idx:
            continue
        fig.add_trace(go.Scattergl(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers",
            name=f"{tname.capitalize()} ({len(idx)})", visible=False,
            marker=dict(size=5, color=track_colors[tname],
                        opacity=0.5 if tname == "main" else 0.8, line=dict(width=0)),
            customdata=_map_custom(papers, idx), hovertemplate=hover,
        ))
        n_track += 1

    buttons = [
        dict(label="By topic", method="update",
             args=[{"visible": [True] * n_topic + [False] * n_track},
                   {"legend.title.text": "topic clusters"}]),
        dict(label="Main vs Findings", method="update",
             args=[{"visible": [False] * n_topic + [True] * n_track},
                   {"legend.title.text": "track"}]),
    ]
    fig.update_layout(base_layout(
        title="The Map of CVPR 2026",
        showlegend=True,
        legend=dict(font=dict(size=11), itemsizing="constant", title=dict(text="topic clusters")),
        height=720,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        hovermode="closest",
        updatemenus=[dict(type="buttons", direction="right", showactive=True,
                          x=1, xanchor="right", y=1.07, yanchor="bottom",
                          pad=dict(t=2, b=2, l=2, r=2), font=dict(size=11),
                          bgcolor="#ffffff", bordercolor=RULE, buttons=buttons)],
    ))
    return fig


def fig_topics(go, px, papers, labels, cluster_labels, k):
    sizes = collections.Counter(labels.tolist())
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    total = len(papers)
    rows = []
    for j in range(k):
        if sizes[j] == 0:
            continue
        terms = cluster_labels[j]
        name = ", ".join(terms[:2]) if terms else f"cluster {j}"
        rows.append((sizes[j], name, palette[j % len(palette)]))
    rows.sort()  # ascending -> largest bar ends up on top in a horizontal bar
    vals = [r[0] for r in rows]
    names = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    text = [f"{v}  ({100 * v / total:.0f}%)" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors),
        text=text, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x} papers<extra></extra>",
    ))
    fig.update_layout(base_layout(
        title=f"Topic clusters by paper count (k={k})",
        height=820, margin=dict(l=20, r=70, t=60, b=40),
        xaxis=dict(title="papers", gridcolor=RULE, zeroline=False),
        yaxis=dict(automargin=True, tickfont=dict(size=12)),
    ))
    return fig


def fig_buzzwords(go, papers):
    blobs = [(p.get("title", "") + " " + p.get("abstract", "")).lower() for p in papers]
    counts = {}
    for label, pats in BUZZWORDS.items():
        rx = [re.compile(pat) for pat in pats]
        counts[label] = sum(1 for b in blobs if any(r.search(b) for r in rx))
    items = sorted(counts.items(), key=lambda x: x[1])
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    total = len(papers)
    pct = [f"{v}  ({100*v/total:.0f}%)" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=ACCENT),
        text=pct, textposition="outside",
        hovertemplate="%{y}: %{x} papers<extra></extra>",
        cliponaxis=False,
    ))
    fig.update_layout(base_layout(
        title=f"Papers mentioning each idea (of {total})",
        height=720, margin=dict(l=20, r=70, t=60, b=40),
        xaxis=dict(title="papers", gridcolor=RULE, zeroline=False),
        yaxis=dict(automargin=True),
    ))
    return fig


def fig_trends(go, px, papers_by_year):
    """Line chart of buzzword document-frequency across years (one line per term)."""
    years = sorted(papers_by_year)
    # Explicit, distinct colors so lines never collide. The NeRF -> 3DGS crossover is
    # the story, so those two are the only red / blue lines and are drawn thicker.
    colors = {
        "Multimodal": "#7b2fbf", "Diffusion models": "#e377c2",
        "Video understanding": "#2ca02c", "Vision-language / VLM": "#ff7f0e",
        "Large language model / LLM": "#8c564b", "Reinforcement learning": "#111111",
        "3D Gaussian Splatting": ACCENT, "NeRF / radiance fields": "#c0392b",
    }
    fig = go.Figure()
    # Compute and order terms by their latest-year share so the legend reads top-down.
    series = {t: [buzz_share(papers_by_year[y], t) for y in years] for t in TREND_TERMS}
    order = sorted(TREND_TERMS, key=lambda t: series[t][-1], reverse=True)
    for t in order:
        vals = series[t]
        highlight = t in ("3D Gaussian Splatting", "NeRF / radiance fields")
        color = colors.get(t, MUTED)
        width = 3.5 if highlight else 2
        label = t.split(" / ")[0]
        fig.add_trace(go.Scatter(
            x=years, y=vals, mode="lines+markers", name=label,
            line=dict(color=color, width=width), marker=dict(size=7),
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(base_layout(
        title="What grew and what faded, 2024 → 2026 (main track)",
        height=620, showlegend=True,
        legend=dict(font=dict(size=11), itemsizing="constant"),
        xaxis=dict(title="", tickmode="array", tickvals=years, gridcolor=RULE, zeroline=False),
        yaxis=dict(title="% of papers mentioning", gridcolor=RULE, zeroline=False,
                   ticksuffix="%"),
        hovermode="x unified",
    ))
    return fig


def fig_code_by_cluster(go, px, papers, labels, cluster_labels, k):
    """Horizontal bar of the code-release rate within each topic cluster."""
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    rows = []
    overall = 100.0 * sum(has_code(p) for p in papers) / len(papers)
    for j in range(k):
        idx = [i for i in range(len(papers)) if labels[i] == j]
        if not idx:
            continue
        rate = 100.0 * sum(has_code(papers[i]) for i in idx) / len(idx)
        terms = cluster_labels[j]
        name = ", ".join(terms[:2]) if terms else f"cluster {j}"
        rows.append((rate, name, len(idx), palette[j % len(palette)]))
    rows.sort()
    fig = go.Figure(go.Bar(
        x=[r[0] for r in rows], y=[r[1] for r in rows], orientation="h",
        marker=dict(color=[r[3] for r in rows]),
        text=[f"{r[0]:.0f}%" for r in rows], textposition="outside", cliponaxis=False,
        customdata=[[r[2]] for r in rows],
        hovertemplate="%{y}<br>%{x:.0f}% of %{customdata[0]} papers release code<extra></extra>",
    ))
    fig.add_vline(x=overall, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text=f"overall {overall:.0f}%", annotation_position="top")
    fig.update_layout(base_layout(
        title="Open-source rate by topic cluster",
        height=820, margin=dict(l=20, r=70, t=60, b=40),
        xaxis=dict(title="% of papers linking code / a project page", gridcolor=RULE,
                   zeroline=False, ticksuffix="%"),
        yaxis=dict(automargin=True, tickfont=dict(size=12)),
    ))
    return fig


def fig_network(go, np, term_sets, freq, top_edges=150, seed=42):
    import networkx as nx
    pairs = cooccur_pairs(term_sets)
    top = pairs.most_common(top_edges)
    G = nx.Graph()
    for (a, b), w in top:
        G.add_edge(a, b, weight=w)
    if G.number_of_nodes() == 0:
        return None
    # Drop tiny stranded islands (e.g. a 2-node {VLA, Robotics} pair) that only
    # connect to each other — spring_layout flings them into a far corner and
    # they read as a glitch. Keep components with at least 3 nodes.
    keep = set()
    for comp in nx.connected_components(G):
        if len(comp) >= 3:
            keep |= comp
    G = G.subgraph(keep).copy()
    # Communities for coloring.
    try:
        comms = list(nx.community.greedy_modularity_communities(G))
    except Exception:
        comms = [set(G.nodes())]
    node_comm = {}
    for ci, com in enumerate(comms):
        for nnode in com:
            node_comm[nnode] = ci
    pos = nx.spring_layout(G, seed=seed, k=0.55, iterations=80, weight="weight")

    # Edge trace.
    ex, ey = [], []
    for a, b in G.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines",
                            line=dict(width=0.6, color="#d7d9d4"),
                            hoverinfo="none", showlegend=False)

    import plotly.express as px
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    nx_, ny_, sizes, colors, texts = [], [], [], [], []
    for node in G.nodes():
        nx_.append(pos[node][0]); ny_.append(pos[node][1])
        f = freq.get(node, 1)
        sizes.append(10 + 3.2 * (f ** 0.5))
        colors.append(palette[node_comm.get(node, 0) % len(palette)])
        texts.append(f"{node}<br>{f} papers · degree {G.degree(node)}")
    node_trace = go.Scatter(
        x=nx_, y=ny_, mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=9, color=MUTED),
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="#ffffff")),
        hovertext=texts, hoverinfo="text", showlegend=False,
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(base_layout(
        title=f"Keyword co-occurrence (top {top_edges} pairs)",
        height=720, margin=dict(l=10, r=10, t=60, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        hovermode="closest",
    ))
    return fig


def fig_naming(go, make_subplots, papers):
    titles = [p["title"] for p in papers]
    total = len(titles)
    # Title patterns.
    all_you_need = sum(1 for t in titles if re.search(r"all you need", t, re.I))
    questions = sum(1 for t in titles if t.strip().endswith("?"))
    colon = sum(1 for t in titles if ":" in t)
    via = sum(1 for t in titles if re.search(r"\bvia\b", t, re.I))
    toward = sum(1 for t in titles if re.match(r"\s*towards?\b", t, re.I))
    pat_names = ["has a colon", "uses 'via'", "'Towards…'", "is a question (?)",
                 "'…all you need'"]
    pat_vals = [colon, via, toward, questions, all_you_need]

    # First-word trends.
    first = collections.Counter()
    for t in titles:
        m = re.match(r"\s*([A-Za-z][A-Za-z0-9-]*)", t)
        if m:
            first[m.group(1).capitalize()] += 1
    fw = [(w, first.get(w, 0)) for w in FIRST_WORD_INTEREST]
    fw = sorted([x for x in fw if x[1] > 0], key=lambda x: x[1])
    fw_names = [w for w, _ in fw]
    fw_vals = [v for _, v in fw]

    # Title length distribution (word count).
    lengths = [len(tokenize(t)) for t in titles]

    # Recurring named-method acronyms (>=2 uppercase letters).
    acro = collections.Counter()
    acro_rx = re.compile(r"\b[A-Za-z0-9]*[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")
    for t in titles:
        for tok in acro_rx.findall(t):
            if 2 <= len(tok) <= 12:
                acro[tok] += 1
    top_acro = [a for a in acro.most_common(40)]
    top_acro = sorted(top_acro, key=lambda x: x[1])[-14:]
    ac_names = [a for a, _ in top_acro]
    ac_vals = [v for _, v in top_acro]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Title patterns (of %d)" % total, "Title length (words)",
                        "How titles open", "Most-reused method names"),
        vertical_spacing=0.16, horizontal_spacing=0.16,
    )
    fig.add_trace(go.Bar(x=pat_vals, y=pat_names, orientation="h",
                         marker_color=ACCENT, showlegend=False,
                         hovertemplate="%{y}: %{x}<extra></extra>"), 1, 1)
    fig.add_trace(go.Histogram(x=lengths, marker_color=ACCENT, showlegend=False,
                               nbinsx=20, hovertemplate="%{x} words: %{y}<extra></extra>"), 1, 2)
    fig.add_trace(go.Bar(x=fw_vals, y=fw_names, orientation="h",
                         marker_color="#3a7ec0", showlegend=False,
                         hovertemplate="%{y}: %{x}<extra></extra>"), 2, 1)
    fig.add_trace(go.Bar(x=ac_vals, y=ac_names, orientation="h",
                         marker_color="#3a7ec0", showlegend=False,
                         hovertemplate="%{y}: %{x}<extra></extra>"), 2, 2)
    fig.update_layout(base_layout(
        title="How CVPR 2026 names things",
        height=780, margin=dict(l=60, r=30, t=80, b=40),
        bargap=0.18,
    ))
    fig.update_xaxes(gridcolor=RULE, zeroline=False)
    fig.update_yaxes(automargin=True)
    return fig


# --------------------------------------------------------------------------------------
# HTML writing
# --------------------------------------------------------------------------------------
def write_fig(fig, out_dir, name, first):
    """Write a figure as a standalone responsive HTML (Plotly via CDN)."""
    import plotly.io as pio
    path = os.path.join(out_dir, name + ".html")
    # Let the chart size itself to whatever the iframe is, instead of a fixed
    # pixel box, so it always fits the page cleanly.
    fig.update_layout(autosize=True, height=None, width=None)
    html = pio.to_html(
        fig, include_plotlyjs="cdn", full_html=True,
        config={"responsive": True, "displayModeBar": False},
        default_width="100%", default_height="100%",
    )
    # Make the chart fill the full iframe (width AND height), clean background.
    html = html.replace(
        "<head>",
        "<head>\n<style>html,body{margin:0;padding:0;height:100%;background:#fff;}"
        ".plotly-graph-div{width:100%!important;height:100%!important;}</style>",
        1,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  wrote {path}")
    return path


def write_png(fig, out_dir, name, width, height, scale=2):
    """Write a figure as a static high-res PNG (for Medium / slides). Needs kaleido."""
    import copy
    path = os.path.join(out_dir, name + ".png")
    f2 = copy.deepcopy(fig)  # don't disturb the figure used for HTML export
    f2.update_layout(autosize=False, width=width, height=height, paper_bgcolor="#ffffff")
    f2.write_image(path, width=width, height=height, scale=scale)
    print(f"  wrote {path}")
    return path


# Per-chart PNG canvas sizes (width, height) tuned for each chart's content.
PNG_SIZES = {
    "01_map": (1280, 1000),
    "02_topics": (1100, 920),
    "03_buzzwords": (1100, 840),
    "04_network": (1180, 980),
    "05_naming": (1180, 880),
    "06_trends": (1100, 760),
    "07_code": (1100, 920),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="charts", help="output directory for chart HTMLs")
    ap.add_argument("--k", type=int, default=24, help="number of topic clusters")
    ap.add_argument("--projection", choices=["auto", "umap", "tsne", "pca"],
                    default="auto", help="2D projection backend for the map")
    ap.add_argument("--no-recompute", action="store_true",
                    help="reuse the cached 2D projection if present")
    ap.add_argument("--data", default="cvpr2026_papers.json")
    ap.add_argument("--proj-cache", default="cvpr2026_projection.npz")
    ap.add_argument("--images", default=None,
                    help="also export static high-res PNGs to this dir (needs kaleido)")
    args = ap.parse_args()

    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading papers from {args.data} ...")
    papers = load(args.data)
    np, emb = _load_emb()
    assert emb.shape[0] == len(papers), f"{emb.shape[0]} embeddings vs {len(papers)} papers"
    print(f"  {len(papers)} papers, embeddings {emb.shape}")

    print(f"Clustering into k={args.k} ...")
    labels = cluster_papers(np, emb, args.k)
    cluster_labels = ctfidf_labels(papers, labels, args.k)
    # Prefer curated human/LLM-authored labels when present (cluster_labels.json maps
    # cluster id -> label). Falls back to the c-TF-IDF labels for any unmapped cluster.
    if os.path.exists("cluster_labels.json"):
        import json
        with open("cluster_labels.json", encoding="utf-8") as f:
            overrides = json.load(f)
        cluster_labels = [[overrides.get(str(j))] if overrides.get(str(j))
                          else cluster_labels[j] for j in range(args.k)]
        print(f"  applied {sum(1 for j in range(args.k) if str(j) in overrides)} curated labels")

    print("Projecting to 2D ...")
    coords, used = project_2d(np, emb, args.projection, args.proj_cache,
                              recompute=not args.no_recompute)
    print(f"  projection: {used}")

    # Prior-year main tracks for the YoY trend chart (optional; scrape_cvf_years.py).
    papers_by_year = {}
    for y in TREND_YEARS:
        if y == 2026:
            papers_by_year[y] = [p for p in papers if p.get("track") == "main"]
        elif os.path.exists(f"cvpr{y}_main_papers.json"):
            papers_by_year[y] = load(f"cvpr{y}_main_papers.json")
    have_trends = len(papers_by_year) >= 2
    if not have_trends:
        print("  (no prior-year files; skipping trends chart)")

    print("Building charts ...")
    if args.images:
        os.makedirs(args.images, exist_ok=True)
    figs = [
        ("01_map", fig_map(go, px, papers, coords, labels, cluster_labels, args.k)),
        ("02_topics", fig_topics(go, px, papers, labels, cluster_labels, args.k)),
        ("03_buzzwords", fig_buzzwords(go, papers)),
        ("04_network", fig_network(go, np, *extract_concepts(papers))),
        ("05_naming", fig_naming(go, make_subplots, papers)),
        ("06_trends", fig_trends(go, px, papers_by_year) if have_trends else None),
        ("07_code", fig_code_by_cluster(go, px, papers, labels, cluster_labels, args.k)),
    ]
    for i, (name, fig) in enumerate(figs):
        if fig is None:
            continue
        if args.images:  # export PNG first, before write_fig strips the fixed size
            w, h = PNG_SIZES.get(name, (1100, 850))
            write_png(fig, args.images, name, w, h)
        write_fig(fig, args.out, name, first=(i == 0))

    print(f"\nDone. {len(papers)} papers · {args.k} clusters · projection={used}")
    print(f"Charts written to {os.path.abspath(args.out)}")
    if args.images:
        print(f"PNGs written to {os.path.abspath(args.images)}")


if __name__ == "__main__":
    main()
