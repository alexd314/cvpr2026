# The Map of CVPR 2026

Code behind **["The Map of CVPR 2026"](https://genawass.github.io/posts/cvpr2026.html)** — a
data-driven tour of all 5,010 accepted CVPR 2026 papers (4,069 main + 941 Findings).
It scrapes the papers from [CVF Open Access](https://openaccess.thecvf.com/), embeds them
with a sentence-transformer, clusters and projects them, and emits 7 interactive Plotly
charts.

## Charts

| # | Chart | What it shows |
|---|-------|---------------|
| 01 | Map | Every paper as a dot (UMAP of embeddings), colored by topic cluster; toggle main vs. Findings |
| 02 | Topics | The 24 clusters by paper count |
| 03 | Buzzwords | Share of papers mentioning each idea (diffusion, VLM, 3DGS, …) |
| 04 | Network | Co-occurrence of concept words in titles |
| 05 | Naming | Title linguistics — colon format, openers, reused acronyms |
| 06 | Trends | Year-over-year idea adoption across the 2024–2026 main tracks |
| 07 | Code | Open-source rate by topic cluster |

## Pipeline

```bash
pip install -r requirements.txt

# 1. Scrape the papers from CVF (main + Findings -> cvpr2026_papers.json/.csv)
python scrape_cvf.py

# 2. (optional) Prior years for the year-over-year trend chart
python scrape_cvf_years.py 2024 2025

# 3. Embed "title. abstract" for every paper (-> cvpr2026_embeddings.npz)
python build_embeddings.py

# 4. Build the 7 interactive charts (reuses a cached UMAP projection with --no-recompute)
python build_dashboard.py --out charts
#    add --images <dir> to also export static PNGs (needs kaleido)
```

`cluster_labels.json` holds the hand-written names for the 24 KMeans clusters (seed=42);
delete it to fall back to automatic c-TF-IDF labels, or regenerate it if you change the
embedding model or `k`.

## Other tools

- `search_cvpr.py` — keyword / TF-IDF / semantic search over the scraped corpus
- `analyze_cvpr.py` — quick corpus stats (topics, co-occurrence, authors, clusters, dupes)

## Notes

- Data is scraped fresh from CVF and is **not** committed (see `.gitignore`); run the
  pipeline above to regenerate it.
- Embeddings use `all-MiniLM-L6-v2` by default. Proximity on the map is semantic text
  similarity, not citations or impact.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
