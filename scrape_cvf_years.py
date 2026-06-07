#!/usr/bin/env python3
"""Scrape prior-year CVPR main tracks from CVF for year-over-year comparison.

Reuses the fetch/parse/enrich pipeline from scrape_cvf.py. Findings is a 2026-only
track, so prior years are main-conference only. Writes cvpr<year>_main_papers.json.

    python scrape_cvf_years.py 2024 2025
"""

import sys
import scrape_cvf as s


def main():
    years = sys.argv[1:] or ["2024", "2025"]
    for y in years:
        url = f"{s.BASE}/CVPR{y}?day=all"
        recs = s.scrape_track("main", url)
        # tag with the year for downstream trend code
        for r in recs:
            r["year"] = int(y)
        s.write_outputs(recs, f"cvpr{y}_main_papers")
        print(f"Wrote cvpr{y}_main_papers.json/.csv ({len(recs)} papers)")


if __name__ == "__main__":
    main()
