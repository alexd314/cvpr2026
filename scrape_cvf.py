#!/usr/bin/env python3
"""Scrape the full CVPR 2026 papers (MAIN + FINDINGS) from CVF Open Access.

paperswithcode.co only indexes the ~2.6K papers that have arXiv/code links. The
authoritative source is openaccess.thecvf.com, reachable from /menu:

    Main Conference -> /CVPR2026?day=all            (4069 papers)
    Findings        -> /CVPR2026_findings?day=all   (941 papers)

The listing pages give title + paper-page URL + PDF URL + authors. Abstracts live
on each paper's individual page, so we fetch those concurrently.

Outputs (schema matches the old scrape_cvpr2026.py, plus a "track" field):
    cvpr2026_main_papers.json / .csv
    cvpr2026_findings_papers.json / .csv
    cvpr2026_papers.json / .csv     <- merged canonical (old one backed up to .pwc.bak)

Usage:
    python scrape_cvf.py
"""

import csv
import html as htmllib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://openaccess.thecvf.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

TRACKS = {
    "main": f"{BASE}/CVPR2026?day=all",
    "findings": f"{BASE}/CVPR2026_findings?day=all",
}

ABS_RE = re.compile(r'id="abstract">(.*?)</div>', re.S)
ARXIV_RE = re.compile(r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})', re.I)
FIELDS = ["title", "link", "keywords", "authors", "abstract",
          "arxiv_id", "url_abs", "url_pdf", "track"]


def fetch(url, retries=4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed {url}: {last}")


def parse_listing(html):
    papers = []
    for blk in re.split(r'<dt class="ptitle">', html)[1:]:
        m = re.search(r'<a href="([^"]+)">(.*?)</a>', blk, re.S)
        if not m:
            continue
        page_url = BASE + m.group(1)
        title = htmllib.unescape(re.sub(r"\s+", " ", m.group(2)).strip())
        authors = [htmllib.unescape(a) for a in
                   re.findall(r'name="query_author" value="([^"]+)"', blk)]
        pdfm = re.search(r'href="([^"]+\.pdf)"', blk)
        pdf_url = BASE + pdfm.group(1) if pdfm else None
        papers.append({"title": title, "page_url": page_url,
                       "pdf_url": pdf_url, "authors": authors})
    return papers


def enrich(p):
    try:
        page = fetch(p["page_url"])
    except Exception as e:
        p["abstract"], p["arxiv_id"], p["_err"] = "", None, str(e)
        return p
    am = ABS_RE.search(page)
    p["abstract"] = htmllib.unescape(re.sub(r"\s+", " ", am.group(1)).strip()) if am else ""
    xm = ARXIV_RE.search(page)
    p["arxiv_id"] = xm.group(1) if xm else None
    return p


def scrape_track(track, url):
    print(f"[{track}] fetching listing...", file=sys.stderr)
    papers = parse_listing(fetch(url))
    print(f"[{track}] parsed {len(papers)} papers; fetching abstracts...", file=sys.stderr)
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(enrich, p) for p in papers]
        for _ in as_completed(futs):
            done += 1
            if done % 300 == 0:
                print(f"[{track}]   {done}/{len(papers)}", file=sys.stderr)
    errs = sum(1 for p in papers if p.get("_err"))
    noabs = sum(1 for p in papers if not p.get("abstract"))
    print(f"[{track}] done. errors={errs} missing_abstract={noabs}", file=sys.stderr)
    return [{
        "title": p["title"], "keywords": [], "link": p["page_url"],
        "abstract": p.get("abstract", ""), "authors": p["authors"],
        "arxiv_id": p.get("arxiv_id"), "url_abs": p["page_url"],
        "url_pdf": p["pdf_url"], "track": track,
    } for p in papers]


def write_outputs(records, base):
    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(f"{base}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for p in records:
            w.writerow([p["title"], p["link"], "; ".join(p["keywords"]),
                        "; ".join(p["authors"]), p["abstract"], p["arxiv_id"],
                        p["url_abs"], p["url_pdf"], p["track"]])


def main():
    all_records = []
    for track, url in TRACKS.items():
        recs = scrape_track(track, url)
        write_outputs(recs, f"cvpr2026_{track}_papers")
        all_records.extend(recs)

    # De-dup by (title, track) preserving order (a paper can't be in both tracks).
    seen, merged = set(), []
    for p in all_records:
        key = (p["title"].lower(), p["track"])
        if key not in seen:
            seen.add(key)
            merged.append(p)

    # Back up the old paperswithcode-sourced canonical file once.
    if os.path.exists("cvpr2026_papers.json") and not os.path.exists("cvpr2026_papers.pwc.bak.json"):
        os.rename("cvpr2026_papers.json", "cvpr2026_papers.pwc.bak.json")
        if os.path.exists("cvpr2026_papers.csv"):
            os.rename("cvpr2026_papers.csv", "cvpr2026_papers.pwc.bak.csv")
        print("Backed up old paperswithcode dataset to *.pwc.bak.*", file=sys.stderr)

    write_outputs(merged, "cvpr2026_papers")
    n_main = sum(1 for p in merged if p["track"] == "main")
    n_find = sum(1 for p in merged if p["track"] == "findings")
    print(f"\nWrote cvpr2026_papers.json/.csv: {len(merged)} papers "
          f"({n_main} main + {n_find} findings)")


if __name__ == "__main__":
    main()
