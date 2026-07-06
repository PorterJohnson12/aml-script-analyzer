"""
Week-1 Join-Yield Probe  ·  Outlier (PRO322)   [parallel edition]
=================================================================

Answers the one number that decides feasibility:
*Of the screenplays I can get, how many join to clean financials AND reception?*
That usable N -- not the raw script count -- is the real dataset size.

For each film (+ optional year) it looks up TMDB, pulls budget / revenue /
vote_average / vote_count, flags usable financials (budget>0 AND revenue>0) and
usable reception (vote_count >= MIN_VOTES), computes ROI, writes a per-film CSV,
and prints a yield summary.

Speed: requests run in a thread pool (I/O-bound), with retry/backoff on HTTP 429.
Progress prints every 25 films instead of one line per film.

Usage
-----
  python join_yield_probe.py                      # smoke test: built-in case-study panel
  python join_yield_probe.py ..\\data\\titles.csv  # real run -> your usable N
  python join_yield_probe.py ..\\data\\titles.csv --workers 12
  python join_yield_probe.py ..\\data\\titles.csv --limit 50   # quick subset while testing

Requires
--------
  pip install requests python-dotenv
  .env with:  TMDB_API_KEY=your_key_here     (UTF-16/BOM from PowerShell is tolerated)
"""

import os
import sys
import csv
import time
import re
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _load_env():
    """Load .env resiliently -- tolerant of Windows/PowerShell UTF-16 or BOM encodings
    that plain python-dotenv chokes on (UnicodeDecodeError on byte 0xff)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    if os.getenv("TMDB_API_KEY"):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(os.getcwd(), ".env"), os.path.join(here, ".env")):
        if not os.path.exists(path):
            continue
        for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
            try:
                with open(path, encoding=enc) as fh:
                    text = fh.read()
            except (UnicodeError, OSError):
                continue
            if "\x00" in text:            # wrong encoding slipped through
                continue
            for line in text.splitlines():
                line = line.strip().lstrip("﻿")
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            if os.getenv("TMDB_API_KEY"):
                return


_load_env()

API_KEY = os.getenv("TMDB_API_KEY")
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
DETAIL_URL = "https://api.themoviedb.org/3/movie/{id}"

MIN_VOTES = 50          # reception is only trustworthy above this many TMDB votes

DEFAULT_PANEL = [
    ("Ratatouille", 2007),       # brave premise, beloved + profitable -> flagship
    ("Coco", 2017),              # Disney/Pixar original, loved
    ("Encanto", 2021),           # Disney original, loved
    ("Inside Out", 2015),        # Disney/Pixar original, loved
    ("Wish", 2023),              # Disney original, underperformed
    ("Strange World", 2022),     # Disney original, flop
    ("Toy Story 2", 1999),       # sequel
    ("Toy Story 3", 2010),       # sequel
    ("Toy Story 4", 2019),       # sequel
    ("The Lighthouse", 2019),    # structurally bold, modest box office
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _get(url, params, tries=4):
    """GET with retry/backoff; handles HTTP 429 (rate limit)."""
    for t in range(tries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 1)) + 0.5)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(0.5 * (t + 1))
    return None


def load_titles(path: str):
    """Return a list of (title, year_or_None) from a .txt or .csv file."""
    titles = []
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            tcol = cols.get("title") or (reader.fieldnames[0] if reader.fieldnames else None)
            ycol = cols.get("year") or cols.get("release_year")
            for row in reader:
                title = (row.get(tcol) or "").strip()
                if not title:
                    continue
                year = None
                if ycol and row.get(ycol):
                    m = re.search(r"\d{4}", str(row[ycol]))
                    year = int(m.group()) if m else None
                titles.append((title, year))
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:
                    title, _, y = line.partition("|")
                    m = re.search(r"\d{4}", y)
                    titles.append((title.strip(), int(m.group()) if m else None))
                else:
                    titles.append((line, None))
    return titles


def tmdb_lookup(title, year):
    """Search TMDB, pick the best match, return its detail dict (or None)."""
    params = {"api_key": API_KEY, "query": title, "include_adult": "false"}
    if year:
        params["primary_release_year"] = year
    data = _get(SEARCH_URL, params)
    results = (data or {}).get("results", [])
    if not results:
        return None
    best = None
    for r in results:
        if _norm(r.get("title")) == _norm(title) or _norm(r.get("original_title")) == _norm(title):
            best = r
            break
    best = best or results[0]
    return _get(DETAIL_URL.format(id=best["id"]), {"api_key": API_KEY})


def process_one(item):
    """item = (idx, title, year) -> (idx, row_dict). Runs in a worker thread."""
    idx, title, year = item
    info = tmdb_lookup(title, year)
    if not info:
        return idx, {"title": title, "year": year, "tmdb_id": None, "matched_title": None,
                     "budget": None, "revenue": None, "roi": None, "vote_average": None,
                     "vote_count": None, "has_financials": False, "has_reception": False,
                     "usable": False}
    budget = info.get("budget") or 0
    revenue = info.get("revenue") or 0
    votes = info.get("vote_count") or 0
    has_fin = budget > 0 and revenue > 0
    has_rec = votes >= MIN_VOTES
    return idx, {"title": title, "year": year, "tmdb_id": info.get("id"),
                 "matched_title": info.get("title"), "budget": budget, "revenue": revenue,
                 "roi": round(revenue / budget, 2) if budget > 0 else None,
                 "vote_average": info.get("vote_average"), "vote_count": votes,
                 "has_financials": has_fin, "has_reception": has_rec,
                 "usable": has_fin and has_rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("titles_path", nargs="?", help="titles .csv/.txt (omit for case-study panel)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only process the first N titles")
    args = ap.parse_args()

    if not API_KEY:
        print("ERROR: No TMDB_API_KEY found. Add it to a .env file in this folder.")
        sys.exit(1)

    if args.titles_path:
        titles = load_titles(args.titles_path)
        source = os.path.basename(args.titles_path)
    else:
        titles = DEFAULT_PANEL
        source = "built-in case-study panel"
    if args.limit:
        titles = titles[:args.limit]

    n = len(titles)
    print(f"\nJoin-yield probe | source: {source} | {n} titles | {args.workers} workers\n")

    items = [(i, t, y) for i, (t, y) in enumerate(titles)]
    results = {}
    matched = fin_ok = rec_ok = usable = done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, it) for it in items]
        for fut in as_completed(futures):
            idx, row = fut.result()
            results[idx] = row
            done += 1
            if row["tmdb_id"] is not None:
                matched += 1
            fin_ok += row["has_financials"]
            rec_ok += row["has_reception"]
            usable += row["usable"]
            if done % 25 == 0 or done == n:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  processed {done:>4}/{n}   matched {matched:>4}   "
                      f"usable {usable:>4}   ({rate:.1f}/s)")

    rows = [results[i] for i in range(n)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"join_yield_{ts}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def pct(x):
        return f"{(100 * x / n):.0f}%" if n else "-"

    no_match = [r["title"] for r in rows if r["tmdb_id"] is None]

    print("\n" + "=" * 58)
    print("  JOIN-YIELD SUMMARY")
    print("=" * 58)
    print(f"  Titles in:                 {n}")
    print(f"  Matched on TMDB:           {matched:>4}  ({pct(matched)})")
    print(f"  Has financials (B & R):    {fin_ok:>4}  ({pct(fin_ok)})")
    print(f"  Has reception (>= {MIN_VOTES} v):    {rec_ok:>4}  ({pct(rec_ok)})")
    print(f"  FULLY USABLE:              {usable:>4}  ({pct(usable)})  <- your real N")
    print("=" * 58)
    print(f"  Detail CSV: {out_csv}    (elapsed {time.time() - t0:.0f}s)")
    print("  Trigger: usable >= ~500 -> on track | < ~500 -> scope/data rethink")
    if no_match:
        preview = ", ".join(no_match[:15])
        print(f"\n  {len(no_match)} unmatched (first 15): {preview}")
    print()


if __name__ == "__main__":
    main()
