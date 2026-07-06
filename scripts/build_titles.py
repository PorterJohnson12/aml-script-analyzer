"""
build_titles.py  ·  aml-mckee-script-classifier
================================================
Flatten the ScriptBase corpus into a single `titles.csv` for the join-yield probe
(and as an early metadata table).

For each `*.tar.gz` in scriptbase_alpha, it opens only the small
`processed/imdb_meta.txt` member and pulls: year, IMDB score, Meta score, genres.
The film title is cleaned from the archive filename (Wikipedia-style "_" -> ":" and
trailing "(… film)" / "(year)" disambiguators removed).

Usage
-----
  python build_titles.py
  python build_titles.py --alpha-dir ../data/scriptbase-master/scriptbase_alpha --out ../data/titles.csv

Output columns: raw_name, title, year, imdb_score, meta_score, genres
"""

import os
import re
import csv
import sys
import glob
import tarfile
import argparse

# repo-root-relative defaults (this file lives in scripts/)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ALPHA = os.path.normpath(os.path.join(HERE, "..", "data", "scriptbase-master", "scriptbase_alpha"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "data", "titles.csv"))

# trailing "(...)" that is a Wikipedia disambiguator, e.g. (film), (1999 film),
# (2009 animated film), (2003), (novel) — strip it. Real year comes from imdb_meta.
_DISAMBIG = re.compile(r"\s*\((?:[^)]*\b(?:film|novel|TV|miniseries|play)\b[^)]*|\d{4})\)\s*$", re.IGNORECASE)


def clean_title(raw_name: str) -> str:
    t = raw_name
    # Wikipedia uses "_" where the real title has ":" (illegal in filenames)
    t = t.replace("_ ", ": ").replace("_", ":")
    # strip one or more trailing disambiguators
    prev = None
    while prev != t:
        prev = t
        t = _DISAMBIG.sub("", t)
    return t.strip()


def parse_imdb_meta(text: str) -> dict:
    """imdb_meta.txt is tab-separated key<TAB>value (keys repeat for genre/keyword/cast)."""
    out = {"year": None, "imdb_score": None, "meta_score": None, "genres": []}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        key, _, val = line.partition("\t")
        key, val = key.strip().lower(), val.strip()
        if not val:
            continue
        if key == "year" and out["year"] is None:
            m = re.search(r"\d{4}", val)
            out["year"] = int(m.group()) if m else None
        elif key == "imdb score" and out["imdb_score"] is None:
            out["imdb_score"] = val
        elif key == "meta score" and out["meta_score"] is None:
            out["meta_score"] = val
        elif key == "genre":
            out["genres"].append(val)
    return out


def read_imdb_meta_from_archive(path: str):
    try:
        with tarfile.open(path, "r:gz") as tf:
            member = next((m for m in tf.getmembers()
                           if m.name.endswith("processed/imdb_meta.txt")), None)
            if member is None:
                return None
            f = tf.extractfile(member)
            return f.read().decode("utf-8", errors="replace") if f else None
    except Exception as e:
        print(f"   ! could not read {os.path.basename(path)}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-dir", default=DEFAULT_ALPHA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    archives = sorted(glob.glob(os.path.join(args.alpha_dir, "*.tar.gz")))
    if not archives:
        print(f"❌ No .tar.gz archives found in {args.alpha_dir}")
        sys.exit(1)

    print(f"\n🎬 Flattening {len(archives)} ScriptBase archives -> {args.out}\n")

    rows = []
    with_year = 0
    for i, path in enumerate(archives, 1):
        raw = os.path.basename(path)[:-len(".tar.gz")]
        title = clean_title(raw)
        meta = parse_imdb_meta(read_imdb_meta_from_archive(path) or "")
        if meta["year"]:
            with_year += 1
        rows.append({
            "raw_name": raw,
            "title": title,
            "year": meta["year"],
            "imdb_score": meta["imdb_score"],
            "meta_score": meta["meta_score"],
            "genres": "|".join(meta["genres"]),
        })
        if i % 200 == 0:
            print(f"   … {i}/{len(archives)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["raw_name", "title", "year",
                                          "imdb_score", "meta_score", "genres"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print("\n" + "=" * 48)
    print(f"  Films written:      {n}")
    print(f"  With a year:        {with_year}  ({100*with_year//n if n else 0}%)")
    print(f"  Missing year:       {n - with_year}")
    print("=" * 48)
    print(f"  -> {args.out}")
    print("  Next: python join_yield_probe.py ../data/titles.csv\n")


if __name__ == "__main__":
    main()
