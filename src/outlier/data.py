"""Loaders for screenplays and turning-point labels.

- Gold labels: 15 test movies with scene-level TP indices (TRIPOD_screenplays_test.csv).
- Silver labels: training movies with projected (distant-supervision) TP scene indices
  (SUMMER labels_train_TRIPOD_silver.pickle).

Both label formats are the same shape: {movie: [tp1_scenes, ..., tp5_scenes]}, where
each tp*_scenes is a list of acceptable scene indices for that turning point.
"""
from __future__ import annotations
import ast
import csv
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .segmentation import segment_screenplay

TPLabels = List[List[int]]  # 5 lists of scene indices


def list_movies() -> List[str]:
    """All movie folders that ship a screenplay."""
    return sorted(p.name for p in config.SCREENPLAYS_DIR.iterdir() if p.is_dir())


def script_path(movie_name: str) -> Optional[Path]:
    """Robust to the folder/file naming quirk (e.g. `48 Hrs_/48 Hrs._script.txt`)."""
    d = config.SCREENPLAYS_DIR / movie_name
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*_script.txt"))
    return hits[0] if hits else None


def get_scenes(movie_name: str) -> List[str]:
    sp = script_path(movie_name)
    return segment_screenplay(sp) if sp else []


def load_gold_labels() -> Dict[str, TPLabels]:
    """{movie: [tp1..tp5]} for the 15 test movies (each tp is a list of scene indices)."""
    out: Dict[str, TPLabels] = {}
    with open(config.GOLD_SCREENPLAYS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["movie_name"]] = [ast.literal_eval(row[f"tp{i}"]) for i in range(1, 6)]
    return out


def load_silver_labels(collapse: bool = True) -> Dict[str, TPLabels]:
    """Silver training labels.

    Keys in the pickle are `Movie_annotatorIndex`. With collapse=True we keep one
    entry per movie (the first/`_0` annotator); with collapse=False the raw
    multi-annotator dict is returned.
    """
    with open(config.SILVER_LABELS_PICKLE, "rb") as f:
        raw = pickle.load(f)
    if not collapse:
        return raw
    out: Dict[str, TPLabels] = {}
    for k in sorted(raw):
        movie = re.sub(r"_\d+$", "", k)
        if movie not in out:
            out[movie] = raw[k]
    return out
