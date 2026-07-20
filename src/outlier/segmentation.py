"""Scene segmentation, faithful to TRIPOD's segmenter.

We replicate TRIPOD's `screenplays_scene_segmentation.py` exactly so that scene
index `i` in our output equals scene index `i` in their gold/silver labels. A new
scene starts at each slugline (a line containing INT./EXT.). Content before the
first slugline is dropped, and — matching their code — the final scene is not
emitted (their gold indices are generated the same way). Verified: every gold
turning-point index falls inside the resulting scene count for all 15 test movies.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import List

from .config import SLUGLINE_MARKERS


def _is_slugline(line: str) -> bool:
    toks = [t for t in re.sub(" +", " ", line).split() if t]
    return any(m in toks for m in SLUGLINE_MARKERS)


def segment_screenplay(path: str | Path) -> List[str]:
    """Return the screenplay as an ordered list of scene texts (index 0 == first scene)."""
    scenes: List[str] = []
    last = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if _is_slugline(line):
                if last != "":
                    scenes.append(last)
                last = line
            elif last != "":
                last += "\n" + line
    # NOTE: TRIPOD does not append the trailing `last` scene; we match that so
    # our indices align with their labels. Do not "fix" this.
    return scenes


def n_scenes(path: str | Path) -> int:
    return len(segment_screenplay(path))
