"""Run the full McKee-conformance pipeline on multiple gold movies for cross-film comparison.

This is the "show real evidence" script — takes a curated set of gold-labeled films
across different genres/structures, runs the emotion + embedding + McKee pipeline on
each, and produces:

    (a) an emotional-arc PNG per movie (saved to notebooks/arcs/)
    (b) a comparison CSV of all McKee-conformance metrics side by side
    (c) a summary table printed to the console
    (d) a plain-language read per movie printed to the console

The comparison table is the artifact that goes into the M3A1 report and the M3P1
presentation as evidence that the tool differentiates structures across films.

Usage (from repo root):
    python scripts/run_multi_movie_arc.py

Runs ~1-2 min per movie on CPU (much faster on GPU). Default movie set below can be
edited or overridden via --movies argument.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from outlier.data import get_scenes, load_gold_labels
from outlier.embeddings import SceneEncoder
from outlier.emotion import EmotionScorer
from outlier.mckee import summarize, plain_language_read, plot_emotional_arc


# Curated cross-film comparison set — each chosen for a distinct structural signature
DEFAULT_MOVIES = [
    "Die Hard",                     # sustained-pressure action, contained space
    "The Breakfast Club",           # single location, real-time-ish, ensemble drama
    "Moon (film)",                  # slow psychological, single character, contained
    "Slumdog Millionaire",          # non-linear episodic, wide reach across settings
    "Panic Room",                   # contained thriller (contrast with Die Hard's action-thriller)
    "Unforgiven",                   # traditional western, character arc-heavy
    "The Crying Game",              # mid-story identity reveal, structural surprise
    "Juno (film)",                  # short-timeline character comedy-drama
]

ARCS_DIR = ROOT / "notebooks" / "arcs"
COMPARE_CSV = ROOT / "notebooks" / "mckee_comparison.csv"


def _load_pipeline(verbose: bool = True):
    """Set up encoder + emotion scorer once, share across all movies."""
    if verbose:
        print("\n[setup] Loading emotion model + scene encoder (this may download models on first run)")
    t0 = time.time()
    encoder = SceneEncoder()
    scorer = EmotionScorer(use_vader=True)
    if verbose:
        print(f"        device: {scorer.device}   elapsed: {time.time() - t0:.1f}s")
    return encoder, scorer


def _process_one(movie: str, gold: dict, encoder: SceneEncoder, scorer: EmotionScorer,
                 verbose: bool = True) -> dict | None:
    """Run the full pipeline on one movie. Returns summary dict or None on failure."""
    if movie not in gold:
        print(f"  ! {movie!r} not in gold labels — skipping")
        return None
    scenes = get_scenes(movie)
    if not scenes:
        print(f"  ! {movie!r} has no scenes (segmenter failed) — skipping")
        return None
    tp_indices = [tp[0] for tp in gold[movie]]

    if verbose:
        print(f"\n  → {movie}   scenes={len(scenes)}   TPs={tp_indices}")

    t0 = time.time()
    scene_scores = scorer.score_script(scenes)
    if verbose:
        print(f"      emotion scoring:   {time.time() - t0:.1f}s")

    t0 = time.time()
    scene_embs = encoder.encode_scenes(scenes)
    if verbose:
        print(f"      scene embeddings:  {time.time() - t0:.1f}s")

    summary = summarize(scene_scores, tp_indices, len(scenes), scene_embeddings=scene_embs)
    summary["_movie"] = movie
    summary["_n_scenes"] = len(scenes)

    # Save the emotional arc PNG
    ARCS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    plot_emotional_arc(scene_scores, tp_scene_indices=tp_indices,
                       title=f"{movie} — emotional arc", ax=ax)
    fig.tight_layout()
    safe_name = movie.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    png_path = ARCS_DIR / f"{safe_name}.png"
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    if verbose:
        print(f"      arc saved:         {png_path}")

    # Also print the plain-language read for this movie
    if verbose:
        print()
        for line in plain_language_read(summary).split("\n"):
            print(f"      {line}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies", nargs="+", default=DEFAULT_MOVIES,
                    help="movie names to process (default: curated cross-genre set)")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Outlier — cross-film McKee-conformance comparison  ({len(args.movies)} films)")
    print("=" * 70)

    gold = load_gold_labels()
    encoder, scorer = _load_pipeline()

    summaries: list[dict] = []
    for movie in args.movies:
        summary = _process_one(movie, gold, encoder, scorer)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("\nNo movies processed successfully.")
        return

    # Build the comparison DataFrame — put _movie first, then key McKee signals
    df = pd.DataFrame(summaries)
    front_cols = ["_movie", "_n_scenes", "flat_scene_rate",
                  "per_scene_swing_slope", "per_act_impact_slope",
                  "reversal_magnitude_slope", "narrative_novelty_slope",
                  "alternation_compliant", "alternation_tp4_close", "alternation_tp5_close",
                  "act_length_std"]
    front_cols = [c for c in front_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    COMPARE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPARE_CSV, index=False)

    print("\n\n" + "=" * 70)
    print("COMPARISON TABLE (key signals) — full data in the CSV")
    print("=" * 70)
    display_cols = ["_movie", "flat_scene_rate", "per_act_impact_slope",
                    "reversal_magnitude_slope", "narrative_novelty_slope",
                    "alternation_compliant", "act_length_std"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nfull comparison table: {COMPARE_CSV}")
    print(f"emotional-arc PNGs:    {ARCS_DIR}/")


if __name__ == "__main__":
    main()
