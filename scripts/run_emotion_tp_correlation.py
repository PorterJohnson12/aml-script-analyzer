"""Sanity check: does the emotion pipeline actually pick up on turning-point moments?

For each of the 8 curated gold movies, we compare per-scene turn magnitude at
labeled TP scenes vs. at non-TP scenes. Three questions:

    1.  Do TP scenes have systematically higher magnitude than non-TP scenes?
        (Mann-Whitney U per film, plus aggregate stats.)
    2.  Is that difference stronger if we look at a small WINDOW around each TP
        (TP ±2 scenes), given that emotional peaks can be adjacent to the labeled
        beat rather than exactly on it?
    3.  Does each specific TP (TP1..TP5) have a consistent magnitude signature
        across films? (McKee's hierarchy would predict TP5 > TP4 > TP3 in the
        idealized case — successive climaxes hit harder.)

Interpretation:
    If TP-scene magnitudes are meaningfully higher than non-TP magnitudes, the
    emotion pipeline is capturing something structurally real and is worth
    building on. If they're indistinguishable, the emotion signal is noise
    relative to the structural anchors and we need to reconsider the approach.

Usage:
    python scripts/run_emotion_tp_correlation.py

Runs ~2 min on the 4070 (mostly emotion scoring across 8 films).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from outlier.data import get_scenes, load_gold_labels
from outlier.emotion import EmotionScorer


MOVIES = [
    "Die Hard",
    "The Breakfast Club",
    "Moon (film)",
    "Slumdog Millionaire",
    "Panic Room",
    "Unforgiven",
    "The Crying Game",
    "Juno (film)",
]

TP_NAMES = ["Opportunity", "Change of Plans", "Point of No Return", "Major Setback", "Climax"]

OUT_CSV = ROOT / "notebooks" / "emotion_tp_correlation.csv"


def _tp_window_indices(tps: list[int], n_scenes: int, radius: int = 2) -> set[int]:
    """Return the set of scene indices within `radius` scenes of any TP."""
    window: set[int] = set()
    for tp in tps:
        for i in range(max(0, tp - radius), min(n_scenes, tp + radius + 1)):
            window.add(i)
    return window


def main() -> None:
    print("=" * 70)
    print("Sanity check — do TP scenes have higher magnitudes than non-TP scenes?")
    print("=" * 70)

    gold = load_gold_labels()
    print(f"\n[setup] loading emotion scorer")
    t0 = time.time()
    scorer = EmotionScorer(use_vader=False)   # VADER not needed here — magnitude is what we test
    print(f"        device: {scorer.device}   elapsed: {time.time() - t0:.1f}s")

    # Per-movie stats
    per_movie_rows: list[dict] = []
    # Per-TP-position: collect magnitudes at TP1..TP5 across all films
    per_tp_magnitudes: dict[int, list[float]] = {i: [] for i in range(1, 6)}
    # Per-TP-position: same but for a TP ±2 window (max magnitude in window)
    per_tp_windowed_magnitudes: dict[int, list[float]] = {i: [] for i in range(1, 6)}

    for movie in MOVIES:
        if movie not in gold:
            print(f"  ! {movie!r} not in gold — skipping")
            continue
        scenes = get_scenes(movie)
        if not scenes:
            print(f"  ! {movie!r} has no scenes — skipping")
            continue
        tp_indices = [tp[0] for tp in gold[movie]]

        print(f"\n  → {movie}   scenes={len(scenes)}   TPs={tp_indices}")
        t0 = time.time()
        scores = scorer.score_script(scenes)
        magnitudes = np.asarray(scores["magnitude"], dtype=np.float64)
        print(f"      scoring: {time.time() - t0:.1f}s")

        # --- STRICT: exact TP scenes vs. all non-TP scenes
        tp_set = set(tp_indices)
        non_tp_set = set(range(len(scenes))) - tp_set
        tp_mags = np.array([magnitudes[i] for i in sorted(tp_set)])
        non_tp_mags = np.array([magnitudes[i] for i in sorted(non_tp_set)])

        # Mann-Whitney U — one-sided (are TP magnitudes greater?)
        try:
            u_stat, p_value = mannwhitneyu(tp_mags, non_tp_mags, alternative="greater")
        except ValueError:
            u_stat, p_value = float("nan"), float("nan")

        # --- WINDOWED: TP ±2 scenes vs. everything else
        window_set = _tp_window_indices(tp_indices, len(scenes), radius=2)
        window_non = set(range(len(scenes))) - window_set
        win_mags = np.array([magnitudes[i] for i in sorted(window_set)])
        winnon_mags = np.array([magnitudes[i] for i in sorted(window_non)])
        try:
            _, p_value_win = mannwhitneyu(win_mags, winnon_mags, alternative="greater")
        except ValueError:
            p_value_win = float("nan")

        # Per-TP-position collection: magnitude AT each labeled TP scene
        for k, tp_idx in enumerate(tp_indices, start=1):
            if 0 <= tp_idx < len(scenes):
                per_tp_magnitudes[k].append(float(magnitudes[tp_idx]))
                # For windowed: max magnitude within ±2 of the TP
                lo = max(0, tp_idx - 2)
                hi = min(len(scenes), tp_idx + 3)
                per_tp_windowed_magnitudes[k].append(float(magnitudes[lo:hi].max()))

        # Per-movie summary row
        per_movie_rows.append({
            "movie": movie,
            "n_scenes": len(scenes),
            "tp_mag_mean": float(tp_mags.mean()),
            "non_tp_mag_mean": float(non_tp_mags.mean()),
            "ratio_strict": float(tp_mags.mean() / non_tp_mags.mean()) if non_tp_mags.mean() > 0 else float("nan"),
            "p_value_strict": float(p_value),
            "win_mag_mean": float(win_mags.mean()),
            "winnon_mag_mean": float(winnon_mags.mean()),
            "ratio_windowed": float(win_mags.mean() / winnon_mags.mean()) if winnon_mags.mean() > 0 else float("nan"),
            "p_value_windowed": float(p_value_win),
        })

        print(f"      STRICT   TP-mag={tp_mags.mean():.3f}  non-TP={non_tp_mags.mean():.3f}"
              f"  ratio={tp_mags.mean() / non_tp_mags.mean():.2f}x  p={p_value:.3f}")
        print(f"      WINDOWED ±2  win-mag={win_mags.mean():.3f}  non-win={winnon_mags.mean():.3f}"
              f"  ratio={win_mags.mean() / winnon_mags.mean():.2f}x  p={p_value_win:.3f}")

    # ---- Per-movie table + save
    df = pd.DataFrame(per_movie_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n\n" + "=" * 70)
    print("PER-MOVIE SUMMARY")
    print("=" * 70)
    show = df[["movie", "tp_mag_mean", "non_tp_mag_mean", "ratio_strict", "p_value_strict",
               "ratio_windowed", "p_value_windowed"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "-" * 70)
    print("AGGREGATE — across all films")
    print("-" * 70)
    strict_wins = int((df["ratio_strict"] > 1).sum())
    win_wins = int((df["ratio_windowed"] > 1).sum())
    sig_strict = int((df["p_value_strict"] < 0.05).sum())
    sig_win = int((df["p_value_windowed"] < 0.05).sum())
    n = len(df)
    print(f"  Films where TP-mag > non-TP-mag (strict):     {strict_wins}/{n}")
    print(f"  Films where p < 0.05 (strict, one-sided):     {sig_strict}/{n}")
    print(f"  Films where windowed win > non-win:           {win_wins}/{n}")
    print(f"  Films where p < 0.05 (windowed, one-sided):   {sig_win}/{n}")
    print(f"  Mean ratio across films (strict):    {df['ratio_strict'].mean():.3f}x")
    print(f"  Mean ratio across films (windowed):  {df['ratio_windowed'].mean():.3f}x")

    print("\n" + "-" * 70)
    print("PER-TP-POSITION SIGNATURES — mean magnitude across all films")
    print("-" * 70)
    print(f"  {'TP':<5} {'name':<22} {'strict_mean':>12} {'windowed_max_mean':>18}  {'n_films':>8}")
    for k in range(1, 6):
        strict_mean = np.mean(per_tp_magnitudes[k]) if per_tp_magnitudes[k] else float("nan")
        win_mean = np.mean(per_tp_windowed_magnitudes[k]) if per_tp_windowed_magnitudes[k] else float("nan")
        print(f"  TP{k}   {TP_NAMES[k-1]:<22} {strict_mean:>12.3f} {win_mean:>18.3f}  {len(per_tp_magnitudes[k]):>8}")

    print(f"\nfull CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
