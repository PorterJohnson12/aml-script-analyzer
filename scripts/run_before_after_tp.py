"""Before/after TP analysis — the professor's specific ask.

For each of 8 gold movies, compute (mean valence, mean intensity, per-emotion
mean probability) for the WINDOW of scenes just BEFORE each TP vs. the WINDOW
just AFTER each TP. Report the shift per TP, aggregated across films.

Also runs a per-emotion analysis: for each of the 7 emotion channels, is the
mean probability at TP scenes different from non-TP scenes across all 8 films?

This tests two hypotheses:

    H1  (windowed shift)   — TPs mark a real change in the surrounding
                             narrative, even if the TP scene itself is
                             sustained-tone. Test: mean(before window) vs
                             mean(after window) across TPs, averaged over films.

    H2  (per-emotion)      — A specific emotion channel (not valence) may
                             be systematically different at TP scenes vs
                             non-TP scenes, even if aggregate magnitude is not.
                             Test: mean(prob[emotion]) at TP scenes vs
                             non-TP scenes per film, aggregated.

The goal is to find AT LEAST ONE real pattern that (a) differentiates TP
context from surrounding-non-TP context and (b) is defensible as a feature
for the trained model.

Usage:
    python scripts/run_before_after_tp.py

Runs ~2 min on the 4070.
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
    "Die Hard", "The Breakfast Club", "Moon (film)", "Slumdog Millionaire",
    "Panic Room", "Unforgiven", "The Crying Game", "Juno (film)",
]

TP_NAMES = ["Opportunity", "Change of Plans", "Point of No Return", "Major Setback", "Climax"]
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

WINDOW = 3  # scenes before/after each TP

OUT_CSV_SHIFT = ROOT / "notebooks" / "before_after_shifts.csv"
OUT_CSV_PER_EMO = ROOT / "notebooks" / "per_emotion_tp.csv"


def _window(idx: int, n: int, radius: int) -> range:
    """Range of scene indices in [idx - radius, idx - 1] (before) or [idx + 1, idx + radius] (after)."""
    return range(max(0, idx), min(n, idx + radius))


def main() -> None:
    print("=" * 72)
    print("Before/after TP analysis — do TPs mark real shifts in surrounding narrative?")
    print("=" * 72)

    gold = load_gold_labels()
    print(f"\n[setup] loading emotion scorer")
    t0 = time.time()
    scorer = EmotionScorer(use_vader=False)
    print(f"        device: {scorer.device}   elapsed: {time.time() - t0:.1f}s")

    # Per-TP-position: collect (before_valence, after_valence, before_intensity, after_intensity,
    # per-emotion before/after distributions) across all films
    per_tp_shifts: dict[int, list[dict]] = {i: [] for i in range(1, 6)}
    # Per-emotion collection: prob at TP scenes vs non-TP scenes (per film)
    per_emotion_rows: list[dict] = []

    for movie in MOVIES:
        if movie not in gold:
            continue
        scenes = get_scenes(movie)
        if not scenes:
            continue
        tps = [tp[0] for tp in gold[movie]]

        print(f"\n  → {movie}   scenes={len(scenes)}   TPs={tps}")
        t0 = time.time()
        scores = scorer.score_script(scenes)
        print(f"      scoring: {time.time() - t0:.1f}s")

        v_close = scores["valence_close"]
        i_close = scores["intensity_close"]
        probs_close = scores["probs_close"]   # (n_scenes, 7)

        for k, tp in enumerate(tps, start=1):
            n = len(scenes)
            before_idx = list(_window(tp - WINDOW, n, WINDOW))
            after_idx = list(range(tp + 1, min(n, tp + WINDOW + 1)))
            if not before_idx or not after_idx:
                continue
            before_v = float(v_close[before_idx].mean())
            after_v = float(v_close[after_idx].mean())
            before_i = float(i_close[before_idx].mean())
            after_i = float(i_close[after_idx].mean())
            before_probs = probs_close[before_idx].mean(axis=0)
            after_probs = probs_close[after_idx].mean(axis=0)
            per_tp_shifts[k].append({
                "movie": movie,
                "tp_scene": tp,
                "before_valence": before_v,
                "after_valence": after_v,
                "valence_shift": after_v - before_v,
                "abs_valence_shift": abs(after_v - before_v),
                "before_intensity": before_i,
                "after_intensity": after_i,
                "intensity_shift": after_i - before_i,
                **{f"before_{lbl}": float(before_probs[j]) for j, lbl in enumerate(EMOTION_LABELS)},
                **{f"after_{lbl}": float(after_probs[j]) for j, lbl in enumerate(EMOTION_LABELS)},
                **{f"shift_{lbl}": float(after_probs[j] - before_probs[j]) for j, lbl in enumerate(EMOTION_LABELS)},
            })

        # Per-emotion at TP vs non-TP scenes for this film
        tp_set = set(tps)
        non_tp = sorted(set(range(len(scenes))) - tp_set)
        tp_list = sorted(tp_set)
        for j, lbl in enumerate(EMOTION_LABELS):
            tp_probs = probs_close[tp_list, j]
            non_tp_probs = probs_close[non_tp, j]
            per_emotion_rows.append({
                "movie": movie,
                "emotion": lbl,
                "tp_mean": float(tp_probs.mean()),
                "non_tp_mean": float(non_tp_probs.mean()),
                "diff": float(tp_probs.mean() - non_tp_probs.mean()),
                "ratio": float(tp_probs.mean() / non_tp_probs.mean()) if non_tp_probs.mean() > 0 else float("nan"),
            })

    # ---- Aggregate per-TP-position shifts across films
    print("\n\n" + "=" * 72)
    print("H1 — WINDOWED SHIFT PER TP (3 scenes before → 3 scenes after)")
    print("=" * 72)
    print("Mean across films of (after-window - before-window):")
    print(f"  {'TP':<5} {'name':<22} {'valence_shift':>16} {'abs_val_shift':>16} {'intensity_shift':>16}  {'n_films':>8}")
    shift_rows = []
    for k in range(1, 6):
        rows = per_tp_shifts[k]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        vshift = df["valence_shift"].mean()
        abs_vshift = df["abs_valence_shift"].mean()
        ishift = df["intensity_shift"].mean()
        print(f"  TP{k}   {TP_NAMES[k-1]:<22} {vshift:>+16.4f} {abs_vshift:>+16.4f} {ishift:>+16.4f}  {len(df):>8}")
        shift_rows.append({
            "tp": f"TP{k}",
            "name": TP_NAMES[k-1],
            "mean_valence_shift": vshift,
            "mean_abs_valence_shift": abs_vshift,
            "mean_intensity_shift": ishift,
            "n_films": len(df),
            **{f"mean_shift_{lbl}": df[f"shift_{lbl}"].mean() for lbl in EMOTION_LABELS},
        })

    shift_df = pd.DataFrame(shift_rows)
    OUT_CSV_SHIFT.parent.mkdir(parents=True, exist_ok=True)
    shift_df.to_csv(OUT_CSV_SHIFT, index=False)

    # Per-emotion shift table (which emotion channel moves most across each TP?)
    print("\n\n" + "-" * 72)
    print("PER-EMOTION SHIFT PER TP (mean(after) - mean(before)) across films")
    print("-" * 72)
    hdr = "  TP    " + "".join(f"{lbl:>12}" for lbl in EMOTION_LABELS)
    print(hdr)
    for row in shift_rows:
        cells = "".join(f"{row[f'mean_shift_{lbl}']:>+12.4f}" for lbl in EMOTION_LABELS)
        print(f"  {row['tp']}  {cells}")

    # ---- H2: per-emotion at TP vs non-TP across films
    print("\n\n" + "=" * 72)
    print("H2 — PER-EMOTION AT TP vs NON-TP (aggregated across films)")
    print("=" * 72)
    emo_df = pd.DataFrame(per_emotion_rows)
    emo_df.to_csv(OUT_CSV_PER_EMO, index=False)

    print(f"  {'emotion':<12} {'tp_mean':>10} {'non_tp_mean':>12} {'diff':>10} {'ratio':>8}  {'films_diff_positive':>22}")
    for lbl in EMOTION_LABELS:
        sub = emo_df[emo_df["emotion"] == lbl]
        tp_mean = sub["tp_mean"].mean()
        non_tp_mean = sub["non_tp_mean"].mean()
        diff = sub["diff"].mean()
        ratio = tp_mean / non_tp_mean if non_tp_mean > 0 else float("nan")
        n_positive = int((sub["diff"] > 0).sum())
        n_total = len(sub)
        print(f"  {lbl:<12} {tp_mean:>10.4f} {non_tp_mean:>12.4f} {diff:>+10.4f} {ratio:>8.2f}x  {n_positive}/{n_total}")

    print(f"\nfull tables:")
    print(f"  {OUT_CSV_SHIFT}")
    print(f"  {OUT_CSV_PER_EMO}")


if __name__ == "__main__":
    main()
