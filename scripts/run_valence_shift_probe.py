"""FIRST LOOK at Week-4 action item 3: does the emotional charge flip at turning points?

The professor's 1-on-1 ask: look at the valence of the ~3 scenes before and after each
turning point, and find patterns useful for both ML and business. This is the smallest
runnable test of that:

    valence delta at a scene = mean valence of the K scenes AFTER - mean of the K before

If |delta| is significantly larger at real turning points than at random scenes, that is
a first real pattern — an ML signal (a per-scene feature) and a business story ("turning
points are where the story's emotional charge shifts").

Scope: EXPLORATORY, gold set only (15 movies), reusing the valence already in
docs/week3/results.json — no re-encoding. Week 4 extends this to the full silver+gold
corpus and to per-emotion deltas (not just valence). Treat the numbers as a first look,
not a validated result.

Usage (from repo root):
    python scripts/run_valence_shift_probe.py            # K=3 (three scenes each side)
    python scripts/run_valence_shift_probe.py --k 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "week3" / "results.json"


def scene_valence(g: dict):
    vo, vc = g.get("valence_open"), g.get("valence_close")
    if vo is None:
        return None
    return (np.array(vo) + np.array(vc)) / 2.0


def delta_at(v: np.ndarray, pos: int, k: int):
    """Mean valence of k scenes after pos minus k before. None if too near an edge."""
    n = len(v)
    if pos - k < 0 or pos + k >= n:
        return None
    return v[pos + 1: pos + 1 + k].mean() - v[pos - k: pos].mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="scenes before/after each TP")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    K = args.k

    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS} — run scripts/run_experiments.py first")
    d = json.loads(RESULTS.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)

    tp_deltas, tp_signed = [], {k: [] for k in range(5)}
    rand_deltas = []
    for g in d["gold"]:
        v = scene_valence(g)
        if v is None:
            continue
        n = g["n"]
        for k, tp in enumerate(g["tps"]):
            dd = delta_at(v, int(round(float(np.mean(tp)))), K)
            if dd is not None:
                tp_deltas.append(dd); tp_signed[k].append(dd)
        interior = np.arange(K, n - K)
        for pos in rng.choice(interior, size=min(5, len(interior)), replace=False):
            dd = delta_at(v, int(pos), K)
            if dd is not None:
                rand_deltas.append(dd)

    tp = np.array(tp_deltas); rnd = np.array(rand_deltas)
    if not len(tp) or not len(rnd):
        raise SystemExit("no valence in results.json — re-run run_experiments.py with the "
                         "emotion_augmented experiment enabled so valence columns are dumped")

    print("=" * 66)
    print(f"VALENCE DELTA (after {K} - before {K}) — |shift| magnitude, gold set")
    print("=" * 66)
    print(f"  at real TPs : mean |delta| = {np.abs(tp).mean():.4f}   (n={len(tp)})")
    print(f"  at random   : mean |delta| = {np.abs(rnd).mean():.4f}   (n={len(rnd)})")
    print(f"  ratio TP/random = {np.abs(tp).mean() / np.abs(rnd).mean():.2f}x")

    obs = np.abs(tp).mean() - np.abs(rnd).mean()
    pool = np.concatenate([np.abs(tp), np.abs(rnd)])
    nA = len(tp)
    perm = np.array([rng.permutation(pool)[:nA].mean() - rng.permutation(pool)[nA:].mean()
                     for _ in range(5000)])
    p = float((np.abs(perm) >= abs(obs)).mean())
    print(f"  permutation p (|shift| TP vs random) = {p:.3f}  "
          f"{'** significant' if p < 0.05 else '(n.s.)'}")

    print("\n" + "=" * 66)
    print("SIGNED delta by turning point (down early / up at the climax?)")
    print("=" * 66)
    names = ["TP1 Opportunity", "TP2 Change of Plans", "TP3 Point of No Return",
             "TP4 Major Setback", "TP5 Climax"]
    for k in range(5):
        arr = np.array(tp_signed[k])
        if len(arr):
            print(f"  {names[k]:<24} mean signed delta = {arr.mean():+.4f}   (n={len(arr)})")
    print(f"\n  random baseline signed delta = {rnd.mean():+.4f}")
    print("\n  NOTE: exploratory, 15 gold movies, TP5 n is small (climax is near the end,")
    print("  so the 3-after window runs off the edge). Week 4 extends to the full corpus.")


if __name__ == "__main__":
    main()
