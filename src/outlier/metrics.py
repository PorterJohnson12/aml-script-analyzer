"""Turning-point evaluation metrics (TRIPOD-style).

A gold entry for a movie is 5 lists of *acceptable* scene indices (one list per TP).
A prediction is 5 single scene indices (one per TP). We report:

- TA  (Total Agreement):   fraction of TPs whose predicted scene is exactly in the
                           gold set for that TP.
- PA  (Partial Agreement): fraction of TPs whose predicted scene is within `tol`
                           (as a fraction of the movie's scene count) of the nearest
                           gold scene.
- D   (Distance):          mean normalized distance from the predicted scene to the
                           nearest gold scene (lower is better).

These follow the turning-point identification metrics described in Papalampidi et al.
(2019). PA tolerance updated to 0.05 (5% of scene count) to match the tolerance
used in the TRIPOD paper; older reports at tol=0.02 will read significantly lower.
"""
from __future__ import annotations
from typing import List, Sequence


def _nearest_dist(pred: int, gold_set: Sequence[int]) -> int:
    return min(abs(pred - g) for g in gold_set)


def distance(pred: Sequence[int], gold: Sequence[Sequence[int]], n_scenes: int) -> float:
    """Mean normalized distance to nearest gold scene, averaged over the 5 TPs."""
    if n_scenes <= 0:
        raise ValueError("n_scenes must be positive")
    ds = [_nearest_dist(p, g) / n_scenes for p, g in zip(pred, gold) if g]
    return sum(ds) / len(ds) if ds else float("nan")


def total_agreement(pred: Sequence[int], gold: Sequence[Sequence[int]]) -> float:
    """Fraction of TPs whose predicted scene is exactly in the gold set."""
    hits = [1.0 if p in g else 0.0 for p, g in zip(pred, gold) if g]
    return sum(hits) / len(hits) if hits else float("nan")


def partial_agreement(
    pred: Sequence[int], gold: Sequence[Sequence[int]], n_scenes: int, tol: float = 0.05
) -> float:
    """Fraction of TPs within `tol` * n_scenes of the nearest gold scene."""
    window = tol * n_scenes
    hits = [1.0 if _nearest_dist(p, g) <= window else 0.0 for p, g in zip(pred, gold) if g]
    return sum(hits) / len(hits) if hits else float("nan")


def per_tp_pa_hits(pred: Sequence[int], gold: Sequence[Sequence[int]], n_scenes: int,
                   tol: float = 0.05) -> List[float]:
    """The per-TP 0/1 hits that `partial_agreement` averages away.

    `partial_agreement` returns one number per movie, which is enough to report but
    not enough to do statistics with. Comparing two models needs the *unpooled*
    decisions, so they can be paired (same movie, same TP) and resampled.

    Pooled across movies this gives n_movies * 5 Bernoulli trials — only 75 for the
    15 gold movies, which is what sets the resolution of every comparison we can
    make. At PA ~= 0.3 the 95% interval is roughly +-0.10 wide, so gaps smaller
    than that are not measurable on gold. See `bootstrap_ci`.
    """
    return [1.0 if _nearest_dist(p, g) <= tol * n_scenes else 0.0
            for p, g in zip(pred, gold) if g]


def bootstrap_ci(values: Sequence[float], n_boot: int = 4000, seed: int = 0,
                 alpha: float = 0.05) -> tuple:
    """Percentile bootstrap CI over a flat array of per-decision values.

    Feed it `per_tp_pa_hits` pooled across movies for a PA interval, or a paired
    difference (hits_a - hits_b, same movies and TPs in the same order) for a
    difference interval — the latter is the honest way to ask "is A better than B"
    on a test set this small.

    Returns (lo, hi). numpy is imported locally to keep this module import-light.
    """
    import numpy as np

    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def evaluate(pred: Sequence[int], gold: Sequence[Sequence[int]], n_scenes: int,
             tol: float = 0.05) -> dict:
    """Convenience: all three metrics for a single movie."""
    return {
        "TA": total_agreement(pred, gold),
        "PA": partial_agreement(pred, gold, n_scenes, tol=tol),
        "D": distance(pred, gold, n_scenes),
    }


def mean_metrics(rows: List[dict]) -> dict:
    """Average a list of per-movie metric dicts (ignores NaNs per key)."""
    keys = ("TA", "PA", "D")
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) == r.get(k)]  # drop NaN
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out
