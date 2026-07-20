"""Is the silver LABEL QUALITY what blocks content learning? (Answer: no.)

Gold gives 15 movies x 5 TPs = 75 decisions, so its 95% interval is ~+-0.10 —
too wide to resolve the effect sizes in play. This script runs the highest-power
test available instead: 5-fold cross-validation over the CLEAN silver movies
(those whose 5 turning points are distinct and strictly ordered), so every clean
movie is predicted exactly once by a model that never trained on it.

  46 clean movies x 5 TPs = 230 decisions -> ~3x the resolution of gold.

Each fold trains twice — real embeddings vs zeroed input — and the two are
compared PAIRED on the identical decisions, which is the only honest way to ask
"does content help?" at this sample size.

Reproduces the §3 "label quality is not the blocker" result in
docs/week3/ml-experimentation-report.md.

Usage (from the repo root):
    python scripts/run_label_quality_cv.py            # 3 seeds (~6 min after prep)
    python scripts/run_label_quality_cv.py --seeds 5

Note: this re-encodes the silver corpus (~3 min) because it needs per-scene
features for movies that are not in the gold set, and results.json only carries
gold. It is a diagnostic, not part of the per-run pipeline.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from outlier.data import get_scenes, load_silver_labels, list_movies
from outlier.embeddings import SceneEncoder
from outlier.model import TPFinder, train_full
from outlier.metrics import per_tp_pa_hits, bootstrap_ci

PA_TOL = 0.05
DEV_HOLDOUT = 8          # small dev slice per fold, for the best-epoch checkpoint
SILVER_NAME_ALIASES = {
    "48 Hrs": "48 Hrs_",
    "Star Wars Episode I- The Phantom Menace": "Star Wars Episode I_ The Phantom Menace",
    "When Harry Met Sally": "When Harry Met Sally___",
}


def is_clean(tps) -> bool:
    """Five distinct, strictly-ordered turning-point scene sets.

    SUMMER's synopsis->scene projection collapsed on ~37% of the corpus, assigning
    two or more TPs to the identical scene set (Gran Torino: TP1=TP2=TP3=scene 25).
    Those movies ask the loss for several different answers and hand it one.
    """
    if len({tuple(sorted(t)) for t in tps}) < 5:
        return False
    centers = [float(np.mean(t)) for t in tps]
    return all(centers[i] < centers[i + 1] for i in range(4))


def load_clean_silver(encoder, device):
    silver = load_silver_labels(collapse=True)
    on_disk = set(list_movies())
    rows = []
    t0 = time.time()
    for m, tps in sorted(silver.items()):
        folder = SILVER_NAME_ALIASES.get(m, m)
        if folder not in on_disk:
            continue
        scenes = get_scenes(folder)
        n = len(scenes)
        if n == 0 or max(max(t) for t in tps) >= n:
            continue
        if not is_clean(tps):
            continue
        rows.append({"movie": folder, "embs": encoder.encode_scenes(scenes),
                     "tps": tps, "n": n})
        if len(rows) % 15 == 0:
            print(f"    encoded {len(rows)} clean movies ({time.time()-t0:.0f}s)")
    return rows


def build(rows, mode):
    return [((np.zeros_like(r["embs"]) if mode == "zeros" else r["embs"]).astype(np.float32),
             r["tps"]) for r in rows]


def hits_and_dists(model, rows, mode, device):
    model.eval()
    H, D = [], []
    with torch.no_grad():
        for (feats, _), r in zip(build(rows, mode), rows):
            t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            preds = model(t).argmax(dim=-1)[0].tolist()
            H.extend(per_tp_pa_hits(preds, r["tps"], r["n"], tol=PA_TOL))
            D.extend(min(abs(p - x) for x in g) / r["n"] for p, g in zip(preds, r["tps"]))
    return H, D


def cv(rows, mode, seed, device):
    """5-fold CV: every movie predicted once by a model that never saw it."""
    rng = np.random.default_rng(500 + seed)
    idx = rng.permutation(len(rows))
    H, D = [], []
    for fold in np.array_split(idx, 5):
        fs = set(fold.tolist())
        tr = [rows[i] for i in idx if i not in fs]
        te = [rows[i] for i in fold]
        torch.manual_seed(seed)
        m = TPFinder(input_dim=384, hidden=128, n_layers=1, dropout=0.1, head="bi-lstm")
        train_full(m, build(tr[DEV_HOLDOUT:], mode), build(tr[:DEV_HOLDOUT], mode),
                   epochs=30, lr=1e-3, device=device, pa_tol=PA_TOL, verbose=False)
        h, d = hits_and_dists(m, te, mode, device)
        H.extend(h); D.extend(d)
    return np.array(H), np.array(D)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device: {device}  ·  PA tolerance {PA_TOL}")
    encoder = SceneEncoder(device=device)
    print("[data] encoding clean silver movies...")
    rows = load_clean_silver(encoder, device)
    print(f"[data] {len(rows)} clean silver movies -> {len(rows)*5} per-TP decisions per seed")

    res = {}
    for mode in ("real", "zeros"):
        Hs, Ds = [], []
        for s in range(args.seeds):
            h, d = cv(rows, mode, s, device)
            Hs.append(h); Ds.append(d)
            print(f"  {mode:<6} seed {s}: PA={h.mean():.3f}  D={d.mean():.4f}")
        res[mode] = (np.stack(Hs), np.stack(Ds))

    print("\n" + "=" * 72)
    print(f"5-FOLD CV ON CLEAN SILVER — {args.seeds} seeds, {len(rows)*5} decisions each")
    print("=" * 72)
    for mode in ("real", "zeros"):
        H, D = res[mode]
        print(f"  {mode:<6} PA={H.mean():.3f} +-{H.mean(1).std():.3f}   "
              f"D={D.mean():.4f} +-{D.mean(1).std():.4f}")

    hr, hz = res["real"][0].mean(0), res["zeros"][0].mean(0)
    dr, dz = res["real"][1].mean(0), res["zeros"][1].mean(0)
    dpa, lo, hi = (hr - hz).mean(), *bootstrap_ci(hr - hz)
    verdict = "SIGNAL" if lo > 0 else ("HARMFUL — content makes it WORSE" if hi < 0 else "no signal")
    print(f"\n  CONTENT LIFT  dPA = {dpa:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]   {verdict}")
    dlo, dhi = bootstrap_ci(dr - dz)
    print(f"  CONTENT LIFT  dD  = {(dr-dz).mean():+.4f} 95% CI [{dlo:+.4f}, {dhi:+.4f}]  "
          f"(negative = better)")
    print("\n  Reading: real embeddings are compared to zeroed input on the SAME decisions.")
    print("  If dPA is not clearly > 0, the scene content is not contributing.")


if __name__ == "__main__":
    main()
