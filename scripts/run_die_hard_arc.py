"""One-shot verification: run the full emotional-arc + McKee-conformance pipeline
on Die Hard and save the plot as a PNG.

Usage (from the repo root):
    python scripts/run_die_hard_arc.py

This prints step-by-step progress so it's easy to spot where anything hangs or
errors. Expected runtime on a 4070 laptop: about 30–60 seconds after model
downloads (which happen once and cache locally).
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

# Make the outlier package importable whether this is run from the repo root or scripts/
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — script writes a PNG, doesn't pop a window
import matplotlib.pyplot as plt

import torch
from outlier.data import get_scenes, load_gold_labels
from outlier.embeddings import SceneEncoder
from outlier.emotion import EmotionScorer
from outlier.mckee import summarize, plain_language_read, plot_emotional_arc


MOVIE = "Die Hard"
OUT_PNG = ROOT / "notebooks" / "die_hard_arc.png"


def main() -> None:
    print("=" * 60)
    print(f"Outlier — emotional arc + McKee readout for: {MOVIE}")
    print("=" * 60)

    # ---- 1. Environment check
    print(f"\n[1/5] Environment")
    print(f"      Python:  {sys.version.split()[0]}")
    print(f"      PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"      CUDA:    {torch.cuda.get_device_name(0)}")
    else:
        print(f"      CUDA:    NOT available — running on CPU (slower but works)")

    # ---- 2. Load Die Hard's scenes and gold labels
    print(f"\n[2/5] Loading {MOVIE}")
    t0 = time.time()
    gold = load_gold_labels()
    if MOVIE not in gold:
        print(f"      ERROR: {MOVIE} not in gold labels. Available: {sorted(gold.keys())}")
        sys.exit(1)
    scenes = get_scenes(MOVIE)
    tp_scene_indices = [tp[0] for tp in gold[MOVIE]]  # first acceptable index per TP
    print(f"      scenes:        {len(scenes)}")
    print(f"      TP indices:    {tp_scene_indices}")
    print(f"      TP acceptable: {gold[MOVIE]}")
    print(f"      elapsed:       {time.time() - t0:.1f}s")

    # ---- 3. Load the emotion scorer (this triggers the DistilRoBERTa download on first run)
    print(f"\n[3/5] Loading DistilRoBERTa emotion classifier + VADER")
    t0 = time.time()
    scorer = EmotionScorer(use_vader=True)
    print(f"      device:  {scorer.device}")
    print(f"      elapsed: {time.time() - t0:.1f}s   (first run may take longer — model downloads ~330MB)")

    # ---- 4. Score every scene
    print(f"\n[4/5] Scoring {len(scenes)} scenes ({2 * len(scenes)} halves total)")
    t0 = time.time()
    scores = scorer.score_script(scenes)
    print(f"      elapsed: {time.time() - t0:.1f}s")
    print(f"      shapes:  " + ", ".join(f"{k}={v.shape}" for k, v in scores.items()))
    print(f"      sample @ scene 0:  valence_open={scores['valence_open'][0]:+.2f}"
          f"  valence_close={scores['valence_close'][0]:+.2f}"
          f"  intensity_close={scores['intensity_close'][0]:.2f}"
          f"  turn={bool(scores['turn'][0])}")
    print(f"      sample @ TP3 ({tp_scene_indices[2]}, Point of No Return):"
          f"  valence_open={scores['valence_open'][tp_scene_indices[2]]:+.2f}"
          f"  valence_close={scores['valence_close'][tp_scene_indices[2]]:+.2f}"
          f"  intensity_close={scores['intensity_close'][tp_scene_indices[2]]:.2f}"
          f"  turn={bool(scores['turn'][tp_scene_indices[2]])}")

    # ---- 5. Compute scene embeddings for the narrative-novelty lens
    print(f"\n[5/6] Computing MiniLM scene embeddings (for narrative-novelty escalation lens)")
    t0 = time.time()
    encoder = SceneEncoder()  # frozen MiniLM
    scene_embs = encoder.encode_scenes(scenes)   # (n_scenes, 384)
    print(f"      shape:   {scene_embs.shape}")
    print(f"      elapsed: {time.time() - t0:.1f}s")

    # ---- 6. Compute the McKee-conformance signals (all four escalation lenses) + plot the arc
    print(f"\n[6/6] Computing McKee signals + plotting arc")
    summary = summarize(scores, tp_scene_indices, len(scenes), scene_embeddings=scene_embs)

    print("\n" + "-" * 60)
    print("McKee-conformance signals — Die Hard")
    print("-" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<38} {v:+.4f}")
        else:
            print(f"  {k:<38} {v}")

    print("\n" + "-" * 60)
    print("Plain-language read")
    print("-" * 60)
    print(plain_language_read(summary))

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_emotional_arc(
        scores,
        tp_scene_indices=tp_scene_indices,
        title=f"{MOVIE} — emotional arc across scenes",
        ax=ax,
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nArc plot saved to: {OUT_PNG}")

    print("\nDone.")


if __name__ == "__main__":
    main()
