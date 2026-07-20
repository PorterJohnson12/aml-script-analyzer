"""M3A1 experiments — trained TP models + rigorous baselines, all MLFlow-tracked.

Trained experiments (all logged to MLFlow; gold touched once per run):
    baseline           bi-LSTM 1 layer, hidden 128, MiniLM only
    larger             bi-LSTM 2 layers, hidden 256, MiniLM only
    emotion_augmented  bi-LSTM 1 layer, hidden 128, MiniLM + standardized emotion features
    zeros_input        CONTROL — baseline architecture trained on ZEROED input vectors.
                       Position without content. If this ties the real-embedding runs,
                       the model only learned a positional prior.
    transformer_nopos  CONTROL — Transformer with no positional encoding, so it is
                       permutation-invariant. Content without position.
    transformer_pos    Transformer + sinusoidal PE. A genuinely different architecture
                       family with position restored.

The two controls bracket the bi-LSTM on both axes, giving a 2x2 over {position, content}:

                       no content            has content
                    +-----------------+---------------------+
      no position   | (uniform)       | transformer_nopos   |
     has position   | zeros_input     | baseline            |
                    +-----------------+---------------------+

Baselines (no training, evaluated on gold directly):
    positional_prior   predict mean silver TP position for every movie (5 constants)
    random             predict random scene indices
    uniform_spaced     predict evenly spaced scenes

Reporting improvements over v1:
    - tol = 0.05, not 0.02  (SEE PA_TOL — this constant is NOT yet verified against
      the paper and it moves the numbers a lot; treat headline PA accordingly)
    - Silver split into train (61) / dev (12); gold (15) held out until final eval
    - Dev-best checkpoint restored before gold eval (train_full handles this)
    - Emotion features standardized to unit variance to match MiniLM scale
    - Movie-name aliasing recovers Star Wars I (the other two aliases resolve to
      folders that segment to 0 scenes / labels out of range, so net +1)
    - Comparison table lists every baseline alongside every trained model
    - Per-run gold predictions dumped to docs/week3/results.json for figures

Usage:
    python scripts/run_experiments.py                   # all experiments + baselines
    python scripts/run_experiments.py --skip-baselines  # trained only

After the run:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1    # http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import mlflow

from outlier.data import get_scenes, load_gold_labels, load_silver_labels, list_movies
from outlier.embeddings import SceneEncoder
from outlier.emotion import EmotionScorer
from outlier.model import TPFinder, train_full, predict_tps, build_emotion_features
from outlier.metrics import evaluate, mean_metrics


EXPERIMENT_NAME = "outlier-tp-finder"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

# PA tolerance (5% of scene count). Commonly-cited value; not confirmed verbatim
# against the TRIPOD paper. Note the theoretical baseline swings substantially with
# this constant (PA 0.13 at tol=0.02 to PA 0.61 at tol=0.1) so any absolute claim
# should be reported alongside the tol used.
PA_TOL = 0.05

# Fixed seed for the silver train/dev split so runs are comparable.
SPLIT_SEED = 1729
DEV_HOLDOUT = 12   # of ~81 silver → ~15% dev, ~66 train


# ------------------------------------------------------------------ silver name aliasing


# SUMMER's silver labels use slightly different movie names than the ScriptBase folders.
# Real folder names (verified by directory inspection). Even with these corrected, only
# Star Wars is recovered — 48 Hrs has n=31 scenes but silver labels reach index 31
# (out of range), and When Harry Met Sally has 0 valid scenes. Net gain: +1 movie.
SILVER_NAME_ALIASES = {
    "48 Hrs": "48 Hrs_",
    "Star Wars Episode I- The Phantom Menace": "Star Wars Episode I_ The Phantom Menace",
    "When Harry Met Sally": "When Harry Met Sally___",
}


# ------------------------------------------------------------------ configs


EXPERIMENT_CONFIGS = {
    "baseline": {
        "description": "bi-LSTM 1 layer, hidden 128, MiniLM only",
        "model_kwargs": {"input_dim": 384, "hidden": 128, "n_layers": 1, "dropout": 0.1, "head": "bi-lstm"},
        "use_emotion": False,
        "epochs": 30,
        "lr": 1e-3,
    },
    "larger": {
        "description": "bi-LSTM 2 layers, hidden 256, MiniLM only",
        "model_kwargs": {"input_dim": 384, "hidden": 256, "n_layers": 2, "dropout": 0.2, "head": "bi-lstm"},
        "use_emotion": False,
        "epochs": 30,
        "lr": 1e-3,
    },
    "emotion_augmented": {
        "description": "bi-LSTM 1 layer, hidden 128, MiniLM + standardized 11-dim emotion features",
        "model_kwargs": {"input_dim": 384 + 11, "hidden": 128, "n_layers": 1, "dropout": 0.1, "head": "bi-lstm"},
        "use_emotion": True,
        "epochs": 30,
        "lr": 1e-3,
    },
    # DIAGNOSTIC CONTROL: same architecture as baseline, trained with input vectors zeroed.
    # If this ties or beats real-embedding trained models, the model is only learning
    # positional prior (script length + sequence position), not content signal.
    "zeros_input": {
        "description": "bi-LSTM 1 layer, hidden 128, INPUT ZEROED (content-signal control)",
        "model_kwargs": {"input_dim": 384, "hidden": 128, "n_layers": 1, "dropout": 0.1, "head": "bi-lstm"},
        "use_emotion": False,
        "zero_input": True,
        "epochs": 30,
        "lr": 1e-3,
    },
    # DIAGNOSTIC CONTROL (the other axis): a Transformer encoder with no positional
    # encoding is permutation-invariant — it sees scene content but cannot know scene
    # ORDER. Where zeros_input is position-without-content, this is content-without-
    # position. Together they bracket the bi-LSTM, which has both.
    "transformer_nopos": {
        "description": "Transformer 1 layer, hidden 128, NO positional encoding (content-only control)",
        "model_kwargs": {"input_dim": 384, "hidden": 128, "n_layers": 1, "dropout": 0.1,
                         "head": "transformer", "pos_encoding": False},
        "use_emotion": False,
        "epochs": 30,
        "lr": 1e-3,
    },
    # Genuinely different architecture family with position restored — the fair
    # Transformer-vs-bi-LSTM comparison. Sinusoidal PE supplies absolute scene index;
    # the bi-LSTM infers position from both directions, so a gap here is a real
    # architectural difference rather than a defect.
    "transformer_pos": {
        "description": "Transformer 1 layer, hidden 128, sinusoidal positional encoding",
        "model_kwargs": {"input_dim": 384, "hidden": 128, "n_layers": 1, "dropout": 0.1,
                         "head": "transformer", "pos_encoding": True},
        "use_emotion": False,
        "epochs": 30,
        "lr": 1e-3,
    },
}


# ------------------------------------------------------------------ data prep


def _valid_silver_movies(silver: dict) -> tuple[list[str], dict]:
    """Return (movie names filtered by valid script + labels-in-range, silver dict remapped to folder names)."""
    on_disk = set(list_movies())
    out_names: list[str] = []
    out_silver: dict = {}
    for m, tps in silver.items():
        folder = SILVER_NAME_ALIASES.get(m, m)
        if folder not in on_disk:
            continue
        n = len(get_scenes(folder))
        if n == 0:
            continue
        if max(max(tp) for tp in tps) >= n:
            continue
        out_names.append(folder)
        out_silver[folder] = tps
    return out_names, out_silver


def _tps_are_distinct(tps: list) -> bool:
    """Return True iff all 5 TP scene sets are distinct (no collapsed labels).

    SUMMER's synopsis-projection sometimes collapses 2+ TPs onto the same scene
    (Gran Torino: TP1=TP2=TP3=25). Training on such labels asks the model to
    produce different distributions for identical targets — an unsolvable
    optimization. This filter keeps only movies with 5 distinct scene sets.
    """
    scene_sets = [tuple(sorted(tp)) for tp in tps]
    return len(set(scene_sets)) == 5


def filter_distinct_tp_movies(silver_names: list[str], silver: dict) -> list[str]:
    """Keep only silver movies where all 5 TP scene sets are distinct."""
    return [m for m in silver_names if _tps_are_distinct(silver[m])]


def prepare_movies(movies: list[str], labels: dict, encoder: SceneEncoder,
                   scorer: EmotionScorer | None, verbose: bool = True) -> list:
    """For each movie return (minilm_embs, emotion_features_or_None, tp_lists, n_scenes)."""
    prepared = []
    for i, m in enumerate(movies, 1):
        scenes = get_scenes(m)
        n = len(scenes)
        if n == 0:
            continue
        embs = encoder.encode_scenes(scenes)
        emo = None
        if scorer is not None:
            scene_scores = scorer.score_script(scenes)
            emo = build_emotion_features(scene_scores)
        if verbose and i % 10 == 0:
            print(f"    prepared {i:>3}/{len(movies)}")
        prepared.append((embs, emo, labels[m], n))
    return prepared


def features_for(prep_entry, use_emotion: bool,
                 emotion_mean: np.ndarray | None = None,
                 emotion_std: np.ndarray | None = None,
                 zero_input: bool = False) -> np.ndarray:
    """Return the input feature matrix.

    Args:
        use_emotion: concatenate standardized emotion features.
        zero_input:  zero out the resulting feature matrix (control experiment).
                     Model still sees a sequence of the correct length, but with
                     no content — isolates positional/length information from
                     content information.
    """
    embs, emo, _, _ = prep_entry
    if use_emotion:
        if emo is None:
            raise ValueError("emotion features requested but not computed")
        e = emo.copy()
        if emotion_mean is not None and emotion_std is not None:
            e = (e - emotion_mean) / (emotion_std + 1e-8)
        feats = np.concatenate([embs, e], axis=-1).astype(np.float32)
    else:
        feats = embs.astype(np.float32)
    if zero_input:
        feats = np.zeros_like(feats)
    return feats


def compute_emotion_norm(train_prep: list) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of emotion features across ALL training scenes for standardization."""
    all_emo = np.concatenate([e[1] for e in train_prep if e[1] is not None], axis=0)
    return all_emo.mean(axis=0), all_emo.std(axis=0)


# ------------------------------------------------------------------ baselines


def eval_predictions(pred_lists: list[list[int]], val_prep: list, tol: float = PA_TOL) -> dict:
    """Given a list of predicted TP scene indices per movie, compute mean TA/PA/D on val."""
    per_movie = []
    for preds, entry in zip(pred_lists, val_prep):
        _, _, tps, n = entry
        per_movie.append(evaluate(preds, tps, n, tol=tol))
    return mean_metrics(per_movie)


def baseline_positional_prior(train_prep: list, val_prep: list) -> tuple[dict, list[list[int]]]:
    """Predict fixed positions: for each TP, the mean fractional position across silver training set."""
    per_tp = {i: [] for i in range(5)}
    for _, _, tps, n in train_prep:
        if n == 0: continue
        for k, tp in enumerate(tps):
            per_tp[k].append(tp[0] / n)
    mean_pos = [float(np.mean(per_tp[k])) for k in range(5)]
    print(f"    positional prior fractions: {[f'{p:.3f}' for p in mean_pos]}")
    preds = []
    for _, _, _, n in val_prep:
        preds.append([int(round(mp * (n - 1))) for mp in mean_pos])
    return eval_predictions(preds, val_prep), preds


def baseline_random(val_prep: list, seed: int = 42) -> tuple[dict, list[list[int]]]:
    """Predict random scene indices."""
    rng = np.random.default_rng(seed)
    preds = []
    for _, _, _, n in val_prep:
        preds.append(sorted(rng.integers(0, n, size=5).tolist()))
    return eval_predictions(preds, val_prep), preds


def baseline_uniform(val_prep: list) -> tuple[dict, list[list[int]]]:
    """Predict evenly-spaced positions (1/6, 2/6, ..., 5/6)."""
    fractions = [1/6, 2/6, 3/6, 4/6, 5/6]
    preds = []
    for _, _, _, n in val_prep:
        preds.append([int(round(f * (n - 1))) for f in fractions])
    return eval_predictions(preds, val_prep), preds


# ------------------------------------------------------------------ trained-model runners


def run_trained(name: str, config: dict, train_data, dev_data, gold_data, device: str) -> dict:
    """Train a model, evaluate on dev per-epoch, report final metrics on both dev and gold."""
    print("\n" + "=" * 72)
    print(f"TRAINED EXPERIMENT: {name}")
    print(f"description: {config['description']}")
    print("=" * 72)

    with mlflow.start_run(run_name=name):
        mlflow.log_param("kind", "trained")
        mlflow.log_param("experiment_name", name)
        mlflow.log_param("description", config["description"])
        mlflow.log_param("epochs", config["epochs"])
        mlflow.log_param("lr", config["lr"])
        mlflow.log_param("use_emotion_features", config["use_emotion"])
        mlflow.log_param("pa_tol", PA_TOL)
        for k, v in config["model_kwargs"].items():
            mlflow.log_param(f"model_{k}", v)

        torch.manual_seed(42)
        model = TPFinder(**config["model_kwargs"])
        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("n_params", n_params)
        print(f"model params: {n_params:,}")

        t0 = time.time()
        loss_curve, val_curves = train_full(
            model, train_data=train_data, val_data=dev_data,
            epochs=config["epochs"], lr=config["lr"], device=device,
            pa_tol=PA_TOL, verbose=True, mlflow_module=mlflow,
        )
        elapsed = time.time() - t0

        # Final metrics on DEV (for model selection)
        final_dev = val_curves[-1]
        best_dev_epoch = int(np.argmax([v["PA"] for v in val_curves]))
        best_dev = val_curves[best_dev_epoch]

        # Final metrics on GOLD (touched once, at end)
        model.eval()
        preds_gold = []
        with torch.no_grad():
            for feats, tps, n in gold_data:
                t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(t)
                preds_gold.append(logits.argmax(dim=-1)[0].tolist())
        gold_metrics = eval_predictions(preds_gold, [(None, None, tps, n) for _, tps, n in gold_data], tol=PA_TOL)

        mlflow.log_metric("dev_final_TA", float(final_dev["TA"]))
        mlflow.log_metric("dev_final_PA", float(final_dev["PA"]))
        mlflow.log_metric("dev_final_D", float(final_dev["D"]))
        mlflow.log_metric("dev_best_PA", float(best_dev["PA"]))
        mlflow.log_metric("dev_best_epoch", best_dev_epoch + 1)
        mlflow.log_metric("gold_final_TA", float(gold_metrics["TA"]))
        mlflow.log_metric("gold_final_PA", float(gold_metrics["PA"]))
        mlflow.log_metric("gold_final_D", float(gold_metrics["D"]))
        mlflow.log_metric("wall_seconds", elapsed)

        print(f"\n{name} results:")
        print(f"  DEV  final: TA={final_dev['TA']:.3f}  PA={final_dev['PA']:.3f}  D={final_dev['D']:.4f}")
        print(f"  DEV  best:  PA={best_dev['PA']:.3f} @ epoch {best_dev_epoch+1}")
        print(f"  GOLD final: TA={gold_metrics['TA']:.3f}  PA={gold_metrics['PA']:.3f}  D={gold_metrics['D']:.4f}")

        return {
            "name": name, "kind": "trained", "n_params": n_params,
            "description": config["description"],
            "dev_final": final_dev, "dev_best": best_dev,
            "gold_final": gold_metrics, "wall_seconds": elapsed,
            # Kept for docs/week3/results.json so the figure script never has to
            # re-encode 88 screenplays (~185s) just to redraw a chart.
            "preds": preds_gold,
            "loss_curve": [float(x) for x in loss_curve],
            "dev_PA_curve": [float(v["PA"]) for v in val_curves],
        }


def run_baseline(name: str, description: str, gold_metrics: dict, preds: list | None = None,
                 n_params: int = 0) -> dict:
    """Log a baseline as an MLFlow run for the comparison table."""
    with mlflow.start_run(run_name=name):
        mlflow.log_param("kind", "baseline")
        mlflow.log_param("experiment_name", name)
        mlflow.log_param("description", description)
        mlflow.log_param("pa_tol", PA_TOL)
        mlflow.log_metric("gold_final_TA", float(gold_metrics["TA"]))
        mlflow.log_metric("gold_final_PA", float(gold_metrics["PA"]))
        mlflow.log_metric("gold_final_D", float(gold_metrics["D"]))
    print(f"  {name:<22} TA={gold_metrics['TA']:.3f}  PA={gold_metrics['PA']:.3f}  D={gold_metrics['D']:.4f}")
    return {
        "name": name, "kind": "baseline", "n_params": n_params,
        "description": description,
        "dev_final": None, "dev_best": None,
        "gold_final": gold_metrics, "wall_seconds": 0,
        "preds": preds, "loss_curve": None, "dev_PA_curve": None,
    }


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENT_CONFIGS.keys()),
                    choices=list(EXPERIMENT_CONFIGS.keys()))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--clean-labels-only", action="store_true",
                    help="filter silver to only movies with 5 distinct TP scene sets "
                         "(drops the ~37%% with SUMMER label collapse)")
    args = ap.parse_args()

    if args.epochs is not None:
        for cfg in EXPERIMENT_CONFIGS.values():
            cfg["epochs"] = args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device: {device}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[setup] MLFlow tracking: {TRACKING_URI}")
    print(f"[setup] PA tolerance:    {PA_TOL} (matches TRIPOD paper)")

    gold = load_gold_labels()
    silver_all = load_silver_labels(collapse=True)
    silver_names, silver = _valid_silver_movies(silver_all)
    print(f"[data] {len(silver)} clean silver movies (after name aliasing)")
    print(f"[data] {len(gold)} gold movies")

    if args.clean_labels_only:
        before = len(silver_names)
        silver_names = filter_distinct_tp_movies(silver_names, silver)
        silver = {m: silver[m] for m in silver_names}
        n_collapsed = before - len(silver_names)
        print(f"[data] --clean-labels-only: dropped {n_collapsed}/{before} silver movies "
              f"with collapsed TP scene sets ({100 * n_collapsed / before:.0f}% of corpus)")
        print(f"[data] remaining clean silver: {len(silver_names)}")

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(silver_names))
    dev_names = [silver_names[i] for i in perm[:DEV_HOLDOUT]]
    train_names = [silver_names[i] for i in perm[DEV_HOLDOUT:]]
    print(f"[data] silver split (seed {SPLIT_SEED}): {len(train_names)} train / {len(dev_names)} dev")

    encoder = SceneEncoder(device=device)
    need_emotion = any(EXPERIMENT_CONFIGS[e]["use_emotion"] for e in args.experiments)
    scorer = EmotionScorer(device=device, use_vader=False) if need_emotion else None

    print(f"\n[data] encoding all movies (silver train + dev + gold)...")
    t0 = time.time()
    train_prep = prepare_movies(sorted(train_names), silver, encoder, scorer)
    dev_prep = prepare_movies(sorted(dev_names), silver, encoder, scorer)
    gold_prep = prepare_movies(sorted(gold.keys()), gold, encoder, scorer)
    print(f"[data] prep complete in {time.time() - t0:.1f}s")

    if need_emotion:
        emo_mean, emo_std = compute_emotion_norm(train_prep)
        print(f"[data] emotion feature standardization computed on train set")
    else:
        emo_mean = emo_std = None

    baselines = []
    if not args.skip_baselines:
        print("\n" + "=" * 72)
        print("BASELINES (no training) - evaluated directly on gold")
        print("=" * 72)
        m, p = baseline_positional_prior(train_prep, gold_prep)
        baselines.append(run_baseline("positional_prior",
            "predict mean silver TP position for every movie", m, p))
        m, p = baseline_random(gold_prep)
        baselines.append(run_baseline("random",
            "predict 5 random scene indices per movie", m, p))
        m, p = baseline_uniform(gold_prep)
        baselines.append(run_baseline("uniform_spaced",
            "predict evenly-spaced positions 1/6 ... 5/6", m, p))

    trained_results = []
    for name in args.experiments:
        cfg = EXPERIMENT_CONFIGS[name]
        zi = cfg.get("zero_input", False)
        train_data = [(features_for(e, cfg["use_emotion"], emo_mean, emo_std, zi), e[2]) for e in train_prep]
        dev_data = [(features_for(e, cfg["use_emotion"], emo_mean, emo_std, zi), e[2]) for e in dev_prep]
        gold_data = [(features_for(e, cfg["use_emotion"], emo_mean, emo_std, zi), e[2], e[3]) for e in gold_prep]
        trained_results.append(run_trained(name, cfg, train_data, dev_data, gold_data, device))

    print("\n\n" + "=" * 84)
    print("COMPARISON TABLE - GOLD metrics (touched once per run)")
    print("=" * 84)
    print(f"  {'name':<22} {'kind':<10} {'n_params':>10}  {'gold TA':>8} {'gold PA':>8} {'gold D':>8}")
    print("-" * 84)
    for r in baselines + trained_results:
        g = r["gold_final"]
        n = r["n_params"] if r["n_params"] else "-"
        print(f"  {r['name']:<22} {r['kind']:<10} {str(n):>10}  "
              f"{g['TA']:>8.3f} {g['PA']:>8.3f} {g['D']:>8.4f}")
    print("\nBaselines are the reference. Trained models are worth their compute only")
    print("if they clearly beat positional_prior AND zeros_input on multiple metrics.")

    # ---- Dump everything the Week-3 figures need, so make_week3_figures.py never
    # has to re-encode the corpus. Valence columns come from the emotion features we
    # already computed (build_emotion_features puts valence_open/close at cols 7/8),
    # which is what lets the Die Hard arc be drawn without a second DistilRoBERTa pass.
    results_path = ROOT / "docs" / "week3" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pa_tol": PA_TOL,
        "split_seed": SPLIT_SEED,
        "n_silver": len(silver_names),
        "n_train": len(train_names),
        "n_dev": len(dev_names),
        "n_gold": len(gold_prep),
        "clean_labels_only": bool(args.clean_labels_only),
        "gold": [
            {
                "movie": m,
                "n": e[3],
                "tps": [list(map(int, t)) for t in e[2]],
                "valence_open": (e[1][:, 7].tolist() if e[1] is not None else None),
                "valence_close": (e[1][:, 8].tolist() if e[1] is not None else None),
            }
            for m, e in zip(sorted(gold.keys()), gold_prep)
        ],
        "runs": {
            r["name"]: {
                "kind": r["kind"],
                "description": r["description"],
                "n_params": r["n_params"],
                "gold_final": r["gold_final"],
                "dev_final": r["dev_final"],
                "dev_best": r["dev_best"],
                "preds": r["preds"],
                "loss_curve": r["loss_curve"],
                "dev_PA_curve": r["dev_PA_curve"],
            }
            for r in baselines + trained_results
        },
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[out] wrote {results_path}")
    with mlflow.start_run(run_name="_results_bundle"):
        mlflow.log_artifact(str(results_path))

    print(f"\nMLFlow: mlflow ui --backend-store-uri {TRACKING_URI} --workers 1")


if __name__ == "__main__":
    main()
