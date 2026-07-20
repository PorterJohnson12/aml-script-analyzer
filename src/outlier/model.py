"""The trained turning-point head — a bi-LSTM over scene embeddings.

This is the one model in the project that gets trained from scratch. Both
sentence-transformer (MiniLM) and emotion (DistilRoBERTa) models are used off the
shelf. Here the input is a sequence of scene embeddings (from MiniLM) and the
output is, for each of the 5 turning points, a probability distribution across
the scenes — TP-k's predicted scene is `argmax` over that distribution.

Shapes:
    Input  scene_embs:  (batch, n_scenes, embed_dim=384)
    Output logits:      (batch, 5, n_scenes)   — softmax over the last dim gives per-TP distributions

Loss:
    For each TP, cross-entropy between the predicted softmax and a target that puts
    uniform mass on the label's set of acceptable scenes (e.g., TP3 = [99, 100] means
    the target is 0.5 at position 99, 0.5 at 100, 0 elsewhere). Total loss is the
    mean of the 5 per-TP cross-entropies, then mean-reduced over the batch.

Overfit-a-single-batch test:
    `overfit_batch(...)` runs the training loop on 2–5 movies for a few hundred
    epochs and returns the loss curve. If the pipeline is wired correctly, loss
    should approach zero on this tiny set. See `notebooks/01_data_validation.ipynb`.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_TPS = 5


# ------------------------------------------------------------------ positional encoding


def _sinusoidal_pe(n_scenes: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal positional encoding → (n_scenes, d_model).

    Built on the fly rather than cached in a buffer: scripts vary from 31 to 230
    scenes and we train one script at a time, so there is no fixed max length.

    Encodes ABSOLUTE scene index. Note that a bi-LSTM gets something strictly
    richer for free — reading in both directions lets it infer position relative
    to the script's end (i and n-i), whereas this only supplies i. That asymmetry
    is a real architectural difference between the two heads, not a bug.
    """
    pos = torch.arange(n_scenes, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d_model)
    pe = torch.zeros(n_scenes, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


# ------------------------------------------------------------------ model


class TPFinder(nn.Module):
    """Sequence classifier over scene features → per-scene, per-TP logits.

    Supports two head architectures:
        head="bi-lstm"     — bidirectional LSTM (default; original architecture)
        head="transformer" — small Transformer encoder

    `input_dim` parameter accepts arbitrary input feature dimension so callers can
    concatenate MiniLM embeddings (384-dim) with additional per-scene features
    (e.g. emotion vectors) before passing in.

    Positional encoding (transformer head only), via `pos_encoding`:
        `nn.TransformerEncoder` has NO positional encoding of its own, so with
        pos_encoding=False this head is *permutation-invariant* — shuffle the
        scene order and every per-scene logit is unchanged. It therefore cannot
        represent "scene 100 of 118" at all, and can only judge a scene by its
        content and the unordered set of its neighbours.

        That is a deliberate experimental arm, not an oversight. Turning points
        in this dataset are overwhelmingly predicted by position, so the two
        settings isolate the two information sources:
            pos_encoding=False → content only  (cannot see order)
            pos_encoding=True  → content + position
        Paired with a zeroed-input bi-LSTM (position only), these complete a
        2x2 over {position, content}. Do not "fix" the False arm.

        The flag is inert for head="bi-lstm", whose recurrence already carries
        order.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden: int = 128,
        n_layers: int = 1,
        n_tps: int = N_TPS,
        dropout: float = 0.1,
        head: str = "bi-lstm",
        pos_encoding: bool = False,
        # Back-compat: allow embed_dim as alias for input_dim
        embed_dim: int | None = None,
    ):
        super().__init__()
        if embed_dim is not None:
            input_dim = embed_dim  # legacy call signature
        self.head_type = head
        self.pos_encoding = bool(pos_encoding) and head == "transformer"
        if head == "bi-lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden,
                num_layers=n_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            output_dim = 2 * hidden
            self.input_proj = None
        elif head == "transformer":
            # Project to hidden dim (Transformer needs consistent d_model)
            self.input_proj = nn.Linear(input_dim, hidden) if input_dim != hidden else nn.Identity()
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            output_dim = hidden
        else:
            raise ValueError(f"unknown head: {head!r} (expected 'bi-lstm' or 'transformer')")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, n_tps)

    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """scene_features: (batch, n_scenes, input_dim) → logits (batch, n_tps, n_scenes)."""
        if self.head_type == "bi-lstm":
            h, _ = self.encoder(scene_features)         # (batch, n_scenes, 2*hidden)
        else:  # transformer
            h = self.input_proj(scene_features)         # (batch, n_scenes, hidden)
            if self.pos_encoding:
                pe = _sinusoidal_pe(h.shape[1], h.shape[2], h.device)
                h = h + pe.unsqueeze(0)                 # broadcast over batch
            h = self.encoder(h)
        h = self.dropout(h)
        logits = self.classifier(h)                     # (batch, n_scenes, n_tps)
        return logits.transpose(1, 2)                   # (batch, n_tps, n_scenes)


# ------------------------------------------------------------------ target construction


def make_target(tp_scenes: Sequence[int], n_scenes: int, device: torch.device) -> torch.Tensor:
    """Uniform target distribution over the acceptable scene set for one TP.

    tp_scenes = [99, 100]  and  n_scenes=118  →  target[99]=target[100]=0.5, others 0.
    Falls back to a uniform-over-all-scenes target if no acceptable index fits.
    """
    target = torch.zeros(n_scenes, device=device)
    valid = [s for s in tp_scenes if 0 <= s < n_scenes]
    if not valid:
        # degenerate — put uniform mass across the whole script so loss is defined
        target[:] = 1.0 / n_scenes
        return target
    for s in valid:
        target[s] = 1.0
    return target / target.sum()


def tp_loss(logits: torch.Tensor, tps: Sequence[Sequence[int]]) -> torch.Tensor:
    """Cross-entropy between predicted softmax and multi-hot uniform target, meaned over the 5 TPs.

    logits: (1, n_tps, n_scenes)   for a single script
    tps:    list of 5 lists of acceptable scene indices
    """
    assert logits.shape[0] == 1, "one script at a time — variable lengths"
    n_scenes = logits.shape[-1]
    log_probs = F.log_softmax(logits[0], dim=-1)   # (n_tps, n_scenes)
    per_tp_losses = []
    for k in range(N_TPS):
        target = make_target(tps[k], n_scenes, device=logits.device)
        # cross-entropy = -sum(target * log_probs)
        per_tp_losses.append(-(target * log_probs[k]).sum())
    return torch.stack(per_tp_losses).mean()


# ------------------------------------------------------------------ overfit-a-single-batch


def overfit_batch(
    model: TPFinder,
    samples: List[Tuple[np.ndarray, Sequence[Sequence[int]]]],
    *,
    epochs: int = 300,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> List[float]:
    """Run the training loop on a tiny set of (scene_embs, tps) pairs for `epochs`.

    Returns the per-epoch loss curve (mean loss across the samples). If the pipeline
    is correct, loss should drop toward ~0 on this tiny fixed set.

    Args:
        model:   an instantiated TPFinder.
        samples: list of (scene_embs_np, tp_lists) — one per movie.
                 scene_embs_np shape: (n_scenes, embed_dim).
                 tp_lists: 5 lists of acceptable scene indices.
        epochs:  training passes over the fixed sample set.
        lr:      learning rate for Adam.
    """
    device = torch.device(device)
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # move samples to device once
    tensor_samples = [
        (torch.tensor(embs, dtype=torch.float32, device=device).unsqueeze(0), tps)
        for embs, tps in samples
    ]

    losses: List[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for embs, tps in tensor_samples:
            opt.zero_grad()
            logits = model(embs)              # (1, 5, n_scenes)
            loss = tp_loss(logits, tps)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= len(tensor_samples)
        losses.append(epoch_loss)
        if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:>3}  loss {epoch_loss:.4f}")
    return losses


# ------------------------------------------------------------------ inference


@torch.no_grad()
def predict_tps(model: TPFinder, scene_embs: np.ndarray, device: str | torch.device = "cpu") -> List[int]:
    """Run inference on a single script → 5 predicted scene indices (argmax per TP)."""
    device = torch.device(device)
    model = model.to(device).eval()
    embs = torch.tensor(scene_embs, dtype=torch.float32, device=device).unsqueeze(0)
    logits = model(embs)                                 # (1, 5, n_scenes)
    preds = logits.argmax(dim=-1)[0].tolist()            # list[int] length 5
    return preds


# ------------------------------------------------------------------ emotion feature augmentation


def build_emotion_features(scene_scores: dict) -> np.ndarray:
    """Extract per-scene emotion feature vector for late-fusion input to the model.

    Returns np.ndarray shape (n_scenes, 11):
        cols 0-6:  probs_close (7-emotion distribution at scene close)
        col 7:     valence_open
        col 8:     valence_close
        col 9:     intensity_open
        col 10:    intensity_close

    Intended to be concatenated with MiniLM embeddings along the feature axis:
        augmented = np.concatenate([minilm_embs, emotion_features], axis=-1)  # (n_scenes, 384+11)
    """
    n = len(scene_scores["valence_open"])
    feats = np.zeros((n, 11), dtype=np.float32)
    feats[:, 0:7] = np.asarray(scene_scores["probs_close"], dtype=np.float32)
    feats[:, 7] = np.asarray(scene_scores["valence_open"], dtype=np.float32)
    feats[:, 8] = np.asarray(scene_scores["valence_close"], dtype=np.float32)
    feats[:, 9] = np.asarray(scene_scores["intensity_open"], dtype=np.float32)
    feats[:, 10] = np.asarray(scene_scores["intensity_close"], dtype=np.float32)
    return feats


# ------------------------------------------------------------------ full training loop with per-epoch eval


def train_full(
    model: TPFinder,
    train_data: List[Tuple[np.ndarray, Sequence[Sequence[int]]]],
    val_data: List[Tuple[np.ndarray, Sequence[Sequence[int]]]],
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    pa_tol: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
    mlflow_module=None,
) -> Tuple[List[float], List[dict]]:
    """Train `model` on `train_data`, evaluate on `val_data` every epoch.

    Args:
        model:        instantiated TPFinder.
        train_data:   list of (scene_features, tp_lists) — one per training movie.
        val_data:     list of (scene_features, tp_lists) — held-out gold movies.
        epochs:       number of passes over the full training set.
        lr:           learning rate for Adam.
        device:       "cuda" or "cpu".
        pa_tol:       tolerance for Partial Agreement (fraction of scene count).
        seed:         random seed for shuffling.
        verbose:      print per-epoch progress.
        mlflow_module: optional mlflow module (pass `mlflow` if you want per-epoch logging).

    Returns:
        loss_curve: per-epoch mean training loss
        val_curves: per-epoch dict of val metrics {TA, PA, D}
    """
    from .metrics import evaluate, mean_metrics  # avoid circular at import time

    device = torch.device(device)
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Move data to device once
    train_tensors = [
        (torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0), tps)
        for feats, tps in train_data
    ]
    val_tensors = [
        (torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0), tps, feats.shape[0])
        for feats, tps in val_data
    ]

    rng = np.random.default_rng(seed)
    loss_curve: List[float] = []
    val_curves: List[dict] = []

    # Track best-dev-PA epoch and its state_dict for checkpoint-restore at end.
    best_state = None
    best_pa = -1.0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(len(train_tensors))
        running_loss = 0.0
        for i in idx:
            feats, tps = train_tensors[i]
            opt.zero_grad()
            logits = model(feats)
            loss = tp_loss(logits, tps)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        running_loss /= len(train_tensors)
        loss_curve.append(running_loss)

        # Evaluate on val_data (gold)
        model.eval()
        per_movie_metrics = []
        with torch.no_grad():
            for feats, tps, n_scenes in val_tensors:
                logits = model(feats)
                preds = logits.argmax(dim=-1)[0].tolist()
                per_movie_metrics.append(evaluate(preds, tps, n_scenes, tol=pa_tol))
        val_avg = mean_metrics(per_movie_metrics)
        val_curves.append(val_avg)

        # Track best-dev checkpoint (by PA — the primary metric)
        if not np.isnan(val_avg["PA"]) and val_avg["PA"] > best_pa:
            best_pa = float(val_avg["PA"])
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(
                f"  epoch {epoch + 1:>3}/{epochs}  "
                f"train_loss={running_loss:.4f}  "
                f"val TA={val_avg['TA']:.3f}  PA={val_avg['PA']:.3f}  D={val_avg['D']:.4f}"
            )

        if mlflow_module is not None:
            mlflow_module.log_metric("train_loss", running_loss, step=epoch)
            mlflow_module.log_metric("val_TA", float(val_avg["TA"]), step=epoch)
            mlflow_module.log_metric("val_PA", float(val_avg["PA"]), step=epoch)
            mlflow_module.log_metric("val_D", float(val_avg["D"]), step=epoch)

    # Restore best-dev-PA checkpoint before returning — this is what downstream code
    # (e.g. gold evaluation) will see. Uses dev-best for model selection instead of
    # final-epoch, which is the honest way to use the dev split.
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  [checkpoint] restored best-dev-PA state from epoch {best_epoch + 1} (PA={best_pa:.3f})")

    return loss_curve, val_curves
