"""Scene-level embeddings via a frozen pretrained sentence-transformer (MiniLM).

MiniLM (`all-MiniLM-L6-v2`) is loaded once and used off the shelf — no fine-tuning.
It converts each scene's raw text into a 384-dimensional vector encoding meaning.
That vector is what the trained bi-LSTM head consumes.

Typical usage:
    from outlier.embeddings import SceneEncoder
    from outlier.data import get_scenes

    enc = SceneEncoder()
    scenes = get_scenes("Die Hard")           # list[str]
    embs   = enc.encode_scenes(scenes)        # np.ndarray shape (N, 384)
"""
from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class SceneEncoder:
    """Thin wrapper around sentence-transformers' MiniLM.

    Args:
        model_name: HuggingFace model id. Defaults to `all-MiniLM-L6-v2` (22M params,
            384-dim output, fast on CPU or a modest GPU).
        device: "cuda", "cpu", or None (auto). None lets sentence-transformers pick.
        batch_size: mini-batch used when calling `encode_scenes`.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_scenes(self, scenes: List[str]) -> np.ndarray:
        """Encode a list of scene texts. Returns np.ndarray of shape (n_scenes, dim).

        Long scenes are truncated by the tokenizer at the model's max input length
        (default 256 tokens for MiniLM); this is fine because we already segmented
        into scenes and rarely need the tail of a very long scene to characterize it.
        """
        if not scenes:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(
            scenes,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
