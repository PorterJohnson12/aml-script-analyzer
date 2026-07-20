"""Per-scene emotional signals from a pretrained emotion classifier + VADER.

Two pretrained models used off the shelf, both frozen — I train neither:
- j-hartmann/emotion-english-distilroberta-base — 7-emotion transformer classifier
- vaderSentiment — lexicon-based sentiment with a calibrated intensity signal

For each scene, the text is split by word count into "open" (first half) and
"close" (second half) halves. Each half is scored independently. That gives every
scene four raw signals — open-valence, close-valence, open-intensity, close-intensity —
plus a boolean for whether the scene "turned" (sign flip between the halves).

A note on what these signals are, and are not:

    Valence is a STATED PROXY for McKee's fuller notion of value. McKee's values
    are broader — love/hate, freedom/slavery, truth/lie, courage/cowardice — but
    every one of them shares a polar structure, and a sign flip is a sign flip
    regardless of which specific value is at stake. The proxy captures the
    countable mechanic McKee names ("a scene turns when the value's polarity
    flips") without pretending to detect which of his values is on the table.

    Intensity is a STATED PROXY for McKee's notion of stakes / degree. It comes
    from two independent sources: the emotion model's non-neutral probability
    mass, and VADER's compound magnitude. Reporting both lets a reader see
    where the two off-the-shelf tools agree and where they diverge.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# The 7 categories the model returns.
# Positive and negative are used for valence; "surprise" is affectively ambiguous
# so it is excluded from valence and instead informs intensity; "neutral" is
# subtracted from 1 to give intensity.
POSITIVE_EMOTIONS: Tuple[str, ...] = ("joy",)
NEGATIVE_EMOTIONS: Tuple[str, ...] = ("anger", "disgust", "fear", "sadness")


# ------------------------------------------------------------------ helpers


def split_scene_halves(text: str) -> Tuple[str, str]:
    """Split a scene's text into (open_half, close_half) by word count.

    Empty or single-word scenes return the same text for both halves — the resulting
    "turn" flag will be False by definition (no shift), which is the right behavior.
    """
    words = text.split()
    if len(words) < 2:
        return text, text
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def emotion_to_valence(probs: Dict[str, float]) -> float:
    """Convert a 7-emotion probability dict to signed valence.

    valence = P(joy) − [P(anger) + P(disgust) + P(fear) + P(sadness)]

    Range: roughly [-1, +1]. Positive scenes read joyful/hopeful; negative scenes
    read angry/afraid/sad. Surprise and neutral are excluded (they inform intensity
    instead — see `emotion_to_intensity`).
    """
    positive = sum(probs.get(e, 0.0) for e in POSITIVE_EMOTIONS)
    negative = sum(probs.get(e, 0.0) for e in NEGATIVE_EMOTIONS)
    return float(positive - negative)


def emotion_to_intensity(probs: Dict[str, float]) -> float:
    """Convert a 7-emotion probability dict to a 0-1 intensity score.

    intensity = 1 − P(neutral)

    High = strongly emotional scene, regardless of which emotion.
    Low = neutral, informational, or descriptive text (a lot of screenplay
    action-line text lands here).
    """
    return float(1.0 - probs.get("neutral", 0.0))


# ------------------------------------------------------------------ scorer


class EmotionScorer:
    """Batch-scores scene halves with DistilRoBERTa + VADER.

    Loads both pretrained models once at construction. Neither is trained.

    Args:
        device: "cuda", "cpu", or None (auto-detect a CUDA device if available).
        use_vader: if True, also compute VADER compound scores alongside emotion valence.
        batch_size: batch size for the transformer pipeline.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        use_vader: bool = True,
        batch_size: int = 32,
        turn_threshold: float = 0.25,
    ):
        """
        Args:
            device:         "cuda", "cpu", or None (auto-detect).
            use_vader:      also compute VADER compound scores.
            batch_size:     batch size for the transformer pipeline.
            turn_threshold: magnitude of |close - open| valence-shift required to count as a
                            "turn" even without a polarity flip. McKee's canonical turn is a
                            polarity flip, but he also emphasizes "perceptible significance"
                            — a scene going from -0.9 to -0.3 (bad → less bad) is a real value
                            shift even without crossing zero. Default 0.25 matches roughly a
                            quarter of the valence range and lands in the "clearly meaningful
                            shift" zone on face-validity spot-checks.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline_device = 0 if device == "cuda" else -1
        self.pipe = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            device=pipeline_device,
            top_k=None,          # return probabilities for ALL classes, not just argmax
            truncation=True,
        )
        self.vader = SentimentIntensityAnalyzer() if use_vader else None
        self.batch_size = batch_size
        self.device = device
        self.turn_threshold = float(turn_threshold)

    def _emotion_probs(self, texts: List[str]) -> List[Dict[str, float]]:
        """Run the emotion pipeline over a list of texts → list of {label: prob} dicts."""
        # transformers returns [[{label,score}, ...], [{label,score}, ...], ...]
        results = self.pipe(texts, batch_size=self.batch_size, truncation=True)
        return [
            {item["label"]: float(item["score"]) for item in per_text}
            for per_text in results
        ]

    def score_script(self, scenes: List[str]) -> Dict[str, np.ndarray]:
        """Score a whole script's scenes at once.

        Returns a dict of numpy arrays, one entry per scene:
            valence_open      shape (n,)   signed
            valence_close     shape (n,)   signed
            intensity_open    shape (n,)   0-1
            intensity_close   shape (n,)   0-1
            magnitude         shape (n,)   |close - open|
            sign_flip         shape (n,)   bool, True iff polarity crossed zero (strict McKee canonical)
            turn              shape (n,)   bool, True iff sign_flip OR magnitude > turn_threshold
                                          (broader — matches McKee's "perceptible significance")

        If use_vader was True at construction, also:
            vader_open        shape (n,)   VADER compound score, signed [-1, +1]
            vader_close       shape (n,)   VADER compound score, signed [-1, +1]
        """
        # Build a flat list of halves: [open_0, close_0, open_1, close_1, ...]
        halves: List[str] = []
        for s in scenes:
            o, c = split_scene_halves(s)
            halves.append(o)
            halves.append(c)

        all_probs = self._emotion_probs(halves)
        open_probs = all_probs[0::2]
        close_probs = all_probs[1::2]

        v_open = np.array([emotion_to_valence(p) for p in open_probs], dtype=np.float32)
        v_close = np.array([emotion_to_valence(p) for p in close_probs], dtype=np.float32)
        i_open = np.array([emotion_to_intensity(p) for p in open_probs], dtype=np.float32)
        i_close = np.array([emotion_to_intensity(p) for p in close_probs], dtype=np.float32)
        magnitude = np.abs(v_close - v_open)

        # Preserve raw 7-emotion probability arrays for late-fusion analysis
        # (kept for pattern-finding across specific emotion channels, feature augmentation, etc.)
        # Order: [anger, disgust, fear, joy, neutral, sadness, surprise]
        emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        probs_open = np.array(
            [[p.get(e, 0.0) for e in emotion_labels] for p in open_probs],
            dtype=np.float32,
        )
        probs_close = np.array(
            [[p.get(e, 0.0) for e in emotion_labels] for p in close_probs],
            dtype=np.float32,
        )

        # Strict sign flip (McKee's canonical "polarity flip" test).
        s_open = np.sign(v_open)
        s_close = np.sign(v_close)
        sign_flip = (s_open != s_close) & (s_open != 0) & (s_close != 0)

        # Broader turn definition: sign flip OR magnitude exceeds threshold.
        # McKee's "perceptible significance" — a scene going from -0.9 to -0.3 is a real
        # value shift even without crossing zero, because the CHARGE of the value moved.
        turn = sign_flip | (magnitude > self.turn_threshold)

        result: Dict[str, np.ndarray] = {
            "valence_open": v_open,
            "valence_close": v_close,
            "intensity_open": i_open,
            "intensity_close": i_close,
            "magnitude": magnitude,
            "sign_flip": sign_flip,
            "turn": turn,
            "probs_open": probs_open,                  # (n_scenes, 7)  raw 7-emotion probabilities
            "probs_close": probs_close,                # (n_scenes, 7)
            "emotion_labels": np.array(emotion_labels),
        }

        if self.vader is not None:
            vader_open = np.array(
                [self.vader.polarity_scores(halves[2 * i])["compound"] for i in range(len(scenes))],
                dtype=np.float32,
            )
            vader_close = np.array(
                [self.vader.polarity_scores(halves[2 * i + 1])["compound"] for i in range(len(scenes))],
                dtype=np.float32,
            )
            result["vader_open"] = vader_open
            result["vader_close"] = vader_close

        return result
