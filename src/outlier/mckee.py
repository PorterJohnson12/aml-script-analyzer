"""McKee-conformance signals — the analysis layer that runs on top of emotion.py.

Four signals per script, each traceable to a specific principle McKee names in
*Story*. None are learned. All are computed from the per-scene valence /
intensity / magnitude / turn arrays that `emotion.EmotionScorer.score_script`
produces, optionally combined with turning-point positions.

    1.  flat_scene_rate   — fraction of scenes that did NOT turn.
                            McKee calls these "nonevents" (Ch. 3, Ch. 11):
                              "If the two notations are the same, the activity
                               between them is a nonevent."

    2.  escalation        — slope and std of turn magnitudes across the script,
                            plus a stricter slope computed only at the act-climax
                            positions. Positive slope = magnitudes grow toward
                            the end, which is McKee's Law of Progression (Ch. 9):
                              "A story must not retreat to actions of lesser
                               quality or magnitude, but move progressively
                               forward to a final action beyond which the audience
                               cannot imagine another."

    3.  alternation       — whether the last two act-climax charges alternate
                            sign. Verbatim from Ch. 9 p. 225:
                              "You cannot set up an up-ending with an up-ending."

    4.  pacing            — inter-TP gaps as fractions of the script. McKee argues
                            act lengths should NOT be uniform; the pacing signal
                            captures this without prescribing a target shape.

Every signal is a STATED PROXY. It captures the countable mechanic McKee names,
but it does not attempt the interpretive judgment he would apply to the same
script (whether the crisis is a real dilemma, whether the antagonist is forcing
something true, etc.). Those live in the capstone.

---

IMPORTANT: emotion valence ≠ McKee value polarity

McKee defines "positive" and "negative" as which side of a value binary the
CHARACTER is on (life/death, love/hate, freedom/slavery, justice/injustice,
victory/defeat, etc. — Ch. 3 p. 34). His "positive" is not the audience's
happy feeling; it's the character being on the desirable side of a specific
value binary.

Our emotion pipeline (DistilRoBERTa) reads AUDIENCE-SIDE affective valence
— "how does this text feel." These two signals CORRELATE in most scenes but
DIVERGE in specific known cases:

    Violent-but-victorious climaxes — e.g. Die Hard's TP5 (Hans falling
        off Nakatomi Plaza). The scene TEXT is violent (dying character,
        chaotic action) so emotion valence reads strongly negative. But
        McKee's "victory/defeat" value at stake for McClane flips from
        NEGATIVE (about to lose) to POSITIVE (evil defeated).

    Melancholic victories — e.g. a bittersweet reunion. Text reads sad
        but the McKee value ("connection") is positive.

    Successful tragedy — a character's willing sacrifice for the greater
        good. Text reads devastating but the McKee value ("meaningful
        sacrifice") is positive.

When our alternation-compliance check flags a script as VIOLATED because
TP5's close-valence is negative, that reading may be:
    (a) genuine — the script actually ends on the same charge as TP4
    (b) proxy limitation — the emotion model misread a violent-but-victorious
        climax as negative, when the McKee value flip is actually positive

The capstone's LLM-weak-labeling layer would resolve this by asking a
story-aware model "which value is at stake in this scene, and which side
does the character end on" — that's exactly the interpretive question
DistilRoBERTa cannot answer but a context-aware LLM can.

For PRO322, we report the proxy honestly and name the limit at every
alternation-check output.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ 1. flat scenes


def flat_scene_rate(scene_scores: Dict[str, np.ndarray]) -> float:
    """Fraction of scenes where valence did NOT flip sign — McKee's nonevents.

    Lower is more McKee-conformant. A perfectly "turning" script would have
    flat_scene_rate = 0.0; a script full of exposition and nonevents would trend
    toward 1.0.

    Returns NaN if the script has no scenes.
    """
    turns = scene_scores["turn"]
    n = int(len(turns))
    if n == 0:
        return float("nan")
    return float((~turns.astype(bool)).sum() / n)


# ------------------------------------------------------------------ 2. escalation


def escalation(
    scene_scores: Dict[str, np.ndarray],
    tp_scene_indices: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Turn-magnitude trend across the script.

    Returns three metrics:

        slope             — linear regression slope of magnitude vs. scene index
                            across ALL scenes. Positive = magnitudes grow toward
                            the end of the script (McKee's Law of Progression).
        std               — standard deviation of the magnitudes. High = the
                            script has dynamic range; low = magnitudes are
                            uniformly small (flat throughout).
        act_climax_slope  — the same slope computed ONLY at the five act-climax
                            scene indices (if `tp_scene_indices` is given).
                            This is the strictest version of McKee's Law of
                            Progression: successive act climaxes, not successive
                            scenes, are what should escalate.

    All three are returned so the reader can see the different lenses.
    """
    magnitudes = np.asarray(scene_scores["magnitude"], dtype=np.float64)
    n = int(len(magnitudes))

    if n < 2:
        return {"slope": float("nan"), "std": float("nan"), "act_climax_slope": float("nan")}

    positions = np.arange(n, dtype=np.float64)
    slope, _ = np.polyfit(positions, magnitudes, 1)

    result: Dict[str, float] = {
        "slope": float(slope),
        "std": float(np.std(magnitudes)),
        "act_climax_slope": float("nan"),
    }

    if tp_scene_indices is not None:
        # Only include TPs that fit inside the script
        tps = [int(i) for i in tp_scene_indices if 0 <= int(i) < n]
        if len(tps) >= 2:
            tp_mags = magnitudes[np.array(tps)]
            tp_slope, _ = np.polyfit(np.arange(len(tps), dtype=np.float64), tp_mags, 1)
            result["act_climax_slope"] = float(tp_slope)

    return result


# ------------------------------------------------------------------ 3. alternation


def alternation_compliance(
    scene_scores: Dict[str, np.ndarray],
    tp4_scene: int,
    tp5_scene: int,
) -> Dict[str, float]:
    """Whether the last two act-climax charges alternate sign.

    McKee's rule (Ch. 9 p. 225, verbatim):
        "You cannot set up an up-ending with an up-ending."

    NOTE ON PROXY LIMIT: this check uses affective valence (emotion polarity of
    the scene text), not McKee's fuller value polarity (which side of the value
    binary the character is on). For scenes where these diverge — most notably
    violent-but-victorious climaxes where the emotion model reads chaotic action
    as negative even when the McKee value flip is positive — this check will
    report VIOLATION when a McKee-informed human would say the script complies.
    Report the number honestly and flag the limitation in the writeup. See the
    module docstring for a fuller treatment.

    Returns:
        compliant   — 1.0 if signs differ, 0.0 if they match, NaN if either TP is out of range.
        tp4_close   — TP4 scene's close-valence.
        tp5_close   — TP5 scene's close-valence.
    """
    v_close = np.asarray(scene_scores["valence_close"], dtype=np.float64)
    n = int(len(v_close))

    if not (0 <= tp4_scene < n and 0 <= tp5_scene < n):
        return {"compliant": float("nan"), "tp4_close": float("nan"), "tp5_close": float("nan")}

    tp4_v = float(v_close[tp4_scene])
    tp5_v = float(v_close[tp5_scene])
    signs_differ = (tp4_v > 0) != (tp5_v > 0)

    return {
        "compliant": float(signs_differ),
        "tp4_close": tp4_v,
        "tp5_close": tp5_v,
    }


# ------------------------------------------------------------------ 4. pacing


def pacing(tp_scene_indices: Sequence[int], total_scenes: int) -> Dict[str, float]:
    """Inter-TP gaps as fractions of the script.

    Returns six fractions summing to ~1.0:

        gap_open_to_tp1   — the "setup" length before TP1
        gap_tp1_to_tp2    — Act I proper
        gap_tp2_to_tp3    — Act IIa
        gap_tp3_to_tp4    — Act IIb
        gap_tp4_to_tp5    — Act III
        gap_tp5_to_end    — resolution

    Plus:
        act_length_std    — std of the FOUR middle gaps. Zero = uniform act
                            lengths (which McKee explicitly argues against —
                            each act should feel different in tempo).
    """
    tps = [int(i) for i in tp_scene_indices]
    n = int(total_scenes)
    if len(tps) != 5 or n <= 0:
        return {}

    boundaries = [0] + tps + [n]
    gaps = np.diff(boundaries) / n

    return {
        "gap_open_to_tp1": float(gaps[0]),
        "gap_tp1_to_tp2": float(gaps[1]),
        "gap_tp2_to_tp3": float(gaps[2]),
        "gap_tp3_to_tp4": float(gaps[3]),
        "gap_tp4_to_tp5": float(gaps[4]),
        "gap_tp5_to_end": float(gaps[5]),
        "act_length_std": float(np.std(gaps[1:5])),
    }


# ------------------------------------------------------------------ escalation lens 2: per-act impact


def per_act_impact(
    scene_scores: Dict[str, np.ndarray],
    tp_scene_indices: Sequence[int],
) -> Dict[str, float]:
    """Mean within-scene magnitude in each of the 5 acts.

    Acts are the scene ranges BETWEEN successive TPs:
        Act 1 = scenes [0        .. TP1)
        Act 2 = scenes [TP1      .. TP2)
        Act 3 = scenes [TP2      .. TP3)
        Act 4 = scenes [TP3      .. TP4)
        Act 5 = scenes [TP4      .. end]

    Growing mean-magnitude across acts = the film sustains bigger emotional shifts as it
    progresses. This is different from per-scene slope: a film can be flat scene-to-scene
    but still have its later acts running at higher average intensity than its earlier ones.
    """
    mags = np.asarray(scene_scores["magnitude"], dtype=np.float64)
    n = len(mags)
    tps = [int(i) for i in tp_scene_indices]
    if len(tps) != 5 or n == 0:
        return {}

    boundaries = [0] + tps + [n]
    result: Dict[str, float] = {}
    per_act = []
    for k in range(5):
        start, end = boundaries[k], boundaries[k + 1]
        if end > start:
            act_mean = float(mags[start:end].mean())
        else:
            act_mean = float("nan")
        result[f"act{k + 1}_mean_magnitude"] = act_mean
        per_act.append(act_mean)

    # Slope across the 5 act means — positive = sustained pressure widens over time
    clean = np.array([v for v in per_act if not np.isnan(v)], dtype=np.float64)
    if len(clean) >= 2:
        slope, _ = np.polyfit(np.arange(len(clean), dtype=np.float64), clean, 1)
        result["per_act_impact_slope"] = float(slope)
    else:
        result["per_act_impact_slope"] = float("nan")
    return result


# ------------------------------------------------------------------ escalation lens 3: reversal magnitude


def reversal_magnitude(
    scene_scores: Dict[str, np.ndarray],
    tp_scene_indices: Sequence[int],
) -> Dict[str, float]:
    """For each act climax, distance of its close-valence from the story's prior baseline.

    Baseline for TP-k = mean close-valence of all scenes BEFORE TP-k.
    reversal_at_tp_k = |close_valence(TP_k) - baseline_before_TP_k|

    Growing across k = each act climax swings farther from the story's "normal so far"
    than the previous one. This is what McKee calls a "major reversal" — an act climax
    doesn't just have high intensity, it *departs* from the story's established state.
    """
    v_close = np.asarray(scene_scores["valence_close"], dtype=np.float64)
    n = len(v_close)
    tps = [int(i) for i in tp_scene_indices]
    if len(tps) != 5 or n == 0:
        return {}

    result: Dict[str, float] = {}
    reversals = []
    for k, tp in enumerate(tps):
        if not (0 <= tp < n) or tp == 0:
            r = float("nan")
        else:
            baseline = float(v_close[:tp].mean())
            r = float(abs(v_close[tp] - baseline))
        result[f"reversal_at_tp{k + 1}"] = r
        reversals.append(r)

    clean = np.array([v for v in reversals if not np.isnan(v)], dtype=np.float64)
    if len(clean) >= 2:
        slope, _ = np.polyfit(np.arange(len(clean), dtype=np.float64), clean, 1)
        result["reversal_magnitude_slope"] = float(slope)
    else:
        result["reversal_magnitude_slope"] = float("nan")
    return result


# ------------------------------------------------------------------ escalation lens 4: narrative novelty


def narrative_novelty(
    scene_embeddings: np.ndarray,
    tp_scene_indices: Sequence[int],
) -> Dict[str, float]:
    """Mean cosine distance of each scene's embedding from the mean of PRIOR scene embeddings,
    aggregated per act.

    Scene 0 gets 0 novelty by definition (nothing prior). Scene k's novelty is
    1 - cos_sim(embedding_k, mean(embedding_0..k-1)). Growing per-act novelty = the
    story reaches into new territory as it progresses — the semantic "reach of
    consequences" that McKee names in Ch. 3 p. 41.

    Args:
        scene_embeddings: shape (n_scenes, embed_dim) from MiniLM.
        tp_scene_indices: 5 TP scene indices defining the 5 acts.

    Returns per-act mean novelty + a slope across the 5 acts.
    """
    E = np.asarray(scene_embeddings, dtype=np.float64)
    if E.ndim != 2 or E.shape[0] == 0:
        return {}
    n = E.shape[0]
    tps = [int(i) for i in tp_scene_indices]
    if len(tps) != 5:
        return {}

    # per-scene novelty: cosine distance to running mean
    running_sum = np.zeros(E.shape[1], dtype=np.float64)
    novelty = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            running_sum += E[i]
            novelty[i] = 0.0
            continue
        prior_mean = running_sum / i
        # cosine distance
        num = float(np.dot(E[i], prior_mean))
        den = float(np.linalg.norm(E[i]) * np.linalg.norm(prior_mean))
        cos_sim = num / den if den > 0 else 0.0
        novelty[i] = 1.0 - cos_sim
        running_sum += E[i]

    boundaries = [0] + tps + [n]
    result: Dict[str, float] = {}
    per_act = []
    for k in range(5):
        start, end = boundaries[k], boundaries[k + 1]
        if end > start:
            act_mean = float(novelty[start:end].mean())
        else:
            act_mean = float("nan")
        result[f"novelty_act{k + 1}_mean"] = act_mean
        per_act.append(act_mean)

    clean = np.array([v for v in per_act if not np.isnan(v)], dtype=np.float64)
    if len(clean) >= 2:
        slope, _ = np.polyfit(np.arange(len(clean), dtype=np.float64), clean, 1)
        result["narrative_novelty_slope"] = float(slope)
    else:
        result["narrative_novelty_slope"] = float("nan")
    return result


# ------------------------------------------------------------------ summarize + plot


def summarize(
    scene_scores: Dict[str, np.ndarray],
    tp_scene_indices: Sequence[int],
    total_scenes: int,
    scene_embeddings: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all McKee-conformance signals for one script → flat dict.

    Ready to drop into a pandas DataFrame row for cross-script comparison.

    Args:
        scene_scores: output of EmotionScorer.score_script.
        tp_scene_indices: 5 TP scene indices (from gold labels or model predictions).
        total_scenes: total number of scenes in the script.
        scene_embeddings: optional MiniLM embeddings shape (n_scenes, embed_dim); when
                          provided, the narrative-novelty escalation lens is included.
    """
    out: Dict[str, float] = {"flat_scene_rate": flat_scene_rate(scene_scores)}

    # Escalation lens 1 — per-scene affective swing (narrow, existing)
    for k, v in escalation(scene_scores, tp_scene_indices=tp_scene_indices).items():
        out[f"per_scene_swing_{k}"] = v

    # Escalation lens 2 — per-act impact (sustained pressure)
    for k, v in per_act_impact(scene_scores, tp_scene_indices).items():
        out[k] = v

    # Escalation lens 3 — reversal magnitude (departure from baseline)
    for k, v in reversal_magnitude(scene_scores, tp_scene_indices).items():
        out[k] = v

    # Escalation lens 4 — narrative novelty (reach of consequences)
    if scene_embeddings is not None:
        for k, v in narrative_novelty(scene_embeddings, tp_scene_indices).items():
            out[k] = v

    tp4, tp5 = int(tp_scene_indices[3]), int(tp_scene_indices[4])
    for k, v in alternation_compliance(scene_scores, tp4, tp5).items():
        out[f"alternation_{k}"] = v
    out.update(pacing(tp_scene_indices, total_scenes))
    return out


def plain_language_read(summary: Dict[str, float]) -> str:
    """Turn the summary dict into a short plain-language paragraph.

    Multi-lens escalation report so the reader can see the shape of the film
    through each dimension of McKee's Law of Progression.
    """
    lines = []

    fs = summary.get("flat_scene_rate", float("nan"))
    if not np.isnan(fs):
        lines.append(
            f"Flat-scene rate: {fs:.0%}. "
            + (
                "Most scenes turn — the script is dense in McKee's sense."
                if fs < 0.3
                else "Around a third of scenes don't turn — some stretches may be nonevents."
                if fs < 0.5
                else "Many scenes don't turn — the script may have long stretches of exposition or scene-level flatness."
            )
        )

    # Escalation — four lenses
    lines.append("")
    lines.append("Escalation (four lenses on McKee's Law of Progression):")

    swing_slope = summary.get("per_scene_swing_slope", float("nan"))
    if not np.isnan(swing_slope):
        lines.append(
            f"  1. Per-scene affective swing slope: {swing_slope:+.4f}. "
            + (
                "Individual moments hit harder as the film progresses."
                if swing_slope > 0
                else "Per-scene affective swing is flat — a possible sign of sustained-pressure structure rather than building crescendo."
            )
        )

    pai_slope = summary.get("per_act_impact_slope", float("nan"))
    if not np.isnan(pai_slope):
        lines.append(
            f"  2. Per-act impact slope (mean magnitude within each act): {pai_slope:+.4f}. "
            + (
                "Later acts sustain higher average intensity than earlier ones — pressure widens."
                if pai_slope > 0
                else "Later acts don't run hotter on average than earlier ones."
            )
        )

    rev_slope = summary.get("reversal_magnitude_slope", float("nan"))
    if not np.isnan(rev_slope):
        lines.append(
            f"  3. Reversal-magnitude slope (climax distance from baseline): {rev_slope:+.4f}. "
            + (
                "Each act climax swings farther from the story's normal than the last — successive reversals grow."
                if rev_slope > 0
                else "Successive act climaxes don't reverse farther from baseline — reversals aren't growing."
            )
        )

    nov_slope = summary.get("narrative_novelty_slope", float("nan"))
    if not np.isnan(nov_slope):
        lines.append(
            f"  4. Narrative-novelty slope (semantic reach per act): {nov_slope:+.4f}. "
            + (
                "Each act reaches into new territory — McKee's expanding reach of consequences."
                if nov_slope > 0
                else "Later acts don't reach into new territory — the story circles back rather than expanding outward."
            )
        )
    lines.append("")

    compliant = summary.get("alternation_compliant", float("nan"))
    tp4c = summary.get("alternation_tp4_close", float("nan"))
    tp5c = summary.get("alternation_tp5_close", float("nan"))
    if not np.isnan(compliant):
        lines.append(
            f"Alternation (TP4 close {tp4c:+.2f}, TP5 close {tp5c:+.2f}): "
            + (
                "COMPLIANT — the last two act climaxes have opposite charge, satisfying McKee's rule that you cannot set up an up-ending with an up-ending."
                if compliant > 0.5
                else "VIOLATED — the last two act climaxes share the same charge, which McKee explicitly warns against. (Note: emotion-model can misread violent-but-victorious climaxes.)"
            )
        )

    return "\n".join(lines)


def plot_emotional_arc(
    scene_scores: Dict[str, np.ndarray],
    tp_scene_indices: Optional[Sequence[int]] = None,
    title: str = "",
    ax=None,
    show_vader: bool = False,
):
    """Plot the valence trajectory across scenes with TP annotations."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))

    v_open = np.asarray(scene_scores["valence_open"])
    v_close = np.asarray(scene_scores["valence_close"])
    n = len(v_open)
    x = np.arange(n)

    ax.fill_between(x, v_open, v_close, alpha=0.18, color="#C9A961", label="scene span")
    ax.plot(x, v_open, alpha=0.5, color="#3E7050", linewidth=1.2, label="open valence")
    ax.plot(x, v_close, alpha=0.85, color="#1B2A3E", linewidth=1.4, label="close valence")

    if show_vader and "vader_close" in scene_scores:
        ax.plot(x, scene_scores["vader_close"], alpha=0.5, color="#A63A3A",
                linewidth=1.0, linestyle=":", label="VADER close (compound)")

    ax.axhline(0, color="black", linestyle="-", alpha=0.35, linewidth=0.8)

    if tp_scene_indices is not None:
        tp_names = ["TP1", "TP2", "TP3", "TP4", "TP5"]
        y_top = ax.get_ylim()[1]
        for tp_idx, name in zip(tp_scene_indices, tp_names):
            if 0 <= tp_idx < n:
                ax.axvline(tp_idx, color="#A63A3A", linestyle="--", alpha=0.55, linewidth=1.2)
                ax.text(tp_idx, y_top * 0.95, name, rotation=90, va="top", ha="right",
                        fontsize=9, color="#A63A3A", alpha=0.85)

    ax.set_xlabel("Scene index")
    ax.set_ylabel("Valence")
    ax.set_title(title or "Emotional arc")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    return ax
