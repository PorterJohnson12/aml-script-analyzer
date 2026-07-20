"""Week-3 presentation/report figures — reads docs/week3/results.json only.

Deliberately decoupled from run_experiments.py: that script dumps every gold
prediction and training curve it produced, so redrawing a chart costs ~1s instead
of the ~185s it takes to re-encode 88 screenplays.

Usage (from the repo root, after run_experiments.py):
    python scripts/make_week3_figures.py

Writes PNGs to docs/week3/figures/.

Chart conventions (see the dataviz method):
  - No dual-axis plots anywhere. Two measures of different scale get two stacked
    panels sharing one x-axis, so no arbitrary scale alignment can invent a
    correlation. This matters most in fig 5, whose whole claim is "loss falls,
    validation doesn't" — on twin axes that claim would rest on how the axes
    were aligned.
  - Categorical hues come from fixed validated slots, assigned by entity, never
    by rank. Slot 1 blue / slot 6 orange: adjacent CVD dE 24.7, normal-vision
    dE 33.6 — both clear their gates on the light surface.
  - Where the story is one comparison, the rest is grayed rather than colored.
  - Hairline solid grid, no dashes, no boxes, thin marks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from outlier.metrics import per_tp_pa_hits, bootstrap_ci
from outlier.data import load_silver_labels

RESULTS = ROOT / "docs" / "week3" / "results.json"
OUTDIR = ROOT / "docs" / "week3" / "figures"

# ---- palette (light surface; validated slots) ------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"    # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 6
BLUE_FILL = "#cde2fb"  # blue ramp step 100

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",     # never dashed
    "axes.grid": False,
    "figure.dpi": 120,
})


def _clean(ax, xgrid=False, ygrid=True):
    """Hairline recessive chrome: no top/right spines, solid grid, muted ticks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.tick_params(length=0)


def load():
    if not RESULTS.exists():
        sys.exit(f"missing {RESULTS} — run scripts/run_experiments.py first")
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


def hits_for(d, run_name):
    """Pooled per-TP PA hits (n_gold * 5 decisions) for one run."""
    tol = d["pa_tol"]
    preds = d["runs"][run_name]["preds"]
    out = []
    for p, g in zip(preds, d["gold"]):
        out.extend(per_tp_pa_hits(p, g["tps"], g["n"], tol=tol))
    return np.array(out, dtype=float)


# ---------------------------------------------------------------- fig 1
def fig_model_comparison(d):
    """Magnitude + uncertainty across runs. Story is one comparison, so the
    no-information control is colored and everything else recedes."""
    order = ["random", "uniform_spaced", "positional_prior", "transformer_nopos",
             "larger", "transformer_pos", "zeros_input", "baseline", "emotion_augmented"]
    order = [r for r in order if r in d["runs"]]
    rows = []
    for name in order:
        h = hits_for(d, name)
        lo, hi = bootstrap_ci(h)
        rows.append((name, h.mean(), lo, hi, d["runs"][name]["kind"]))
    rows.sort(key=lambda r: r[1])

    ref = next(r[1] for r in rows if r[0] == "zeros_input")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ys = np.arange(len(rows))
    for i, (name, pa, lo, hi, kind) in enumerate(rows):
        is_ctrl = name in ("zeros_input", "transformer_nopos")
        c = ORANGE if is_ctrl else (BLUE if kind == "trained" else MUTED)
        ax.barh(i, pa, height=0.5, color=c, zorder=3)
        ax.plot([lo, hi], [i, i], color=INK_2, lw=1.6, zorder=4, solid_capstyle="butt")
        ax.plot([lo, lo], [i - .11, i + .11], color=INK_2, lw=1.6, zorder=4)
        ax.plot([hi, hi], [i - .11, i + .11], color=INK_2, lw=1.6, zorder=4)
        ax.text(hi + .012, i, f"{pa:.3f}", va="center", ha="left",
                fontsize=9, color=INK_2)

    ax.axvline(ref, color=ORANGE, lw=1.2, zorder=2)
    ax.text(ref, len(rows) - 0.35, "  zeros_input — a model that sees nothing",
            color=ORANGE, fontsize=9, va="bottom", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK_2)
    ax.set_xlabel("Gold Partial Agreement (PA)  ·  bars = point estimate, rules = 95% bootstrap CI")
    ax.set_xlim(0, 0.66)
    # Precise claim: every model with access to POSITION has a CI spanning
    # zeros_input's estimate. transformer_nopos (the content-only arm) does not —
    # it is the one model that cannot see position, and it sits last.
    ax.set_title("Every model that can see position ties the control that sees nothing",
                 loc="left", color=INK, fontweight="bold", pad=14)
    _clean(ax, xgrid=True, ygrid=False)

    handles = [plt.Line2D([], [], color=BLUE, lw=6, label="trained on real features"),
               plt.Line2D([], [], color=ORANGE, lw=6, label="diagnostic control"),
               plt.Line2D([], [], color=MUTED, lw=6, label="untrained baseline")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
              labelcolor=INK_2)
    fig.text(0.01, 0.015,
             f"15 gold movies x 5 turning points = {len(hits_for(d,'baseline'))} decisions. "
             f"PA tolerance {d['pa_tol']}. The only model whose interval clears zeros_input is "
             f"transformer_nopos — the content-only arm — and it clears it downwards, below random.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(OUTDIR / "01_model_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_ablations(d):
    """Two CONTROLLED ablations, each a matched pair within one architecture.

    A single 2x2 over {position, content} is tempting but dishonest with the models
    we have: the content axis only has a matched pair in the bi-LSTM (which cannot
    ignore scene order), and the position axis only in the transformer. Crossing them
    — e.g. transformer_nopos 0.147 vs bi-LSTM zeros_input 0.320 — confounds the effect
    with the architecture. So two panels, each toggling exactly one thing with the
    parameter count held fixed."""
    panels = [
        # (title, full_run, ablated_run, feature, arch, params)
        ("Does scene CONTENT help?", "baseline", "zeros_input", "content",
         "bi-LSTM · 527k params · input zeroed", "content"),
        ("Does scene POSITION help?", "transformer_pos", "transformer_nopos", "position",
         "Transformer · 182k params · positional encoding removed", "position"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    rnd = hits_for(d, "random").mean()

    for ax, (title, full, ablated, feat, sub, _) in zip(axes, panels):
        pf, pa_ = hits_for(d, full), hits_for(d, ablated)
        mf, ma = pf.mean(), pa_.mean()
        lof, hif = bootstrap_ci(pf); loa, hia = bootstrap_ci(pa_)
        xs = [0, 1]
        ax.bar(0, mf, width=0.62, color=BLUE, zorder=3)
        ax.bar(1, ma, width=0.62, color=ORANGE, zorder=3)
        for xx, m, lo, hi in [(0, mf, lof, hif), (1, ma, loa, hia)]:
            ax.plot([xx, xx], [lo, hi], color=INK_2, lw=1.6, zorder=4)
            ax.plot([xx - .06, xx + .06], [lo, lo], color=INK_2, lw=1.6, zorder=4)
            ax.plot([xx - .06, xx + .06], [hi, hi], color=INK_2, lw=1.6, zorder=4)
            ax.text(xx, m / 2, f"{m:.3f}", ha="center", va="center", color="white",
                    fontsize=13, fontweight="bold", zorder=5)
        ax.axhline(rnd, color=MUTED, lw=1, ls="-", zorder=1)
        ax.text(1.62, rnd, f"random {rnd:.3f}", color=MUTED, fontsize=8,
                va="center", ha="right")
        # the effect = full - ablated, stated as the number the panel is about
        eff = mf - ma
        ax.annotate("", xy=(0, mf), xytext=(1, ma),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        ax.text(0.5, max(mf, ma) + .07, f"{feat} is worth  {eff:+.3f}",
                ha="center", fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"with\n{feat}", f"{feat}\nremoved"], fontsize=9, color=INK_2)
        ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=10, fontsize=11.5)
        ax.text(0, -0.16, sub, transform=ax.transAxes, fontsize=8, color=MUTED)
        ax.set_xlim(-0.6, 1.75)
        _clean(ax)

    axes[0].set_ylabel("Gold Partial Agreement (PA)")
    axes[0].set_ylim(0, 0.52)
    fig.suptitle("Position carries the score; content contributes nothing",
                 x=0.01, ha="left", color=INK, fontweight="bold", fontsize=13, y=0.99)
    fig.text(0.01, 0.015,
             "Each panel is a matched pair — same architecture, same parameter count, one thing "
             "toggled. Bars = point estimate, rules = 95% bootstrap CI.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUTDIR / "02_ablations.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_die_hard(d):
    """Familiar-movie evidence: the emotional arc, then where each method thinks
    the turning points are. Two stacked panels share the scene axis."""
    g = next((x for x in d["gold"] if x["movie"] == "Die Hard"), None)
    if g is None or g["valence_close"] is None:
        print("  [skip] Die Hard valence unavailable (run with emotion_augmented enabled)")
        return
    n = g["n"]
    vo = np.array(g["valence_open"]); vc = np.array(g["valence_close"])
    scenes = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.2})

    mean_v = (vo + vc) / 2
    # Raw per-scene valence is spiky enough to hide the shape, so show it faint and
    # overlay a rolling mean. Both are drawn — the smoothing is a reading aid, not a
    # claim, and the raw series stays visible underneath.
    w = 9
    pad = np.pad(mean_v, (w // 2, w // 2), mode="edge")
    smooth = np.convolve(pad, np.ones(w) / w, mode="valid")[:n]

    ax1.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax1.plot(scenes, mean_v, color=BLUE, lw=0.8, alpha=.35, zorder=3,
             label="per-scene valence (raw)")
    ax1.fill_between(scenes, 0, smooth, color=BLUE_FILL, alpha=.55, zorder=1)
    ax1.plot(scenes, smooth, color=BLUE, lw=2.2, zorder=4,
             label=f"rolling mean ({w} scenes)")
    for k, tp in enumerate(g["tps"]):
        for s in tp:
            ax1.axvline(s, color=INK, lw=0.9, alpha=.5, zorder=5)
        # TP4/TP5 sit 2 scenes apart in Die Hard, so same-height labels collide.
        # Stagger them INSIDE the axes (valence is negative up here, so it's empty)
        # rather than above it, where they would run into the title.
        yoff = 0.95 if k % 2 == 0 else 0.85
        ax1.text(float(np.mean(tp)), yoff, f"TP{k+1}", transform=ax1.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax1.set_ylabel("valence  (joy+surprise − anger/disgust/fear/sadness)")
    ax1.set_title("Die Hard — the emotional arc is real; the turning-point predictions are not",
                  loc="left", color=INK, fontweight="bold", pad=12)
    ax1.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2, ncol=2)
    _clean(ax1)

    gi = d["gold"].index(g)
    # Two-line labels: the single-line versions overflowed the figure's left edge.
    tracks = [
        ("gold\n(ground truth)", [float(np.mean(t)) for t in g["tps"]], INK, "o"),
        ("baseline\n(reads script)", d["runs"]["baseline"]["preds"][gi], BLUE, "s"),
        ("zeros_input\n(sees nothing)", d["runs"]["zeros_input"]["preds"][gi], ORANGE, "D"),
        ("positional_prior\n(5 constants)", d["runs"]["positional_prior"]["preds"][gi], MUTED, "^"),
    ]
    for i, (label, xs, color, marker) in enumerate(tracks):
        y = len(tracks) - 1 - i
        ax2.plot(xs, [y] * len(xs), marker, color=color, markersize=8,
                 markeredgecolor=SURFACE, markeredgewidth=1.6, linestyle="none", zorder=3)
    for tp in g["tps"]:
        ax2.axvline(float(np.mean(tp)), color=INK, lw=0.9, alpha=.25, zorder=1)
    ax2.set_ylim(-0.6, len(tracks) - 0.4)
    # yticks rather than in-plot text: keeps the labels out of the data area and lets
    # the shared x-axis start at 0 instead of reserving 40 scenes of dead space.
    ax2.set_yticks(range(len(tracks)))
    ax2.set_yticklabels([t[0] for t in tracks][::-1], fontsize=9, color=INK_2)
    ax2.set_xlabel("scene index  (Die Hard has 118 scenes)")
    ax2.set_xlim(-2, n + 2)
    _clean(ax2, ygrid=False)
    ax2.spines["left"].set_visible(False)

    # State only what the numbers support. Per-movie the model does NOT sit on top of
    # the prior (they differ by up to 21 scenes here); what converges is the AVERAGE
    # position across all 15 gold movies. The Die-Hard-specific fact is the TP3 miss.
    tp3_pred = d["runs"]["baseline"]["preds"][gi][2]
    tp3_gold = float(np.mean(g["tps"][2]))
    fig.text(0.01, 0.028,
             f"The model puts the Point of No Return at scene {tp3_pred}; it is scene "
             f"{tp3_gold:.0f} — McClane hearing Ellis die — a miss of "
             f"{abs(tp3_pred-tp3_gold):.0f} scenes, a third of the film. It also puts TP1 and TP2 "
             f"on the same scene ({d['runs']['baseline']['preds'][gi][0]}).",
             fontsize=8.5, color=MUTED)
    fig.text(0.01, 0.006,
             "Averaged over all 15 gold movies, the model that reads the script and the model "
             "that sees nothing predict the same relative positions.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(OUTDIR / "03_die_hard.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_per_tp(d):
    """Two series across five ordered categories -> grouped bars."""
    tol = d["pa_tol"]

    def per_tp(run):
        acc = np.zeros(5); cnt = np.zeros(5)
        for p, g in zip(d["runs"][run]["preds"], d["gold"]):
            h = per_tp_pa_hits(p, g["tps"], g["n"], tol=tol)
            for k, v in enumerate(h):
                acc[k] += v; cnt[k] += 1
        return acc / cnt

    real = per_tp("baseline"); zero = per_tp("zeros_input")
    x = np.arange(5); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2 - 0.01, real, w, color=BLUE, label="baseline (real MiniLM embeddings)", zorder=3)
    ax.bar(x + w / 2 + 0.01, zero, w, color=ORANGE, label="zeros_input (sees nothing)", zorder=3)
    for i in range(5):
        ax.text(i - w / 2 - 0.01, real[i] + .015, f"{real[i]:.2f}", ha="center",
                fontsize=8.5, color=INK_2)
        ax.text(i + w / 2 + 0.01, zero[i] + .015, f"{zero[i]:.2f}", ha="center",
                fontsize=8.5, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(["TP1\nOpportunity", "TP2\nChange of Plans", "TP3\nPoint of No Return",
                        "TP4\nMajor Setback", "TP5\nClimax"], fontsize=9, color=INK_2)
    ax.set_ylabel("Gold PA for this turning point")
    ax.set_ylim(0, 1.0)
    # Claim only what THIS chart shows. The TP5 gap is the readable one; the other
    # four differences are 1-2 movies each and swing both directions, so "content
    # hurts the early TPs" is NOT assertable here — that finding comes from the
    # 230-decision silver cross-validation, and belongs with that experiment.
    ax.set_title("TP5 is nearly free. On the other four, both models are near chance.",
                 loc="left", color=INK, fontweight="bold", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    _clean(ax)
    rnd = per_tp("random")
    fig.text(0.01, 0.028,
             "The climax sits at the end of every script, so 'guess the last scene' scores well. "
             "Averaging all five turning points into one PA hides that: drop TP5 and baseline "
             f"falls {real.mean():.3f} → {real[:4].mean():.3f}, zeros_input {zero.mean():.3f} → "
             f"{zero[:4].mean():.3f}  (random: {rnd.mean():.3f} → {rnd[:4].mean():.3f}).",
             fontsize=8.5, color=MUTED)
    fig.text(0.01, 0.006,
             "Each bar is one turning point over 15 movies = 15 decisions (0.27 is 4 movies, 0.13 is 2). "
             "Only the TP5 gap is bigger than the noise.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUTDIR / "04_per_tp.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_overfitting(d):
    """Two measures, two scales -> two stacked panels sharing the epoch axis.
    Never a dual y-axis: the claim is 'loss falls, validation doesn't', and on
    twin axes that claim would depend on how the two scales were aligned."""
    runs = [("baseline", BLUE, "baseline (real embeddings)"),
            ("zeros_input", ORANGE, "zeros_input (sees nothing)")]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                   gridspec_kw={"hspace": 0.16})
    for name, color, label in runs:
        lc = d["runs"][name]["loss_curve"]
        ax1.plot(np.arange(1, len(lc) + 1), lc, color=color, lw=2, label=label, zorder=3)
    ax1.set_ylabel("training loss")
    ax1.set_title("Training loss collapses. Validation does not move.",
                  loc="left", color=INK, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    _clean(ax1)

    for name, color, label in runs:
        pc = d["runs"][name]["dev_PA_curve"]
        ax2.plot(np.arange(1, len(pc) + 1), pc, color=color, lw=2, label=label, zorder=3)
    ax2.set_ylabel("dev PA")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0, max(max(d["runs"][r]["dev_PA_curve"]) for r, _, _ in runs) * 1.25)
    _clean(ax2)

    b_lc = d["runs"]["baseline"]["loss_curve"]
    z_lc = d["runs"]["zeros_input"]["loss_curve"]
    ax1.annotate(f"{b_lc[0]:.2f} → {b_lc[-1]:.2f}\nmemorising 61 movies",
                 xy=(len(b_lc), b_lc[-1]), xytext=(-96, 26), textcoords="offset points",
                 fontsize=8.5, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax1.annotate(f"{z_lc[0]:.2f} → {z_lc[-1]:.2f}\nnothing to memorise",
                 xy=(len(z_lc), z_lc[-1]), xytext=(-96, -30), textcoords="offset points",
                 fontsize=8.5, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    fig.text(0.01, 0.015,
             "Both panels share the epoch axis and keep their own honest scale. "
             "The blue model memorises the training set; the orange one cannot — "
             "and their validation curves are indistinguishable.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUTDIR / "05_overfitting.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 6
def fig_label_collapse(d):
    """One series over an ordered count -> single-hue bars, clean bar emphasised."""
    silver = load_silver_labels(collapse=True)
    counts = {}
    for m, tps in silver.items():
        k = len({tuple(sorted(t)) for t in tps})
        counts[k] = counts.get(k, 0) + 1
    ks = sorted(counts)
    total = sum(counts.values())
    clean = counts.get(5, 0)

    fig, ax = plt.subplots(figsize=(8.6, 5))
    for k in ks:
        c = BLUE if k == 5 else ORANGE
        ax.bar(k, counts[k], width=0.6, color=c, zorder=3)
        ax.text(k, counts[k] + 0.7, str(counts[k]), ha="center", fontsize=10, color=INK_2)
    ax.set_xticks(ks)
    ax.set_xlabel("distinct turning-point scene sets per movie  (5 = every TP on its own scene)")
    ax.set_ylabel("silver movies")
    ax.set_title(f"{total - clean} of {total} silver movies ({100*(total-clean)/total:.0f}%) "
                 f"put two or more turning points on the same scene",
                 loc="left", color=INK, fontweight="bold", pad=14)
    handles = [plt.Line2D([], [], color=BLUE, lw=6, label="usable supervision"),
               plt.Line2D([], [], color=ORANGE, lw=6, label="collapsed projection")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    _clean(ax)
    fig.text(0.01, 0.015,
             "SUMMER's synopsis→scene projection collapsed. Gran Torino puts TP1, TP2 and TP3 "
             "all on scene 25 — the loss asks for three different answers and the label gives one.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(OUTDIR / "06_label_collapse.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 7 (Week-4 first look)
def fig_valence_shift(d):
    """Week-4 preview: does the emotional charge shift more at turning points?

    Two honest panels (separate scales, no dual-axis). Left: is the shift real —
    |valence delta| at TPs vs random. Right: the per-TP signed pattern (down early,
    up at the climax). Same computation as scripts/run_valence_shift_probe.py, which
    is the canonical/tested version and runs the permutation test."""
    K = 3
    def scene_val(g):
        vo, vc = g.get("valence_open"), g.get("valence_close")
        return None if vo is None else (np.array(vo) + np.array(vc)) / 2.0
    def delta_at(v, pos):
        n = len(v)
        if pos - K < 0 or pos + K >= n:
            return None
        return v[pos + 1:pos + 1 + K].mean() - v[pos - K:pos].mean()

    rng = np.random.default_rng(0)          # seed matches the probe so numbers agree
    tp_signed = {k: [] for k in range(5)}
    tp_abs, rand_abs = [], []
    for g in d["gold"]:
        v = scene_val(g)
        if v is None:
            continue
        for k, tp in enumerate(g["tps"]):
            dd = delta_at(v, int(round(float(np.mean(tp)))))
            if dd is not None:
                tp_signed[k].append(dd); tp_abs.append(abs(dd))
        interior = np.arange(K, g["n"] - K)
        for pos in rng.choice(interior, size=min(5, len(interior)), replace=False):
            dd = delta_at(v, int(pos))
            if dd is not None:
                rand_abs.append(abs(dd))
    if not tp_abs:
        print("  [skip] no valence in results.json (run with emotion_augmented enabled)")
        return
    tp_abs, rand_abs = np.array(tp_abs), np.array(rand_abs)
    # permutation test (same as the probe) so the figure states its own p-value
    obs = tp_abs.mean() - rand_abs.mean()
    pool = np.concatenate([tp_abs, rand_abs]); nA = len(tp_abs)
    perm = np.array([rng.permutation(pool)[:nA].mean() - rng.permutation(pool)[nA:].mean()
                     for _ in range(5000)])
    pval = float((np.abs(perm) >= abs(obs)).mean())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.7),
                                   gridspec_kw={"width_ratios": [1, 1.7]})

    # ---- left: is the shift real?
    mt, mr = tp_abs.mean(), rand_abs.mean()
    axL.bar(0, mt, width=0.6, color=BLUE, zorder=3)
    axL.bar(1, mr, width=0.6, color=MUTED, zorder=3)
    for x, m in [(0, mt), (1, mr)]:
        axL.text(x, m + 0.006, f"{m:.3f}", ha="center", fontsize=11, color=INK_2)
    axL.set_xticks([0, 1]); axL.set_xticklabels(["at turning\npoints", "at random\nscenes"], fontsize=9, color=INK_2)
    axL.set_ylabel("mean |valence shift|  (3 after − 3 before)")
    axL.set_ylim(0, max(mt, mr) * 1.25)
    axL.set_title(f"The shift is real — {mt/mr:.2f}× bigger at turning points",
                  loc="left", color=INK, fontweight="bold", fontsize=11, pad=8)
    axL.text(0.5, max(mt, mr) * 1.13, f"permutation p = {pval:.3f}", ha="center",
             fontsize=9.5, color="#0a7d3c" if pval < 0.05 else MUTED, fontweight="bold")
    _clean(axL)

    # ---- right: the per-TP signed pattern (diverging around zero)
    signed = [np.mean(tp_signed[k]) if tp_signed[k] else 0.0 for k in range(5)]
    ns = [len(tp_signed[k]) for k in range(5)]
    x = np.arange(5)
    colors = [ORANGE if s < 0 else BLUE for s in signed]   # orange = darkens, blue = lifts
    axR.axhline(0, color=AXIS, lw=1, zorder=2)
    axR.bar(x, signed, width=0.62, color=colors, zorder=3)
    for i, s in enumerate(signed):
        va = "top" if s < 0 else "bottom"
        off = -0.006 if s < 0 else 0.006
        axR.text(i, s + off, f"{s:+.2f}", ha="center", va=va, fontsize=9.5, color=INK_2)
    axR.set_xticks(x)
    axR.set_xticklabels(["TP1\nOpportunity", "TP2\nChange of\nPlans", "TP3\nPoint of\nNo Return",
                         "TP4\nMajor\nSetback", "TP5\nClimax"], fontsize=8.5, color=INK_2)
    axR.set_ylabel("signed valence shift  (after − before)")
    axR.set_title("It darkens at the Change of Plans, lifts at the Climax",
                  loc="left", color=INK, fontweight="bold", fontsize=11, pad=8)
    yl = max(abs(min(signed)), abs(max(signed))) * 1.45
    axR.set_ylim(-yl, yl)
    axR.text(0.99, 0.04, "orange = story darkens   ·   blue = story lifts",
             transform=axR.transAxes, ha="right", fontsize=8.5, color=MUTED)
    _clean(axR)

    fig.suptitle("Week-4 first look — the emotional charge shifts at turning points",
                 x=0.01, ha="left", color=INK, fontweight="bold", fontsize=13, y=0.99)
    fig.text(0.01, 0.015,
             "Exploratory · 15 gold movies · valence = joy+surprise − anger/disgust/fear/sadness. "
             "TP5 n=9 (climax runs near the end). Reproduce: run_valence_shift_probe.py.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUTDIR / "07_valence_shift.png")
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = load()
    for fn in (fig_model_comparison, fig_ablations, fig_die_hard,
               fig_per_tp, fig_overfitting, fig_label_collapse, fig_valence_shift):
        fn(d)
        print(f"  wrote {fn.__name__}")
    print(f"\nfigures -> {OUTDIR}")


if __name__ == "__main__":
    main()
