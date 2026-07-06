# Outlier — 5-Week Schedule (PRO322 · M1A1)

**Author:** Joe Porter · Solo · PRO322 — Applied AI Projects II (Machine Learning)

A high-level map of the core requirements onto five weeks. Detailed rationale lives in `proposal.md`
and `concrete-attack.md`; this file is the at-a-glance plan and the weekly checkpoint.

---

## Core requirements (definition of done)

The project is complete when all of the following exist and are demonstrated:

- **R1 — Data pipeline:** ScriptBase screenplays segmented into scenes; TRIPOD labels loaded and aligned.
- **R2 — Core model:** a fine-tuned turning-point identifier that locates the five act-structure
  turning points, with **reported agreement against TRIPOD's gold labels**.
- **R3 — Secondary principles:** scene-turn detection (validated on a ~15-script hand-labeled gold set)
  and an escalation/progression measure derived from it.
- **R4 — Soundness scoring:** a per-script structural read — are the beats present and well-placed, and
  does the story escalate — plus an escalation profile across the acts.
- **R5 — Experiment tracking:** all training/evaluation runs tracked in MLflow; best model registered.
- **R6 — Demo + docs:** a Streamlit app that takes a screenplay and shows highlighted turning points,
  flagged flat scenes, and the escalation curve, readable by a non-technical user; documentation.

---

## Week-by-week

| Week | Focus | Milestone (what "done" means) | Behind-schedule signal |
|---|---|---|---|
| **1** | Foundation & setup | Env + GPU working; TRIPOD downloaded and its **baseline model reproduced**; ScriptBase scene segmentation running (R1) | Can't reproduce the TRIPOD baseline, or the GPU training env isn't working |
| **2** | Core model | **Turning-point identifier fine-tuned** on TRIPOD gold + silver; evaluated vs. gold (R2) | Model doesn't beat a trivial positional baseline, with no diagnostic why |
| **3** | Secondary principles + pilot | Scene-turn detection + **~15-script hand-validated pilot**; escalation features derived from the turn signal (R3) | Turn detection disagrees badly with hand labels and I can't explain it |
| **4** | Soundness scoring + tracking | Structural-soundness read + escalation profile; tuning; **MLflow-tracked**, best model registered; error analysis (R4, R5) | No registered model, or evaluation too unstable to trust |
| **5** | Demo + communication | **Streamlit demo** (paste script → turning points + flat scenes + escalation curve); docs; short writeup (R6) | Demo can't run end-to-end from a fresh clone |

> Weeks 1–2 are setup + data validation per the assignment; meaningful technical milestones (a trained,
> evaluated model) begin at Week 2.

---

## Dependencies & risk

- The **core (R2)** depends only on script text + TRIPOD labels — **both already in hand**, so the
  biggest data risk is retired before Week 1.
- **Escalation (R3)** rides on the scene-turn signal, so it costs no extra labeling — which is why it's
  a committed goal, not a stretch.
- Stretch goals (spine coherence, forces of antagonism) are added only if R1–R6 land ahead of schedule.
- Box-office / de-risking analysis is **out of scope** for these five weeks (capstone).
