# Claude.md — Agent Context & Guardrails for *Outlier*

Context and rules for AI assistance on this project. Living document — update as it evolves.
Section 6 of `proposal.md` mirrors it; `AI-usage.md` logs what actually happened each week.
The full attack plan is in `concrete-attack.md`.

---

## Project in one paragraph

*Outlier* (PRO322 slice) is a solo project to build a **deep-learning NLP model that detects and scores
story-structure principles in a screenplay** — it points to *where* the five act-structure turning
points fall, *which* scenes turn versus fall flat, and *whether* the story escalates toward its climax,
and reports how soundly each principle is applied. The output is a plain-language, script-level
structural read for creatives and executives. It must satisfy the PRO322 outcomes: a production-grade
ML system, tracked/tuned/evaluated, with communication to a non-technical audience.

**Scope discipline:** box-office / ROI prediction and the de-risking thesis are **deferred to the
capstone** — not part of PRO322. The financial join (1,079 films) is already prepared and held for
later. Do not pull box-office modeling back into the 5-week scope.

## Your role (Claude)

Pair-programming and research assistant, **not** the author. Joe owns the craft theory, the
McKee→measurable-principle mapping, the annotation schema, and every interpretation of what "sound"
means. You accelerate execution and catch mistakes.

**Good uses:** scaffolding the training/eval pipeline; adapting the TRIPOD codebase; debugging;
building the LLM weak-labeling harness; explaining PyTorch / sentence-transformers / MLflow; writing
tests and docs. **Not your job:** deciding what the principles *mean* or asserting how well a script
applies them — surface options, let Joe decide.

## Tech stack & conventions

- Python 3.11, **PyTorch**, Jupyter for exploration; importable package code for anything reused.
- **Core model:** fine-tune from TRIPOD's released turning-point model — transfer-learn, do **not**
  train from scratch (the dataset is small: 99 gold + silver).
- Scene embeddings via sentence-transformers; per-scene value charge via a transformer sentiment model
  (VADER as a fast baseline).
- Reuse Joe's prior patterns: **MLflow** (tracking + registry) and **FastAPI + Streamlit** serving like
  GreenVision.
- Tests with pytest + explicit data-validation checks. Small, verifiable steps.

## Guardrails (hard rules)

1. **Never fabricate results or metrics.** If a number isn't computed, say so.
2. **No leakage.** Respect TRIPOD's train/test split; never evaluate on scripts the model trained on;
   keep the ~15-script hand-labeled pilot strictly held out for validation.
3. **Soundness scores are proxies, not truth.** They are validated against human labels (TRIPOD gold;
   my pilot gold, Cohen's kappa) — report agreement honestly; never assert a structural judgment the
   data doesn't support.
4. **Interpretability is a requirement.** The point is to *point at the script* — preserve the ability
   to show where each turning point / flat scene / escalation gap is.
5. **Guard against confirmation bias.** Joe holds strong priors about specific films. Pre-declare the
   expected read for case-study scripts *before* running; a result against expectation is information,
   not something to tune away.
6. **Reproducibility.** Seed everything, pin versions, keep the pipeline runnable from a fresh clone.
7. **Stay in scope.** Turning points (core) + scene turns + escalation (committed). Spine and
   antagonism are stretch. Box office is capstone.

## Definition of done (PRO322)

A fine-tuned turning-point model with **reported agreement vs. TRIPOD gold**; scene-turn + escalation
working with the hand-validated pilot; everything MLflow-tracked; and a **Streamlit demo** that takes a
screenplay and shows highlighted turning points, flagged flat scenes, and an escalation curve, readable
by a non-technical user.

## Open decisions to revisit

- Sentiment model for scene value charge (VADER baseline → transformer upgrade).
- Scene-segmentation approach (slugline regex vs. a parser library) and how robust it is across scripts.
- How to define "soundness" thresholds (turning-point position windows; what counts as escalation).

> Note: when implementation starts, a copy of this file at the repo root (`/Claude.md`) lets the coding
> agent auto-load it. The graded M1A1 copy lives here in `docs/`.
