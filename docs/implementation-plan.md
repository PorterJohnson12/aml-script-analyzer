# Outlier — Implementation Plan

*Living 5-week execution plan for PRO322. Additive log — new entries appended over time; older entries not rewritten. See the change log at the bottom for revision history.*

---

## The end state (what "done" looks like)

A Streamlit app that takes a screenplay text file, runs two model branches in parallel over it, and returns a single unified structural read:

- **Structural branch** (MiniLM → trained bi-LSTM head) predicts the five turning-point scenes and highlights them in the actual script text.
- **Emotional branch** (DistilRoBERTa over open/close halves of each scene) produces an emotional-arc curve across the whole script.
- **McKee-conformance layer** merges the two branches into three signals derived directly from principles Robert McKee names in *Story*: flat-scene rate, escalation slope across act climaxes, and alternation compliance on the final two act climaxes.
- **Business-value validation:** correlation between the McKee-conformance signals and IMDB scores across the 1,276 ScriptBase films with metadata.

Everything is MLflow-tracked, evaluated on the 15 held-out gold movies with TRIPOD's TA/PA/D metrics for the trained TP model, and runnable end-to-end from a fresh clone of the repo.

---

## Architecture — one pipeline, two branches

```
Screenplay text file
       │
       ▼
   Segmenter (src/outlier/segmentation.py)  ──► List of scene strings
       │
       ├──► Structural branch
       │        MiniLM encoder (frozen, pretrained)      ──► scene embeddings (N × 384)
       │        bi-LSTM head (trained on 72 silver)      ──► 5 per-TP distributions ──► argmax ──► predicted TP scenes
       │
       └──► Emotional branch
                DistilRoBERTa emotion classifier          ──► 7 emotion probs per open/close half of each scene
                Valence formula                           ──► per-scene open-valence and close-valence
                                                              │
       ┌──────────────────────────────────────────────────────┘
       ▼
   McKee-conformance layer (src/outlier/mckee.py — Week 3)
       ├── Flat-scene rate (emotional branch only)
       ├── Escalation slope (emotional magnitudes at TP-defined act climaxes)
       └── Alternation compliance (signs of TP4 & TP5 valences)
       │
       ▼
   Streamlit UI (src/outlier/app.py — Week 5)
```

Only the bi-LSTM head is trained. MiniLM and DistilRoBERTa are used off the shelf. The McKee-conformance layer is rule-based arithmetic on the two branches' outputs. Standard transfer-learning shape.

---

## Week-by-week execution

### Week 1 — Foundation (DONE)

- Python 3.11 + PyTorch environment on both laptop and desktop (5070 Ti available; 4060 in laptop for backup CPU/GPU runs).
- TRIPOD downloaded and inspected — `TRIPOD_screenplays_test.csv` (15 gold movies), `TRIPOD_synopses_test/train.csv` (synopsis-level TPs), `labels_train_TRIPOD_silver.pickle` (SUMMER's 84-movie silver corpus).
- Segmenter (`src/outlier/segmentation.py`) written as a faithful port of TRIPOD's slugline-based scene splitter. Verified against gold indices on all 15 test movies — Die Hard produces 118 scenes and its labeled Point of No Return at scene 100 lands on the correct scene in the text.
- Data loaders (`src/outlier/data.py`) for gold, silver, and ScriptBase corpora.
- Metrics module (`src/outlier/metrics.py`) implementing TRIPOD's TA (Total Agreement), PA (Partial Agreement), D (Distance) evaluation.
- ScriptBase corpus flattened to `data/titles.csv` and probed against TMDB for financial + reception coverage (`scripts/join_yield_probe.py`).

### Week 2 — Data understanding + trained-model foundation (DONE)

- Full audit of gold + silver labels against the segmenter. 72 clean silver + 15 gold = 87 usable script/label pairs, ~10,925 scene-level examples. Zero overlap between train and test.
- Positional priors computed from the 15 gold movies: TP1 mean 15%, TP2 38%, TP3 64%, TP4 85%, TP5 97%. TP5 is trivially positional; TP2 and TP3 have the widest variance.
- Data-quality issues named openly: segmenter fails on 6 non-standard-format scripts (Alien, Jaws, Jurassic Park, My Girl, Saw, The Exorcist); 3 silver movies have off-by-a-few labels (Bridesmaids, Minority Report, Salton Sea); 3 name mismatches (48 Hrs, Star Wars I, When Harry Met Sally). All filtered out for now.
- EDA visualizations added to `notebooks/eda-notebook.ipynb`: TP-position box plot, scene-count histogram, multi-annotator agreement heatmap.
- `src/outlier/embeddings.py` — MiniLM wrapper for scene encoding.
- `src/outlier/model.py` — TPFinder (bi-LSTM head), tp_loss (cross-entropy vs. multi-hot target), overfit_batch (training loop), predict_tps (inference).
- Overfit-a-single-batch test run in the notebook (§8) on 3 gold movies — training curve confirms the pipeline is correctly wired end-to-end.
- Data-understanding report written and saved to `docs/week2/data-understanding-report.md`.

### Week 3 — McKee-conformance layer (NOW)

**Goal:** ship the piece that visibly differentiates the project from TRIPOD. Even without a trained TP model yet, we can produce the emotional arc + all three McKee signals using positional priors as anchor stand-ins.

- **`src/outlier/emotion.py`** — DistilRoBERTa wrapper: given a list of scene texts, split each into open/close halves, run the emotion classifier on each, return the 7-emotion probability distribution per half. Batch-run for efficiency.
- **Valence formula:** `valence = joy + surprise − anger − disgust − fear − sadness` (neutral is dropped). One signed number per scene half.
- **Per-scene turn detection:** a scene "turns" if `sign(open_valence) != sign(close_valence)`. Store the turn boolean and the turn magnitude `|close − open|` per scene.
- **`src/outlier/mckee.py`** — the three conformance signals:
  - `flat_scene_rate(valences)` → fraction of scenes that didn't turn.
  - `escalation_slope(magnitudes, tp_scenes)` → linear-regression slope of turn magnitudes at the act-climax positions.
  - `alternation_compliance(open_close_pairs, tp4_scene, tp5_scene)` → binary; True if signs of TP4 and TP5 close-valences differ.
- **Emotional-arc plot for Die Hard** — matplotlib line chart of per-scene valence across the whole script, with the five TP positions annotated as vertical lines. This is the visible demo artifact for the Week 3 check-in.
- **First-pass McKee readout for Die Hard** — print the three conformance signals in plain language: e.g., *"flat-scene rate: 22% · escalation slope: +0.4 · alternation: compliant (TP4 close −0.6, TP5 close +0.3)."*
- Anchors used this week come from the gold labels (for gold movies) or from the positional priors (for unseen scripts). Full trained TP model comes in Week 4.

**Behind-schedule signal:** face-validity check on emotion outputs fails — e.g., Die Hard's Point of No Return (Ellis's death) doesn't read as strongly negative. If that happens, the emotion model isn't a good enough proxy for scene-level "value" and we need to reconsider the whole emotional branch.

### Week 4 — Trained TP model + business validation

- Train the bi-LSTM head on the 72 clean silver movies. Adam optimizer, cross-entropy loss vs. multi-hot targets, batch size 1 (variable-length sequences), 20–50 epochs.
- **Hyperparameter tuning** with MLflow: sweep hidden size {64, 128, 256}, learning rate {5e-4, 1e-3, 3e-3}, dropout {0.0, 0.1, 0.3}. Track every run.
- Evaluate best config on the 15 held-out gold movies using TA / PA / D. Compare against three baselines: positional prior (predict mean position), TRIPOD's paper numbers, random.
- Register the best model in MLflow's model registry.
- **Business validation** — for the 1,276 ScriptBase films with IMDB metadata, run the full pipeline (structural + emotional + McKee-conformance signals) and compute Pearson correlation between each signal and the IMDB score. Report signs and confidence intervals honestly. This is the "stock-price analog" — objective evidence that structure tracks reception, without pretending correlation is causation.
- Error analysis: which movies does the TP model fail on? Are there patterns (short scripts, non-standard genres, sequels)?

**Behind-schedule signal:** the trained bi-LSTM can't beat the positional-prior baseline on TA. Fallback plan below.

### Week 5 — Streamlit demo + writeup

- **`src/outlier/app.py`** — Streamlit app. Paste-a-screenplay input; output panel shows highlighted TP scenes (with the actual scene text excerpted), the emotional-arc plot, a flat-scene list, and a plain-language conformance read.
- README + repo-level docs cleaned up so a fresh clone runs end-to-end.
- Short writeup summarizing what was built, what was learned, and what would come next in the capstone.

**Behind-schedule signal:** demo can't run end-to-end from a fresh clone.

---

## Fallback plans (documented so they don't have to be invented on the fly)

**If the bi-LSTM doesn't beat the positional baseline:** drop the trained TP model from the pipeline entirely. Use the positional priors (mean position per TP from the 15 gold movies) as the act-climax locator for the McKee-conformance layer. The tool still produces the emotional arc, the three McKee signals, and the Streamlit demo — just with less ambitious anchor detection. Still deep-learning end-to-end via the emotion model. A smaller working thing beats an aspirational broken thing.

**If the emotion model's face-validity check fails:** try a different pretrained emotion classifier (e.g., `SamLowe/roberta-base-go_emotions` with 28 categories instead of 7). If that also fails, fall back to VADER as the sentiment source and document the limitation openly in the writeup.

**If IMDB correlation shows no signal:** report it honestly. A null result is a finding — it either means the McKee-conformance signals don't track reception in this dataset, or that other drivers (budget, stars, marketing) swamp structure's contribution. Capstone would then use budget-and-star controls to isolate structure's share of variance.

---

## Explicitly out of scope for PRO322

- Hand-labeled scene-turn pilot (Cohen's kappa validation) — deferred to capstone.
- Deeper "which McKee value is at stake" detection (love/hate vs. justice/injustice etc.) — deferred to capstone; requires LLM weak-labeling + hand validation.
- Box-office / de-risking analysis controlling for budget and stars — deferred to capstone.
- Spine coherence and forces-of-antagonism signals — stretch goals only if R1–R6 land ahead of schedule.
- Fine-tuning MiniLM or DistilRoBERTa — kept frozen; only the bi-LSTM head is trained.

---

## Change log

- **2026-07-06 (Week 2 Day 1):** Initial draft based on the Week-1 proposal and post-review reframe.
- **2026-07-10 (Week 2 Day 5, end of week):** Reordered Week 3 and Week 4. McKee-conformance layer (formerly Week 4) now precedes trained TP model (formerly Week 3). Rationale: after M2P2, the professor's biggest critique was that the visible portion of the project overlaps with TRIPOD. Building the McKee-conformance layer first makes the differentiator visible earlier and defensible at the Week 3 check-in. The trained bi-LSTM is small enough to complete in Week 4 without missing the Week 5 demo deadline.
- **2026-07-10:** Added the fallback plan for the case where the trained model doesn't beat the positional baseline. This was implicit before; now written down so it doesn't get invented on the fly.
- **2026-07-12 (Week 2 weekend):** Dropped the hand-labeled scene-turn pilot from PRO322 scope; deferred to capstone. Rationale: the ~15-script hand-labeling is real work and its Cohen's-kappa validation only affects the scene-turn secondary signal, which the McKee-conformance layer will use in a face-validity mode (a few known scenes) rather than a full agreement study for PRO322.
- **2026-07-12:** Added `src/outlier/embeddings.py` and `src/outlier/model.py`. Overfit-a-single-batch test added to `notebooks/eda-notebook.ipynb` §8.
