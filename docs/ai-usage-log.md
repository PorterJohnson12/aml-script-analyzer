# ai-usage-log.md — Living Log of Human–AI Interaction

A weekly record of how AI was actually used on *Outlier*. Per the assignment, this is **not graded for
correctness** — it documents intent and reality. Updated each week. For each week: tasks AI assisted
with, prompts/context that worked well, and where AI output needed correction.

---

## Week 1 — Scoping, proposal, and foundation setup

**Tasks AI assisted with:**
- Brainstorming and pressure-testing the project direction, then grounding it against existing tools (ScriptBook, Cinelytic/Callaia, Prescene) and academic work (TRIPOD, ScriptBase, emotional arcs).
- Building the M1A1 package: `proposal.md`, `schedule.md`, `claude.md`, and `concrete-attack.md`.
- Writing the feasibility probe (`scripts/join_yield_probe.py`) and the ScriptBase titles extractor (`scripts/build_titles.py`); confirmed a usable dataset before committing.
- Reviewing the TRIPOD paper and repo, then porting their scene-segmentation algorithm (`screenplays_scene_segmentation.py`) into `src/outlier/segmentation.py` while verifying the resulting scene indices match TRIPOD's gold labels on all 15 test movies.
- Scaffolding the initial package layout — `src/outlier/{config,data,segmentation,metrics}.py`.
- Debugging the join-yield probe against TMDB's rate limits and the ScriptBase archive naming quirks.

**Prompts / context that worked well:**
- Giving the assistant my tech background, prior repos, and the actual PRO322 outcomes up front, so scope advice was grounded in what I've already built rather than generic.
- Asking it to separate "what industry tools already do" from "what's still open" — that surfaced the interpretable, structure-first angle.
- After the pitch review, feeding it the professor's feedback and asking for a concrete, feasible attack — that produced the TRIPOD-based reframe.
- Handing the assistant the TRIPOD source file directly rather than describing what it did; the port then went cleanly.
- "Verify my segmenter matches gold indices on all 15 test movies" as a check rather than "does it look right" — turned a subjective ask into a runnable test.

**Where AI output needed correction / direction:**
- **Scope pivot after review.** The first plan centered on box-office/ROI prediction. Instructor feedback flagged the annotation burden and steered toward "identify a principle and how well it's applied" with deep-learning NLP. I redirected the assistant to reframe around **TRIPOD turning-point detection** as the core (existing labels, no mass annotation), with box office deferred to the capstone. The proposal, schedule, and agent file were rewritten to match.
- Corrected an over-broad initial claim that the tool measures "how a story feels" — narrowed it to measuring the *countable mechanics* (turns, escalation, turning-point positions) validated against human labels.
- First segmenter version appended a trailing empty scene that TRIPOD's version omits — caught by the gold-index check when Die Hard came out with 119 scenes instead of 118. Fixed to match their exact behavior.

---

## Week 2 — Data understanding + trained-model foundation

**Tasks AI assisted with:**
- Full inspection of `TRIPOD_screenplays_test.csv`, `TRIPOD_synopses_*.csv`, and `labels_train_TRIPOD_silver.pickle` — extracting the actual data schema (what each column means, what each `tpN` list contains, why silver has 3-scene windows) rather than describing it in the abstract.
- Data-quality audit: cross-referencing every silver movie against the segmenter to identify the 9 broken cases (Alien, Jaws, Jurassic Park, My Girl, Saw, The Exorcist, Bridesmaids, Minority Report, Salton Sea) and the 3 name mismatches. Writing the audit up in `docs/data-understanding.md` / `docs/week2/data-understanding-report.md`.
- Computing the positional priors (TP1 15%, TP2 38%, TP3 64%, TP4 85%, TP5 97%) from the 15 gold movies as the baseline the trained model has to beat.
- Writing `src/outlier/embeddings.py` (MiniLM wrapper) and `src/outlier/model.py` (TPFinder bi-LSTM + tp_loss + overfit_batch + predict_tps) so the overfit-a-single-batch requirement for M2A1 has code behind it.
- Adding EDA visualizations to the notebook: TP-position box plot across gold + silver, scene-count histogram, multi-annotator agreement heatmap.
- Rewriting the M2P2 deck around a cleaner Braintrust-scaling framing after the initial pitch check-in flagged the business-value story wasn't landing.
- Restructuring `docs/M2P2/talking-points.md` as one-paragraph explainers for every concept in the project (MiniLM, bi-LSTM, target variable, two-branch pipeline shape, the three models) after realizing during M2P2 that dense reference docs weren't giving me a defensible mental model.

**Prompts / context that worked well:**
- "Open the file and tell me exactly what's in it" — moving from conceptual descriptions to file-level facts. This surfaced the semantic sanity check on Die Hard (scene 100 = McClane hears Ellis die = Point of No Return), which became the strongest single point in the presentation.
- "Explain MiniLM in one paragraph — no jargon" produced the analogy (*"MiniLM is eyes that convert scene text into signals the bi-LSTM can process"*) that finally made the two-stage architecture click for me. I use that framing now for every architecture question.
- Asking for cluster-organized answers to the Week-1 notes (five clusters: what TRIPOD does, scenes, value, ground truth, good/bad) rather than answering them note-by-note. That reorganization made the responses coherent instead of scattered.
- "Rewrite this doc in one-paragraph explainers, each at the level of 'can you say this out loud in one breath.'"

**Where output needed correction / direction:**
- **Overly defensive framing.** Replaced some answers with direct, factual answers to how the model differs from TRIPOD.
- **Docs too dense.** The first M2A1 report and the initial M2P2 talking-points doc were over-complete — every angle covered, hard to skim, hard to defend under pressure because specific facts were buried in prose. Corrected by rewriting talking-points as one-paragraph explainers and marking the long docs as ammo-only.
- **Conflating scene turns with what the trained model does.** Initial framing muddled the McKee-conformance layer (rule-based, uses DistilRoBERTa) with the TP-prediction model (trained bi-LSTM). Corrected by explicitly writing the architecture as "one pipeline, two branches that merge at the McKee-conformance layer," and naming three separate models (MiniLM + bi-LSTM in the structural branch, DistilRoBERTa in the emotional branch).
- **VADER as the primary sentiment tool.** The initial secondary-signals plan used VADER — a lexicon-based sentiment tool designed for social media. The Week-2 check-in note about needing a deep-learning approach to have landed cleanly; upgraded to DistilRoBERTa emotion classifier as the primary tool, with VADER kept only as a fast comparison baseline.

---

## Week 3 — McKee-conformance layer + emotional arc
*(to be filled)*

---

## Week 4 — Trained TP model + business validation
*(to be filled)*

---

## Week 5 — Streamlit demo + writeup
*(to be filled)*
