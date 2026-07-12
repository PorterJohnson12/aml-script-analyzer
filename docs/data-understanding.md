# Outlier — Data Understanding Report (PRO322 · M2A1)

**Author:** Porter Johnson · Solo · PRO322

*inspection of files on disk*

---

## 1. Data source & ingestion

Three datasets live in `data/`, all in-hand:

- **TRIPOD** (Papalampidi et al., EMNLP 2019) — `data/TRIPOD-master/`.
  Contains the 99 screenplays (`Screenplays_and_imdb_meta/<movie>/<movie>_script.txt`) and the gold turning-point labels (`Synopses_and_annotations/TRIPOD_screenplays_test.csv`).
- **SUMMER** (Papalampidi's follow-up) — `data/SUMMER-master/dataset/labels_train_TRIPOD_silver.pickle`.
  Contains silver-standard training labels — turning-point scene indices projected from synopsis annotations via SUMMER's method.
- **ScriptBase** — `data/scriptbase-master/scriptbase_alpha/` (1,276 `.tar.gz` archives). Held for secondary work and the capstone; not required for PRO322.

No API dependency, no update frequency to manage; all static, versioned artifacts.

## 2. What TP1–TP5 actually mean in the CSVs

**File:** `TRIPOD_screenplays_test.csv`
**Columns:** `movie_name, tp1, tp2, tp3, tp4, tp5`
**Row count:** 15 (the 15 gold-annotated test movies)

Each `tpN` cell is a **Python literal — a list of integer scene indices** — representing the *set of acceptable scenes* that human annotators identified as that turning point. Multiple entries reflect either multiple annotators agreeing on multiple adjacent scenes, or one annotator marking a short range around the beat.

Concrete example — the raw row for *Die Hard*:

```
movie_name  = Die Hard
tp1  = [11, 12]              → Opportunity (annotators marked scenes 11 or 12)
tp2  = [26, 28, 30]          → Change of Plans
tp3  = [99, 100]             → Point of No Return
tp4  = [114, 115]            → Major Setback
tp5  = [116, 117]            → Climax
```

The five names come from TRIPOD's paper (Papalampidi et al. 2019) which draws on Cutting (2016)'s empirical narrative-arc scheme: **Opportunity → Change of Plans → Point of No Return → Major Setback → Climax**.

**Semantic sanity check on Die Hard.** Scene 100 (TP3, Point of No Return) is the scene where McClane hears Ellis being executed on the terrorist radio — the exact point after which McClane cannot back down. That's a canonical Point-of-No-Return by any story theory's definition. TP1 scene 11 lands on the calm setup with Holly at Ellis's office, immediately before the terrorists arrive. **The labels correspond to real story beats — not arbitrary positional markers.**

## 3. Silver labels — different shape

**File:** `labels_train_TRIPOD_silver.pickle` (unpickle → `dict`)
**Keys:** `<Movie_annotatorIndex>` — e.g. `"10 Things I Hate About You_0"` — 128 total.
**Unique movies:** 84 (62 with 1 annotator, 22 with 3 annotators — the `_0`, `_1`, `_2` suffixes).
**Values:** `[tp1, tp2, tp3, tp4, tp5]` — same shape as gold, but each `tpN` is a **fixed 3-scene window** `[n, n+1, n+2]` around the projected point (SUMMER's method produces windowed projections from synopsis sentences, so silver labels are noisier and always window-shaped, unlike gold's variable-width sets).

Example silver entry: `"10 Things I Hate About You_0" = [[20,21,22], [29,30,31], [43,44,45], [66,67,68], [84,85,86]]`

**Zero overlap** between gold and silver movie sets — the 84 silver movies and the 15 gold movies are disjoint. **Train/test split is clean out of the box.**

## 4. Screenplay text

99 movie folders under `Screenplays_and_imdb_meta/`, each containing a `<movie>_script.txt`. Scene segmentation uses sluglines (`INT./EXT. LOCATION - TIME`) — a hard industry formatting rule. My segmenter (`src/outlier/segmentation.py`) matches TRIPOD's exactly on all 15 gold movies (verified: Die Hard produces 118 scenes; the labeled Point of No Return at scene 100 lands on the correct scene in the text).

**Scene count distribution (15 gold):** min 42 (*The Breakfast Club*) · max 230 (*The Shining*, *Soldier*) · mean 139 · median 132.

## 5. Data-quality audit — what actually cleans up

I audited every movie's labels against its segmented scene count:

| Category | Count |
|---|---|
| Silver movies with labels + script matching (clean training corpus) | **72** |
| Silver movies with data-quality issues (TP indices exceed actual scene count, or segmenter fails) | 9 |
| Silver movies with no matching script folder (naming mismatch) | 3 |
| Gold movies (all 15 clean) | 15 |
| Total training corpus (silver-clean) | **72 movies · ~10,925 scenes** |
| Held-out gold for evaluation | **15 movies** |

**Problem cases I found:**

- **Segmenter returns 0 scenes** for *Alien*, *Jaws*, *Jurassic Park (weak)*, *My Girl*, *Saw*, *The Exorcist* — these scripts don't use standard `INT./EXT.` sluglines (they're either early drafts, source-website preambled files, or use unusual formatting). Fixable with a more robust slugline detector, but for PRO322 I filter them out and use the 72 clean.
- **Off-by-a-few TP indices** in *Bridesmaids*, *Minority Report*, *The Salton Sea* — silver labels reference scenes 1–10 past the actual scene count. SUMMER used a slightly different segmenter than mine on those. Also filtered out for now.
- **Naming mismatches** — *48 Hrs* (silver) vs *48 Hrs_* (folder), *Star Wars Episode I- The Phantom Menace* (silver) vs the folder name, *When Harry Met Sally* (silver) vs the folder name. Trivially fixable with a normalization step.

**Bottom line:** ~92% of the silver corpus is usable clean; the remaining 8% is either fixable or filterable.

## 6. Where the turning points actually land (positional priors)

Computed from the 15 gold movies, TP position as fraction of total scenes:

| TP | Name | min | mean | std | max |
|---|---|---:|---:|---:|---:|
| tp1 | Opportunity | 0.03 | **0.15** | 0.10 | 0.33 |
| tp2 | Change of Plans | 0.17 | **0.38** | 0.14 | 0.64 |
| tp3 | Point of No Return | 0.33 | **0.64** | 0.18 | 0.86 |
| tp4 | Major Setback | 0.65 | **0.85** | 0.11 | 0.99 |
| tp5 | Climax | 0.92 | **0.97** | 0.03 | 0.99 |

**Interpretation:**
- **TP5 (Climax) is extremely consistent** — always in the last 8% of the script. A model that always predicts "last scene" gets close on TP5.
- **TP1 is usually early** (mean 15%) but with real variance — some films delay the inciting event significantly.
- **TP2 and TP3 are the highest-variance beats** — this is where the model has to actually read the text; positional priors don't cut it.
- **TP4 (Major Setback) usually right before the climax** (mean 85%, std 11%).

**These distributions are also the target for the "positional deviation" secondary signal** — the model's TP predictions can be compared to these expected positions to flag structurally unusual scripts.

## 7. EDA visualizations planned for the notebook

*(To be produced in `notebooks/01_data_validation.ipynb` before submission.)*

1. **TP-position box plot per TP** across gold + silver — confirms the distributions above, shows the outliers, and gives a visual anchor for the "positional deviation" metric.
2. **Scene-count histogram** — shows the range (42–230) and identifies short scripts that may be harder to model.
3. **Multi-annotator agreement heatmap** on the 22 silver movies with 3 annotators — how consistent are annotators about where each TP falls?

## 8. Preprocessing decisions and rationale

- **Scene segmentation** = sluglines. Chosen because it's TRIPOD's operationalization and verifiable against gold indices. Alternatives (parser libraries, scene-detection ML) are more brittle and don't align with the gold labels.
- **Silver labels filtered** to movies whose labels are within their script's scene count and whose segmenter returns a non-empty sequence. This removes 9 movies (~10% of silver). Documented as a known dataset limitation.
- **Naming mismatches** normalized via a small lookup (`48 Hrs` ↔ `48 Hrs_`, etc.) — trivial fix, recovers 3 movies.
- **Sentence encoding** — pretrained sentence-transformer (MiniLM-L6-v2 or all-mpnet-base-v2), frozen. Deep-learning backbone, not a lexicon model. See §10 for why VADER is being downgraded from the plan.

## 9. Deep-learning pipeline architecture (M2A1 §4)

**Ingestion pipeline (PyTorch):**

```
raw .txt screenplay
  → slugline segmentation (List[str] of scenes)
  → per-scene sentence-transformer embedding (frozen, batched on GPU)
  → padded/truncated scene sequence (max 300 scenes; median 132)
  → PyTorch Dataset yielding (scene_embeddings, tp_labels_multi_hot)
  → DataLoader with dynamic batching by scene-count bucket
```

**Model head** (trained from scratch):
- Bidirectional Transformer encoder or bi-LSTM over the scene-embedding sequence.
- Output: 5 softmax distributions over the scene positions (per-scene per-TP probability).
- Loss: cross-entropy per TP against the multi-hot gold/silver acceptable set.

**Transformation & standardization for text:**
- Tokenization / truncation handled inside the sentence-transformer (default 384-token limit per scene). Long scenes get truncated at the encoder; ~92% of scenes in the corpus fit under that limit. Reported as a known transformation loss.
- Max scene-sequence length capped at 300 to bound memory; 100% of scripts fit under that cap.

**"Overfit a single batch" test** — planned for Week 2 as the pipeline correctness check. Take 2 scripts, train the head with a high learning rate for ~200 steps, verify loss → ~0. Curve to be included in the notebook.

## 10. Deep learning on the secondary signals — replacing VADER

VADER was in the original plan as the fast baseline for scene value charge. **The plan is being upgraded to a transformer-based sentiment/emotion model** as the primary approach, with VADER kept only as a comparison baseline. Reasons:

1. VADER is a lexicon-based sentiment model tuned for social media — screenplays don't look like tweets, and dialogue-heavy scenes with subtext (character says one thing but means the opposite) are exactly where lexicon models fail.
2. A transformer emotion model (e.g. `SamLowe/roberta-base-go_emotions` — 28 emotion categories, or `j-hartmann/emotion-english-distilroberta-base` — 7 categories) reads context and handles subtext. Per scene half (open / close), it produces an emotion distribution, from which valence is derived. This is deep learning applied to the secondary signal, not a lexicon rule.
3. Alternative deep-learning path already available at zero extra cost: the **scene embedding delta** — compute cosine change between the sentence-transformer embedding of the open vs. close half of each scene. Distance from the mean-scene direction gives magnitude; sign along a learned valence axis (trained from a small sentiment-labeled subset) gives polarity.

Both approaches are deep learning. VADER stays as a comparison baseline so I can report agreement between the fast rule and the transformer read — if they agree, either is fine; if they diverge, the transformer answer is the trusted one.

## 11. Revised core requirements (M2A1 §5)

- **R1** — Data pipeline: ScriptBase segmented, TRIPOD gold + silver labels loaded, filtered to 72 clean silver + 15 gold. **DONE** (verified this week).
- **R2** — Core model: trained turning-point classifier (frozen sentence-transformer + trained sequence head), evaluated on 15 gold movies with TRIPOD's TA/PA/D metrics; pipeline proven by overfit-a-single-batch test.
- **R3** — Secondary signals: scene-turn detection via **transformer emotion model** (open/close halves), validated against ~15-script hand-labeled pilot with Cohen's kappa; escalation slope + alternation compliance derived from the turn signal.
- **R4** — Structural conformance read: three McKee-stated constraints (flat-scene rate, escalation slope across act climaxes, alternation of final two act-climax charges). Reported as conformance, not a grade.
- **R5** — Experiment tracking + hyperparameter tuning in MLflow; best model registered.
- **R6** — Streamlit demo + docs; end-to-end runnable from a fresh clone.

## 12. Known risks and mitigations

- **Small labeled dataset (72 clean silver movies).** Mitigated by transfer learning — the sentence-transformer backbone is pretrained on billions of tokens; only the head trains from scratch. Overfit-a-single-batch test proves the pipeline before real training. Cross-validation on the 72 silver used for training-set robustness.
- **Silver labels are noisy** (fixed 3-scene windows from a distant-supervision projection, not human-annotated). Mitigation: evaluate primarily on TRIPOD gold; silver is training signal, not ground truth.
- **Segmenter fails on non-standard scripts.** Mitigation: filter to the 72 that work; document limitation. A more robust segmenter is a stretch goal.
- **Sentiment ≠ McKee value** — sentiment polarity is a stated proxy for the polar flip McKee's values share, not the values themselves. Reported as a proxy. Deeper value detection (which specific McKee value is at stake) is capstone work.
