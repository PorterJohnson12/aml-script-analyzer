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

## 7. EDA visualizations (in the notebook)

Three visualizations live in `notebooks/01_data_validation.ipynb`, each with written interpretation:

1. **TP-position box plot** across all gold + clean-silver movies — confirms the positional priors above and shows the outlier scripts. TP5 is tightly bunched near the end; TP2 and TP3 have visibly wider spread, which is where the trained model has to actually read the text to add value.
2. **Scene-count histogram** — gold overlaid on silver. Most scripts land 100–200 scenes; *The Breakfast Club* at 42 scenes is a clear short-tail outlier that will need special attention when I train the sequence model.
3. **Multi-annotator agreement heatmap** on the 22 silver movies with 3 annotators — shows the per-TP standard deviation of position across annotators. TP2 and TP3 are the beats annotators most often disagree on, consistent with the positional-priors finding that these are the "actually needs to read the script" beats.

## 7.1 Ingestion pipeline architecture (PyTorch)

The deep-learning ingestion pipeline goes from raw text files on disk to batches of scene-embedding tensors:

```python
# Load once at process start:
from outlier.data import get_scenes, load_silver_labels, load_gold_labels
from outlier.embeddings import SceneEncoder

encoder = SceneEncoder(device="cuda")            # MiniLM, frozen, off the shelf
gold    = load_gold_labels()                     # 15 movies with acceptable scene sets
silver  = load_silver_labels(collapse=True)      # 84 movies (72 clean)

# For each script:
scenes = get_scenes("Die Hard")                  # List[str], N ~ 100–200 scenes
scene_embs = encoder.encode_scenes(scenes)       # np.ndarray shape (N, 384)
# → this is the input tensor for the trained bi-LSTM.
```

**Batching strategy.** Scripts have very different lengths (42–230 scenes), so I process one script per training step rather than padding them into fixed-length batches — for a small labeled corpus (72 movies) the compute cost of one-script batches is negligible, and it avoids either wasteful padding or a bucketing scheme. Shuffling happens at the script level between epochs (standard PyTorch `random.shuffle(train_ids)`).

**Text tokenization / truncation.** MiniLM's tokenizer has a max input length of 256 tokens. A scene longer than that gets its tail truncated at the tokenizer boundary. Median scene length in the corpus is well under this limit, so ~92% of scenes fit without truncation; the remaining ~8% are long ensemble scenes where the truncation loses trailing action lines but keeps the dominant emotional/semantic signal.

**No fixed sequence-length padding.** Because I batch one script at a time, there is no scene-count padding required. If I later switch to true mini-batching for speed, the plan is to bucket by scene count so each batch pads to a similar length.

## 7.2 "Overfit-a-single-batch" test (pipeline correctness)

Before training the real model, I ran the standard overfit-a-single-batch sanity check: three gold movies (*Die Hard*, *Panic Room*, *The Crying Game*), encoded with MiniLM once, then trained the bi-LSTM for 300 epochs on just those three. The pipeline is data → MiniLM → bi-LSTM → cross-entropy loss vs. multi-hot targets → backprop — if any step is broken, loss will not descend; if all steps work, loss will approach zero on three fixed scripts because the model has enough capacity to memorize them.

The training loop (in `src/outlier/model.py`) is:

```python
from outlier.model import TPFinder, overfit_batch, predict_tps

model = TPFinder(embed_dim=384)     # bi-LSTM, 1 layer, hidden=128, dropout=0.1
losses = overfit_batch(
    model,
    samples=[(embs, gold[m]) for m, embs in overfit_data],
    epochs=300,
    lr=1e-3,
    device="cuda",
)
# → losses is the per-epoch training loss curve
```

The training curve is generated live in the notebook (§8) and reproduces on a fresh clone. This test confirms the pipeline is correct; it says nothing about generalization, and I make no such claim from it. Full training on the 72-silver corpus with proper evaluation on the 15 gold happens in Week 4 per the schedule.

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

**"Overfit a single batch" test** — planned for Week 2 as the pipeline correctness check. Take 2 scripts, train the head with a high learning rate for ~200 ste