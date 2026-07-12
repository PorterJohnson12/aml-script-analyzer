# Outlier — Project Proposal (PRO322 · M1A1)

**Author:** Joe Porter · **Mode:** Solo · **Course:** PRO322 — Applied AI Projects II (Machine Learning)

> **Scope note (revised after Week-1 review).** The PRO322 deliverable is a **deep-learning NLP model
> that detects and scores story-structure principles in a screenplay** — it *points to* the structural
> moments that make or break a film. The box-office / de-risking analysis is deliberately deferred to
> the capstone (see Appendix); this keeps PRO322 a focused, finishable applied-ML task. The full
> technical attack plan is in `concrete-attack.md`; detailed answers to the Week-1 review notes are in
> `notes-resolutions.md`.

---

## 1. What?

Hollywood has a quiet standardization problem. Most new scripts are measured against Blake Snyder's
*Save the Cat* beat sheet, and the AI tools studios use are built to predict a success score so
executives can *avoid losses*. The effect is a conservatism engine: anything off-template reads as
"risky" and doesn't get made. Underneath that sits a **translation gap** — a creative can *feel* why
a story works but can't defend it in a language an executive trusts, so original scripts default to
"too risky" and die there. Even the gold-standard human process for closing that gap — a
Pixar-Braintrust-style room of trusted creatives — is expensive and doesn't scale beyond flagship
titles.

**Outlier** (the PRO322 slice) is a deep-learning NLP tool that reads a screenplay and points to
specific structural moments — where the five act-structure turning points fall, which scenes
genuinely *turn* versus fall flat, and whether the story *escalates* toward its climax. For each, it
reports conformance to specific structural principles McKee names, without inventing a "good/bad"
verdict the theory refuses to give. The output is an annotated, plain-language read of the script —
a structured diagnostic a creative can use to defend an unconventional choice and a development
executive can use to defend a greenlight, without either side relying on "you can just feel it."

*ML approach in one sentence:* a **scene-sequence neural model (bi-LSTM head) trained on TRIPOD's
turning-point labels** on top of a frozen pretrained sentence-transformer (MiniLM) — standard
transfer learning — combined with a separate emotional-signal branch (pretrained DistilRoBERTa
emotion classifier over open/close halves of each scene) that produces the McKee-conformance
readout. Three models total: two pretrained tools (MiniLM, DistilRoBERTa) used off the shelf, plus
one trained model (the bi-LSTM) that is mine. The bi-LSTM is the ML content of the project — same
shape as fine-tuning a plant classifier head on top of a pretrained image backbone.

*Pipeline shape:* one pipeline, two branches that merge at the output. The **structural branch**
(MiniLM → bi-LSTM) predicts the five turning points. The **emotional branch** (DistilRoBERTa over
scene halves → per-scene valence) produces the raw signal for the McKee-conformance layer. The two
branches merge at the analysis step — escalation and alternation each need both a TP location and a
valence signal — and the final Streamlit UI shows the unified read.

---

## 2. Why?

I have spent years studying *why* stories work — McKee's *Story*, Frank and Ollie's *The Illusion of
Life*, structure theory — long before I could build software. What I've always wanted to do is point
*real* machine learning at the parts of a script that make or break it: not to replace the creative's
intuition, but to make it **legible and defensible**. When a model can say "here is your Point of No
Return, and here is a stretch of scenes that don't turn," that's the evidence a creative needs to
fight for an unconventional structure, and the evidence an executive needs to say yes to it. This is
a **Braintrust-scaling** tool — it doesn't replace the room of trusted creatives; it runs the
mechanical pass they'd do in their heads, so their taste gets applied to the questions that need it.

The long-term north star (capstone territory) is bigger: showing how a structurally strong original
*seeds* long-term value — the goodwill that becomes sequels, theme-park attractions, and merchandise
years later. That vision is why the work matters to me.

---

## 3. Your Takeaway

The capability I want to prove is that I can build a **deep-learning NLP model that operationalizes a
craft theory into detectable, *validated* structure** — transfer learning and sequence modeling over
long text, plus honest evaluation against human labels. My prior projects gave me a strong ML
*systems* foundation (transfer learning + MLflow registry + FastAPI/Streamlit serving; AirAlert:
orchestration; DemandCast: leakage-safe forecasting). The gap I'm closing here is **applied NLP /
deep learning on text**, and the discipline of **measuring a fuzzy concept against a human gold
standard** rather than asserting it.

---

## 4. Tech Stack

| Layer | Technology | Familiar / New | Plan to get up to speed |
|---|---|---|---|
| Language / env | **Python 3.11**, Jupyter, **PyTorch** | Familiar (Python); PyTorch partly new | Reproduce TRIPOD's baseline first, learn the codebase by running it |
| Script text | **ScriptBase** corpus (1,276 screenplays, in hand) | In hand | Already downloaded; scene-segment via slugline parsing |
| Structure labels (core) | **TRIPOD** — 99 gold + silver scene labels | New | The core X/Y; read the paper + repo; train the sequence head on the labels (the pretrained network from their paper is not loaded — a different architecture) |
| Model backbone | **Frozen sentence-transformer** (MiniLM / MPNet) for scene embeddings | New | Reuse via `sentence-transformers` library — leaves-and-plants pattern |
| Model head | **Trained** scene-sequence classifier (Transformer encoder or bi-LSTM) | New | Standard PyTorch training; MLflow-tracked |
| Turn / value signal | Transformer **sentiment** for per-scene value charge (VADER as fast baseline) | Partly new | Baseline first, upgrade once the pipeline runs |
| Secondary labels (pilot) | **~15-script hand-labeled gold** + LLM weak-labeling as stretch | New | I author the schema; validate a gold subset by hand; Cohen's kappa vs a second annotator |
| Compute | Local **GPU (RTX 5070 Ti)** | Familiar | Desktop for fine-tuning; laptop for everything else |
| Experiment tracking | **MLflow** (tracking + registry) | Familiar | Reuse GreenVision setup |
| Serving / demo | **FastAPI** + **Streamlit** | Familiar | Reuse GreenVision serving pattern |
| Testing | **pytest** + data-validation checks + **overfit-a-single-batch** test | Familiar | — |
| *(Capstone only)* | TMDB financials + reception (**1,079-film join already prepared**) | Ready | Held for the capstone's de-risking analysis |

---

## 5. Proposed Schedule (5 weeks)

*(A standalone, high-level version of this plan is in `schedule.md`.)*

**Week 1 — Foundation.** Env + PyTorch/GPU; download TRIPOD and **reproduce its baseline turning-point
model** as a reference number; scene-segment ScriptBase scripts.
*Behind-schedule signal:* can't reproduce the TRIPOD baseline, or the GPU training env isn't working.

**Week 2 — Core model.** Build the PyTorch DataLoader; **"overfit a single batch"** test proving the
pipeline works before real training; **train my own turning-point head** on TRIPOD gold + silver
(frozen sentence-transformer backbone); evaluate vs. gold with TRIPOD's standard agreement metric.
*Behind-schedule signal:* the model doesn't beat a trivial positional baseline, with no diagnostic why.

**Week 3 — Secondary principles + pilot.** **Scene-turn** detection (value sign-flips); a ~15-script
**hand-validated pilot** with Cohen's kappa; **escalation** features (turn-magnitude trend + semantic
novelty) derived from the turn signal.
*Behind-schedule signal:* turn detection disagrees badly with my hand labels and I can't explain it.

**Week 4 — Soundness scoring + hyperparameter tuning.** **Structural-conformance read** (turning-point
presence + position; escalation profile; alternation of act-climax charges — the McKee-stated
constraints); explicit **hyperparameter tuning** with MLflow-tracked experiments; best model
registered; error analysis.
*Behind-schedule signal:* no registered model, or evaluation is too unstable to trust.

**Week 5 — Demo + communication.** **Streamlit** app — paste a script → highlighted turning points,
flagged flat scenes, and an **escalation curve across the acts**, with a plain-language read for a
non-technical user; docs; short writeup.
*Behind-schedule signal:* the demo can't run end-to-end from a fresh clone.

---

## 6. Claude & AI Usage Plan

Full detail in `Claude.md`; logged weekly in `AI-usage.md`. **AI-assisted:** scaffolding the
training/evaluation pipeline, adapting TRIPOD code, debugging, the LLM weak-labeling harness (stretch),
and documentation. **Mine to own:** the McKee→measurable-principle mapping, the annotation schema for
the pilot, evaluation design, and every claim about what "structural conformance" means. **Guardrails
(in `Claude.md`):** no fabricated metrics; respect TRIPOD's train/test split (no leakage); soundness
signals are *proxies* validated against human labels, never asserted; interpretability stays central.

---

## 7. Scope Justification (pairs only)

Not applicable — this is a **solo** project.

---

## Supplementary — What gets measured (the results)

- **Turning-point identification:** agreement with TRIPOD's gold labels (their standard TA/PA/D metrics) — can
  the model locate each of the five turning points?
- **Scene-turn detection:** agreement with my ~15-script hand-labeled gold (Cohen's kappa).
- **Structural conformance read:** three McKee-stated constraints, each countable — every scene should
  turn (flat-scene rate); successive act climaxes should escalate in magnitude (escalation slope); the
  last two act-climax charges should alternate sign ("you cannot set up an up-ending with an up-ending"
  — verbatim, Ch. 9). Reported as conformance / violation, **not** as a grade.
- **Honesty discipline:** for a small case-study set (e.g., a tight 3-act film vs. a sprawling 8-act
  one), I write down the expected read *before* running the model — a result against expectation is
  information to report, not to tune away.

## Supplementary — Principles as a ladder

| Tier | Principle | Detection | Labels |
|---|---|---|---|
| **Core** | 5 act-structure turning points (TRIPOD operationalization) | scene-sequence neural model | TRIPOD (existing) |
| **Secondary (committed)** | Scene turns (flip a value, or fall flat?) | per-scene sentiment → sign-flip | ~15-script pilot |
| **Secondary (committed)** | Escalation / progression | turn-magnitude trend + novelty | none (reuses turn signal) |
| **Secondary (committed)** | Alternation of act-climax charges | sign check on final two act climaxes | derived from TP + turn signals |
| Stretch | Spine / through-line coherence | scene-embedding drift | none (computed) |
| Stretch | Forces of antagonism | LLM weak-labeling | small validated set |

Escalation and alternation are *committed*, not maybes, because they reuse the scene-turn signal at
no extra labeling cost — and each maps to a specific principle McKee names verbatim in Ch. 9. The
showcase visual output is an **escalation profile across the acts** — the cleanest read of whether an
unconventional structure is actually sound.

## A framing note on "five turning points"

The five-TP scheme (Opportunity / Change of Plans / Point of No Return / Major Setback / Climax) is
the **TRIPOD / Syd Field operationalization** of a broader tradition McKee sits inside. McKee himself
uses a five-*part* form (Inciting Incident → Progressive Complications → Crisis → Climax → Resolution)
and treats turning points as *hierarchical and numerous* — one per scene ideally, then sequence, act,
and story climax. TRIPOD's five-TP scheme is a published, defensible operationalization compatible
with McKee, but it isn't literally *"McKee's five turning points."* This proposal separates them
cleanly: **TRIPOD provides the labeled operationalization** the core model is trained against;
**McKee provides the principles the secondary signals actually measure** (turn, progression,
alternation of act-climax charges).

---

## Appendix — Capstone Trajectory (context, not in scope for PRO322)

PRO322 builds the **structure detector**. The **capstone** (10 weeks) adds three layers on top of a
detector that already works:

1. **Deeper value detection** — LLM weak-labeling of *which* McKee value is at stake in each scene
   (love/hate, freedom/slavery, truth/lie, etc.), validated against a hand-labeled gold set, then
   distilled into a trained classifier.
2. **The Braintrust-facing explanation layer** — an LLM that surfaces *the questions a Braintrust
   would ask about this beat*, grounded on the model's structural findings and required to cite the
   script. Not "here is your score" — "here are the questions your taste should be applied to." The
   trained core keeps the LLM honest on scripts it doesn't know.
3. **Aggregate credibility for execs** — relate structural reads to reception using the 1,079-film
   join already prepared, **controlling for budget and stars**, so structure's *own* share of variance
   is what's measured. Correlational, honest, not a good/bad verdict.

Same vision, correctly sequenced.

## References

- Papalampidi et al. (EMNLP 2019), *Movie Plot Analysis via Turning Point Identification* — **TRIPOD**.
- Reagan et al. (2016), *The emotional arcs of stories are dominated by six basic shapes.*
- Gorinski & Lapata — **ScriptBase**.
- Robert McKee, *Story* (1997) — the principles being operationalized.
