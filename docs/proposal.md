# Outlier — Project Proposal (PRO322 · M1A1)

**Author:** Joe Porter · **Mode:** Solo · **Course:** PRO322 — Applied AI Projects II (Machine Learning)

> **Scope note (revised after Week-1 review).** The PRO322 deliverable is a **deep-learning NLP model
> that detects and scores story-structure principles in a screenplay** — it *points to* the structural
> moments that make or break a film. The box-office / de-risking analysis is deliberately deferred to
> the capstone (see Appendix); this keeps PRO322 a focused, finishable applied-ML task. The full
> technical attack plan is in `concrete-attack.md`.

---

## 1. What?

Hollywood has a quiet standardization problem. Most new scripts are measured against Blake Snyder's
*Save the Cat* beat sheet, and the AI tools studios use are built to predict a success score so
executives can *avoid losses*. The effect is a conservatism engine: anything off-template reads as
"risky" and doesn't get made. Underneath that sits a **translation gap** — a creative can *feel* why a story works but can't defend it in a language an executive trusts, so original scripts default to "too risky" and die there.

**Outlier** (the PRO322 slice) attacks that gap with a **deep-learning NLP tool that reads a screenplay
and points to specific structural moments**: where the five act-structure turning points fall, which
scenes genuinely *turn* versus fall flat, and whether the story *escalates* toward its climax. For each,
it reports **how soundly the principle is applied**. The output is an annotated, plain-language read of
the script — a creative can see and defend the craft; an executive can see it too — without anyone
relying on "you can just feel it."

*ML approach in one sentence:* a **scene-sequence neural model, fine-tuned from an existing
turning-point model (TRIPOD)**, that identifies act-structure principles in a screenplay and scores
their structural soundness, surfaced as a script-level structural readout for creatives and executives.

---

## 2. Why?

I have spent years studying *why* stories work — McKee's *Story*, Frank and Ollie's *The Illusion of
Life*, structure theory — long before I could build software. What I've always wanted to do is point
*real* machine learning at the parts of a script that make or break it: not to replace the creative's
intuition, but to make it **legible and defensible**. When a model can say "here is your Point of No
Return, and here is a stretch of scenes that don't turn," that's exactly the evidence a creative needs to fight for an unconventional structure, and the evidence an executive needs to say yes.

The long-term north star (potentially capstone territory) is bigger: showing how a structurally strong original *seeds* long-term value — the goodwill that becomes sequels, theme-park attractions, and merchandise years later. That vision is why the work matters to me

---

## 3. Your Takeaway

The capability I want to prove is that I can build a **deep-learning NLP model that operationalizes a
craft theory into detectable, *validated* structure** — transfer learning and sequence modeling over
long text, plus honest evaluation against human labels. My prior projects gave me a strong ML *systems* foundation (transfer learning + MLflow registry + FastAPI/Streamlit serving; AirAlert:
orchestration; DemandCast: leakage-safe forecasting). The gap I'm closing here is **applied NLP / deep learning on text**, and the discipline of measuring a fuzzy concept against a human gold standard rather than asserting it.

---

## 4. Tech Stack

| Layer | Technology | Familiar / New | Plan to get up to speed |
|---|---|---|---|
| Language / env | **Python 3.11**, Jupyter, **PyTorch** | Familiar (Python); PyTorch partly new | Reproduce TRIPOD's baseline first, learn the codebase by running it |
| Script text | **ScriptBase** corpus (1,276 screenplays, in hand) | In hand | Already downloaded; scene-segment via slugline parsing |
| Structure labels (core) | **TRIPOD** — 99 gold + silver scene labels + pre-trained teacher/student models | New | The core X/Y; read the paper + repo, fine-tune from released weights |
| Model | transformer **scene encoder / sequence model** fine-tuned from TRIPOD; **sentence-transformers** for scene embeddings | New | Transfer-learn, don't train from scratch (handles the small dataset) |
| Turn / value signal | transformer **sentiment** for per-scene value charge (VADER as fast baseline) | Partly new | Baseline first, upgrade once the pipeline runs |
| Secondary labels (pilot) | **LLM weak-labeling** (narrative-blueprint pattern) + ~15-script hand-validated gold | New | I author the schema; validate a gold subset by hand |
| Compute | local **GPU (RTX 5070 Ti)** | Familiar | Desktop for fine-tuning; laptop for everything else |
| Experiment tracking | **MLflow** (tracking + registry) | Familiar | Reuse GreenVision setup |
| Serving / demo | **FastAPI** + **Streamlit** | Familiar | Reuse GreenVision serving pattern |
| Testing | **pytest** + data-validation checks | Familiar | — |
| *(Capstone only)* | TMDB financials + reception (**1,079-film join already prepared**) | Ready | Held for the capstone's de-risking analysis |

---

## 5. Proposed Schedule (5 weeks)

*(A standalone, high-level version of this plan is in `schedule.md`.)*

**Week 1 — Foundation.** Env + PyTorch/GPU; download TRIPOD and **reproduce its baseline turning-point
model**; scene-segment ScriptBase scripts.
*Behind-schedule signal:* can't reproduce the TRIPOD baseline, or the GPU training env isn't working.

**Week 2 — Core model.** Fine-tune the **turning-point identifier** on TRIPOD gold + silver; evaluate
against gold with the standard agreement metric.
*Behind-schedule signal:* the model doesn't beat a trivial positional baseline, with no diagnostic why.

**Week 3 — Secondary principles + pilot.** **Scene-turn** detection (value sign-flips); a ~15-script
**hand-validated pilot**; **escalation** features (turn-magnitude trend + semantic novelty) derived from
the turn signal.
*Behind-schedule signal:* turn detection disagrees badly with my hand labels and I can't explain it.

**Week 4 — Soundness scoring + tracking.** **Structural-soundness** read (turning-point presence +
position; escalation profile); tuning; MLflow experiments; error analysis.
*Behind-schedule signal:* no registered model, or evaluation is too unstable to trust.

**Week 5 — Demo + communication.** **Streamlit** app — paste a script → highlighted turning points,
flagged flat scenes, and an **escalation curve across the acts**, with a plain-language read for a
non-technical user; docs; short writeup.
*Behind-schedule signal:* the demo can't run end-to-end from a fresh clone.

---

## 6. Claude & AI Usage Plan

Full detail in `Claude.md`; logged weekly in `AI-usage.md`. **AI-assisted:** scaffolding the training/evaluation pipeline, adapting TRIPOD code, debugging, the LLM weak-labeling harness, and documentation.
**Mine to own:** the McKee→measurable-principle mapping, the annotation schema for the pilot, evaluation design, and every claim about how "sound" a script's structure is. **Guardrails (in `Claude.md`):** no fabricated metrics; respect TRIPOD's train/test split (no leakage); soundness scores are *proxies* validated against human labels, never asserted; interpretability stays central.

---

## 7. Scope Justification (pairs only)

Not applicable — this is a **solo** project.

---

## Supplementary — What gets measured (the results)

- **Turning-point identification:** agreement with TRIPOD's gold labels (their standard metric) — can
  the model locate each of the five turning points?
- **Scene-turn detection:** agreement with my ~15-script hand-labeled gold (Cohen's kappa).
- **Structural-soundness read:** are the five beats present and well-positioned, and does the story
  escalate (the escalation profile) rather than flatten or repeat?
- **Honesty discipline:** for a small case-study set (e.g., a tight 3-act film vs. a sprawling 8-act
  one), I write down the expected read *before* running the model — a result against expectation is
  information to report, not to tune away.

## Supplementary — Principles as a ladder

| Tier | Principle | Detection | Labels |
|---|---|---|---|
| **Core** | 5 act-structure turning points | scene-sequence neural model | TRIPOD (existing) |
| **Secondary (committed)** | Scene turns (flip a value, or fall flat?) | per-scene sentiment → sign-flip | ~15-script pilot |
| **Secondary (committed)** | Escalation / progression | turn-magnitude trend + novelty | none (reuses turn signal) |
| Stretch | Spine / through-line coherence | scene-embedding drift | none (computed) |
| Stretch | Forces of antagonism | LLM weak-labeling | small validated set |

Escalation is *committed*, not a maybe, because it reuses the scene-turn signal at no extra labeling
cost — and its showcase output (an escalation profile across the acts) is the cleanest visual test of
whether an unconventional structure is actually sound.

---

## Appendix — Capstone Trajectory (context, not in scope for PRO322)

PRO322 builds the **structure detector**. The **capstone** (10 weeks) feeds its outputs into the
earned-vs-loved / de-risking analysis using the **financial + reception join already prepared (1,079
films)**, toward the long-term-goodwill thesis — how structurally strong originals seed franchise,
theme-park, and merchandise value over time. Same vision, correctly sequenced.

## References

- Papalampidi et al. (EMNLP 2019), *Movie Plot Analysis via Turning Point Identification* — **TRIPOD**.
- Reagan et al. (2016), *The emotional arcs of stories are dominated by six basic shapes.*
- Gorinski & Lapata — **ScriptBase**.
- Robert McKee, *Story* — the principles being operationalized.
