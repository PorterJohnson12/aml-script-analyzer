# Outlier — Concrete Attack (PRO322, revised after review)

*A one-page response to the review feedback: what the model is, how I get training data, why it fits
five weeks, and what I can start processing next week.*

---

## The revised project, in one line

A **deep-learning NLP tool that points to specific structural moments in a screenplay** — where the
story's key turning points fall, and where scenes are working versus falling flat — so a creative can
see and defend the craft, and an executive can see it too, without anyone relying on "you can just
feel it."

The core ML question, in the professor's framing: **can a model identify a storytelling principle in
a script and assess how well it's applied?** I am *not* predicting box office in this course — that
noisy, granular piece moves to the capstone. PRO322 is about the model's structural results.

---

## The core task and its X / Y (the annotation question, answered)

**Task:** identify the **five act-structure turning points** in a screenplay and score how soundly
they're placed. In plain terms, the five are the standard beats screenwriters use:

1. **Opportunity** – the inciting event after the setup
2. **Change of Plans** – the goal of the story gets defined
3. **Point of No Return** – the protagonist fully commits
4. **Major Setback** – everything falls apart
5. **Climax** – the final confrontation

**Where the labels come from — I do *not* hand-annotate 1,000 scripts.** There is a published,
peer-reviewed dataset — **TRIPOD** (Papalampidi et al., Edinburgh) — that already provides these five
turning points annotated for **99 screenplays at the scene level**, plus **silver-standard labels for
a larger training set and pre-trained models**. So:

- **X** = the screenplay, segmented into scenes and encoded (embeddings).
- **Y** = TRIPOD's turning-point labels (which scene is each turning point).
- **Model** = a scene-sequence neural network (transformer/encoder), **fine-tuned from TRIPOD's
  released model** — not trained from scratch, which is how the small dataset stays workable.

That directly answers "would you have to annotate each script to get anything out of it?" — for the
core task, **no**. The labels exist.

## Why this fits five weeks

- **No mass annotation** — the heavy labeling is already done (TRIPOD).
- **No training from scratch** — transfer-learn from released weights; my desktop GPU (5070 Ti)
  handles fine-tuning easily.
- **The core needs only script text + TRIPOD labels** — it does *not* depend on the box-office join,
  so a whole category of data risk is removed from this course.
- **Deep learning is the right tool here** (unlike a small tabular box-office model): finding
  structure in *text* is a genuine NLP sequence problem, exactly what these models are for.

## Scope: principles as a ladder (major ones first, stretch goals if fast)

| Tier | Principle | How it's detected | Labels |
|---|---|---|---|
| **Core** | 5 act-structure turning points | scene-sequence neural model | TRIPOD (existing) |
| **Secondary (committed)** | Scene "turns" — does a scene flip a value, or fall flat? | per-scene sentiment → sign-flip; flags weak/non-event scenes | ~15-script hand-validated pilot |
| **Secondary (committed)** | Escalation / progression — do the turns escalate toward the climax, or flatten/repeat? | turn-magnitude trend + semantic novelty of successive beats | none (rides on the turn signal) |
| Stretch | Spine / through-line coherence | scene-embedding drift | none (computed) |
| Stretch | Forces of antagonism | LLM weak-labeling | small validated set |

The two **secondary** principles are where the "help creatives find weak spots" payoff lives — flat
scenes that don't turn, and stretches that fail to escalate. Both rest on one small piece of
hand-validation (~15 scripts), matching the "one or two principles on ~10 scripts" suggestion from
review. **Escalation is cheap because it reuses the scene-turn signal — no extra labels** — which is
why it's a committed goal, not a maybe. Its showcase output is an **escalation profile across the
acts**: on an 8-act film you can see at a glance whether each act ratchets the stakes up or the story
front-loads and coasts — the cleanest visual test of whether an unconventional structure is actually
sound.

## What the results look like (what gets graded)

- **Identification accuracy** against TRIPOD's gold labels (their standard agreement metric) — can the
  model locate each turning point?
- **A "structural soundness" read** per script — are all five beats present, well-separated, and in
  structurally-sound positions? Plus, from the secondary principle, which scenes don't turn.
- **A demo:** paste a screenplay → the model highlights each turning point and flags flat scenes, with
  a plain-language summary a creative *or* an executive can read. This is the "point to parts of the
  script" tool.

## Data readiness — can I start processing next week?

**Yes.**

- **Screenplays:** ScriptBase corpus already downloaded (1,276 scripts, titles + metadata flattened).
- **Labels + models:** TRIPOD is a direct download — scene-level turning-point labels, silver-standard
  training labels, and pre-trained teacher/student models.
- **Secondary pilot:** ~15 scripts, LLM-labeled then hand-checked by me.

Nothing needs to be scraped or hand-built *before* I can begin. Week 1 is environment + loading
TRIPOD + reproducing its baseline; modeling starts Week 2.

## Deferred to the capstone (10 weeks), on purpose

- Box-office / ROI prediction and the earned-vs-loved analysis.
- The long-term **goodwill** thesis — how a structurally strong original seeds franchise, theme-park,
  and merchandise value over time.

PRO322 builds the **structure detector**; the capstone uses its outputs to make the de-risking and
long-term-value argument. Same vision, correctly sequenced.

---

## Answers to your specific questions

- **Is it a whole research question / doable in 5 weeks?** The de-risking *thesis* is capstone-scale;
  the PRO322 slice — detect and score turning points from existing labels — is a focused, finishable
  applied-ML task.
- **How do I get X and Y for the model?** X = encoded scenes; Y = TRIPOD's existing turning-point
  labels. Fine-tuned neural model, no from-scratch training.
- **Would I annotate every script?** No for the core (TRIPOD). Only a ~15-script hand-validated pilot
  for the one secondary principle.
- **Compute/time?** Transfer-learning on ~100 gold + silver scripts on my GPU — very manageable.
- **Focus on results, not box office?** Agreed — box office is out of scope for PRO322.
- **Deep learning for NLP?** Yes — a scene-sequence transformer is the core model.
