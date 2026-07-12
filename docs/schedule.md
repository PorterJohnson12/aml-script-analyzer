# Outlier — 5-Week Schedule (PRO322)

**Author:** Joe Porter · Solo · PRO322 — Applied AI Projects II (Machine Learning)

*Revised at end of Week 2. Trajectory reset around building the visible differentiator (McKee-conformance layer) first, then the trained TP model, then the demo.*

---

## Core requirements (definition of done)

The project is complete when all of the following exist and are demonstrated:

- **R1 — Data pipeline:** screenplays segmented into scenes; TRIPOD gold + silver labels loaded and aligned. Verified against gold scene indices.
- **R2 — McKee-conformance layer:** three signals per script — flat-scene rate, escalation slope across act climaxes, alternation compliance on the final two act climaxes — computed from a pretrained transformer emotion model over open/close halves of each scene. Emotional-arc curve as the visible artifact.
- **R3 — Core model:** a trained turning-point identifier — bi-LSTM head on top of a frozen sentence-transformer (MiniLM) backbone — that predicts the five turning-point scene indices. Evaluated against TRIPOD's gold set with their TA/PA/D metrics. Pipeline correctness proven by an overfit-a-single-batch test before real training.
- **R4 — Business validation:** correlation between structural conformance signals and IMDB scores across the 1,276 ScriptBase films with metadata. Reported honestly with confounds acknowledged.
- **R5 — Experiment tracking + hyperparameter tuning:** all training/evaluation runs tracked in MLflow; explicit tuning pass; best model registered.
- **R6 — Demo + docs:** a Streamlit app that takes a screenplay and returns the predicted turning points, an emotional-arc curve, a list of flat scenes, and a plain-language McKee-conformance read. Runnable from a fresh clone.

---

## Week-by-week

| Week | Status | Focus | Milestone (what "done" means) | Behind-schedule signal |
|---|---|---|---|---|
| **1** | DONE | Foundation & setup | Env + GPU; TRIPOD downloaded; segmenter verified against all 15 gold movies (R1) | Can't reproduce the TRIPOD baseline, or the GPU env isn't working |
| **2** | DONE | Data understanding | Full audit of gold + silver labels; positional priors computed; data-quality report; M2A1 + M2P2 submitted | Can't explain the data at the file level |
| **3** | NOW | McKee-conformance layer + emotional arc | Emotion pipeline (DistilRoBERTa) running per scene; flat-scene rate, escalation slope, alternation compliance computed on Die Hard; emotional-arc plot as a PNG (R2) | Emotion outputs don't match a face-validity check on known scenes |
| **4** | NEXT | Trained TP model + business validation | bi-LSTM head trained on 72 silver movies; evaluated against 15 gold with TA/PA/D; overfit-a-single-batch test passes; hyperparameter tuning tracked in MLflow; IMDB correlation on ScriptBase (R3 + R4 + R5) | Model can't beat the positional baseline, or the correlation can't be computed cleanly |
| **5** | THEN | Streamlit demo + writeup | Streamlit app end-to-end (paste script → TPs + arc + flat scenes + McKee read); documentation; short writeup (R6) | Demo can't run end-to-end from a fresh clone |

---

## Why the reset — Week 3 builds the differentiator first

The Week-2 check-in flagged that the visible portion of the project overlaps too much with what TRIPOD already published. The mechanical response is to ship the differentiator earlier — the McKee-conformance layer is the piece TRIPOD does not compute, and it's cheap to build because it uses off-the-shelf pretrained models (no training loop, no dataset). Getting an emotional-arc plot for Die Hard done in Week 3 makes the "how is this different from TRIPOD" question defensible with an actual demo instead of a plan.

The trained TP model then follows in Week 4. Even if it lands at modest quality, the McKee layer stands on its own — the fallback is to use TRIPOD's expected positional priors (from the 15 gold movies) as the act-climax locations for the alternation and escalation checks, which is still deep-learning end-to-end via the emotion model.

---

## Dependencies & risk

- **R2** depends only on script text + a pretrained emotion model — both in hand.
- **R3** depends on R1 (data), MiniLM (in hand), and PyTorch training loop (standard).
- **R4** depends on `join_yield_probe.py` output — already generated in Week 1.
- **Fallback for R3:** if the trained bi-LSTM doesn't beat the positional baseline, use the positional baseline (mean TP positions from the 15 gold movies) as the act-climax locator and run the McKee-conformance layer on top. Still deep-learning end-to-end. Smaller working thing beats an aspirational broken one.
- Stretch goals (spine coherence, forces of antagonism, hand-labeled scene-turn pilot with Cohen's kappa) are added only if R1–R6 land ahead of schedule.
- Box-office / de-risking analysis is out of scope for PRO322 (capstone).

---

## Change log

- **Week 2 revision:** dropped the hand-labeled scene-turn pilot from PRO322 scope (deferred to capstone). Reordered Week 3 and Week 4 — McKee-conformance layer now precedes trained model so the differentiator is visible earlier. Added IMDB-score correlation as R4 for business validation.
