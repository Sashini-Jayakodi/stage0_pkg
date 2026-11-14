# POC-3 Implementation Plan (Revision 9): Sign-Level Certification Layer

**Working title:** Sign-Level Certification of Sequential Commit Decisions in Traffic Sign Recognition: A Model-Agnostic Learn-then-Test Layer with Pre-Registered Multi-Dataset Confirmation
**Target venue:** IEEE Access
**Purpose of this document:** a self-contained build plan for the third proof of concept (POC-3). A new chat or a new Claude Code session should be able to start from this file alone.
**No code is included on purpose.** Each stage lists what to build, what it must output, its checks, and what to do if a check fails.
**This plan does not replace** `docs/poc2-implementation-plan-r8.md`. That file is frozen (SHA-256 recorded at F1) and must not be edited. POC-3 is a new study on top of the POC-1 and POC-2 record.

Confidence tags: **[Certain]** = verified or forced by the maths. **[Likely]** = strong inference. **[Guessing]** = an estimate to be checked during the build.

---

## 0. How to use this document

1. Save this file as `docs/poc3-implementation-plan-r9.md` in `stage0_pkg`, commit it, and record its SHA-256 in `LOG.md`.
2. In Claude Code say: "Read CLAUDE.md, docs/HANDOFF.md and docs/poc3-implementation-plan-r9.md. Help me implement POC-3 Stage N."
3. Work one stage at a time. Record each stage in `LOG.md` with the same template as POC-2 (plan R8 section 13).
4. **Read sections 2 and 3 before anything else.** They explain why POC-1 and POC-2 lost their score and how POC-3 is built so the same thing cannot happen.

---

## 1. What POC-1 and POC-2 showed

### What held (keep and build on)
- A certificate computed per physical sign is valid on linked frames. On real GTSRB data at alpha = 1%, frame-level Learn-then-Test broke its promise in 20.0% of resamples (budget 10%). The sign-level certificate broke it in 1.4%. [Certain, POC-2 Stage 3b]
- Controlling the accuracy gap (Ringel et al. 2024) does not control the real error: 2.3% to 5.0% real error at a 1% gap target. [Certain, POC-2 Stage 6]
- The uncertified sequential conformal method (Boursinos and Koutsoukos 2022) went above its own error target in 1 to 3 of 5 re-splits per variant, up to 5.58% for a 2% target. [Certain, POC-2 Stage 6]
- Where the sign-level certificate was issued, the Test error stayed inside the promise in every re-split. [Certain, POC-2 Stage 4]

### What failed (do not repeat)
| Failure | Real cause |
|---|---|
| POC-1: "hard pairs are riskier" was false | The whole contribution rested on one guess about the data |
| POC-2 D1 failed | A check on a claim the paper did not need |
| POC-2 P2, P3, P4 failed (0 of 5) | The primary alpha (1%) was picked from a 261-track design set that happened to have 0 errors. The Hoeffding-Bentkus test is loose for rare errors. The test order could only relax, so the 11 strictest thresholds were never tried |
| POC-2 Stage 6: our commit score lost to B6 | Our own score (summed Gaussian margin) was sold as the method. It is weaker early on than the published conformal score |
| Certificate B not certifiable | The target (10% late) could not be met when about 11% of signs escalate |

### What changes in POC-3
1. **The contribution is the certification layer, not our score.** The layer wraps any commit score, including the published one that beat us. If the best score is someone else's, the layer still wins, because it is what turns that score into a certified one.
2. **No research question can "fail".** Every question asks how much, how often or how early. Whatever the numbers are, they are the answer.
3. **A sharper test.** An exact binomial test on the committed signs replaces Hoeffding-Bentkus as the primary test.
4. **A better test order.** Testing starts at the strictest threshold that still has enough committed signs, then relaxes.
5. **Honest alpha selection.** The primary alpha is picked with a safety margin for the small design set, so it is the alpha that is expected to certify, not the one we hope for.
6. **Fresh data.** All 1,307 GTSRB tracks have been used. Confirmation runs on BelgiumTSC (and DITS if it can be downloaded).

---

## 2. The rule for numbers (what "adjusting our numbers" means here)

Numbers may be chosen freely, but only at the right time.

| When | What is allowed |
|---|---|
| **Design phase** (design poles/tracks of a fresh dataset only) | Try any alpha grid, threshold grid, score variant, window size, start point. Change them as often as needed. Nothing here is a result. |
| **Freeze** | Write every chosen value to a config file, commit, push, record the hash. |
| **Confirm phase** (Cert and Test of the fresh dataset) | Nothing changes. Results are reported as they come out. |
| **GTSRB** | All tracks have been seen. Anything new on GTSRB is labelled **exploratory (post-hoc)**. The POC-2 results stay as recorded. |

Never allowed: changing a frozen value after seeing a Cert or Test number, dropping a re-split or baseline because of its result, or weakening a baseline. The git history is public and timestamped, so this would be visible.

Fixed in every phase: delta = 0.1. The Stage 2 engine and its tests are frozen before any real data is run through it.

---

## 3. How POC-3 avoids the old pitfalls (design rules)

**Rule 1: three kinds of checks, and only one kind can stop the project.**
- **V-checks (validity).** These must pass. They hold by mathematics if the code is right, so a failure means a bug, not a research failure. Example: the sign-level certificate's violation rate stays within delta in simulation.
- **F-gates (feasibility).** Run on design data or on dataset structure, before the freeze. Each has a written fallback. Example: BelgiumTSC pole IDs can be recovered.
- **R-results (research results).** Reported, never scored as pass or fail. Example: the smallest certifiable alpha for each score.

There are no P-checks in POC-3. POC-2's P2, P3 and P4 were R-results wrongly written as pass levels.

**Rule 2: no claim depends on our own score winning.** Four scores are wrapped. The answer to "which score is cheapest to certify" is a finding whichever score it is.

**Rule 3: the primary alpha comes from a conservative rule.** On the design set, use the upper 90% Clopper-Pearson bound of the committed error (not the raw error) in the prediction, and the real Cert size. The primary alpha is the smallest grid value predicted certifiable under that bound.

**Rule 4: hypotheses are written as expectations, not gates.** They are stated in the paper with the outcome beside them.

**Rule 5: every limitation found so far is stated up front** (section 12).

---

## 4. Research questions (two, WH form, answerable either way)

| ID | Question | Answered in |
|---|---|---|
| RQ1 | How often do frame-level (view-level) and sign-level Learn-then-Test certificates break their promise at delta = 0.1, on GTSRB approach tracks and on BelgiumTSC pole groups, and how does the size of the difference follow the within-sign error correlation and design effect? | POC-2 Stage 3 (GTSRB, done), Stage 6 (BelgiumTSC) |
| RQ2 | Under one sign-level certificate, what is the smallest error budget alpha in {0.5, 1, 2, 3, 5}% at which each of four commit scores can be certified, and what sign coverage, commit frame and sign size does each score reach at that budget, compared with the same scores without a certificate? | Stage 3 (GTSRB, exploratory), Stage 6 (BelgiumTSC), Stage 7 (DITS, if available) |

Secondary analysis inside RQ2 (not a third question): how many more signs the exact binomial test certifies compared with Hoeffding-Bentkus at the same alpha and delta.

Expectations (not gates):
- E1: frame-level certification over-violates on GTSRB (already measured). On BelgiumTSC, with about 3 views per sign, the design effect is about 1.9 at rho = 0.45, so the difference may be small or absent. Either result answers RQ1. [Likely]
- E2: the wrapped conformal score (S2) certifies at an earlier frame than the Gaussian sum (S1). [Likely, from POC-2 Stage 6]
- E3: the exact binomial test certifies 1% in cases where Hoeffding-Bentkus could not. [Likely, section 6]

---

## 5. The method

### 5.1 The certification layer
Input: any per-frame score and any rule that turns the frames seen so far into (label, confidence number). A sign commits at the first frame where the confidence number passes a threshold lambda. If it never passes, the sign escalates.

For a fixed lambda, on the calibration signs: n_c = number of committed signs, k = number of those that are wrong.
- **Primary test (exact binomial):** p = P(Binomial(n_c, alpha) <= k). Valid because, for a fixed lambda, each committed calibration sign is wrong or right independently of the others. [Likely; Stage 2 checks it by simulation]
- **Comparator test (as in POC-2):** Hoeffding-Bentkus on the loss L = wrong - alpha*committed + alpha.
- **Fixed-sequence testing** over a grid of thresholds at level delta, in an order fixed on design data (5.3).

Promise: with probability at least 1 - delta over the calibration draw, P(wrong | committed) <= alpha.

### 5.2 The four scores that are wrapped
| ID | Score | Source |
|---|---|---|
| S1 | Summed tied-Gaussian log-likelihood margin (POC-2 V1) | Ours |
| S2 | ECDF-combined conformal p-values over a window, credibility/confidence score | Boursinos and Koutsoukos 2022 (POC-2 B6 code) |
| S3 | Majority-vote count margin | POC-2 B4 |
| S4 | Last-frame (or last-view) margin | POC-2 B3 |

S2 needs its own calibration data (ICP scores and ECDF). In POC-3 this comes from the **design set only**, split in two disjoint halves by sign, so that all Cert signs stay available for the certificate. (a, b) for S2 are also fixed on the design set.

### 5.3 Threshold grid and order (fixed on design data)
- Grid: about 30 thresholds per score covering the full range from "commit almost everything" to "commit about 30% of signs". The range is wider than in POC-2 on purpose.
- **Power-aware start:** start at the strictest grid point whose design-set commit fraction, times the Cert size, times alpha, is at least 4. Then relax one grid point at a time. Stop at the first failure.
- The start point is computed per alpha, so strict budgets start at stricter thresholds.

### 5.4 What is reported per score and alpha
Certified or not; Test coverage; error among committed signs; median commit frame; median sign size at commit; D90 size (as in POC-2 Stage 5); escalation rate.

---

## 6. Sample-size table (exact binomial, delta = 0.1)

Largest number of errors k among n_c committed calibration signs that still certifies. "none" means not certifiable even with 0 errors. [Certain: computed with scipy.stats.binom; recompute in Stage 2]

| n_c | alpha 0.5% | alpha 1% | alpha 2% | alpha 3% | alpha 5% |
|---|---|---|---|---|---|
| 784 | 1 | 3 | 10 | 17 | 30 |
| 700 | 0 | 3 | 8 | 14 | 27 |
| 560 | 0 | 2 | 6 | 11 | 20 |
| 470 | 0 | 1 | 5 | 8 | 17 |
| 420 | none | 1 | 4 | 7 | 14 |
| 300 | none | 0 | 2 | 4 | 9 |
| 250 | none | 0 | 1 | 3 | 7 |

Smallest n_c that certifies with zero errors: 460 (0.5%), 230 (1%), 114 (2%), 76 (3%), 45 (5%).

What this means: with about 700 committed signs, 1% tolerates 3 errors. In POC-2 the Hoeffding-Bentkus test tolerated about 1. This is why 1% failed there. 0.5% stays out of reach on BelgiumTSC-sized data and is reported as such.

---

## 7. Datasets

| Dataset | Role in POC-3 | Status |
|---|---|---|
| GTSRB training release (1,307 tracks, 39,209 images as counted in POC-1 Stage 0) | Motivation (POC-1, POC-2 results as recorded) and an exploratory run of the layer with four scores | All tracks used. New results are exploratory |
| BelgiumTSC (62 classes; 4,575 train / 2,520 test images as distributed) | **Primary confirmation.** Fresh. Unit = pole (one physical sign, about 3 camera views taken at the same time) | Pole IDs and train/test overlap must be verified in Stage 1 |
| DITS (Italian signs; tracks of 15 frames; day, night, fog) | **Secondary confirmation of earliness**, only if it can be downloaded | Access unverified. Not required for POC-3 to succeed |

Notes:
- BelgiumTSC views are not time-ordered. On BelgiumTSC "commit time" is the number of views used (1, 2 or 3) in a fixed camera order. It can confirm validity and certifiable alpha, not approach-time earliness. [Certain, from Atif et al. 2022]
- Atif et al. 2022 report 39,210 GTSRB training images; POC-1 counted 39,209 from the official zip. Report our count and its source.

---

## 8. Baselines and the final comparison

| ID | Method | Promise | Role |
|---|---|---|---|
| Ours | Sign-level layer on S1, S2, S3, S4 | Certified real error among committed signs | The contribution |
| B1 | Frame-level (view-level) Learn-then-Test | Certified per frame, frames assumed independent | Shows why the sign must be the unit |
| B2 | One random frame per sign, Learn-then-Test | Certified real error | Valid but wasteful (the "single subsampling" idea of Dunn et al. 2023) |
| B5, B5c | Ringel et al. 2024, marginal and conditional | Certified accuracy gap | Shows gap control is not error control |
| B6 | Boursinos and Koutsoukos 2022, uncertified | None | The same score as S2 without the layer |

### Comparison table to fill (the deliverable of POC-3)
One table per dataset, at the primary alpha, median and range over 5 re-splits.

| Row | Ours S1 | Ours S2 | Ours S3 | Ours S4 | B1 | B2 | B5 | B5c | B6 |
|---|---|---|---|---|---|---|---|---|---|
| Type of promise | | | | | | | | | |
| Violation rate vs delta (resampling) | | | | | | | | | |
| Certified in how many re-splits | | | | | | | | | |
| Test coverage | | | | | | | | | |
| Test error among committed | | | | | | | | | |
| Worst re-split error vs target | | | | | | | | | |
| Smallest certifiable alpha | | | | | | | | | |
| Median commit frame (or views) | | | | | | | | | |
| Median sign size at commit | | | | | | | | | |
| Can escalate | | | | | | | | | |

### What can be claimed [be honest in the write-up]
- **Safe:** the layer gives a real-error certificate that holds on linked frames, for any score. No other method in the table has that.
- **Safe if S2 certifies:** the published score keeps most of its earliness and coverage once certified, and loses its target overruns. The table shows exactly what the certificate costs (S2 against B6).
- **Do not claim:** that our own score S1 is the best score, or that the certificate is the first use of sequence-level calibration (Ringel et al. 2024 calibrate over whole sequences).
- **Novelty wording to use:** "a certified real-error commit-or-escalate rule for traffic-sign tracks, calibrated at the sign level, model-agnostic over the commit score, with a real-data audit of frame-level against sign-level certification."

---

## 9. Stages

Hours are with Claude Code writing the code. [Guessing on all hours]

### Stage 0: Repo setup (0.5 h)
- Tag `poc2-final`. Create branch `poc3`. Save this plan in `docs/`, commit, record its SHA-256.
- Add to `CLAUDE.md`: POC-3 code goes in `src/r9_*.py` and `tests/test_r9_*.py`; no POC-1 or POC-2 file is edited; POC-3 outputs go in `runs/20261003_stage1/poc3/` and `runs/btsc/`.
- Update `docs/HANDOFF.md` with the POC-3 status table.

### Stage 1: Data audit (2 to 3 h) — F-gates
**Build**
- Download BelgiumTSC (training and testing zips). Read the readme. Work out how the pole number and camera are encoded in file names.
- Count: poles and images per pole in each release; classes; whether any pole appears in both releases.
- Try to download DITS. Record whether it works and how tracks are stored.

**F-gates**
| ID | Gate | If it fails |
|---|---|---|
| F-A1 | BelgiumTSC poles can be recovered and at least 600 test-release poles exist | Fallback 1 (section 11) |
| F-A2 | No pole is shared between the train and test release (or shared poles can be removed) | Remove shared poles; if fewer than 600 remain, Fallback 1 |
| F-A3 | DITS can be downloaded and has track IDs | Drop Stage 7. Not a failure |

### Stage 2: Certification engine v2 (3 to 4 h) — V-checks
**Build** (new file, no real data)
- Exact binomial p-value on committed signs; fixed-sequence walk; power-aware start (section 5.3).
- The Hoeffding-Bentkus comparator through the existing `src.certify.hb_pvalue`.
- A synthetic simulator with sign-level difficulty (reuse the POC-2 Stage 3a model) for 30-frame tracks and for 3-view clusters.

**V-checks (must pass; a failure is a bug)**
- V1: the table in section 6 is reproduced exactly.
- V2: in 2,000 simulated runs per setting (rho in {0, 0.3, 0.45}; cluster sizes 3 and 30; alpha 1% and 2%), the sign-level exact-binomial certificate's violation rate has an upper 95% Clopper-Pearson bound at most delta + 0.02.
- V3: hand-computed p-values match to 1e-9.
- V4: a threshold with 0 committed signs is never certified.

**Reported, not scored:** the frame-level violation rate in the same simulations; how often each test certifies.

**Then:** commit, push, record the engine's SHA-256. The engine is frozen from here.

### Stage 3: GTSRB exploratory run of the layer (3 to 4 h) — R-results
Labelled **exploratory (post-hoc)** everywhere. Uses the POC-2 Stage 4 splits (regenerated and checked against the Stage 4 coverages, as in POC-2 Stage 5).

**Build**
- Fit S2's ICP and ECDF calibration on the 261 design tracks (two disjoint halves). Fix the grids and power-aware starts for S1 to S4 from the design tracks. Commit these before any Cert number.
- Certify S1 to S4 with the exact binomial test and with Hoeffding-Bentkus at every alpha, on the 5 re-splits.
- Report the section 5.4 metrics and the smallest certifiable alpha per score.

**Why this stage comes early:** it shows within a day whether the layer works with all four scores at GTSRB scale. It cannot fail, because it only reports.

### Stage 4: BelgiumTSC model and cache (4 to 6 h)
- Cache crops at 112x112. Fine-tune a fresh ResNet-18 once on the BelgiumTSC training release (62 classes) with the POC-1 light augmentation rules. Record the checkpoint hash.
- Split the test-release poles by pole, stratified by class where possible: design 20%, pool 80%. Fix the design poles forever.
- Extract embeddings for all test-release views. Fit the tied-Gaussian head on training-release embeddings.
- **V-check:** image and class counts match Stage 1; no design pole is in the pool; every script hash-checks its inputs.

### Stage 5: BelgiumTSC design phase and freeze F3 (3 to 4 h) — design poles only
**Build**
- For S1 to S4 on the design poles: commit fraction and committed error across a wide threshold grid; S2's calibration; view order = camera index.
- Measure the within-pole error correlation and design effect on design poles.
- **Primary alpha by Rule 3:** for each alpha in the grid and each score, predict certifiability at the real Cert size using the upper 90% Clopper-Pearson bound of the design committed error. Primary alpha = the smallest alpha at which at least one score is predicted certifiable with commit fraction at least 70%.
- Grids, power-aware starts, re-split seeds 0 to 4 (Cert 75% / Test 25% of the pool), 2,000 resampling draws.

**F-gate**
| ID | Gate | If it fails |
|---|---|---|
| F-D1 | Some alpha in {1, 2, 3, 5}% is predicted certifiable for at least one score | Add 10% to the grid before the freeze and say so. If still none, Fallback 2 |

**Then:** write `configs/frozen_r9_btsc.yaml` and the pre-registration block in `LOG.md`; commit; push; record the hash. **F3 closed.** This is the last point at which any number can be changed.

### Stage 6: BelgiumTSC confirmation (3 to 4 h) — R-results
**Build**
- **Validity (RQ1):** treat the pool poles as the population. Draw Cert-size samples with replacement 2,000 times. Certify with the sign-level layer and with view-level B1. Count how often the true risk is above alpha. Report next to the measured design effect and the GTSRB numbers.
- **Certification (RQ2):** S1 to S4 on the 5 re-splits at every alpha, with both tests. B2, B5, B5c and uncertified B6 on the same splits.
- Fill the comparison table for BelgiumTSC.

**V-check:** the sign-level violation rate has an upper 95% Clopper-Pearson bound at most delta + 0.02. This is the one confirm-phase result that would sink the project if it failed (section 12), and by the Stage 2 simulations it should not.

**Everything else is reported.** If view-level B1 does not over-violate with 3 views per pole, that is the RQ1 answer for small clusters.

### Stage 7: DITS earliness (4 to 6 h, only if F-A3 passed)
Same pattern as Stages 4 to 6 by track, at alpha in {2, 5}% (the dataset is small). Reports commit frame out of 15 for S1 to S4 under the certificate. Not needed for POC-3 to count as done.

### Stage 8: Final tables (2 to 3 h)
- The three comparison tables (GTSRB exploratory, BelgiumTSC confirmatory, DITS if available).
- The exact-binomial against Hoeffding-Bentkus table (secondary analysis).
- The S2 against B6 "cost of the certificate" table: coverage, commit time and worst re-split error, with and without the layer.
- Optional, descriptive only: the six adverse cells from POC-2 with the layer on S2.

### Stage 9: Idea text and gates (1 to 2 h)
Update the idea text with the measured results, the limitations in section 12 and the three closest papers (section 13). Run both gate prompts in a fresh chat.

### Order and effort
Stages 0 to 3: about 9 to 12 h. They already show the layer working on four scores.
Stages 4 to 6: about 10 to 14 h. They give the fresh-data confirmation.
Stages 7 to 9: about 7 to 11 h.
Total: about 26 to 37 h.

---

## 10. When POC-3 counts as done ("doable")

POC-3 has shown the idea is doable when all of these are true:
1. All V-checks pass (Stage 2 and Stage 6).
2. The layer has been run on four scores on GTSRB and on BelgiumTSC and the comparison tables are filled.
3. Both research questions have a measured answer.
4. Every result, good or bad, is in `LOG.md` and pushed.

None of these depends on a particular number coming out a particular way.

---

## 11. Fallbacks (decided now)

| Situation | What to do |
|---|---|
| **Fallback 1:** BelgiumTSC poles not usable (F-A1 or F-A2) | Use DITS as the fresh dataset if F-A3 passed. If neither is usable, POC-3 stops at Stage 3 and the paper is the GTSRB study with the layer labelled exploratory, plus the POC-2 record. State the single-dataset limit. |
| **Fallback 2:** nothing predicted certifiable on BelgiumTSC even at 10% | Report the certifiability curve against the number of poles. This answers RQ2 for that dataset ("not certifiable at this size"). |
| S2 does not certify at a stricter alpha than S1 | Report it. The layer claim does not depend on which score is best. |
| View-level B1 does not over-violate on BelgiumTSC | Report it with the design effect. It is the RQ1 answer for clusters of 3. |
| DITS not available | Earliness evidence is GTSRB only, labelled exploratory. State this as a limitation. |

---

## 12. Limitations to state up front, and the one real risk

Limitations (written into the idea text and the paper):
- The promise is an average over signs, not per class or per weather condition.
- Signs are assumed independent of each other. Signs from the same road or the same day may not be.
- BelgiumTSC views are simultaneous, so it cannot show approach-time earliness.
- All GTSRB results that use the POC-3 layer are exploratory, because every GTSRB track had been used before.
- alpha = 0.5% is not certifiable at these sample sizes.
- The escalation fallback (what the vehicle does next) is out of scope.
- The parts of the method exist separately in prior work. The contribution is the combination and the audit.

**The one result that would sink POC-3:** the sign-level certificate itself breaking its promise on fresh BelgiumTSC data by more than the simulation error. Stage 2 exists to make this very unlikely before any real data is touched. [Likely]

**Softer risks:** BelgiumTSC train and test share poles (handled by F-A2); the pole count is lower than expected (the section 6 table shows what is still certifiable).

---

## 13. Closest related papers (for Gate 1) and how POC-3 differs

1. **L. Ringel, R. Cohen, D. Freedman, M. Elad, Y. Romano, "Early Time Classification with Accumulated Accuracy Gap Control," ICML 2024.** Calibrates a stopping rule with Learn-then-Test over whole sequences, and controls the accuracy gap to the full-length classifier. It never escalates. POC-3 controls the real error among committed signs and can escalate. In our own run their rule met a 1% gap target with 2.3% to 5.0% real error.
2. **D. Boursinos, X. Koutsoukos, "Selective Classification of Sequential Data Using Inductive Conformal Prediction," IEEE ICAA 2022.** Combines conformal p-values over GTSRB frames and rejects low-confidence predictions, with no certificate on the accepted error. POC-3 wraps their score in a sign-level certificate and measures what the certificate costs.
3. **J. Zhou, M. Wang, "False Sense of Safety in Selective Signal Classification," arXiv 2606.15153, 2026.** Audits selective-risk certificates and finds overruns under grouped deployment on sound and image-forensics data. POC-3 audits the same failure for near-duplicate frames of one physical object, gives the fix (the sign as the unit) and adds the commit-time dimension.

Also cite in the introduction: Lee, Barber and Willett (hierarchical conformal prediction), Dunn, Wasserman and Ramdas (one draw per group), Boursinos and Koutsoukos DDDAS 2020 (query frames until confident), Fermanian, Hebiri and Salmon (p-value aggregation over several images of one object).

---

## 14. Honest score estimate

[Guessing, informed by two literature reviews] With POC-3 completed as planned and BelgiumTSC usable: Gate 1 about 72 to 78%, Gate 2 about 8 / 7 / 8 / 8, with novelty 8 possible if the S2-against-B6 table is clean. 80% is not something this plan can promise. The ceiling is novelty: sequence-level calibration of a stopping rule already exists, so the reviewer will see the work as a solid combination with a strong audit, not a new method.
