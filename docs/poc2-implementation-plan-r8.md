# POC-2 Implementation Plan (Revision 8): Certify the Sign, Not the Frame

**Project:** Certify the Sign, Not the Frame: Track-Level Certified Commit Rules and Certified Readable Size for Traffic Sign Recognition, Tested on German and Belgian Benchmarks
**Target venue:** IEEE Access
**Purpose of this document:** a self-contained build plan for the second proof of concept (POC-2). A new chat should be able to start from this file alone.
**No code is included on purpose.** Each stage lists what to build, what it must output, the checks it must pass and what to do if it fails.
**This plan replaces** the baseline list and the scenario list in `claude/proposal-r8-certify-the-sign.md` (softmax and fixed-K baselines dropped; 6 adverse cells instead of 10).

Confidence tags: **[Certain]** = verified or mathematically forced. **[Likely]** = strong inference. **[Guessing]** = an estimate to be checked during the build.

---

## 0. How to use this document in a new chat

1. Open a new chat in the *FYP Visual Distance* project and say: "Read `claude/poc2-implementation-plan-r8.md` from the project and help me implement Stage N."
2. Work one stage at a time. A stage is done only when its **Checks** pass. Record each result in `LOG.md` (template in section 13).
3. **Read section 2 before anything else.** It says which numbers may be chosen, and when. This is what protects the project from a second failed result and from a reviewer attack.
4. Deliver code as a zip with short comments, as before.

---

## 1. What carries over from POC-1 and what is new

### Reused as is (do not rebuild) [Certain]
- Stage 0 data cache and track index: GTSRB training release, 39,209 frames, 1,307 tracks (1,306 of 30 frames, 1 of 29), with ROI boxes and frame index per image.
- The frozen ResNet-18 checkpoint (SHA-256 `8652735...0bcc`), fine-tuned once on the official test release.
- Cached embeddings and logits for all 39,209 track frames (`train_release.npz`) and the 11,344 fine-tune frames.
- The tied-Gaussian head (43 class means, one Ledoit-Wolf shared covariance, whitening transform).
- The open-set gate rule (split-conformal threshold, beta = 5%).
- The Hoeffding-Bentkus p-value function and fixed-sequence engine, with its synthetic tests.
- The scenario generator and `configs/frozen_g2.yaml` (severities frozen at commit `00e1664`).
- The latency benchmark.

### New in POC-2
- The decision unit is the **physical sign (track)**, not the frame.
- Evidence is **added up over the frames seen so far**. The system **commits** to a label when the evidence is strong enough, otherwise it **escalates** (flagged as uncertain, reason logged).
- The certificate is on the **error among committed signs**, certified over tracks.
- A **joint certificate** on error and earliness gives a "certified readable size".
- A new split protocol with a fixed **design set** (section 3).
- A second dataset, **BelgiumTSC**, for the validity result only.
- Six adverse cells instead of ten.

### What POC-1 taught us (keep in mind)
- The certifier is valid; frame-level certification is not (synthetic: 8.7% violations against a 5% budget).
- Frames in a track are strongly linked: rho = 0.446, design effect 13.9.
- Frame errors: 3,176 errors spread over 438 of the 1,307 tracks. Most tracks with errors are wrong on only some frames (about 7 wrong frames per affected track on average). [Likely] Many of those wrong frames are the early, small ones, which is exactly what adding up evidence should fix.
- The failed part was a prediction about the data (that confusable pairs carry more risk). POC-2 does not depend on any such prediction.

---

## 2. The rule for numbers: choose on the design set, freeze, then confirm

The aim is a study whose research questions get answered, not one that can fail on a guess. The way to do that honestly is **not** to change numbers after seeing results. It is to choose every number on data that is never used for the results, then freeze.

### Three phases
1. **Design phase (stages 1 to 2).** Uses the **design set only** (261 tracks, section 3). Here you are free to try score variants, alpha levels, grids and earliness targets as many times as you like. Nothing in this phase is a result.
2. **Freeze (gate F1).** Write every chosen value to `configs/frozen_r8.yaml`, write the pre-registration block in `LOG.md`, commit, record the hash.
3. **Confirm phase (stages 3 onward).** Uses Cert and Test tracks. **Nothing may change after F1.** If a check fails here, it is reported as it is.

### What you may choose in the design phase
- The score variant (section 5).
- The lambda grid.
- The primary alpha and the alpha grid.
- The earliness target (size s and late fraction gamma).
- The pass levels of checks P2 to P4 (section 8), derived from design-set numbers by the rules given there.

### What stays fixed in every phase
- delta = 0.1 and beta = 5%. [Certain: changing delta after POC-1 would look like loosening the promise.]
- The backbone checkpoint, the head, and the frozen scenario severities.
- The loss definitions in section 6.

### What counts as cheating (never do)
- Changing any frozen value after a Cert or Test number has been seen.
- Dropping a re-split, a cell or a baseline because of its result.
- Reporting a failed check as passed, or leaving it out.

### Disclosure that goes in the paper [Certain]
- The POC-1 criteria and their failure, as a pilot.
- All 1,307 tracks were seen at frame level in POC-1, so aggregate frame errors were known. The new commit rule's outcomes on Cert and Test were not.
- The design set, the freeze commit hash, and the list of design choices.

---

## 3. Split protocol (changed from R6, declared)

| Set | Tracks | Role |
|---|---|---|
| Design set | 261 (the seed-0 Fit split from POC-1) | All design choices, gate threshold tau, threshold ordering. Fixed forever. Never in Cert or Test. |
| Cert | 784 | Certification. Redrawn in every re-split. |
| Test | 262 | All reported results. Redrawn in every re-split. |

- The other 1,046 tracks (1,307 minus 261) are split by track, stratified by class, into Cert 784 and Test 262. [Certain: 784 + 262 = 1,046.]
- POC-2 uses 5 re-splits (seeds 0 to 4). The full study uses 100. Seeds 0 to 4 are the first five of the 100.
- **Why the change:** the design set was already used for the G2 freeze, so it is already "spent". Keeping it out of Cert and Test in every re-split means no design choice can leak into a result.
- Nadeau-Bengio correction factor for the full study: 1/100 + 262/784 = 0.344. [Certain]
- Zero-error floor at n = 784 and delta = 0.1: alpha = ln(10)/784 = 0.29%. [Certain]

---

## 4. Datasets and adverse cells

### GTSRB (core)
All research questions. Frames in a track are ordered by frame index, and the sign grows as the car approaches.

### BelgiumTSC (second dataset, validity result only)
- 62 classes; 4,575 training and 2,520 test images; sign sizes 11x10 to 562x438. [Certain, from published tables]
- Images with the same pole number show one physical sign, about 3 views each. [Likely; verify in Stage 7a from the dataset readme before any other work]
- Used only for RQ1 (validity and cost). Not used for earliness: too few views and no approach order.

### Six adverse cells (subset of the frozen ten, chosen by rule, not by result)
| Cell | Why kept |
|---|---|
| Rain severe | Severe band reached; frozen recertification cell |
| Side view severe | Severe band reached; frozen recertification cell |
| Fog severe | Severe band reached |
| Night severe | Severe band reached |
| Rain mild | Mild partner of a recertification cell |
| Side view mild | Mild partner of a recertification cell |

- **Rule:** keep every scenario whose severe band was reached at G2 (motion blur was not, so it is dropped), plus the mild level of the two frozen recertification scenarios.
- Severities stay exactly as in `frozen_g2.yaml`. Log this subset as a declared amendment before any scenario result on Cert or Test exists.

### Optional: real night and fog data
DITS (Italian signs) has real night and fog sequences. Stage 9 is a one-hour check of whether its classification images carry sequence IDs. Do not plan around it until that check passes. [Guessing]

---

## 5. The commit rule

For a track with frames t = 1..n (ordered by frame index):

1. **Gate.** A frame passes if its Mahalanobis++ score is at most tau (tau from the design set, one random frame per track, beta = 5%). Frames that fail add no evidence. A track with no passing frame is **rejected as unknown**.
2. **Per-frame evidence.** For each class k, the log-likelihood under the tied-Gaussian head: minus half the squared Mahalanobis distance from the frame's embedding to class mean k.
3. **Accumulated evidence.** S_t(k) = log prior(k) + sum of the per-frame log-likelihoods of all passing frames up to t.
4. **Margin.** M_t = largest S_t(k) minus second largest.
5. **Commit.** At the first t with M_t >= lambda, output the top class and stop. Record the commit frame and the ROI size (shorter side, original pixels) at that frame.
6. **Escalate.** If no frame reaches lambda, the sign is flagged as uncertain.

### Score variants to compare in the design phase
| Variant | Definition | Why it might help |
|---|---|---|
| V1 Sum (default) | As above | Closest to a sequential likelihood ratio test |
| V2 Mean | S_t divided by the number of passing frames | Less overconfident when frames are near-copies |
| V3 Window | Sum over the last 5 passing frames only | Forgets early, tiny, noisy frames |

- **Selection rule (fixed now):** at the primary alpha, pick the variant with the highest design-set commit fraction among thresholds whose design-set selective error is at most alpha/2. Ties go to V1.
- **Why summing correlated frames is still valid [Certain]:** correlation makes the raw sum overconfident, but the threshold is certified on held-out tracks, so the certificate absorbs it. Validity never depends on the Gaussian model being right; only coverage does.

---

## 6. Losses and certificates (one number per track)

For a threshold lambda and track i:
- committed_i = 1 if the track commits, else 0.
- wrong_i = 1 if it commits and the label is wrong.
- late_i(s) = 1 if the track has **not** committed at any frame whose ROI size is at most s pixels. (A track that never reaches size s counts as late only if it never commits.)

### Certificate A: error among committed signs (primary)
- Promise: with probability at least 1 - delta over the Cert draw, P(wrong | committed) <= alpha.
- Per-track loss: L_i = wrong_i - alpha * committed_i + alpha, in [0, 1]. Test H0: mean L >= alpha with the existing Hoeffding-Bentkus p-value over tracks.
- Fixed-sequence testing over the lambda grid at level delta, in the order fixed on the design set: start at the design-set best threshold and relax one grid point at a time; stop at the first failure (same rule as section 6.3 of the first plan).
- Note: the committed label can change with lambda, so losses are computed **per lambda** (a tracks-by-lambdas table), not from one score per track.

### Certificate B: joint error and earliness ("certified readable size")
- Promise: with probability at least 1 - delta, P(wrong | committed) <= alpha **and** P(late(s)) <= gamma.
- For each lambda: p_error from Certificate A's test, p_late from a Hoeffding-Bentkus test of H0: mean late_i(s) >= gamma. Use p = max(p_error, p_late). [Certain: the maximum is a valid p-value for "at least one promise is broken", so delta is not split.]
- Fixed-sequence testing in a design-set order: rank grid thresholds by their design-set plug-in p-value, smallest first. Among the thresholds that pass, report the one with the smallest design-set late fraction.
- The pair (s, gamma) is chosen in the design phase and frozen. gamma default 10%.

### Rejected-as-unknown tracks
Count as not committed in both certificates. Report the track-level false-rejection rate separately.

---

## 7. Research questions, hypotheses and where each is answered

| ID | Question | Answered in |
|---|---|---|
| RQ1 | How often does a per-frame certificate break its promise when used on linked frames, and what does certifying per sign cost or gain in sign coverage? (GTSRB and BelgiumTSC) | Stages 3, 4, 7 |
| RQ2 | How early, in frame number and sign pixel size, can a sign be committed under a certified error rate, compared with last-frame and majority-vote rules? (GTSRB) | Stages 4, 5 |
| RQ3 | How do the certified threshold, coverage and earliness change in six adverse cells, and how many labelled tracks are needed to recertify? (GTSRB) | Stage 8 |

| ID | Hypothesis | Why it is expected |
|---|---|---|
| H1 | Per-frame certification breaks the delta budget on linked frames; track-level does not | Design effect 13.9; already shown on synthetic data [Certain for synthetic, Likely for real] |
| H2 | At the same certified alpha, the commit rule covers more signs than certifying one frame per sign | Many frames carry more evidence than one [Likely] |
| H3 | The median certified commit happens before the last frame | Frame accuracy is 92.6%, so evidence builds fast [Likely] |
| H4 | Severe cells raise the smallest certifiable alpha and delay commit; recertifying needs about ln(1/delta)/alpha tracks when there are no errors | Arithmetic of the bound [Certain for the formula, Likely for the data] |

Every RQ is a "how much / how early / how many" question, so it has an answer whichever way the numbers fall. The hypotheses are directional and can still fail; a failure is reported.

---

## 8. Gates and checks

### D-gate: design go/no-go (design set only, before the freeze)
| ID | Check | If it fails |
|---|---|---|
| D1 | Track error of the accumulated rule at the last frame (commit everything) is clearly below the single-frame error of about 7.4% | See section 11, fallback 1 |
| D2 | Plug-in test: some alpha in {0.5, 1, 2, 3, 5}% is predicted certifiable at n = 784 with design-set commit fraction at least 70% | Try V2, V3; if still failing, fallback 1 |
| D3 | Number of design-set tracks wrong on at least 25 of their frames, and their accumulated margin. This is the error floor at full coverage. | Informational; it sets the realistic alpha |

The D-gate costs nothing in integrity, because it touches no Cert or Test track. **This is where the project is protected from a second failure.** Do not go past it until D1 and D2 pass.

**Plug-in test for D2:** for each lambda, take the design-set commit fraction c and committed-error rate e, form the expected mean loss alpha*(1 - c) + c*e, and compute the Hoeffding-Bentkus p-value as if n = 784. Call alpha "predicted certifiable" if some lambda has p <= delta/10 (the factor 10 is a safety margin) and c >= 0.70.

**Primary alpha:** the smallest alpha in the grid that is predicted certifiable. [Guessing: 1% or 2%.]

### F1: freeze
`configs/frozen_r8.yaml` holds: score variant, lambda grid, threshold order, alpha grid, primary alpha, (s, gamma), tau, the six cells, the pass levels of P2 to P4. Commit and record the hash in `LOG.md`.

### P-checks: confirm phase (5 re-splits)
| ID | Check | Pass level (written at F1) |
|---|---|---|
| P1 | Validity. Synthetic and resampling tests (Stage 3): track-level violation rate at most delta; frame-level clearly above delta | Fixed here: track-level upper Clopper-Pearson bound <= delta + 0.02; frame-level lower bound > delta |
| P2 | Certifiability. The primary alpha is certified, with Test sign coverage at least c_min | In at least 4 of 5 re-splits; c_min = design-set predicted coverage minus 10 points |
| P3 | Advantage. At the primary alpha, commit-rule coverage is higher than one-random-frame-per-track coverage | In at least 4 of 5 re-splits |
| P4 | Earliness. Median commit frame is before the last frame | In at least 4 of 5 re-splits |

---

## 9. Baselines and the final comparison

All run on the same cached per-frame outputs and the same splits.

| ID | Existing method | How it is run here | What it lacks |
|---|---|---|---|
| B1 | Frame-level Learn-then-Test (Geifman and El-Yaniv 2017 style; Angelopoulos et al. 2025) | Threshold on the per-frame margin, frames treated as independent | Not valid on linked frames |
| B2 | One-random-frame-per-track Learn-then-Test | Valid, uses one frame per sign | Throws evidence away |
| B3 | Last-frame-only Learn-then-Test | Valid, uses the largest frame | Latest possible decision |
| B4 | Majority vote over all frames (Gross et al. 2023 fusion) with a track-level certificate on the vote margin | Valid | Needs every frame before deciding |
| B5 | Ringel et al. 2024 stopping rule (marginal accuracy-gap control) | Their rule on the same accumulated score | Controls the gap to the full-sequence answer, not the real error; never escalates |
| B6 | Boursinos and Koutsoukos 2022 p-value combination | Sliding-window combination (arithmetic mean and ECDF-min, K = 5), threshold from the risk-coverage curve as they describe | No certified error |

### Comparison table to fill at the end (the deliverable of POC-2)
| Metric | Ours | B1 | B2 | B3 | B4 | B5 | B6 |
|---|---|---|---|---|---|---|---|
| Type of promise (certified real error / gap / none) | | | | | | | |
| Valid on linked frames (violation rate vs delta, Stage 3) | | | | | | | |
| Sign coverage at the primary alpha (Test) | | | | | | | |
| Error among committed signs (Test) | | | | | | | |
| Smallest certifiable alpha at n = 784 | | | | | | | |
| Median commit frame | | | | | | | |
| Median sign size at commit (pixels) | | | | | | | |
| Can escalate | | | | | | | |

### Which claims are safe [be honest in the write-up]
- **Safe:** valid where B1 is not; certified where B6 is not; a promise on real error where B5 only bounds a gap.
- **Likely:** more coverage than B2 and B3 at the same alpha; earlier than B3 and B4.
- **Do not claim:** more coverage than B4. Majority vote sees every frame, so its final accuracy may match ours. Our edge over B4 is deciding earlier, with a certificate.
- **B5:** present as "what each method promises", plus B5's measured real error among stopped signs. Do not present it as a win on B5's own target.

---

## 10. Stages

Hours are for your time with Claude writing the code. [Guessing on all hours]

### Stage 1: Accumulation and track table (2 to 3 h)
**Build**
- Per-frame class log-likelihoods for all 39,209 frames from the cached embeddings and the head.
- Running sums, margins and top class per frame, for V1, V2, V3.
- A per-track table: for each lambda in a wide grid, committed, wrong, commit frame, ROI size at commit.

**Checks**
- The frame-level label from one frame's log-likelihood plus prior reproduces the POC-1 head accuracy (92.6%) exactly.
- At the last frame of V1, the top class equals the argmax of the summed log-likelihoods computed a second, independent way.
- Frame order: within tracks, ROI size rises with frame index (report the median per-track Spearman correlation; expect clearly positive).
- The 29-frame track is handled; nothing hard-codes 30.
- Commit fraction never rises when lambda rises.

### Stage 2: Design phase and D-gate (3 to 4 h, design set only)
**Build**
- Error-versus-frame curve of the accumulated rule on the 261 design tracks.
- For each variant and lambda: commit fraction, committed-error rate, median commit frame and size.
- D1, D2, D3. The plug-in table for every alpha in the grid.
- Choice of variant, primary alpha, lambda grid (about 30 points around the useful range), threshold order, and (s, gamma). For s, take the design-set median commit size at the best threshold for the primary alpha, rounded up to the next 5 pixels.
- Pass levels for P2 to P4.

**Checks**
- A code assertion that no track outside the design set is read in this stage.
- D1 and D2 pass.

**Then:** write `frozen_r8.yaml` and the pre-registration block, commit. **F1 closed.**

### Stage 3: Validity tests (3 to 4 h)
**3a Synthetic.** Extend the POC-1 Beta-binomial simulator to the commit rule: 30-frame tracks, a latent per-track difficulty, rho_sim in {0, 0.3, 0.45}, 784 calibration tracks, 1,000 runs, a 200k-track population for the true risk. Compare the track-level certifier with B1.

**3b Resampling on real data.** Treat the 1,046 non-design tracks as the population, so the true risk of any threshold is known exactly. Draw 784 tracks with replacement 2,000 times; certify with the track-level rule and with B1; count how often the true risk exceeds alpha.

**Checks:** P1. Hand-check three p-values. Confirm the Certificate B max-p rule on a synthetic case where only the lateness promise is broken.

**If B1 does not visibly break on real data:** report it as measured. The synthetic result and the design effect still stand, and the claim becomes "frame-level certificates overstate the evidence by a factor of about 14" with the measured violation rate beside it.

### Stage 4: Certificate A on five re-splits (3 to 4 h)
**Build:** Cert/Test splits of the 1,046 tracks (seeds 0 to 4); certify at every alpha in the grid; Test coverage, committed error, commit frame and size; the same for B2, B3, B4.
**Checks:** P2, P3, P4. Splits are disjoint, every class is in Cert, the design set is in neither.

### Stage 5: Certificate B, the certified readable size (2 to 3 h)
**Build:** joint certificate at the frozen (s, gamma); Test check of both promises; a descriptive curve of commit fraction against sign size on Test.
**Checks:** on Test, both realised rates are within their targets in at least 4 of 5 re-splits (descriptive at 5 re-splits). If the joint test certifies nothing, report the pair as "not certifiable at n = 784" and show the curve; do not move s or gamma.

### Stage 6: Published baselines B5 and B6, and the comparison table (4 to 6 h)
**Build:** B5 and B6 as in section 9; fill the comparison table for the clean data.
**Checks:** B5's own target (accuracy gap <= its alpha) holds on Test, which shows it was implemented correctly. B6's combined p-values reproduce the direction reported in its paper (combination improves calibration).

### Stage 7: BelgiumTSC (5 to 8 h)
- **7a (1 h, do first):** download; read the readme; confirm how pole numbers are encoded; count poles and images per pole in each release; confirm no pole is in both releases. **Stop if poles cannot be recovered.**
- **7b:** cache at 112x112; fine-tune a fresh ResNet-18 once on the training release (62 classes) with the same light augmentation rules; fit the head.
- **7c:** split the test-release poles into design / Cert / Test (about 20 / 60 / 20 by pole); measure the within-pole error correlation and design effect; run Stage 3b and Stage 4 with the pole as the unit and the views in file order.
- **Checks:** assertions on image and class counts against the published numbers; the design effect is reported whatever its size.
- **Honest limit:** about 3 views per sign. This tests "the sign is the unit", not earliness.

### Stage 8: Six adverse cells and recertification (5 to 7 h)
**Build**
- Scenario embeddings for all tracks in the six cells with the frozen generator (GPU, minutes to an hour). [Likely]
- **Stress test:** apply the clean-certified threshold in each cell; Test coverage, committed error, commit frame and size.
- **Recertification curve** in rain severe and side view severe: subsample 100, 200, 400 and 784 Cert tracks of the shifted data, certify, record the smallest certifiable alpha and the Test coverage.
**Checks:** image hashes reproduce; the design set is used only to preview commit fractions, never to change a frozen value.
**If a cell makes the certificate nearly empty:** that is the answer to RQ3 for that cell. Report it.

### Stage 9: Optional extras
- **9a (1 h):** DITS check for sequence IDs and real night/fog classification crops.
- **9b (1 to 2 h):** latency of the commit rule (one running-sum update per frame; expect well under the 1 ms budget). [Likely]
- **Full study only:** 100 re-splits, corrected t-tests with Holm, a second backbone, public pre-registration.

### Order and effort
Stages 1 to 6 are the core POC-2: about 17 to 24 h. With BelgiumTSC and the adverse cells: about 27 to 39 h.
**Do Stages 1 and 2 first and stop at the D-gate.** Everything else depends on it.

---

## 11. Fallbacks (decided now, not after the data)

| Situation | What it means | What to do |
|---|---|---|
| **Fallback 1:** D1 or D2 fails on the design set | Adding up evidence does not lower track error enough, usually because some tracks are confidently wrong on every frame | Nothing is lost: no Cert or Test track was touched. Options, in order: V2 or V3; a higher primary alpha (3% or 5%); the fallback paper (track-level versus frame-level validity plus the recertification cost curve, no earliness claim). |
| P2 fails | The design set predicted too well | Report the certifiability curve (smallest certifiable alpha against n). Do not move alpha. |
| P3 fails | One good frame is as useful as many | Report it. The paper leans on validity, earliness and recertification. |
| P4 fails | Signs only become readable at the end | Report the size at which they become readable; that is still the answer to RQ2. |
| BelgiumTSC poles not recoverable | No second grouped dataset | Drop Stage 7; state the single-dataset limit; try 9a. |

---

## 12. Risk register

| # | Risk | Likelihood [Guessing] | Impact | Early signal | Mitigation |
|---|---|---|---|---|---|
| R1 | Tracks that are wrong on every frame set an error floor above the target alpha | Medium | High | D3 count on the design set | Choose alpha from D2; V2/V3; fallback 1 |
| R2 | Head overconfidence (fitted on frames at 99.96% accuracy) makes summed margins huge on wrong tracks | Medium | Medium | Design-set error does not fall as lambda rises | V2 or V3; the certificate stays valid either way |
| R3 | Losses computed from one score per track instead of per lambda | Medium (easy bug) | Critical | Committed label differs across lambdas | Stage 1 builds the tracks-by-lambdas table; unit test with a track whose top class flips |
| R4 | Design-set leakage into Cert or Test | Low | Critical | Assertion in every stage | Fixed design set; assertion on track IDs |
| R5 | B1 does not visibly break on real data | Medium | Medium | Stage 3b | Report measured rate; keep the synthetic and design-effect evidence |
| R6 | Joint certificate certifies nothing | Medium | Medium | Stage 2 plug-in for Certificate B | Choose (s, gamma) from the design set with the same safety margin |
| R7 | BelgiumTSC pole IDs or split assumptions wrong | Medium | Medium | Stage 7a | Stop at 7a; fallback row above |
| R8 | Frame order is not approach order for some tracks | Low | Medium | Stage 1 Spearman check | Report; earliness uses frame index as recorded |
| R9 | Values changed after the freeze | Medium (human) | Critical for review | Diff of `frozen_r8.yaml` | Commit hash in `LOG.md`; section 2 |
| R10 | Effort overrun | Medium | Medium | Stage 6 past 6 h | Core first (Stages 1 to 6); 7 and 8 after |

---

## 13. `LOG.md` template

One entry per stage or decision:
- Date / stage / run tag
- Config hash / checkpoint hash / git commit
- What was done (one line)
- Result numbers (copied from the results JSON)
- Checks: pass or fail for each
- Decision taken and why (cite the section of this plan)
- Deviation from this plan? yes/no; if yes, what and why
- Next step

### Pre-registration block to write at F1 (before Stage 3)
1. Score variant and why it was chosen (with the design-set table).
2. Lambda grid and threshold order.
3. Alpha grid and primary alpha.
4. (s, gamma) for Certificate B.
5. Gate threshold tau.
6. The six cells and the rule that selected them.
7. Pass levels of P1 to P4.
8. The baselines B1 to B6 and how each is run.
9. The fallbacks in section 11.
