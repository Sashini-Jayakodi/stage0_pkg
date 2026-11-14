## POC-3 (R9) Stage 3: GTSRB run of the layer on four scores (exploratory, post-hoc)

### Part A: fixed definitions (committed before any Stage 3 number exists)

Plan: `docs/poc3-implementation-plan-r9.md` (git blob 0abe7fa51af3b499cd79218e46c83790dbfa2e99), sections 2, 3, 5 and
9 Stage 3. **Every result of this stage is exploratory (post-hoc).** All 1,307 GTSRB tracks were used in POC-1 and
POC-2, so nothing here is a confirmation, and the POC-2 results stay as recorded. delta = 0.1 throughout. Stage 3
produces R-results only: they are reported, never scored as pass or fail. The integrity checks in (k) are not
results; if one fails, the run stops and the failure is logged.

#### (a) Scope and inputs
- GTSRB only. No BelgiumTSC file is opened (`data/btsc_raw/`, `runs/btsc/`); a code assertion and a test check this
  (hard rule 3: no BelgiumTSC pool-pole number before F3).
- Design set: the 261 seed-0 Fit tracks (`track_split == 0` in `splits/seed0.npz`), read through
  `src.r8_stage2.load_inputs` (hash-checked against `stage1_summary.json`; refuses `track_table_pool.npz`). None of the
  261 is a track without a passing frame (`LOG_r8_stage2_entry.md`, D1: "0 tracks with no passing frame").
- Pool: the 1,046 other tracks. The 5 tracks with no passing frame (Stage 1) are therefore all in the pool. They stay
  in the pool and escalate under every score: they count in n and in the Test denominator, never in n_c.
- Engine: `src/r9_certify.py` (frozen, git blob 158a9c9e32ab8ec1f568f75b5c35c9992711db74), called through
  `certify(k, n_c, n, alpha, f_design, test, n_design=...)` and `start_for(...)`, with the grid ordered loosest first.
  Nothing in the engine is changed or copied.

#### (b) Freeze check (`src/r9_frozen.py`, new)
- Before anything else, every Stage 3 script that uses the engine calls `check_frozen()`. It compares the git blob ID
  of the engine (158a9c9e...), the engine tests (ecb6e044...) and plan R9 (0abe7fa5...) with the IDs recorded in
  `LOG.md` (Stage 0 and Stage 2 entries) and stops on any difference. The blob ID is computed in Python as git does
  (SHA-1 of `blob <length>\0` followed by the bytes) after CRLF is turned into LF, so a line-ending-only difference
  passes and does not depend on the machine's git settings. A test checks that it equals `git hash-object`.
- It also checks that the imported `src.r9_certify` module is the file it checked.
- `configs/frozen_r9_btsc.yaml` is added to the list at Stage 5. `src/r9_stage2_vchecks.py` stays as recorded and
  does not call the guard.

#### (c) Re-splits (POC-2 Stage 4, regenerated)
- Seeds 0 to 4. Per seed, `src.r8_stage4.make_split(y_track, seed, 262)` over the pool tracks in track-code order gives
  262 Test and 784 Cert tracks. Split membership was never saved, so it is regenerated.
- Before use: `src/r8_stage4.py` SHA-256 must equal fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2
  and `stage4_summary.json` SHA-256 e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc
  (`LOG_r8_stage4_entry.md` Part B). The regenerated splits must reproduce, exactly, the Stage 4 Test coverage of every
  certified (seed, alpha, method) cell that Stage 4 recorded, as `src/r8_stage5.py` does. If not, stop.
- n = 784 Cert tracks in every re-split; the Test denominator is 262.

#### (d) The scores (plan 5.1, 5.2)
All scores are sequential: per track and frame t (frame order) there is a confidence number and a label, or no value
yet. A track commits at the first frame t with a value and confidence >= lambda; its label there is its answer. If no
frame qualifies, it escalates. Gate-failing frames give no evidence (Stage 1 gate, tau = 43.8435).
- **S1** (ours, POC-2 V1): summed tied-Gaussian log-likelihood plus log prior over passing frames up to t; confidence
  = top minus runner-up; label = top class (`src.r8_stage1.running_scores`, `margins`). No value before the first
  passing frame.
- **S2** (Boursinos and Koutsoukos 2022, POC-2 B6 code `src.r8_stage6_methods`, K = 5): nonconformity per frame
  ([BK] eq. 9), ICP p-values ([BK] eq. 10), combined over the last 5 passing frames up to t, ECDF-transformed;
  confidence = a * credibility + b * confidence; label = class with the largest combined p-value. A value exists only
  at passing frames from the 5th passing frame on. Two variants:
  - **S2-sum (primary)**: ECDF Sum. **S2-min (secondary)**: ECDF Min. Both are computed and reported.
  - The primary is chosen now, before any Cert number (author's decision). POC-2 Stage 6 had no primary B6 variant:
    both K = 5 variants were equal columns of its comparison table (`LOG_r8_stage6_entry.md` Part A (h), line 116).
  - Commit rule: confidence >= lambda (the layer's rule), where POC-2 B6 used k > theta. K = 9 is not run.
- **S3** (POC-2 B4, made sequential): running majority vote over the single-frame top classes of passing frames up to
  t; confidence = votes(label) - votes(runner-up); label = the class with most votes, ties to the smaller class id.
  No value before the first passing frame.
- **S4** (POC-2 B3, made sequential): the margin of the current frame alone (top minus runner-up of the V1
  single-frame score, `src.r8_stage3_b1design.frame_margins`); label = its top class. A value at passing frames only.
- POC-2's B3 and B4 decided at the last frame only. Their POC-2 results stay on record unchanged; S3 and S4 here are
  the sequential forms that plan 5.1 requires, so their numbers are not comparable one-to-one with POC-2 B3 and B4.

#### (e) S2 calibration on the design set (plan 5.2)
- The 261 design tracks are split into two disjoint halves by track: within each class, the class's design tracks (in
  track-code order) are permuted with `numpy.random.default_rng([20261114, 3, class_id])` and assigned A, B, A, B, ...
  (half A gets the extra track in classes with an odd count).
- Half A: ICP calibration scores (true-class nonconformity of every passing frame). Half B: the ECDF of the combined
  p-value of the true class over every full window, then (a, b) by POC-2's rule (grid -1.0 to 1.0 in steps of 0.1,
  minimum AURC over the half-B frames with full windows, ties to the first pair in ascending order). Fitted with
  `r8_stage6_methods.b6_fit` with parts icp = A, ecdf = B, select = B; its theta values are not used. Fitted
  separately for Sum and Min.
- **Difference from POC-2, stated:** in POC-2, B6 was calibrated on Cert thirds (Stage 6 Amendment 1: ICP / ECDF /
  selection, 261 / 261 / 262). Here S2 is calibrated on the design set only, so all 784 Cert tracks stay available for
  the certificate (plan R9 section 5.2). S2 therefore sees about 130 tracks per calibration part instead of 261.
- Stated now: S2's grid, start and reference in (f), (g), (i) are computed on design tracks that also fitted S2 (half B
  also chose (a, b) and the ECDF), so S2's design commit fractions and errors are partly in-sample. This can move S2's
  grid, start and reference, not the validity of its certificate, because the Cert tracks are disjoint from the design
  set. [Likely]

#### (f) Threshold grids (plan 5.3; design tracks only)
- Per score, per design track i: C_i = the largest confidence over the frames where the track has a value (-inf if it
  never has one). With N = 261 and f_j = 1.00 - 0.70 * j / 29:
  - lambda_0 = -inf: commit at the first frame with a value (the loosest point).
  - lambda_j (j = 1..29) = the m_j-th largest C_i, m_j = ceil(f_j * N). At least m_j design tracks then commit
    (more if values tie), so design commit fractions run from about 100% to about 30%.
- Repeated values (a discrete score such as S3's vote margin gives few distinct values) are removed, keeping the order.
  The number of distinct grid points per score is reported.
- Design commit fraction f_d(j) = committed design tracks / 261 at lambda_j (denominator: all 261 tracks); design
  committed error e_d(j) = wrong / committed (undefined if nothing commits).
- Grids are the same for every alpha and every re-split. This rule is wider than the POC-2 working grids (which kept
  c >= 70% only).

#### (g) Starts (plan 5.3; design values, n = 784, n_design = 261)
- Exact binomial (primary): `power_start(f_d, 784, alpha)`, the strictest j with f_d(j) * 784 * alpha >= 4.
  Floors on f_d: 51.0% at 1%, 25.5% at 2%, 17.0% at 3%, 10.2% at 5%.
- **alpha 0.5%: no start possible at n = 784 (needs design commit fraction >= 102%).** Nothing is tested at 0.5% with
  the exact binomial; it is reported as "no start possible", not as tested and failed (consistent with plan section 12:
  alpha = 0.5% is not certifiable at these sample sizes).
- Hoeffding-Bentkus comparator, all three starts (Stage 2 Amendment 2): `hb` (primary; zero-error pass at the lower 90%
  Clopper-Pearson bound of f_d, n_design = 261), `hb_a1` (Amendment 1, zero margin), `hb_bs` (the binomial start).
- Starts per (score, alpha, rule) depend only on design values, 784 and alpha. They are written to the design-values
  file and into this Part A before any Cert number exists.

#### (h) Certification
- For each seed (0 to 4), score (S1, S2-sum, S2-min, S3, S4), alpha in {0.5, 1, 2, 3, 5}% and rule in
  {binom, hb, hb_a1, hb_bs}: on the 784 Cert tracks, k(j) and n_c(j) for every grid point, then `certify(...)`. Losses
  come from the tracks-by-lambdas table (hard rule 4), never from one score per track.
- Output per cell: the certified grid index (or none) and its lambda.

#### (i) Reported per (seed, score, alpha, rule) (plan 5.4), on the 262 Test tracks
- Certified or not; Test coverage = committed / 262; error among committed (wrong / committed); escalation rate =
  1 - coverage; median commit frame (1-based position in the track, committed tracks only); median ROI size at commit
  (shorter side min(x2 - x1, y2 - y1) in original pixels, Stage 1 convention); D90 size as in `LOG_r8_stage5_entry.md`
  Part A (smallest s, in multiples of 5 px, with at least 90% of Test signs committed; Test description, not a
  certificate).
- Smallest certifiable alpha per (seed, score, rule): the smallest alpha in the grid at which a threshold is certified.
- **Uncertified reference (RQ2, "the same score without a certificate")**, per score and alpha: the loosest grid point
  j with e_d(j) <= alpha on the design tracks (none if no grid point qualifies), evaluated on the same Test tracks with
  the same metrics. Fixed in the design phase; no Cert data is used. For S2 this is not POC-2's B6 (different
  calibration data, commit rule and selection set).
- Summaries: median and range over the 5 re-splits.
- Secondary analysis (inside RQ2): per (score, alpha), the number of re-splits certified by the exact binomial and by
  each HB start, and the certified grid index under each.
- ROI sizes come from `data/cache/train_meta.csv`, checked against SHA-256
  4c06c3ad07a3b3b89faabcd49b26e8f6dbfff6460f25bf5b66730140de6a132e (recorded in `LOG_r8_stage4_entry.md` Part B) and
  aligned with the embeddings row by row as in `src.r8_stage4.roi_sizes`.

#### (j) Not in Stage 3
- B1, B2, B5, B5c and B6 are not re-run here. Their GTSRB numbers stay as recorded in POC-2; the comparison tables are
  assembled in Stage 8.

#### (k) Integrity checks (stop on failure; not results)
- `check_frozen()` passes; the design loader refused the pool table and only design rows reached the design step;
  no BelgiumTSC path was opened.
- The recomputed gate flags and the stored Stage 1 design commit tables (V1 to V3) reproduce exactly
  (`src.r8_stage2.reproduces`), so S1 on the design tracks is the POC-2 V1 rule.
- The design-values file read by the runner has the SHA-256 recorded in this Part A (hard rule 2).
- The regenerated splits reproduce the Stage 4 coverages (c).

#### (l) Order of work
1. This Part A is committed.
2. `src/r9_frozen.py` and its tests are committed.
3. `src/r9_stage3_scores.py` (score sequences, halves, grids, design statistics, starts, reference; no file I/O) and
   `src/r9_stage3_design.py` (design tracks only) with tests are committed, then the design script is run: S2 halves
   and calibration, grids, f_d, e_d, starts and references. Its output file and SHA-256 go into this Part A in a
   separate commit, and these commits are pushed before any Cert number is computed.
4. `src/r9_stage3.py` (runner) and synthetic tests with brute-force references and tamper tests; full suite; commit.
5. Real run; the whole printout is kept; then stop. Part B is filled after the author has reviewed the printout.

#### (m) Code, tests and outputs
- `src/r9_frozen.py`, `src/r9_stage3_scores.py`, `src/r9_stage3_design.py`, `src/r9_stage3.py`;
  `tests/test_r9_frozen.py`, `tests/test_r9_stage3_design.py`, `tests/test_r9_stage3.py`. POC-1 and POC-2 modules are
  imported, never edited.
- Outputs in `runs/20261003_stage1/poc3/stage3/`: `stage3_design.json` (step 3), `stage3_summary.json` (step 5), each
  with the SHA-256 of the code and of every file read. `README_r9_stage3.md` gives the run commands.

#### Design-set values (step 3; design tracks only, exploratory)
Computed 2026-11-14 after the Part A commit (9be85ac), with the code committed first (05c1df5;
`src/r9_stage3_design.py` SHA-256 1984e00d04fc7eeee75ac838de1cbab9888ea3714c9ce12d7630f9423fa48b65;
`src/r9_stage3_scores.py` SHA-256
ba6c3a745855d3216885cb832937fa69cef552bf37b00b53840aedaae2fcee0d). Command
`py -3.12 -m src.r9_stage3_design --run runs/20261003_stage1`. Output
`runs/20261003_stage1/poc3/stage3/stage3_design.json` SHA-256
798585428c4cc345a760e347ded0c37f16c46f65a2ecf7dfa50473bacfb8e62b. Freeze check passed (all three blob IDs); design
tables and gate flags reproduce Stage 1; files opened: seed0.npz, train_release.npz, scores_train_release.npz,
frame_evidence.npz, head.npz, track_table_design.npz (not the pool table).
- S2 halves: A (ICP) 141 tracks, B (ECDF and a, b) 120 tracks; the "about 130 per part" in (e) was an estimate: half A
  gets the extra track of every class with an odd count. ICP calibration scores 4,031; ECDF values 2,994.
  - S2-sum (primary): a = +0.3, b = +0.8, design AURC 0.00223. S2-min (secondary): a = +0.1, b = +0.9, AURC 0.00242.
- Grids (lambda_0 = -inf in every score):
  | Score | Distinct points | lambda_1 .. lambda_max | f_d range | e_d at lambda_0 | Tracks with no value |
  |---|---|---|---|---|---|
  | S1 | 30 | 2230.44 .. 24423.1 | 100.00 .. 30.27% | 14.94% (39/261) | 0 |
  | S2-sum | 30 | 0.816132 .. 1.05461 | 99.62 .. 30.27% | 10.00% (26/260) | 1 |
  | S2-min | 30 | 0.895792 .. 0.984803 | 99.62 .. 30.27% | 10.38% (27/260) | 1 |
  | S3 | 14 | 9 .. 30 | 100.00 .. 61.30% | 14.94% (39/261) | 0 |
  | S4 | 30 | 388.62 .. 1009.82 | 100.00 .. 30.27% | 14.94% (39/261) | 0 |
  - S3's vote margin is an integer of at most 30, and 61.30% of design tracks reach 30, so its strictest point commits
    61.30%, not about 30%.
  - The S2 track with no value has fewer than 5 passing frames.
- Starts (grid index, loosest = 0; -1 = no start possible) and the uncertified reference (loosest j with e_d <= alpha):
  | Score | alpha | binom | hb | hb_a1 | hb_bs | reference |
  |---|---|---|---|---|---|---|
  | S1 | 0.5 / 1 / 2 / 3 / 5% | -1 / 20 / 29 / 29 / 29 | 4 / 12 / 19 / 23 / 28 | 5 / 14 / 21 / 24 / 29 | -1 / 20 / 29 / 29 / 29 | 4 / 3 / 1 / 1 / 1 |
  | S2-sum | 0.5 / 1 / 2 / 3 / 5% | -1 / 20 / 29 / 29 / 29 | 4 / 12 / 19 / 23 / 28 | 5 / 14 / 21 / 24 / 29 | -1 / 20 / 29 / 29 / 29 | 3 / 3 / 2 / 1 / 1 |
  | S2-min | 0.5 / 1 / 2 / 3 / 5% | -1 / 20 / 29 / 29 / 29 | 4 / 12 / 19 / 23 / 28 | 5 / 14 / 21 / 24 / 29 | -1 / 20 / 29 / 29 / 29 | 3 / 3 / 2 / 2 / 1 |
  | S3 | 0.5 / 1 / 2 / 3 / 5% | -1 / 13 / 13 / 13 / 13 | 4 / 11 / 13 / 13 / 13 | 6 / 11 / 13 / 13 / 13 | -1 / 13 / 13 / 13 / 13 | 4 / 3 / 3 / 1 / 1 |
  | S4 | 0.5 / 1 / 2 / 3 / 5% | -1 / 20 / 29 / 29 / 29 | 4 / 12 / 19 / 23 / 28 | 5 / 14 / 21 / 24 / 29 | -1 / 20 / 29 / 29 / 29 | 14 / 6 / 4 / 3 / 1 |
- alpha 0.5%: no start possible at n = 784 for the exact binomial (needs design commit fraction >= 102%). The HB
  starts `hb` and `hb_a1` do exist at 0.5% (index 4 to 6), so HB is tested at 0.5% while the exact binomial is not.
- Full per-point tables (lambda, f_d, e_d, wrong, median commit position) are in the JSON and the printout.
- No Cert or Test track was read. The runner checks this file's SHA-256 before use.

### Amendment 1 (14 Nov 2026, after the design-set values, before any Cert number)
- alpha 0.5%: no exact-binomial start at n = 784, because the power-aware start rule needs f_d x 784 x 0.005 >= 4.
  This is caused by the start rule; plan section 6 shows the test itself could certify at n_c >= 460 with 0 errors.
  HB starts exist at 0.5%, so HB results at 0.5% are not compared with the exact binomial.
- Design facts recorded: S3's grid has 14 points and stops at 61.3% commit because the vote margin is an integer of at
  most 30; design error is 0 from grid point 5 (S1), 4 (S2-sum, S2-min) and 14 (S4) on, so how far the walk relaxes
  depends on Cert data.
- This replaces the bracket "consistent with plan section 12: alpha = 0.5% is not certifiable at these sample sizes"
  in (g). No rule, grid, start or other value changes.

### Part B: results (exploratory, post-hoc; filled from the printout after review)
**Every number in this Part B is exploratory (post-hoc):** all GTSRB tracks were used before (Part A preamble).

- Date: 2026-11-14 / Stage: R9 Stage 3 / Run tag: runs/20261003_stage1/poc3/stage3
- Git commit of Part A: 9be85ac (definitions), 58cacfe (design-set values), 0c255f0 (Amendment 1). Code: 05c1df5
  (scores, design script), 27ed9a7 (runner; `src/r9_stage3.py` SHA-256
  45a712ca65c0bf55c012cb0a41984a0a6a00411083540bf040628bb9d243ff7c). Run from a clean tree on 27ed9a7:
  `py -3.12 -m src.r9_stage3 --run runs/20261003_stage1 --config configs/poc.yaml` (2.6 s).
- Output: `stage3_summary.json` SHA-256 aad2c03ff94487bbc4743dbedf6b76a3e51a4167f50a8d3b831ba7360fa052e2; whole
  printout `stage3_printout.txt` SHA-256 21d57b80b500a5c5c56fc0dbfd28f05a4621b8daf5bc87e8624817ffb6283d69 (same
  directory). Full suite before the run: 368 passed.

#### Integrity (all passed)
- Freeze check: plan R9 0abe7fa5, engine 158a9c9e, engine tests ecb6e044 (blob IDs) — PASS.
- Hash checks: stage3_design.json, stage4_summary.json, src/r8_stage4.py, baselines_design.json, train_meta.csv and
  every POC-2 pool input — PASS.
- Split reproduction: the five Stage 4 re-splits regenerated; all 68 certified Stage 4 Test coverages reproduce
  exactly — PASS.
- POC-2 checks: frozen_grid_matches_stage1_grid, gate_flags_reproduce, pool_commit_table_reproduces,
  b1_top_equals_stored_top1, meta_roi_matches_stage1_sizes, b2_frames_all_pass_gate — all PASS. Design starts
  recomputed equal the design file — PASS.
- Pool 1,046 tracks; Cert 784 / Test 262 per re-split. Pool tracks with no value: S1 5, S2-sum 10, S2-min 10, S3 5,
  S4 5 (the 5 tracks with no passing frame, plus 5 with fewer than 5 passing frames for S2).

#### Certification (grid index certified per seed 0-4; "-" = none)
| Score | alpha | binom | hb | hb_a1 | hb_bs |
|---|---|---|---|---|---|
| S1 | 0.5% | - - - - - | - - - - - | - - - - - | - - - - - |
| S1 | 1% | 5 5 7 7 7 | 6 10 10 10 10 | 6 10 10 10 10 | - - - - - |
| S1 | 2% | 3 3 3 3 4 | 4 4 4 4 4 | - - - - - | - - - - - |
| S1 | 3% | 3 3 3 3 2 | 3 3 3 3 3 | - - - - - | - - - - - |
| S1 | 5% | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 1 |
| S2-sum | 0.5% | - - - - - | - - - - - | - - - - - | - - - - - |
| S2-sum | 1% | 4 4 4 4 4 | 4 4 12 12 12 | - - 12 12 - | - - - - - |
| S2-sum | 2% | 3 3 3 3 2 | 3 3 4 3 3 | 3 3 4 3 3 | - - - - - |
| S2-sum | 3% | 2 2 2 2 2 | 2 3 2 2 2 | 2 3 2 2 2 | - - - - - |
| S2-sum | 5% | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 1 |
| S2-min | 0.5% | - - - - - | 4 4 - - - | 4 4 - - - | - - - - - |
| S2-min | 1% | 4 4 4 4 4 | 4 4 - - - | - - 13 - - | - - - - - |
| S2-min | 2% | 3 3 3 3 3 | 3 3 4 4 3 | 3 3 4 4 3 | - - - - - |
| S2-min | 3% | 2 2 2 2 2 | 2 3 2 3 2 | 2 3 2 3 2 | - - - - - |
| S2-min | 5% | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 1 |
| S3 | 0.5% | - - - - - | - - - - - | - - - - - | - - - - - |
| S3 | 1% | 5 6 6 6 6 | 7 - - - - | 7 - - - - | - - - - - |
| S3 | 2% | 3 4 3 4 4 | 4 4 4 4 4 | 4 4 4 4 4 | 4 4 4 4 4 |
| S3 | 3% | 3 3 3 3 3 | 3 3 3 3 3 | 3 3 3 3 3 | 3 3 3 3 3 |
| S3 | 5% | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 1 |
| S4 | 0.5% | - - - - - | - - - - - | - - - - - | - - - - - |
| S4 | 1% | 11 18 19 19 19 | - - - - - | - - - - - | - - - - - |
| S4 | 2% | 4 10 10 11 4 | 10 18 - - - | - - - - - | - - - - - |
| S4 | 3% | 3 4 3 3 3 | 4 4 4 - - | 4 4 - - - | - - - - - |
| S4 | 5% | 1 1 1 1 1 | 2 2 1 1 1 | 2 2 1 1 1 | 2 2 1 1 1 |

Smallest certifiable alpha per seed 0-4:
| Score | binom | hb | hb_a1 | hb_bs |
|---|---|---|---|---|
| S1 | 1% 1% 1% 1% 1% | 1% 1% 1% 1% 1% | 1% 1% 1% 1% 1% | 5% 5% 5% 5% 5% |
| S2-sum | 1% 1% 1% 1% 1% | 1% 1% 1% 1% 1% | 2% 2% 1% 1% 2% | 5% 5% 5% 5% 5% |
| S2-min | 1% 1% 1% 1% 1% | 0.5% 0.5% 2% 2% 2% | 0.5% 0.5% 1% 2% 2% | 5% 5% 5% 5% 5% |
| S3 | 1% 1% 1% 1% 1% | 1% 2% 2% 2% 2% | 1% 2% 2% 2% 2% | 2% 2% 2% 2% 2% |
| S4 | 1% 1% 1% 1% 1% | 2% 2% 3% 5% 5% | 3% 3% 5% 5% 5% | 5% 5% 5% 5% 5% |
- alpha 0.5%: no exact-binomial start at n = 784 (Amendment 1); the HB values of 0.5% for S2-min are not compared with
  the exact binomial.

#### Test metrics, exact binomial, median (min-max) over the 5 re-splits (all certified 5/5 at 1% to 5%)
| Score | alpha | Coverage % | Error % | Commit frame | Size px |
|---|---|---|---|---|---|
| S1 | 1% | 81.30 (79.77-87.40) | 0.00 (0.00-0.88) | 17.0 (15.0-18.0) | 33.0 (31.0-36.0) |
| S1 | 2% | 93.51 (90.08-94.27) | 0.82 (0.42-1.22) | 9.0 (8.0-13.0) | 26.0 (25.0-27.0) |
| S1 | 3% | 93.89 (93.13-94.66) | 1.21 (0.81-3.63) | 9.0 (7.0-9.0) | 26.0 (22.0-26.0) |
| S1 | 5% | 96.95 (95.80-98.09) | 3.11 (1.97-3.98) | 5.0 (5.0-5.5) | 23.0 (22.0-24.0) |
| S2-sum | 1% | 90.84 (87.02-91.22) | 0.00 (0.00-0.44) | 5.0 (5.0-5.0) | 24.0 (23.0-25.0) |
| S2-sum | 2% | 92.75 (91.98-94.27) | 0.83 (0.00-2.44) | 5.0 (5.0-5.0) | 23.0 (22.0-24.0) |
| S2-sum | 3% | 94.27 (93.89-95.80) | 1.59 (0.81-2.44) | 5.0 (5.0-5.0) | 23.0 (22.0-24.0) |
| S2-sum | 5% | 96.18 (95.04-98.47) | 2.35 (1.98-3.61) | 5.0 (5.0-5.0) | 23.0 (22.0-23.0) |
| S2-min | 1% | 89.31 (85.50-90.08) | 0.00 (0.00-0.45) | 5.0 (5.0-7.0) | 25.0 (24.0-26.0) |
| S2-min | 2% | 92.75 (92.37-93.89) | 0.83 (0.00-1.24) | 5.0 (5.0-5.0) | 23.0 (23.0-24.5) |
| S2-min | 3% | 95.04 (93.89-96.18) | 1.59 (1.19-2.44) | 5.0 (5.0-5.0) | 23.0 (22.0-24.0) |
| S2-min | 5% | 96.18 (95.80-98.85) | 2.75 (1.98-3.98) | 5.0 (5.0-5.0) | 23.0 (22.0-23.0) |
| S3 | 1% | 85.88 (85.50-87.79) | 0.00 (0.00-0.87) | 21.0 (20.0-21.0) | 42.0 (39.0-43.0) |
| S3 | 2% | 91.22 (87.79-92.75) | 0.84 (0.42-1.67) | 18.0 (16.0-18.0) | 35.0 (32.0-36.0) |
| S3 | 3% | 91.60 (88.93-92.75) | 0.83 (0.43-1.67) | 16.0 (16.0-16.0) | 33.0 (32.0-34.0) |
| S3 | 5% | 97.33 (95.04-97.71) | 3.14 (1.95-3.61) | 9.0 (9.0-9.0) | 26.0 (25.0-26.0) |
| S4 | 1% | 56.49 (53.82-72.90) | 0.00 (0.00-1.57) | 2.0 (1.0-3.5) | 24.0 (22.0-25.0) |
| S4 | 2% | 77.10 (75.19-88.93) | 1.02 (0.00-3.00) | 2.0 (2.0-3.0) | 23.0 (21.0-24.0) |
| S4 | 3% | 93.13 (90.08-94.27) | 2.07 (0.85-3.27) | 2.0 (1.0-2.0) | 21.0 (20.0-23.0) |
| S4 | 5% | 96.18 (95.42-98.47) | 3.20 (1.98-4.37) | 1.0 (1.0-1.0) | 21.0 (20.0-21.0) |
Escalation rate = 100% - coverage. D90 sizes, the HB rows and the uncertified reference rows per seed are in the
printout and the JSON.

#### Test error and descriptive 95% Clopper-Pearson interval, every certified exact-binomial cell
Descriptive, added at Part B, changes nothing. wrong / committed Test signs, error %, [two-sided 95% CP] %; seeds 0-4.
- S1 1%: 2/227 0.88 [0.11, 3.15]; 1/229 0.44 [0.01, 2.41]; 0/210 0.00 [0.00, 1.74]; 0/213 0.00 [0.00, 1.72]; 0/209 0.00 [0.00, 1.75]
- S1 2%: 3/247 1.21 [0.25, 3.51]; 2/246 0.81 [0.10, 2.91]; 3/245 1.22 [0.25, 3.54]; 2/244 0.82 [0.10, 2.93]; 1/236 0.42 [0.01, 2.34]
- S1 3%: 3/247 1.21 [0.25, 3.51]; 2/246 0.81 [0.10, 2.91]; 3/245 1.22 [0.25, 3.54]; 2/244 0.82 [0.10, 2.93]; **9/248 3.63 [1.67, 6.78]**
- S1 5%: 8/254 3.15 [1.37, 6.11]; 7/255 2.75 [1.11, 5.57]; 5/254 1.97 [0.64, 4.53]; 8/257 3.11 [1.35, 6.04]; 10/251 3.98 [1.93, 7.20]
- S2-sum 1%: 1/228 0.44 [0.01, 2.42]; 1/233 0.43 [0.01, 2.37]; 0/238 0.00 [0.00, 1.54]; 0/239 0.00 [0.00, 1.53]; 0/238 0.00 [0.00, 1.54]
- S2-sum 2%: 3/241 1.24 [0.26, 3.59]; 2/242 0.83 [0.10, 2.95]; 0/243 0.00 [0.00, 1.51]; 1/247 0.40 [0.01, 2.23]; **6/246 2.44 [0.90, 5.23]**
- S2-sum 3%: 4/247 1.62 [0.44, 4.09]; 2/247 0.81 [0.10, 2.89]; 3/249 1.20 [0.25, 3.48]; 4/251 1.59 [0.44, 4.03]; 6/246 2.44 [0.90, 5.23]
- S2-sum 5%: 5/251 1.99 [0.65, 4.59]; 5/252 1.98 [0.65, 4.57]; 6/255 2.35 [0.87, 5.05]; 8/258 3.10 [1.35, 6.02]; 9/249 3.61 [1.67, 6.75]
- S2-min 1%: 1/224 0.45 [0.01, 2.46]; 1/230 0.43 [0.01, 2.40]; 0/235 0.00 [0.00, 1.56]; 0/236 0.00 [0.00, 1.55]; 0/234 0.00 [0.00, 1.56]
- S2-min 2%: 3/242 1.24 [0.26, 3.58]; 2/242 0.83 [0.10, 2.95]; 0/244 0.00 [0.00, 1.50]; 1/246 0.41 [0.01, 2.24]; 3/243 1.23 [0.26, 3.57]
- S2-min 3%: 4/249 1.61 [0.44, 4.06]; 3/248 1.21 [0.25, 3.49]; 4/251 1.59 [0.44, 4.03]; 3/252 1.19 [0.25, 3.44]; 6/246 2.44 [0.90, 5.23]
- S2-min 5%: 6/252 2.38 [0.88, 5.11]; 5/252 1.98 [0.65, 4.57]; 7/255 2.75 [1.11, 5.57]; 10/259 3.86 [1.87, 6.99]; 10/251 3.98 [1.93, 7.20]
- S3 1%: 2/230 0.87 [0.11, 3.11]; 0/228 0.00 [0.00, 1.60]; 0/225 0.00 [0.00, 1.63]; 0/224 0.00 [0.00, 1.63]; 1/224 0.45 [0.01, 2.46]
- S3 2%: 4/239 1.67 [0.46, 4.23]; 1/236 0.42 [0.01, 2.34]; 3/243 1.23 [0.26, 3.57]; 2/239 0.84 [0.10, 2.99]; 1/230 0.43 [0.01, 2.40]
- S3 3%: 4/239 1.67 [0.46, 4.23]; 2/240 0.83 [0.10, 2.98]; 3/243 1.23 [0.26, 3.57]; 2/241 0.83 [0.10, 2.97]; 1/233 0.43 [0.01, 2.37]
- S3 5%: 8/255 3.14 [1.36, 6.09]; 5/256 1.95 [0.64, 4.50]; 5/255 1.96 [0.64, 4.52]; 8/255 3.14 [1.36, 6.09]; 9/249 3.61 [1.67, 6.75]
- S4 1%: **3/191 1.57 [0.33, 4.52]**; 1/150 0.67 [0.02, 3.66]; 0/148 0.00 [0.00, 2.46]; 0/141 0.00 [0.00, 2.58]; 0/144 0.00 [0.00, 2.53]
- S4 2%: **5/233 2.15 [0.70, 4.94]**; 2/197 1.02 [0.12, 3.62]; 2/202 0.99 [0.12, 3.53]; 0/198 0.00 [0.00, 1.85]; **7/233 3.00 [1.22, 6.09]**
- S4 3%: 5/242 2.07 [0.67, 4.76]; 2/236 0.85 [0.10, 3.03]; 5/244 2.05 [0.67, 4.72]; 6/247 2.43 [0.90, 5.21]; **8/245 3.27 [1.42, 6.33]**
- S4 5%: 6/251 2.39 [0.88, 5.13]; 5/253 1.98 [0.64, 4.55]; 8/250 3.20 [1.39, 6.21]; 9/258 3.49 [1.61, 6.52]; 11/252 4.37 [2.20, 7.68]
- HB cells certified at an index the exact binomial did not certify (their Test errors are in the JSON): the only one
  above its alpha is S4, hb, 2%, seed 0: 5/200 2.50 [0.82, 5.74]; its lower CP bound is also below alpha.

#### Test cells above alpha (exact binomial; bold above)
- **Six** certified cells have a Test error above alpha: S1 3% seed 4 (3.63%); S2-sum 2% seed 4 (2.44%); S4 1% seed 0
  (1.57%); S4 2% seed 0 (2.15%); S4 2% seed 4 (3.00%); S4 3% seed 4 (3.27%). The author's review list named five; S4
  2% seed 0 was missed there and is added here.
- In every certified cell the lower 95% CP bound is below alpha, so none of these Test estimates is evidence that the
  true error exceeds alpha.
- The certificate states that, with probability at least 0.9 over the Cert draw, the true error among committed signs
  is at most alpha. A Test estimate from about 140 to 260 committed signs moves by about 0.4 to 0.7 points per wrong
  sign, so single Test estimates can exceed alpha without the certificate being broken. Validity is measured by
  violation rates over many draws (POC-2 Stage 3, R9 Stage 2, Stage 6), not by five Test estimates.

#### Readings (exploratory statements, not results scored against anything)
- **E2 (S2 earlier than S1):** at 1% the exact-binomial median commit frame is 5.0 (5.0-5.0) for S2-sum against 17.0
  (15.0-18.0) for S1, with coverage 90.84% against 81.30% (medians). At 2% to 5% S2-sum stays at frame 5 while S1 moves
  from 9 to 5.
- **E3 (exact binomial against HB):** at every alpha from 1% to 5%, in every score and seed, the exact binomial
  certified whenever any HB start certified, and at the same or a looser grid point. HB at 0.5% (S2-min, seeds 0 and 1)
  is not compared (Amendment 1). Re-splits certified out of 5 at 1%, binom / hb / hb_a1 / hb_bs: S1 5/5/5/0, S2-sum
  5/5/2/0, S2-min 5/2/1/0, S3 5/1/1/0, S4 5/0/0/0.
- **Cost of the certificate, S2-sum at 1%:** certified coverage 90.84% (87.02-91.22) with worst Test error 0.44%
  (seed 0, 1/228); uncertified reference (loosest point with design error <= 1%, j = 3) coverage 92.37% (91.98-94.27)
  with worst Test error 1.24% (seed 0, 3/241). The certificate costs about 1.5 points of median coverage here, and the
  reference's worst error is above its 1% target.
- **1% certified for every score in 5 of 5 re-splits:** POC-2's commit rule certified 0 of 5 at 1%. This stage changed
  the test (exact binomial), the grid (30 points down to about 30% commit) and the start (power-aware) together, so the
  difference cannot be assigned to any one change.
- **Commit frames sit at each score's floor:** S2 commits at frame 5 (the first full window, K = 5) and S4 at frames
  1 to 3 (a single frame) at most alphas. S1 and S3 commit later and move earlier as alpha grows (median frame S1 17
  to 5, S3 21 to 9 from 1% to 5%) without reaching their first possible frame. Differences in earliness between S2 or
  S4 and the others therefore partly reflect where each score can first give a value.

#### Computation note for the CP intervals and the E3 check
Computed after the review from `stage3_summary.json` (hash above) with scipy.stats.beta (two-sided 95%
Clopper-Pearson). No Cert or Test outcome was recomputed; nothing in Part A or the code changed.

- Decision taken and why: Stage 3 reported as run. All integrity checks passed; the results are R-results (plan R9
  section 3), reported, not scored.
- Deviation from the plan? Yes (declared in Part A and Amendment 1): S3 and S4 made sequential; S2 in two variants
  with S2-sum primary; S2 calibrated on design halves (141 / 120 tracks); alpha 0.5% has no exact-binomial start.
  Process: the review list of above-alpha cells missed one (added above).
- Next step: Stage 4 (BelgiumTSC model and cache), only when the author says so.
