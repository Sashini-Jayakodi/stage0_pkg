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
