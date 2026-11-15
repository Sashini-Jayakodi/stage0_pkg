## POC-3 (R9) Stage 6: BelgiumTSC confirmation (confirm phase, after F3)

### Part A: fixed definitions (committed before any Stage 6 number exists)

Plan: `docs/poc3-implementation-plan-r9.md` (git blob 0abe7fa51af3b499cd79218e46c83790dbfa2e99), sections 2, 3, 5.4,
8, 11, 12 and 9 Stage 6. This is the confirm phase of the fresh dataset. Every value Stage 6 uses is read from
`configs/frozen_r9_btsc.yaml` (F3 commit e94e6670987f7e9abad8382d848012a6a928407a, git blob
f4d37e5d73ef230e91689be5303e5eac51b15450) or recomputed from frozen inputs by a frozen rule and checked equal to the
yaml. No value is chosen in Stage 6. delta = 0.1. Stage 6 has one V-check (section (e)); everything else is an
R-result, reported and not scored (plan Rule 1). Definitions that refer to "Stage 5 (x)" mean `LOG_r9_stage5_entry.md`
Part A section (x) (commit 019f782), whose settings the yaml freezes.

#### (a) Inputs and hash checks (hard rule 2)
- `configs/frozen_r9_btsc.yaml`: checked by git blob ID through `check_frozen()` (b) before it is read; parsed with a
  safe YAML loader; `-.inf` is read as minus infinity and `null` as "none" (Stage 5 Part B, design file and yaml).
- `runs/btsc/stage4/stage4_summary.json`: SHA-256 1c66a0119d39473f818f1add56c8a878d32ddfcc91aef4b0f5a9bd76568a3a31
  (or blob 3d0d79c42e3a782e301656acdb0a24515646ebab, Stage 5 (a) rule for text files). Every other Stage 4 file is
  checked against its `files_sha256` entry before it is read; text files also pass on the blob IDs recorded in the
  yaml (`sources.stage4_inputs_blob`).
- `runs/btsc/stage5/stage5_design.json`: SHA-256 115f98025f60b14aedad9c0d75df6d5deafc568d77b427e2bf9804be1d854a11 or
  blob b06a09f8c64bbd789bd02addab0b518ecde66d0e. Read only for the reproduction check in (k); every value used comes
  from the yaml.
- Files opened in Stage 6: the yaml, `stage4_summary.json`, `split_poles.json`, `test_views.csv`,
  `embeddings/test_design.npz`, `embeddings/test_pool.npz`, `head/head.npz`, `head/scores_test_design.npz`,
  `head/scores_test_pool.npz`, `stage5_design.json`. Not opened: the image caches, the checkpoint, training-release
  files. The runner records every file it opens and prints the list.
- Testing labels (design and pool) are read through `src.r9_btsc_guard.GuardedTestLabels`; the runner asserts
  `f3_closed()` is true before the first pool label is requested.

#### (b) Freeze check
- Every Stage 6 script calls `src.r9_frozen.check_frozen()` first: 4 files by git blob ID (plan R9, engine, engine
  tests, `configs/frozen_r9_btsc.yaml`). If it fails, nothing runs.
- The engine `src/r9_certify.py` is used unchanged: `certify` (which computes the start from f_d, n and alpha by
  the frozen rule), `start_for`, `max_errors`. Clopper-Pearson intervals: `src.r8_stage3_engine.cp_bounds`
  (two-sided 95%, the function behind the Stage 2 V2 bound).
- Scores, windows, S2 combination, B2 draw and B6 rule: the Stage 5 functions in `src/r9_stage5_scores.py`, imported
  unchanged, with the frozen S2 fit (ECDFs and (a, b) from the yaml; ICP calibration scores recomputed from half A and
  checked against `icp_cal_sha256`).

#### (c) Units and the tracks-by-lambdas tables (hard rule 4)
- Unit = (pole, class); view order = ascending YYYYY; commit time = views used (1-based); no gate (yaml).
- Pool: 530 poles, 626 units, 1,860 views (Stage 4 Part B). Sign size per view = the `size` column; size at commit =
  size of the committing view.
- For each score (S1, S2-sum, S2-min, S3, S4) and each pool unit i, at every frozen grid point j: committed_ij,
  wrong_ij (committed and label wrong), commit view pos_ij and size at commit. All losses below come from these
  tables, never from one score per unit.
- B1 table: each pool view is a unit; committed = S4 single-view margin >= lambda_j on the frozen 30-point B1 grid;
  wrong = committed and head top-1 wrong.

#### (d) RQ1 resampling (yaml `rq1`, Stage 5 (k))
- Population: the 530 pool poles with all their units (and views, for B1).
- 2,000 draws, d = 0 .. 1,999. Draw d: `rng = numpy.random.default_rng([20261115, 6, d])`; pole indices
  `rng.integers(0, 530, size=397)` into the pool poles sorted by ascending pole ID; with replacement, a pole drawn
  twice contributes all its units (and views) twice. One draw serves every cell (common draws across scores, B1,
  alphas and rules).
- Sign-level certificate per (score, alpha, rule): k(j), n_c(j) summed over the drawn units; n = the number of units
  drawn; `certify(k, n_c, n, alpha, f_d, test=rule, n_design=157)` with the score's frozen grid and f_d. The start
  therefore follows the draw's n by the frozen rule (as the frozen per-seed starts followed n(s)).
- View-level B1 per (alpha, rule): k(j), n_c(j) summed over the drawn views; n = views drawn; the B1 grid and f_d from
  the yaml; `n_design = 465` (yaml `b1.n_design_views`).
- Rules: `binom` and `hb` (HB primary start). Alphas: every alpha in {0.5, 1, 2, 3, 5}%. A draw whose start is -1
  ("no start possible") certifies nothing and is counted as such; a cell with no start in any draw is printed as
  "no start possible", not as tested.
- True risk of grid point j: R(j) = wrong / committed over the whole pool population, each unit counted once
  (626 units; for B1, 1,860 views). The pool curve R(j) and pool commit fraction per grid point are printed for every
  score and B1 (descriptive).
- Violation (Stage 2 definition): a grid point was certified and its true risk R(j) > alpha. Certifying nothing is
  not a violation. Violation rate = violations / 2,000.
- Reported for every cell (scores and B1, every alpha, both rules): certify count (of 2,000); violation count;
  violation rate with its two-sided 95% Clopper-Pearson interval; mean true risk R of the certified grid point over
  the certifying draws ("none" if 0 certify); the distribution of the certified grid index (counts per index).
- Printed next to the RQ1 table: the design-phase within-unit rho 0.631 [0.126, 0.887] and design effect 2.24
  [1.25, 2.74] (yaml `correlation_design`), and the GTSRB numbers as recorded in POC-2 Stage 3b (alpha 1%: frame-level
  B1 20.0% [18.3, 21.8]%, track-level 1.4% [0.9, 2.0]%). The two datasets differ in cluster size, alpha and test; the
  comparison is descriptive.

#### (e) Stage 6 V-check (yaml `stage6_v_check`; pre-registration item 8)
- Scored cells, exactly 5: S1, S2-sum, S2-min, S3, S4, each at the primary alpha 3% with the exact binomial (`binom`),
  from the RQ1 resampling (d).
- Pass: the upper end of the two-sided 95% Clopper-Pearson interval of the sign-level violation rate is <= 0.12
  (delta + 0.02). The V-check passes if all 5 cells pass.
- Printed next to each scored cell: its certify count (of 2,000), its violation count and its mean true risk. A cell
  that rarely certifies has few chances to violate, so its pass carries little evidence; the certify count makes this
  visible. The check is still scored exactly as pre-registered, whatever the certify counts are (a cell that never
  certifies has 0 violations, upper bound 0.18%, and passes).
- All other RQ1 cells (other alphas, HB, B1) are reported, not scored.
- A V-check failure means a bug by plan Rule 1. The code is searched for the bug; if one is found, it is fixed and
  the run is repeated under (l). If none is found, the V-check is reported as failed (plan section 12, the one
  result that would sink POC-3). It is never re-scored on other cells.

#### (f) RQ2 certification on the 5 frozen re-splits
- Seeds 0 to 4; Cert and Test pole lists from the yaml (`resplits.split`). n(s) = Cert units of that seed (471, 472,
  464, 471, 468).
- For each seed, score, alpha in {0.5, 1, 2, 3, 5}% and rule in {binom, hb, hb_a1, hb_bs}: k(j), n_c(j) on the Cert
  units; `certify(..., n = n(s), n_design = 157)`. The start the engine computes must equal the frozen start in the
  yaml for that (seed, score, alpha, rule); a mismatch stops the run (k). Output: certified grid index (or none) and
  its lambda.
- **Uncertified reference** per score and alpha: the frozen index in the yaml (`scores.<S>.reference`; -1 = none),
  evaluated on the same Test units with the same metrics.

#### (g) Metrics on Test units (plan 5.4), per (seed, score, alpha, rule) and per reference
- Certified or not; Test coverage = committed / Test units; error among committed = wrong / committed, with a
  descriptive two-sided 95% Clopper-Pearson interval (not a certificate); escalation rate = 1 - coverage; median commit
  view (committed units only); median size at commit; D90 size (Stage 3 (i) rule: the smallest s, in multiples of
  5 px from 5 px up to the largest pool view size rounded up to 5, at which committed with size at commit <= s covers
  at least 90% of Test units; "not reached" otherwise). On BelgiumTSC views are simultaneous, so D90 and commit view
  describe size and views used, not approach time (plan section 12).
- Smallest certifiable alpha per (seed, score, rule): the smallest alpha in the grid with a certified grid index;
  "none" if none. Summarised as median and range over seeds, and counts of seeds per alpha.
- Test cells whose error exceeds alpha are listed with their interval (as Stage 3 Part B).
- Secondary analysis (RQ2): per (score, alpha), seeds certified by the exact binomial and by each HB start, and the
  certified index under each.
- Summaries: median and range over the 5 re-splits.

#### (h) Baselines, exactly as the yaml (Stage 5 (k))
- **B1** (view-level Learn-then-Test, S4 single-view margin, view as unit): in the RQ1 resampling (d) only; not run
  on the re-splits (decision Q1).
- **B2** (one random view per unit): Cert and Test draws per seed with `default_rng([20261115, 7, s])` over that
  seed's units in the frozen unit order (`src.r9_stage5_scores.b2_draw`); the frozen B2 grid, f_d and per-seed starts
  (checked equal to the engine's starts); certified with the exact binomial and the three HB starts, n = n(s),
  n_design = 157; same Test metrics as (g), the committed view's position as commit view.
- **B5 and B5c** (Ringel et al. 2024): `src.r8_stage6_methods`, imported; classifier output = S1 accumulated score;
  Y_full = S1 top class after the unit's last view; pi = F_design(S1 margin) from the frozen 465 design margins
  (`b5.f_design`); grid 0, 0.01, ..., 1 and infinity; exact binomial p-value; delta = 0.1; alpha = the gap tolerance.
  B5 calibrates on all Cert units of the seed. B5c: thresholds for views 1, 2, 3 (units with more than 3 views use
  lambda_3 from view 4 on); D_cal-1 = the first 199 Cert poles of a permutation from `default_rng([20261116, s])`,
  D_cal-2 = the other 198. Reported per seed and alpha: certified or not, Test gap, Test error (real error, with
  interval), coverage (100%; B5 and B5c never escalate), median commit view, median size at commit.
- **B6** (Boursinos and Koutsoukos 2022, uncertified): S2-sum and S2-min as fitted; a unit commits at its first view
  with confidence > theta(alpha), theta from the yaml (0.09875 at 0.5% and 1%; minus infinity, that is commit at view
  1, at 2%, 3% and 5%); no certificate; target r = alpha. Reported per seed and alpha: Test coverage, Test error with
  interval, error above r or not, median commit view, median size at commit.
- No baseline is dropped or weakened because of its result (plan section 2).

#### (i) Comparison table for BelgiumTSC (plan section 8), primary alpha 3%
- Columns: Ours S1, S2 (S2-sum primary; S2-min in a second line), S3, S4; B1; B2; B5; B5c; B6. Values: median and
  range over the 5 re-splits unless the row says otherwise.
- Rows and sources:
  | Row | Source |
  |---|---|
  | Type of promise | Fixed text: certified real error among committed (Ours, B1 per view, B2); certified gap (B5, B5c); none (B6) |
  | Violation rate vs delta (resampling) | (d) at 3%, exact binomial (Ours, B1); B2, B5, B5c, B6: "not in the pre-registered resampling" (decision Q2) |
  | Certified in how many re-splits | (f), (h) at 3%, exact binomial (B5, B5c: their own test); B6: "no certificate" |
  | Test coverage | (g), (h) |
  | Test error among committed | (g), (h) |
  | Worst re-split error vs target | largest Test error over seeds against 3% (B5, B5c: against the gap tolerance and as real error; B6: against r = 3%) |
  | Smallest certifiable alpha | (g), exact binomial; B5, B5c: smallest gap tolerance certified, labelled as a different promise; B6: none |
  | Median commit view | (g), (h) |
  | Median sign size at commit | (g), (h) |
  | Can escalate | Fixed: yes (Ours, B1, B2, B6), no (B5, B5c) |
- B1 column: type of promise, violation rate (d) and can escalate are filled; every re-split row reads "not run (B1
  is pre-registered for the resampling only)" (decision Q1). B1's certify counts per alpha in (d) are shown under the
  table instead.
- Where a method is not certified in a seed, its Test metrics in that seed are "not certified" (nothing commits) and
  the median and range are over the certified seeds, with the count shown.
- The same table is also printed at every other alpha (descriptive).

#### (j) S2 against B6: cost of the certificate
- Per S2 variant (S2-sum primary, S2-min secondary) and every alpha, with 3% as the headline row (decision Q3): S2 under the layer (exact
  binomial, (f)) against B6 at r = alpha (h), and the uncertified reference (f) beside them. Columns: seeds certified
  (layer only); Test coverage; median commit view; median size at commit; Test error (median, range); worst re-split
  error against alpha; seeds with error above alpha.
- Stated in advance: with the frozen theta, B6 commits every unit at view 1 at 2%, 3% and 5%, and S2's grid has 3
  points (Stage 5 Part B), so at 3% the comparison is between "commit all at view 1" and at most two stricter points.

#### (k) Integrity checks (stop on failure; not results)
- `check_frozen()` passes (4 files); every input matches (a); the file-open list holds only the files of (a);
  `f3_closed()` is true before the first pool label.
- Design reproduction: recomputed on the 157 design units, every score's committed and wrong counts and f_d, e_d per
  grid point, the B1 and B2 design f_d and e_d, the S2 ICP calibration SHA-256 and the reference indices equal the
  yaml (and `stage5_design.json`).
- Recomputed per-view head top-1 and margins on pool views equal `head/scores_test_pool.npz` exactly in float32 (as
  Stage 5 Part B for design views).
- Pool structure: 530 poles, 626 units, 1,860 views; no design pole in the pool; per seed the Cert and Test pole lists
  partition the 530 pool poles and give the yaml's Cert / Test unit counts.
- Starts: for every (seed, score, alpha, rule) and for B2, the engine's start equals the frozen start.

#### (l) Order of work and reruns
1. This Part A is committed (and pushed) before any Stage 6 number exists.
2. `src/r9_stage6_methods.py` (tables, RQ1 resampling, re-split certification, metrics, baselines, comparison and
   cost tables; no file I/O) and `src/r9_stage6.py` (runner), with `tests/test_r9_stage6.py`: synthetic data with
   brute-force references (resampling counts, violation rule, certify via direct p-values and walk, metrics, D90,
   B2/B5/B5c/B6 on hand-made units), tamper tests (every input type changed by one byte is refused; a yaml changed by
   one byte is refused by `check_frozen()`; a wrong start or a non-partitioning split stops the run), and a test that
   the runner calls `check_frozen()` before reading anything. Full suite; commit.
3. One run on real data; the whole printout is kept (`stage6_printout.txt`); then stop.
4. No rerun to change a number. If a bug is found after the run, it is fixed in a separate commit with its tests, the
   run is repeated once with the fix, and both runs (printouts, summaries, SHA-256, the bug and the fix) are recorded
   in Part B; the fixed run's numbers are the results and the first run's are shown beside them.
5. Part B is filled after the author has reviewed the printout; committed with `stage6_summary.json`; entry appended
   to `LOG.md`; HANDOFF updated.

#### (m) Code, tests and outputs
- New: `src/r9_stage6_methods.py`, `src/r9_stage6.py`, `tests/test_r9_stage6.py`, `README_r9_stage6.md`. POC-1, POC-2
  and earlier POC-3 modules are imported, never edited.
- Outputs in `runs/btsc/stage6/`: `stage6_summary.json` (code SHA-256, input SHA-256 and blob IDs, files opened, every
  cell of (d) to (j)), `stage6_printout.txt`.

#### Choices the plan leaves open (decided by the author before this Part A was committed)
1. Q1 B1 beyond the resampling: B1 is filled from the RQ1 resampling only; its re-split rows in (i) read "not run
   (B1 is pre-registered for the resampling only)". The yaml freezes B1 for the resampling only (no per-seed starts),
   and plan Stage 6 runs B1 only there.
2. Q2 Resampling rows for B2, B5, B5c and B6: "not in the pre-registered resampling". None is in the RQ1 scope of the
   yaml.
3. Q3 Cost table alphas: every alpha, with 3% as the headline row (j).
4. Ways of carrying out the yaml that it does not spell out (no number is set by them):
   - Draw call: `rng.integers(0, 530, size=397)` into the pool poles sorted by ascending pole ID (the pattern of
     `src.r9_stage5_scores.pole_boot_weights`) (d).
   - Common draws: one draw serves every RQ1 cell (scores, B1, alphas, rules) (d).
   - Start per draw: `certify()` computes the start from the draw's n by the frozen rule; it is not a stored value (d).
   - Medians in the comparison table are taken over the seeds where the method was certified, with the count shown
     (i).

### Part B: results (filled from the printout after the run)

- Date: 2026-11-15 / Run tag: runs/btsc/stage6 / Code commit: 34c3492 (`src/r9_stage6.py` SHA-256
  c6b7a89c0eda47c2b7a8bc6e0abf84d2ac82bc47b0d302c37f99bebafff7b108, `src/r9_stage6_methods.py` SHA-256
  d0681dcf82e6ee2ff4d35af26511d0b3d2ebbb977277cdab6847435c3c10c6fd) / Part A commit: 89a8c42. Full suite before the
  run: 505 passed. One run, `py -3.12 -m src.r9_stage6`, 24.7 s, exit 0; not rerun.
- `stage6_summary.json` SHA-256 71be795779554a62546b1e03089222b53c35bc5a8ffac4bb2406c470b39e185f (git blob
  28ceaa6ae6cdd8964b0018510612fe2840dab3e2); `stage6_printout.txt` SHA-256
  b2d6eef1a8f11d673882b01d08ff4ecca0d6c4eb57bdad04bfbd844c1898f30c (git blob
  4dd6c061f9c80345ffc941a90aecb235d1ed4c2b).

#### Integrity (k): all checks True
- Freeze check PASS, 4 files by git blob ID (plan R9, engine, engine tests, `configs/frozen_r9_btsc.yaml`).
- Stage 4 inputs and `stage5_design.json` matched by SHA-256; both pool files matched `stage4_summary.json`. Files
  opened: the ten of (a), no others.
- The recomputed design phase wrote `configs/frozen_r9_btsc.yaml` again byte for byte (LF).
- Pool rows match the view table; no design pole in the pool; `f3_closed()` true before the first pool label; pool
  head top-3 and margins equal the stored Stage 4 scores exactly; pool 530 poles, 626 units, 1,860 views.
- Splits partition the pool and give the yaml's Cert / Test unit counts; every engine start (scores and B2) equals
  the frozen start.

#### V-check (e): PASS, 5 of 5 cells (alpha 3%, exact binomial, upper 95% CP <= 0.12)
| Score | Certified (of 2,000) | Violations | Rate [95% CP] | Mean true risk | Result |
|---|---|---|---|---|---|
| S1 | 2000 | 0 | 0.00% [0.00, 0.18] | 1.31% | PASS |
| S2-sum | 1196 | 65 | 3.25% [2.52, 4.12] | 1.85% | PASS |
| S2-min | 1920 | 65 | 3.25% [2.52, 4.12] | 1.05% | PASS |
| S3 | 1989 | 0 | 0.00% [0.00, 0.18] | 1.31% | PASS |
| S4 | 2000 | 0 | 0.00% [0.00, 0.18] | 1.16% | PASS |

Every scored cell certified in at least 1,196 of 2,000 draws, so no pass rests on a cell that rarely certifies. All
65 S2 violations (each variant) are draws that certified grid point 0 (commit at view 1), whose true pool error is
3.35% for S2. The check is scored as pre-registered.

#### RQ1 (d): violation rate, sign-level against view-level B1
Measured on the design poles (Stage 5): within-unit rho 0.631 [0.126, 0.887], design effect 2.24 [1.25, 2.74] at
2.96 views per unit. GTSRB record (POC-2 Stage 3b, alpha 1%, HB): frame-level 20.0% [18.3, 21.8]%, track-level 1.4%
[0.9, 2.0]%. Pool true risk at grid point 0 (commit at view 1): S1, S3, S4 1.44%; S2-sum, S2-min 3.35%; B1 1.45% per
view. Items per draw: 441 to 503 units (mean 468.6); 1,309 to 1,490 views for B1.

Rate % [95% CP] (certified draws of 2,000); exact binomial / HB primary start:
| alpha | S1 | S2-sum | S2-min | S3 | S4 | B1 (views) |
|---|---|---|---|---|---|---|
| 0.5% | no start / 0.00 (0) | no start / 0.00 (0) | no start / 0.00 (0) | no start / 0.00 (0) | no start / 0.00 (0) | **14.00 [12.51, 15.60] (1083)** / 4.00 [3.18, 4.95] (958) |
| 1% | 1.45 [0.97, 2.08] (1646) / 0.05 [0.00, 0.28] (958) | 0.20 [0.05, 0.51] (4) / 0.00 (0) | 0.00 (120) / 0.00 (13) | 0.65 [0.35, 1.11] (390) / 0.10 [0.01, 0.36] (94) | 1.25 [0.81, 1.84] (1648) / 0.10 [0.01, 0.36] (958) | 8.60 [7.41, 9.92] (1642) / 3.15 [2.43, 4.01] (946) |
| 2% | 0.00 (2000) / 0.00 (1219) | 0.05 [0.00, 0.28] (285) / 0.00 (67) | 0.05 [0.00, 0.28] (1234) / 0.00 (658) | 0.00 (1644) / 0.00 (1253) | 0.00 (1632) / 0.00 (962) | 0.00 (1998) / 0.00 (1646) |
| 3% | 0.00 (2000) / 0.00 (937) | 3.25 [2.52, 4.12] (1196) / 0.80 [0.46, 1.30] (710) | 3.25 [2.52, 4.12] (1920) / 0.90 [0.53, 1.42] (1667) | 0.00 (1989) / 0.00 (1936) | 0.00 (2000) / 0.00 (1147) | 0.00 (2000) / 0.00 (1856) |
| 5% | 0.00 (2000) / 0.00 (1994) | 0.00 (1993) / 0.00 (1955) | 0.00 (2000) / 0.00 (1998) | 0.00 (2000) / 0.00 (2000) | 0.00 (2000) / 0.00 (1784) | 0.00 (2000) / 0.00 (2000) |

0.5% with HB: 1 of 2,000 draws had a start for each score; none certified. Every rate of 0.00 has the interval
[0.00, 0.18].
- View-level B1 exceeds delta at 0.5%: 14.00% [12.51, 15.60]% with the exact binomial (HB 4.00%).
- At 1% B1 is 8.60% [7.41, 9.92]%, below delta but about 6 times the largest sign-level rate (S1 1.45%).
- At 2% and above, B1, S1, S3 and S4 have no violations, because their pool error at view 1 is 1.44% (1.45% per view
  for B1). S2-sum and S2-min, whose pool error at view 1 is 3.35%, have 1 violation each at 2% (0.05%) and 65 each at
  3% (3.25%), within delta.
- No sign-level cell exceeds delta at any alpha or rule; the largest sign-level rate is 3.25% [2.52, 4.12]%.

#### RQ2 (f), (g): smallest certifiable alpha per score and rule (seeds 0 to 4; "-" = none)
| Score | binom | hb | hb_a1 | hb_bs |
|---|---|---|---|---|
| S1 | 1, 1, 1, 1, 1% | 5, 5, 5, 5, 1% | -, -, 5, 5, 1% | -, -, -, -, 1% |
| S2-sum | 3, 3, 5, 5, 3% | 5, 5, 5, 5, 5% | 5, 5, 5, 5, 5% | 5, 5, 5, 5, 5% |
| S2-min | 2, 2, 3, 2, 2% | 3, 3, 3, 3, 2% | 3, 3, 3, 3, 2% | 3, 3, 3, 3, 2% |
| S3 | 2, 2, 2, 2, 2% | 2, 2, 3, 3, 2% | 2, 2, 3, 3, 2% | 2, 2, 3, 3, 2% |
| S4 | 1, 1, 1, 1, 1% | 5, 5, 3, 5, 1% | -, -, -, -, 1% | -, -, -, -, 1% |
| B2 | 1, 1, 1, 1, 1% | 5, 2, 2, 5, 1% | -, -, 2, -, 1% | -, -, -, -, 1% |

E3: the exact binomial certified 1% for S1 and S4 in all 5 re-splits; Hoeffding-Bentkus did so only in seed 4 under
each of its three starts. In no (seed, score, alpha) cell did HB certify where the exact binomial did not.

#### Every Test cell above alpha (error among committed, 95% CP)
- Layer (certified): S1 and S4, seed 4, 1%, all four rules: 1.32% (2/152) [0.16, 4.67], coverage 96.20%. No other
  certified layer cell is above alpha.
- B2 (certified), all in seed 4: 1% binom 1.32% (2/152) [0.16, 4.67]; 1% hb, hb_a1, hb_bs 1.34% (2/149)
  [0.16, 4.76]; 2% binom 3.80% (6/158) [1.41, 8.08]; 2% hb and hb_a1 3.18% (5/157) [1.04, 7.28]; 3% binom and hb
  3.80% (6/158) [1.41, 8.08].
- B6 S2-sum (against r = alpha): 0.5% and 1% in all 5 re-splits (1.94% (3/155) [0.40, 5.55], 1.30% (2/154)
  [0.16, 4.61], 1.23% (2/162) [0.15, 4.39], 1.29% (2/155) [0.16, 4.58], 1.90% (3/158) [0.39, 5.45] for seeds 0 to 4);
  2% in all 5 (3.23% (5/155) [1.06, 7.37], 2.60% (4/154) [0.71, 6.52], 2.47% (4/162) [0.68, 6.20], 3.23% (5/155)
  [1.06, 7.37], 2.53% (4/158) [0.69, 6.35]); 3% in seeds 0 and 3 (3.23% (5/155) [1.06, 7.37]).
- B6 S2-min: 0.5% in all 5 (0.65% (1/153) [0.02, 3.59], 0.65% (1/153) [0.02, 3.59], 0.62% (1/160) [0.02, 3.43],
  0.66% (1/152) [0.02, 3.61], 1.92% (3/156) [0.40, 5.52]); 1% in seed 4 (1.92% (3/156) [0.40, 5.52]); 2% in all 5
  (same values as S2-sum at 2%); 3% in seeds 0 and 3 (3.23% (5/155) [1.06, 7.37]).
- B5 (real error above alpha; gap 0.00% in each): 0.5% in all 5 (1.29% [0.16, 4.58], 0.65% [0.02, 3.56], 0.62%
  [0.02, 3.39], 0.65% [0.02, 3.54], 1.90% [0.39, 5.45]); 1% in seeds 0 (1.29%) and 4 (1.90%).
- B5c (real error above alpha; gap 0.00% in each): 0.5% in all 5 (same values as B5); 1% in seed 4 (1.90%
  [0.39, 5.45]).

#### Comparison table (i), BelgiumTSC, primary alpha 3%: median (min-max; seeds counted)
At 3%, most methods certify committing every unit at view 1 because the pool error at view 1 is 1.44%; the
primary-alpha table therefore does not separate the methods. The primary alpha is not changed after the run. All
alphas were pre-registered to be run and reported; the 1% results are reported alongside as pre-registered,
non-scored cells.

| Row | S1 | S2-sum | S2-min | S3 | S4 | B1 | B2 | B5 | B5c | B6 S2-sum | B6 S2-min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Type of promise | real error, sign | real error, sign | real error, sign | real error, sign | real error, sign | real error per view | real error, one view per sign | accuracy gap, marginal | accuracy gap, conditional | none | none |
| Violation rate (resampling) | 0.00% [0.00, 0.18] (2000) | 3.25% [2.52, 4.12] (1196) | 3.25% [2.52, 4.12] (1920) | 0.00% [0.00, 0.18] (1989) | 0.00% [0.00, 0.18] (2000) | 0.00% [0.00, 0.18] (2000) | not in RQ1 | not in RQ1 | not in RQ1 | not in RQ1 | not in RQ1 |
| Certified re-splits | 5/5 | 3/5 | 5/5 | 5/5 | 5/5 | not run | 5/5 | 5/5 (finite lambda) | 5/5 (finite lambda) | no certificate | no certificate |
| Test coverage | 100% | 100% (3) | 98.73 (98.06-99.35) | 100% | 100% | not run | 100% | 100% | 100% | 100% | 100% |
| Test error | 0.65 (0.62-1.90) | 1.90 (1.30-1.94; 3) | 0.65 (0.62-1.92) | 0.65 (0.62-1.90) | 0.65 (0.62-1.90) | not run | 1.85 (0.65-3.80) | 0.65 (0.62-1.90) real | 0.65 (0.62-1.90) real | 2.60 (2.47-3.23) | 2.60 (2.47-3.23) |
| Worst error vs 3% | 1.90, 0 above | 1.94, 0 above | 1.92, 0 above | 1.90, 0 above | 1.90, 0 above | not run | 3.80, 1 above | gap 0.65; real 1.90, 0 above | gap 0.65; real 1.90, 0 above | 3.23, 2 above | 3.23, 2 above |
| Smallest certifiable alpha | 1% (5/5) | 3% (3-5%) | 2% (2-3%) | 2% (5/5) | 1% (5/5) | not run | 1% (5/5) | 0.5% (gap tolerance) | 1% (gap tolerance) | none | none |
| Median commit view | 1 | 1 | 1 | 1 | 1 | not run | 2 | 1 | 1 | 1 | 1 |
| Median size at commit (px) | 71 (62-75.5) | 70 (62-75.5) | 71.5 (63-77) | 71 (62-75.5) | 71 (62-75.5) | not run | 67 (60-70.5) | 71 (62-75.5) | 71 (62-75.5) | 71 (62-75.5) | 71 (62-75.5) |
| Can escalate | yes | yes | yes | yes | yes | yes | yes | no | no | yes | yes |

Errors and coverage in %. B1 re-split rows: "not run (B1 is pre-registered for the resampling only)". B1 certify
counts in the resampling (exact binomial): 0.5% 1083, 1% 1642, 2% 1998, 3% 2000, 5% 2000.

#### Comparison table at 1% (pre-registered, reported, not the primary alpha)
Same format. Assembled after the run from `stage6_summary.json` by `src/r9_stage6_tables.py`
(`py -3.12 -m src.r9_stage6_tables --alpha 0.01`, commit 2392433; deviation 2 below).

| Row | S1 | S2-sum | S2-min | S3 | S4 | B1 | B2 | B5 | B5c | B6 S2-sum | B6 S2-min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Violation rate (resampling) | 1.45% [0.97, 2.08] (1646) | 0.20% [0.05, 0.51] (4) | 0.00% [0.00, 0.18] (120) | 0.65% [0.35, 1.11] (390) | 1.25% [0.81, 1.84] (1648) | 8.60% [7.41, 9.92] (1642) | not in RQ1 | not in RQ1 | not in RQ1 | not in RQ1 | not in RQ1 |
| Certified re-splits | 5/5 | 0/5 | 0/5 | 0/5 | 5/5 | not run | 5/5 | 5/5 (finite lambda) | 5/5 (finite lambda) | no certificate | no certificate |
| Test coverage | 93.55 (91.36-96.20) | n/a | n/a | n/a | 95.48 (93.55-97.40) | not run | 92.86 (86.45-96.20) | 100% | 100% | 100% | 98.73 (98.06-99.35) |
| Test error | 0.00 (0.00-1.32) | n/a | n/a | n/a | 0.00 (0.00-1.32) | not run | 0.00 (0.00-1.32) | 0.65 (0.62-1.90) real | 0.65 (0.62-1.90) real | 1.30 (1.23-1.94) | 0.65 (0.62-1.92) |
| Worst error vs 1% | 1.32, 1 above | n/a | n/a | n/a | 1.32, 1 above | not run | 1.32, 1 above | gap 0.65; real 1.90, 2 above | gap 0.65; real 1.90, 1 above | 1.94, 5 above | 1.92, 1 above |
| Smallest certifiable alpha | 1% (5/5) | 3% (3-5%) | 2% (2-3%) | 2% (5/5) | 1% (5/5) | not run | 1% (5/5) | 0.5% (gap tolerance) | 1% (gap tolerance) | none | none |
| Median commit view | 2 | n/a | n/a | n/a | 1 | not run | 2 | 1 | 2 (1-3) | 1 | 1 |
| Median size at commit (px) | 65.5 (62.5-71) | n/a | n/a | n/a | 73 (68-79.5) | not run | 68.5 (61-73) | 71 (62-75.5) | 66 (61-71) | 71 (62-75.5) | 71.5 (63-77) |
| Can escalate | yes | yes | yes | yes | yes | yes | yes | no | no | yes | yes |

Type of promise as in the 3% table.

#### S2 against B6 (j): cost of the certificate, every alpha (median commit view 1 wherever anything commits)
Coverage %, error % median (min-max), worst error, re-splits above alpha. Reference = uncertified reference.

| Variant | alpha | Layer (exact binomial) | B6 (r = alpha) | Reference |
|---|---|---|---|---|
| S2-sum | 0.5% | 0/5 certified | 100; 1.30 (1.23-1.94); worst 1.94; 5 above | commits nothing |
| S2-sum | 1% | 0/5 certified | 100; 1.30 (1.23-1.94); worst 1.94; 5 above | commits nothing |
| S2-sum | 2% | 0/5 certified | 100; 2.60 (2.47-3.23); worst 3.23; 5 above | 100; 1.30 (1.23-1.94); worst 1.94; 0 above |
| S2-sum | **3%** | **3/5; 100; 1.90 (1.30-1.94); worst 1.94; 0 above** | **100; 2.60 (2.47-3.23); worst 3.23; 2 above** | 100; 1.30 (1.23-1.94); worst 1.94; 0 above |
| S2-sum | 5% | 5/5; 100; 2.60 (1.23-3.23); worst 3.23; 0 above | 100; 2.60 (2.47-3.23); worst 3.23; 0 above | 100; 2.60 (2.47-3.23); worst 3.23; 0 above |
| S2-min | 0.5% | 0/5 certified | 98.73 (98.06-99.35); 0.65 (0.62-1.92); worst 1.92; 5 above | commits nothing |
| S2-min | 1% | 0/5 certified | 98.73 (98.06-99.35); 0.65 (0.62-1.92); worst 1.92; 1 above | 98.73; 0.65 (0.62-1.92); worst 1.92; 1 above |
| S2-min | 2% | 4/5; 98.72 (98.06-99.35); 0.66 (0.65-1.92); worst 1.92; 0 above | 100; 2.60 (2.47-3.23); worst 3.23; 5 above | 98.73; 0.65 (0.62-1.92); worst 1.92; 0 above |
| S2-min | **3%** | **5/5; 98.73 (98.06-99.35); 0.65 (0.62-1.92); worst 1.92; 0 above** | **100; 2.60 (2.47-3.23); worst 3.23; 2 above** | 98.73; 0.65 (0.62-1.92); worst 1.92; 0 above |
| S2-min | 5% | 5/5; 100 (98.73-100); 2.60 (0.62-3.23); worst 3.23; 0 above | 100; 2.60 (2.47-3.23); worst 3.23; 0 above | 100; 2.60 (2.47-3.23); worst 3.23; 0 above |

Where both commit, the certificate costs no commit view (view 1 for both). At 3% it costs S2-sum 2 of 5 re-splits
and S2-min about 1.3 points of coverage; B6 exceeds its target in 2 of 5 re-splits at 3% and in 5 of 5 at 2%
(both variants).

#### B5 and B5c: accuracy gap against real error (Test, per seed 0 to 4)
| Method | alpha | lambda | Gap % | Real error % |
|---|---|---|---|---|
| B5 | 0.5% | 0.08 (all seeds) | 0.00 in all | 1.29, 0.65, 0.62, 0.65, 1.90 |
| B5 | 1% | 0.01 (seed 3: 0.00) | 0.00, 0.00, 0.00, 0.65, 0.00 | 1.29, 0.65, 0.62, 0.65, 1.90 |
| B5 | 2%, 3%, 5% | 0.00 | 0.00, 0.00, 0.00, 0.65, 0.00 | 0.65, 0.65, 0.62, 0.65, 1.90 |
| B5c | 0.5% | all infinite (stop at the last view) | 0.00 in all | 1.29, 0.65, 0.62, 0.65, 1.90 |
| B5c | 1% | seeds 0, 3: (0, 0, 0); seeds 1, 2: (inf, inf, 0); seed 4: (inf, 0, 0) | 0.00, 0.00, 0.00, 0.65, 0.00 | 0.65, 0.65, 0.62, 0.65, 1.90 |
| B5c | 2%, 3%, 5% | (0, 0, 0) | 0.00, 0.00, 0.00, 0.65, 0.00 | 0.65, 0.65, 0.62, 0.65, 1.90 |

The gap stays at or below the tolerance in every cell (largest 0.65%); the real error reaches 1.90% (seed 4) at every
tolerance. Coverage is 100% throughout (Ringel never escalates).

#### Deviations, bugs and reruns
1. The printout header and `stage6_summary.json` (`worktree_clean: false`) report a tree that is not clean. Cause: the
   runner creates `runs/btsc/stage6/` and opens the printout there before it calls `git status`, so the run's own
   untracked output folder is counted. The tree was clean before the run (checked after commit 34c3492); the code
   files equal that commit. No number changed; not rerun.
2. Part A (i) says the comparison table is also printed at every other alpha. The runner printed it at 3% only. The 1%
   table above was assembled after the run by `src/r9_stage6_tables.py` (commit 2392433), which reads the
   hash-checked `stage6_summary.json` and calls the runner's own `comparison_table` function; no number is recomputed
   from data. Its test (`tests/test_r9_stage6_tables.py`) checks that the 3% table it builds equals the printed one line
   for line. Tables at 0.5%, 2% and 5% were not assembled; their cells are all in `stage6_summary.json` and the
   printout, and the same script builds them.
3. Part A does not define "certified" for B5 and B5c. The runner reads it as a finite threshold (B5) or at least one
   finite threshold (B5c); with no finite threshold Ringel's rule stops every unit at its last view.
4. "Test cells above alpha" lists B5 and B5c where the real error exceeds alpha, although their promise is the gap.
- No bug was found; there was one run and no rerun.
- Next step: Stage 8 (final tables), only when the author says so. Stage 7 (DITS) is dropped (F-A3 failed).
