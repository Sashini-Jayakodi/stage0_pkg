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

- Date: ____ / Run tag: runs/btsc/stage6 / Code commit: ____ / Part A commit: ____
- `stage6_summary.json` SHA-256: ____ ; `stage6_printout.txt` SHA-256: ____
- Integrity (k): ____
- RQ1 (d): ____
- V-check (e), S1 / S2-sum / S2-min / S3 / S4 at 3%, binom (certify count; violations; rate [95% CP]; PASS/FAIL):
  ____
- RQ2 (f), (g): ____
- Baselines (h): ____
- Comparison table (i): ____
- S2 against B6 (j): ____
- Deviations, bugs and reruns: ____
