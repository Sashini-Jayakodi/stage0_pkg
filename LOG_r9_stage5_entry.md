## POC-3 (R9) Stage 5: BelgiumTSC design phase and freeze F3 (design poles only)

### Part A: fixed definitions (committed before any Stage 5 number exists)

Plan: `docs/poc3-implementation-plan-r9.md` (git blob 0abe7fa51af3b499cd79218e46c83790dbfa2e99), sections 2, 3, 5, 6,
11 and 9 Stage 5. This is the design phase of the fresh dataset: values may be chosen freely on the 133 design poles
until `configs/frozen_r9_btsc.yaml` is committed (F3). No Cert or Test number and no label-dependent number on pool
poles exists before F3 (hard rule 3). delta = 0.1 throughout. Stage 5 has one F-gate (F-D1, section (i)); the checks
in (m) are integrity checks: if one fails, the run stops and the failure is logged.

#### (a) Inputs and hash checks (hard rule 2)
- `runs/btsc/stage4/stage4_summary.json` must have SHA-256
  1c66a0119d39473f818f1add56c8a878d32ddfcc91aef4b0f5a9bd76568a3a31 (Stage 4 Part B). Every other Stage 4 file is
  checked against its `files_sha256` entry before it is read.
- Text inputs (CSV, JSON) are line-ending sensitive: the recorded SHA-256 values are of the bytes written on this
  machine (CRLF). A text input passes if its SHA-256 equals the recorded value, or if its git blob ID (computed after
  CRLF -> LF, as `src.r9_frozen.blob_id`) equals the blob ID committed in `7a1d985`; otherwise the run stops. Blob IDs:
  `stage4_summary.json` 3d0d79c42e3a782e301656acdb0a24515646ebab, `split_poles.json`
  a7affda109adab1332994484c974c07b5fc3eabe, `test_views.csv` e486d6f1a58d52aed01f5366673b7c07a32a3d5b. Binary inputs
  (`*.npz`) are checked by SHA-256 only.
- Files opened in Stage 5: `stage4_summary.json`, `split_poles.json`, `test_views.csv`, `embeddings/test_design.npz`,
  `head/head.npz`, `head/scores_test_design.npz` (cross-check of the recomputed per-view margins).
- Never opened in Stage 5: `embeddings/test_pool.npz`, `head/scores_test_pool.npz`, the caches and the checkpoint. A
  code assertion records every file opened and stops if a pool score or embedding file appears; a test checks it.
- Pool rows of `test_views.csv` are used for structure only: pole, unit key (pole, class folder), view count. The
  class of a pool unit is used only for the rarest-class strata of the Cert/Test split (g), as Stage 4 did for the
  design/pool split. No pool prediction exists in Stage 5, so no pool error can be formed.
- Testing labels of design poles are read through `src.r9_btsc_guard.GuardedTestLabels` only (pool poles refused
  while F3 is open).
- **Disclosure (structure only, before this Part A was committed):** while drafting this Part A, the views per unit
  and units per pole were counted from `test_views.csv` for both sides, using the pole, class-folder and row-count
  columns only; no label was joined with a prediction and no pool score or embedding file was opened. Pool counts:
  views per unit 1 view 11 units, 2 views 8, 3 views 599, 4 views 6, 6 views 2; units per pole 1 unit 448 poles,
  2 units 69, 3 units 12, 4 units 1. The design counts are in (c). These counts motivated decision Q2.

#### (b) Freeze check
- Every Stage 5 script calls `src.r9_frozen.check_frozen()` first (plan R9, engine, engine tests by git blob ID).
- The engine `src/r9_certify.py` is used unchanged: `max_errors`, `start_for`, `certify`, `cp_lower`.

#### (c) Units, views and view order
- Unit = (pole, class), as Stage 4. View = one image. View order within a unit = ascending YYYYY (Stage 1 Amendment 2;
  there is no camera index). "Commit time" on BelgiumTSC = the number of views used, 1-based.
- Design: 133 poles, 157 units, 465 views (Stage 4 Part B). Views per design unit (structure): 1 view 3 units,
  2 views 1, 3 views 152, 4 views 1. Units per design pole: 1 unit 110 poles, 2 units 22, 3 units 1.
- Sign size per view = the `size` column (min(ROI width, ROI height), original pixels). Size at commit = size of the
  committing view.
- Per-view evidence: the tied-Gaussian head of Stage 4 on the l2-normalised embeddings.
- **No gate (decision Q1):** every view gives evidence. Stage 4 fixed no gate threshold for BelgiumTSC; adding one
  would need a new threshold choice. This differs from GTSRB (Stage 1 gate, tau = 43.8435) and is stated.

#### (d) The scores (plan 5.1, 5.2)
All scores are sequential over the views of a unit in view order. A unit commits at the first view v with
confidence >= lambda; its label there is its answer. If no view qualifies, the unit escalates.
- **S1** (ours, POC-2 V1): summed head log-likelihood plus log prior over views 1..v; confidence = top minus
  runner-up; label = top class.
- **S2** (Boursinos and Koutsoukos 2022): per-view nonconformity ([BK] eq. 9) and ICP p-values ([BK] eq. 10) as in
  `src.r8_stage6_methods`; combined over the window of views max(1, v - 2)..v, that is all views seen so far up to 3
  and the last 3 after that (decision Q2; the GTSRB K = 5 cannot apply with 3 to 4 views per unit); ECDF-transformed;
  confidence = a * credibility + b * confidence; label = class with the largest combined p-value. Two variants, chosen
  now: **S2-sum (primary)**, ECDF Sum; **S2-min (secondary)**, ECDF Min.
  - **ECDF per window length (decision Q2):** the null distribution of a combined p-value depends on how many p-values
    are combined, so three ECDFs are fitted per variant, one each for windows of 1, 2 and 3 views. A window is
    transformed with the ECDF of its own length. Units with more than 3 views use the last 3 views from view 4 on
    (design: 1 unit; pool: 8 units by the disclosed count).
  - The POC-2 B6 functions assume a fixed K, so the variable-window combination is new code in
    `src/r9_stage5_scores.py`; `src/r8_stage6_methods.py` is imported, not changed.
- **S3** (POC-2 B4, sequential): running majority vote over per-view head top-1 classes, views 1..v (all views, no
  window); confidence = votes(label) - votes(runner-up); label = most votes, ties to the smaller class ID. Stated now:
  with 3 to 4 views the vote margin takes only the values 1 to 4 on design units, so S3's grid has at most 4 distinct
  points and its strictest point is coarse.
- **S4** (POC-2 B3, sequential): margin of the current view alone (top minus runner-up of the single-view head score);
  label = its top class.

#### (e) S2 calibration on the design poles (plan 5.2)
- Halves by pole, with the Stage 4 rule: `src.r9_stage4_data.split_by_pole(design_units, (1, 2), seed=20261115)`
  selects round-half-up(133 / 2) = 67 poles by rarest-class strata and largest remainder. Selected poles = half B
  (ECDFs and (a, b)); the other 66 poles = half A (ICP calibration scores). Every unit of a pole is on its pole's side.
- Half A: true-class nonconformity of every view. Half B: for each window length 1, 2, 3, the ECDF of the combined
  true-class p-value over every window of that length in every unit (windows as in (d)); then (a, b) by POC-2's rule
  (grid -1.0 to 1.0 in steps of 0.1, minimum AURC over all half-B unit-view positions, ties to the first pair in
  ascending order). Fitted separately for Sum and Min.
- Stated now: S2's grid, start and reference in (f), (h), (k) use design units that also fitted S2, so S2's design
  values are partly in-sample. This can move S2's grid, start and predicted certifiability, not the validity of its
  certificate, because Cert poles are disjoint from the design poles. [Likely] With about 78 units in half B, each
  ECDF rests on about 78 windows, far fewer than on GTSRB (2,994 windows over 120 tracks, Stage 3).

#### (f) Threshold grids (plan 5.3; design units only)
- As Stage 3 (f), with N = 157: per score and design unit i, C_i = the largest confidence over the unit's views
  (-inf if none); lambda_0 = -inf; lambda_j (j = 1..29) = the m_j-th largest C_i, m_j = ceil(f_j * 157),
  f_j = 1.00 - 0.70 * j / 29. Repeated values are removed, keeping the order; the number of distinct points per score
  is reported (expected near 30 for S1, S2, S4; at most 4 for S3).
- Design commit fraction f_d(j) = committed design units / 157; design committed error e_d(j) = wrong / committed.
- Grids are the same for every alpha and every re-split.

#### (g) Cert/Test re-splits (plan Stage 5)
- Seeds 0 to 4. Per seed s: Test poles = `split_by_pole(pool_units, (1, 4), seed=s)` over the 626 pool units, that
  is round-half-up(530 / 4) = 133 Test poles chosen by rarest-class strata (strata counted over pool units) and
  largest remainder; Cert = the other 397 poles. Every unit of a pole is on its pole's side.
- Split membership (pole lists per seed) is written to the design-values file and frozen at F3, so Stage 6 reads it
  instead of regenerating it.
- Cert size in units n(s) and Test size in units are reported per seed (structure only). n_min = min over s of n(s).

#### (h) Starts (plan 5.3; design values, n, n_design = 157)
- Exact binomial (primary): `power_start(f_d, n(s), alpha)`; HB comparator: `hb` (primary, Stage 2 Amendment 2,
  n_design = 157), `hb_a1`, `hb_bs`, all three reported.
- **n (decision Q3):** each seed's own Cert unit count n(s) for the starts and for certification in that seed, so
  starts can differ between seeds; all of them are frozen at F3.
- Floors on f_d for the binomial start at n = 470 (for orientation): 4 / (470 * alpha) = 170% at 0.5% (no start),
  85.1% at 1%, 42.6% at 2%, 28.4% at 3%, 17.0% at 5%. alpha 0.5% therefore has no exact-binomial start (as Stage 3
  Amendment 1); it is reported as "no start possible", not as tested and failed.

#### (i) Primary alpha (plan Rule 3), F-D1 and Fallback 2
- For each score and alpha in {0.5, 1, 2, 3, 5}%: **predicted certifiable** if some grid point j with f_d(j) >= 70%
  has UCB90(j) * n_c(j) <= k_max(n_c(j), alpha), where UCB90(j) = the upper one-sided 90% Clopper-Pearson bound of
  e_d(j) from (wrong, committed) design units, n_c(j) = floor(f_d(j) * n_min) (decisions Q3 and Q4: the smallest Cert
  size of the five seeds, rounded down), and k_max = `src.r9_certify.max_errors` (no k_max, that is not certifiable
  even with 0 errors, means not predicted).
- **Primary alpha** = the smallest alpha predicted certifiable for at least one score.
- What the rule can pick, from arithmetic alone (n_design = 157, n between 460 and 480 units, no design outcome used;
  recomputed at the real n_min in the design step): 0.5%, 1% and 2% can never be predicted certifiable, because UCB90
  from at most 157 units stays above the room k_max leaves even with 0 errors; 3% is predicted only if a point with
  f_d >= 70% has 0 committed design errors; 5% is predicted only if such a point has at most 2. So the primary alpha is
  3%, 5% or (after F-D1) 10% by construction. This is the conservatism Rule 3 asks for. It limits only which alpha is
  called primary: Stage 6 still certifies and reports every alpha in the grid.
- **Walk path (decision Q5):** Rule 3 checks a single grid point. The walk starts at the power-aware start and must
  pass every point from there down to the certified point; the prediction does not model this path. Recorded as a
  limitation of the prediction, not corrected.
- **F-D1** passes if some alpha in {1, 2, 3, 5}% is predicted certifiable for at least one score. If it fails: 10% is
  added to the alpha grid before the freeze and this is stated in Part B and in the pre-registration block; Rule 3 is
  rerun with 10%. If still none: Fallback 2 (plan section 11), the certifiability curve against the number of poles is
  reported as the RQ2 answer for BelgiumTSC ("not certifiable at this size"). F-D1 is an F-gate: it cannot fail the
  project.

#### (j) Within-sign error correlation and design effect (design poles only; reported, no pass/fail)
- Error indicator per view = head top-1 wrong (no threshold).
- Within-unit rho: one-way ANOVA ICC with unequal cluster sizes (`src.stage5.icc_anova`, POC-1, imported), views
  clustered by unit; 95% interval by a pole-cluster bootstrap, 2,000 draws, `numpy.random.default_rng([20261115, 5, 1])`.
  Design effect = 1 + (m - 1) * rho with m = mean views per design unit (465 / 157 = 2.96), and at m = 3 for the plan's
  comparison; with interval.
- Within-pole rho: the same ICC on unit-level errors (last view in view order), units clustered by pole; same
  bootstrap with `default_rng([20261115, 5, 2])`. Only 23 design poles hold more than one unit, so this interval is
  expected to be wide; it is reported as measured.
- With about 14 wrong views in 465 (Stage 4 Part B), both rho values are expected to be imprecise.

#### (k) Stage 6 settings frozen here (plan Stage 6)
- **RQ1 resampling:** population = the 530 pool poles with every unit's per-lambda (committed, wrong) from the
  tracks-by-lambdas table (hard rule 4). 2,000 draws, each 397 poles with replacement (the Cert size in poles); a pole
  drawn twice contributes its units twice. Sign-level certificate counted by unit, with n = the number of units drawn;
  violation = the true pool risk (pool committed error at the certified lambda) above alpha. View-level B1: each view
  treated as an independent unit with the S4 single-view margin, view grid built by rule (f) from the 465 design views
  (N = 465), n_design = 465 views; violation = true pool view-level committed error above alpha. Draw seed
  `default_rng([20261115, 6, d])`, d = draw index.
- **RQ1 scope (decision Q6, fixed before any design number):** the resampling runs for every score variant (S1,
  S2-sum, S2-min, S3, S4) and for B1, at every alpha with a start, with the exact binomial and with HB under its
  primary start (`hb`). All cells are reported.
- **Stage 6 V-check scope (decision Q6):** the V-check (upper 95% Clopper-Pearson bound of the sign-level violation
  rate <= delta + 0.02 = 0.12) is scored only for every score variant at the primary alpha, exact binomial. All other
  cells (other alphas, HB, B1) are reported, not scored. Fixed now, before any design number, so the primary alpha
  that it depends on is chosen by Rule 3 and not after a Stage 6 number.
- Reported next to the measured design effect (j) and the GTSRB numbers (POC-2 Stage 3).
- **Certification (RQ2):** S1, S2-sum, S2-min, S3, S4 on the 5 re-splits at every alpha, with the exact binomial and HB
  (three starts). Reported per (seed, score, alpha, rule) on Test units: certified or not; coverage; error among
  committed; escalation rate; median commit view; median size at commit; D90 size (Stage 3 (i) rule; on BelgiumTSC a
  description of size, not of approach time); smallest certifiable alpha.
- **Uncertified reference** per score and alpha: the loosest grid point with e_d <= alpha on the design units, as
  Stage 3 (i), fixed now.
- **Baselines (decision Q7), on the same 5 re-splits at every alpha:**
  - **B2** (one random view per unit, Learn-then-Test): each unit is represented by one view drawn uniformly from its
    views; confidence = that view's S4 margin; the unit commits if it is >= lambda, at that view's position. Cert and
    Test draws per seed with `default_rng([20261115, 7, s])`; design draw with `default_rng([20261115, 7, 100])`; grid
    by rule (f) from the design draw (N = 157); certified with the same engine (exact binomial; HB under three starts),
    n = n(s), starts from the design draw's f_d.
  - **B5 and B5c** (Ringel et al. 2024), POC-2 Stage 6 code (`src.r8_stage6_methods`, imported) and its rules
    (`LOG_r8_stage6_entry.md` Part A (b) to (d)), with views as the time axis: classifier output = the S1 accumulated
    score; Y_full = the S1 top class after the unit's last view (its own t_max); pi = F_design(S1 margin), the
    empirical CDF of the S1 margins over all 465 design views; grid {0, 0.01, ..., 1} plus infinity; exact binomial
    p-value; delta = 0.1; alpha = the gap tolerance. B5 calibrates on all Cert units. B5c: one threshold per view index
    t = 1, 2, 3, units with more than 3 views use lambda_3 from view 4 on; Cert poles split into D_cal-1 (the first
    round-half-up(397 / 2) = 199 poles of a permutation drawn with `default_rng([20261116, s])`) and D_cal-2 (the other
    198), by pole. B5 and B5c never escalate.
  - **B6** (Boursinos and Koutsoukos 2022, uncertified): S2-sum and S2-min exactly as fitted in (e); a unit commits at
    its first view with k > theta, label = the largest combined p-value; theta = the smallest value whose unit-level
    selective error on half B is <= r, with r = alpha for each alpha; if none, B6 commits nothing at that alpha. No
    certificate. Declared difference from POC-2: POC-2 kept a separate selection third (Stage 6 Part A (e)); here half
    B serves the ECDFs, (a, b) and theta, so theta is chosen in-sample.

#### (l) Freeze F3
- `configs/frozen_r9_btsc.yaml` holds: delta; alpha grid (with 10% if F-D1 added it); primary alpha and the Rule 3
  table behind it (with n_min); per score the grid (lambda values), f_d, e_d and the starts per (seed, alpha, rule); S2
  halves (pole lists), the ICP calibration scores' SHA-256, the three ECDFs per variant, (a, b) per variant; re-split
  seeds, pole lists and n(s); RQ1 draw count, seed rule and scope; the Stage 6 V-check scope; the B1 view grid; the
  baseline settings of (k) with B2's design grid, B5's F_design and B6's theta per alpha; the SHA-256 and blob ID of
  the design-values file and of every Stage 4 input.
- The F3 commit contains the yaml, the pre-registration block appended to `LOG.md`, and `configs/frozen_r9_btsc.yaml`
  added to `FROZEN_BLOBS` in `src/r9_frozen.py` with its blob ID. From that commit on the guard lets pool labels
  through, so nothing in the yaml can change. The yaml's SHA-256 and blob ID are recorded in `LOG.md`. The F3 commit
  and its push wait for the author's explicit go.

#### (m) Integrity checks (stop on failure; not results)
- `check_frozen()` passes; every input matches (a); no pool score or embedding file was opened; every label request
  went through the guard and only design poles were served.
- Recomputed per-view head margins and top-1 classes on design views equal `head/scores_test_design.npz` (1e-9).
- The design-values file read by the freeze step has the SHA-256 recorded in this Part A (hard rule 2).
- Stage 4 design facts reproduce: 157 units, 465 views, 451 / 465 views and 150 / 157 units correct at the last view.

#### (n) Order of work
1. This Part A is committed.
2. `src/r9_stage5_scores.py` (BelgiumTSC sequences, variable-window S2 with per-length ECDFs, halves, grids, design
   statistics, Rule 3, correlation, baseline design fits; no file I/O) and `src/r9_stage5_design.py` (design poles
   only; writes `stage5_design.json`), with synthetic tests (brute-force references for the variable-window
   combination, Rule 3, grids and the re-split rule; tamper tests for every input type; a test that the pool files
   are refused). Full suite; commit.
3. The design script is run; the whole printout is kept; its output file and SHA-256 go into this Part A in a separate
   commit ("Design-pole values").
4. `src/r9_stage5_freeze.py` writes `configs/frozen_r9_btsc.yaml` from the hash-checked design-values file; the
   pre-registration block is drafted; both are shown to the author; F3 commit and push on the author's go.
5. Part B is filled after the author has reviewed the printout; the entry is appended to `LOG.md`; HANDOFF updated.

#### (o) Code, tests and outputs
- New: `src/r9_stage5_scores.py`, `src/r9_stage5_design.py`, `src/r9_stage5_freeze.py`;
  `tests/test_r9_stage5.py`. POC-1, POC-2 and earlier POC-3 modules are imported, never edited, except the one-line
  `FROZEN_BLOBS` addition in `src/r9_frozen.py` at F3 (planned there since Stage 3).
- Outputs in `runs/btsc/stage5/`: `stage5_design.json`, `stage5_printout.txt`, `stage5_summary.json` (code SHA-256,
  input hashes, files opened). `README_r9_stage5.md` gives the run commands.

#### Choices the plan leaves open (decided by the author before this Part A was committed)
1. Q1 Gate: none; every view gives evidence (c).
2. Q2 S2 window and ECDF: one ECDF each for windows of 1, 2 and 3 views; units with more than 3 views use the last 3
   (d), (e).
3. Q3 Cert size n: each seed's own Cert unit count for starts and certificates; the smallest of the five (n_min) for
   Rule 3 (h), (i).
4. Q4 Rounding in Rule 3: n_c rounded down (i).
5. Q5 Rule 3 path: single grid point; recorded that it ignores the walk path (i).
6. Q6 RQ1 and V-check scope: resampling for all score variants and B1 at every alpha with a start, exact binomial and
   HB primary start, all reported; the Stage 6 V-check scored only on every score variant at the primary alpha, exact
   binomial (k).
7. Q7 Baselines: B2, B5, B5c and B6 as fixed in (k).
8. Also decided: S2 halves seed 20261115 (e); Cert/Test 397 / 133 poles per seed with the pole lists frozen at F3 (g);
   bootstrap seeds [20261115, 5, 1] and [20261115, 5, 2] (j); RQ1 draw seeds [20261115, 6, d] (k); B2 seeds
   [20261115, 7, s] and [20261115, 7, 100], B5c seed [20261116, s] (k).

#### Design-pole values (step 3; design poles only)
Computed 2026-11-15 after the Part A commit (019f782), with the code committed first (b63546a;
`src/r9_stage5_design.py` and `src/r9_stage5_scores.py`, SHA-256 in the output file). Command
`py -3.12 -m src.r9_stage5_design`. Output `runs/btsc/stage5/stage5_design.json` SHA-256
115f98025f60b14aedad9c0d75df6d5deafc568d77b427e2bf9804be1d854a11, git blob b06a09f8c64bbd789bd02addab0b518ecde66d0e;
printout `stage5_printout.txt` SHA-256 3e6b8e5f01719c4873356f426382c0185f4e73a40ba6fa852ada78a3bad8280c.
- Integrity (m): freeze check PASS (3 files by blob ID); all six inputs matched by SHA-256; files opened: the six of
  (a), no pool score or embedding file; design rows, poles, views and sizes match the view table; guarded labels equal
  the stored design labels; recomputed head top-3 and margins equal the stored float32 values exactly (the float64
  margins differ from the stored float32 ones by at most 6.07e-05, rounding only); Stage 4 facts reproduce (133 poles,
  157 units, 465 views, 451 views and 150 units correct; pool 530 poles, 626 units).
- S2 halves: A 66 poles / 77 units / 226 views (226 ICP scores); B 67 poles / 80 units / 239 views; ECDF sizes
  80 / 80 / 79 for windows of 1 / 2 / 3 views. S2-sum a = -0.0, b = +0.1, AURC 0.00005; S2-min the same a, b and AURC.
- Re-splits (Cert / Test units): seed 0 471 / 155, 1 472 / 154, 2 464 / 162, 3 471 / 155, 4 468 / 158; 397 / 133 poles
  in every seed; n_min = 464.
- Distinct grid points: S1 30, S2-sum 3, S2-min 3, S3 4, S4 30. Design committed error at lambda_0 (commit at view 1):
  S1, S3, S4 2.55% (4 / 157); S2-sum and S2-min 4.46% (7 / 157).
- **S2 on BelgiumTSC is nearly degenerate.** Its confidence takes only two values on the design units (0.09875 and
  0.1), so its grid has 3 points (lambda_0, 0.09875, 0.1) and its strictest point commits 97.45% (S2-sum) / 95.54%
  (S2-min). Mechanism: (a, b) = (0, 0.1) makes the confidence 0.1 x (1 - second-largest combined p-value); the
  second-largest ICP p-value is 0 for 418 of 465 views, and each ECDF has only about 80 values, so the transformed
  values sit on a few steps. Its 3 extra wrong units at view 1 (7 against 4 for S4) come from ties: the largest
  combined p-value is tied for 9 of 157 units at view 1, and 6 of the 7 wrong units are ties broken to the smaller
  class ID (POC-2 `cred_conf`, argmax). The tie and view counts come from a check run after the recorded design run, on
  design units only, with the recorded S2 fit; it is not in the printout.
- Starts (grid index; binom / hb / hb_a1 / hb_bs; seeds 0, 1, 3: n = 471, 472, 471; seed 2: n = 464; seed 4: n = 468):
  - S1 and S4 (same commit counts per point): 0.5% -1 / -1 / 0 / -1 in every seed; 1% 6 / 6 / 8 / 6 (seed 2:
    5 / 6 / 7 / 5); 2% 24 / 14 / 16 / 24 (seed 2: 23 / 13 / 16 / 23; seed 4: 23 / 14 / 16 / 23); 3%
    29 / 18 / 20 / 29; 5% 29 / 22 / 25 / 29 (seed 4: 29 / 22 / 24 / 29).
  - S2-sum, S2-min: 0.5% -1 / -1 / 1 / -1; 1% to 5% 2 / 2 / 2 / 2, every seed.
  - S3: 0.5% -1 / -1 / 1 / -1; 1% to 5% 3 / 3 / 3 / 3, every seed.
- Rule 3 (n_min = 464; deciding point j, wrong / committed, UCB90, n_c, UCB90 x n_c, k_max):
  | Score | 0.5% | 1% | 2% | 3% | 5% |
  |---|---|---|---|---|---|
  | S1 | no (j 11, 0/116, 1.965%, 342, 6.72, none) | no (j 6, 0/135, 1.691%, 398, 6.73, 1) | no (j 6, 6.73 > 4) | **yes** (j 2, 0/150, 1.523%, 443, 6.75 <= 8) | **yes** (j 1, 1/154, 2.502%, 455, 11.39 <= 16) |
  | S2-sum | no (j 2, 2/153, 3.441%, 452, 15.55, none) | no (15.55 > 1) | no (15.55 > 4) | no (15.55 > 8) | **yes** (15.55 <= 16) |
  | S2-min | no (j 2, 1/150, 2.568%, 443, 11.38, none) | no (11.38 > 1) | no (11.38 > 4) | no (11.38 > 8) | **yes** (11.38 <= 15) |
  | S3 | no (j 3, 2/148, 3.556%, 437, 15.54, none) | no (15.54 > 1) | no (15.54 > 4) | no (15.54 > 8) | no (15.54 > 15) |
  | S4 | as S1 | as S1 | as S1 | **yes** (as S1) | **yes** (as S1) |
- **F-D1: PASS** (3% and 5% predicted). **Primary alpha: 3%** (S1 and S4). 10% not added; Fallback 2 not used.
- Correlation (j): within-unit rho 0.6312 [0.1256, 0.8868] (157 units, 465 views, 14 wrong views); design effect
  2.2382 [1.2465, 2.7396] at m = 2.96 and 2.2624 [1.2513, 2.7735] at m = 3. Within-pole rho 0.5146 [-0.6269, 1.0000]
  (133 poles, 157 units, 7 wrong units; 23 poles with more than one unit). No undefined bootstrap draws.
- Baselines: B1 view grid 30 points (f_d 100.00 to 30.11%, e_d at lambda_0 3.01%); B2 design draw 30 points (f_d
  100.00 to 30.57%, e_d at lambda_0 3.18%); B5 F_design 465 S1 margins; B6 theta on half B, both variants: 0.0988 at
  0.5% and 1%, -inf (commit at view 1) at 2%, 3% and 5%.
- Uncertified reference (loosest j with e_d <= alpha), 0.5 / 1 / 2 / 3 / 5%: S1 and S4 2 / 1 / 1 / 0 / 0; S2-sum
  -1 / -1 / 2 / 2 / 0; S2-min -1 / 2 / 2 / 2 / 0; S3 -1 / -1 / 2 / 0 / 0.
- No pool label, prediction, Cert or Test number was read or computed. Nothing is frozen yet (F3 open).

### Part B: results (filled from the printout after the run)
- Date: 2026-11-15 (design run 05:57:12 UTC) / Stage: R9 Stage 5 / Run tag: runs/btsc/stage5
- Git commit of Part A: 019f782 | Code: b63546a (design run HEAD; `src/r9_stage5_design.py` SHA-256
  65df87fedc15128a58adafcc819d434a5ae6d990f11fdeeb1fdaa85c2773b382, `src/r9_stage5_scores.py` SHA-256
  9de1e24808d950f5cc1ce5ea65b7bf1c46c9bad4feb260d385480a1e43c0b12f) | Design-pole values: ec85800 | Freeze code: 986a8e7
  | **F3 commit: e94e6670987f7e9abad8382d848012a6a928407a** | Tests after F3: 1355c7d.
- Run commands: `py -3.12 -m src.r9_stage5_design` (HEAD b63546a; 0.5 s), then `py -3.12 -m src.r9_stage5_freeze`
  (HEAD 986a8e7). Environment: Python 3.12.10, numpy 2.5.2. Both began with the freeze check (PASS, 3 files by git
  blob ID before F3).
- Input hash checks: all six inputs matched their recorded SHA-256 (the blob-ID fallback for text files was not
  needed). Files opened: `stage4_summary.json`, `test_views.csv`, `split_poles.json`, `embeddings/test_design.npz`,
  `head/head.npz`, `head/scores_test_design.npz`; no pool score or embedding file.
- Integrity checks (m): all PASS (details in the design-pole values above). The head margins were compared exactly in
  float32, the stored precision, instead of within 1e-9; float64 values differ by at most 6.07e-05.
- S2 halves: A 66 poles / 77 units / 226 views; B 67 poles / 80 units / 239 views. (a, b) = (-0.0, +0.1) for S2-sum and
  for S2-min (AURC 0.00005 each); ECDF sizes 80 / 80 / 79 for windows of 1 / 2 / 3 views.
- Grids (distinct points; f_d range; e_d at lambda_0): S1 30, 100.00 to 30.57%, 2.55%; S2-sum 3, 100.00 to 97.45%,
  4.46%; S2-min 3, 100.00 to 95.54%, 4.46%; S3 4, 100.00 to 94.27%, 2.55%; S4 30, 100.00 to 30.57%, 2.55%.
- Cert / Test units per seed (397 / 133 poles each): 471 / 155, 472 / 154, 464 / 162, 471 / 155, 468 / 158; n_min 464.
- Starts per (seed, score, alpha, rule): as listed in the design-pole values; full table in the yaml.
- Rule 3: as tabled in the design-pole values. Predicted certifiable at 3%: S1, S4; at 5%: S1, S2-sum, S2-min, S4; S3
  at no alpha (5%: 15.54 > 15).
- **Primary alpha: 3%** | **F-D1: PASS** (10% added: no) | Fallback 2: no.
- Within-unit rho 0.6312 [0.1256, 0.8868], design effect 2.2382 [1.2465, 2.7396] (m = 2.96), 2.2624 [1.2513, 2.7735]
  (m = 3); within-pole rho 0.5146 [-0.6269, 1.0000].
- Baseline design values: B1 view grid 30 points; B2 design draw 30 points; B5 F_design 465 S1 margins; B6 theta (both
  variants) 0.09875 at 0.5% and 1%, -inf (commit at view 1) at 2%, 3% and 5%.
- `configs/frozen_r9_btsc.yaml` SHA-256 93e030fdf85e6416691df1ddc0185286095be2c8254e571ed4d8897f3f645932 (working-copy
  bytes), git blob f4d37e5d73ef230e91689be5303e5eac51b15450; in `FROZEN_BLOBS` since e94e667. After F3,
  `check_frozen()` passes with 4 files and the guard reports F3 closed.
- **S2 resolution (author's decision (a), before F3):** S2 is frozen as fitted and not amended; its low resolution on
  BelgiumTSC is reported as a design-phase finding (pre-registration block in `LOG.md`).
- **Post-run tie check (after the recorded design run, not in the printout):** on design units only, with the
  recorded S2 fit, the largest combined p-value at view 1 is tied for 9 of 157 units, 7 units are wrong at view 1,
  and 6 of those 7 are ties broken to the smaller class ID; the same counts for S2-sum and S2-min. The second-largest
  ICP p-value is 0 for 418 of 465 views. The S2-sum check ran before the design-pole values commit, the S2-min check
  before the F3 commit.
- **Design file and yaml:** `stage5_design.json` stores every non-finite number as null, so a B6 threshold of -inf and
  None ("commit nothing") are not distinguishable in that file. The freeze step recomputed the design phase from the
  hash-checked Stage 4 inputs; the recomputation equalled the hash-checked design file in every field except the run
  fields (PASS), and the yaml was written from it, keeping -inf and null apart.
- **Tests edited at F3:** `tests/test_r9_btsc_guard.py` (from Stage 4) was edited at F3 (commit 1355c7d): the pre-F3
  tests now pin an explicit open state (`blobs={}`), and a new test checks that F3 is closed in the real repo. The
  guard code `src/r9_btsc_guard.py` is unchanged (SHA-256 d77d87365674560ff8013fc84a6ae51c5c1c93866c90cb3f72483c1ce8664d8d,
  as in Stage 4). In the same commit, one Stage 4 guard test in `tests/test_r9_stage4.py` pins the open state the same
  way, and `tests/test_r9_stage5.py` reaches the overwrite checks with an explicit pre-F3 state and gains two tests
  that the design and freeze steps refuse to run after F3. Full suite: 472 passed before F3; after F3, 9 tests that
  read the live `FROZEN_BLOBS` failed as expected; 474 passed after the edits.
- Checks: integrity (m) PASS | F-D1 PASS | freeze-step reproduction PASS | check_frozen after F3 PASS (4 files).
- Facts about the run: the design file records `worktree_clean False`: the runner creates `runs/btsc/stage5/` and its
  printout before it checks git status, so the new untracked outputs counted; no tracked file was changed (HEAD
  b63546a), as in Stage 4.
- Deviation from the plan? Yes: (1) S2 combines at most the last 3 views with one ECDF per window length (Q2), not a
  fixed K = 5; (2) Rule 3 uses n_min and checks a single grid point (Q3 to Q5); (3) the view order is ascending YYYYY,
  not camera index (Stage 1 Amendment 2); (4) Part A (o) named a `stage5_summary.json`; it was not written: the code
  SHA-256, input hashes and files opened are in `stage5_design.json`, and the yaml `sources` block holds the hashes of
  the design file and the Stage 4 inputs; (5) the integrity comparison of head margins was exact in float32 instead
  of within 1e-9 (Part A (m)); (6) the post-run tie check above is not in the printout.
- Next step: Stage 6 (BelgiumTSC confirmation), only when the author says so.
