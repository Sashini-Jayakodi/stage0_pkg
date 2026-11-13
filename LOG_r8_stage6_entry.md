## POC-2 (R8) Stage 6: published baselines B5 (Ringel et al.) and B6 (Boursinos and Koutsoukos), comparison table

### Part A: fixed definitions (committed BEFORE any B5/B6 number was computed)
Confirm phase (F1 commit 3de4cb9). Nothing in `configs/frozen_r8.yaml` changes. delta = 0.1 throughout. Every decision
below was made by the author before any B5 or B6 number existed. Every value fixed from data is fixed from the 261
design tracks only (values block below, computed after this rules block was committed).

#### Papers (local copies in `docs/papers/`, git-ignored, not redistributed)
- **[R]** L. Ringel, R. Cohen, D. Freedman, M. Elad, Y. Romano, "Early Time Classification with Accumulated Accuracy Gap
  Control," Proc. 41st Int. Conf. on Machine Learning (ICML), PMLR vol. 235, pp. 42584-42600, 2024. Published PMLR
  version, from https://proceedings.mlr.press/v235/ringel24a.html via its PDF link
  https://raw.githubusercontent.com/mlresearch/v235/main/assets/ringel24a/ringel24a.pdf (the guessed
  .../v235/ringel24a/ringel24a.pdf returned 404). Local `docs/papers/ringel24a.pdf`, SHA-256
  cc1ffe7245b21ab282569a522b3b308493c1b1e982cc20bc06098359614b1039. Page numbers below are the PDF's printed pages 1-17.
- **[BK]** D. Boursinos, X. Koutsoukos, "Selective Classification of Sequential Data Using Inductive Conformal
  Prediction," 2022 IEEE Int. Conf. on Assured Autonomy (ICAA), pp. 46-55, doi:10.1109/ICAA52185.2022.00015 (year
  confirmed 2022 on the PDF). From
  https://cs.vuse.vanderbilt.edu/koutsoxd/www/Publications/Selective_Classification_of_Sequential_Data_Using_Inductive_Conformal_Prediction.pdf
  (IEEE Xplore print). Local `docs/papers/boursinos_koutsoukos_selective_classification.pdf`, SHA-256
  af0f13ce31e6c8fadad7e13f04d8445bf811c183569c10b7be2f022a1ba5b557. Page numbers below are printed pages 46-55.

#### Guiding rule for every point the papers leave open (author's decision)
Choose the option most faithful to the paper and not weaker than the paper's own setup, fixed from the 261 design tracks
only, and write the reason. Where we cannot be faithful, it is a declared difference.

(a) **Freeze check and inputs.** Every Stage 6 script calls `load_frozen()` first. All inputs are hash-checked as in
Stages 4 and 5 (pool loader, train_meta.csv cross-check, stage4_summary.json e66e45f2...d2cc, r8_stage4.py
fae67390...45c2, baselines_design.json, b1_design.json, stage3a/3b summaries); the Stage 4 splits are regenerated with
the hash-checked Stage 4 code and must reproduce every Stage 4 Test coverage exactly. The two PDFs are hash-checked
against the SHA-256 values above. Cert = calibration, Test = evaluation, in every re-split (seeds 0 to 4).

(b) **Common to B5 and B5c (Ringel et al.).**
  - Classifier output per frame t: the V1 accumulated score of Stage 1 ("their rule on the same accumulated score",
    plan section 9). Prediction Y_t = V1 top class at t. Before a track's first passing frame there is no evidence:
    no prediction, and no stop is possible there (as in Stage 1).
  - Y_full = the V1 top class after the track's last frame (its own t_max: 30 frames, or 29 for the one short track).
    The paper's rule always stops at t_max at the latest ([R] eq. 6, p. 3), so B5/B5c never escalate.
  - Gate: gate-failing frames add no evidence (same accumulated score as the commit rule). A track with no passing
    frame has no prediction at any frame: it is counted as not committed (rejected as unknown) and its gap loss is 0
    (the full prediction cannot be correct).
  - Loss: L_gap = (1[Y = Y_full] - 1[Y = Y_early])_+ per track ([R] eq. 3, p. 2).
  - **Score in [0, 1].** [R] uses a confidence in [0, 1], e.g. the largest class probability ([R] p. 3, §5.2 p. 8).
    Our accumulated V1 margins are in log-likelihood units and reach thousands, so the largest posterior saturates at
    1.0 (share measured on the design set below). Faithful fix: pi(X_t) = F_design(M_t), the empirical CDF of the V1
    margins M_t over all design-set frames with evidence (F(x) = share of design margins <= x); pi = -1 before the first
    passing frame (cannot stop). This is a monotone transform, so the stopping rule is the paper's rule on the same
    score, the [0, 1] scale and the grid resolution are the paper's, and the guarantee holds for any score.
  - Grid: Lambda = {0, 0.01, ..., 1} plus infinity (never stop early), Delta = 0.01 as in [R] §5, p. 7.
  - p-value: CDF_bin(n * R_hat; n, alpha), the exact binomial tail for a 0/1 loss ([R] §3, p. 4).
  - alpha = the gap tolerance, run at every alpha of the frozen grid {0.5, 1, 2, 3, 5}%; the primary alpha 1% is the
    pre-registered setting ([R] does not require a value; its experiments use 10% and 5%, Table 1 p. 8, Table C.3
    p. 13). delta = 0.1 (project-wide). **Declared difference:** [R] uses delta = 1% (§5, p. 7).

(c) **B5 = Ringel marginal method** (pre-registered as named in plan section 9). [R] §3 p. 4 and Algorithm A.3 p. 12:
one threshold lambda for all frames; stop at the first t with pi(X_t) >= lambda, or at t_max. Fixed-sequence test over
Lambda from the largest value down at level delta, stop at the first p > delta; lambda_hat = the smallest value that
passed, or infinity if the first fails. Calibration set = all 784 Cert tracks ([R] §5.1: D_cal = D_cal-1 union D_cal-2).

(d) **B5c = Ringel conditional method** (declared additional baseline: [R] presents the marginal method as a warm-up,
§3 p. 4, and calls the conditional one its main contribution, p. 2). [R] §4, Algorithms 1 and 2 (pp. 5-7), with one
threshold per frame index t = 1..30 (lambda_t; a track stops at its own t_max at the latest).
  - Cert split into D_cal-1 and D_cal-2 of 392 tracks each (equal halves, as in [R] §5.1 p. 7): the first 392 tracks of
    a permutation of the Cert tracks drawn with `np.random.default_rng([20261116, seed])`.
  - Stage 1 (D_cal-1): for t = 1..30, lambda_t = the smallest grid value with empirical accumulated gap R^t_gap <= alpha
    among tracks halted by t; infinity if none or if no track halts by t (Algorithm 1).
  - Stage 2 (D_cal-2): test the configurations from t = 30 down to 1, each through its finer hypotheses t' >= t, with
    the max-p rule and the binomial p-value; stop at the first failure (Algorithm 2).

(e) **B6 = Boursinos and Koutsoukos selective classifier** ([BK] §IV pp. 47-48, §VI pp. 49-52).
  - **Per-frame p-values (ICP).** Nonconformity of class j for a frame = d_j / min_{k != j} d_k, the nearest-centroid
    ratio of [BK] eq. 9 (p. 50), with d_k = the head's Mahalanobis distance to class mean k (sqrt of `mahal_sq`).
    p_j = share of calibration nonconformity scores >= the frame's score for class j ([BK] eq. 10, p. 50; no +1).
    **Declared difference:** [BK] computes the distances in a Siamese-network embedding ([BK] §V, p. 48); we use the
    frozen ResNet-18 embedding with the head's whitening (Mahalanobis = Euclidean after whitening), the same formula.
  - **Data roles, design set only** (all disjoint by track; [BK] §VII-A p. 52 also keeps calibration and ECDF
    sequences disjoint): the 261 design tracks are permuted with `np.random.default_rng(20261117)` and split into three
    parts of 87 tracks: ICP calibration (nonconformity scores of the true class), ECDF calibration (the combined
    null p-values for eq. 22), and selection (a, b and theta). Only gate-passing frames are used in all three.
    The head was fitted on the fine-tune frames, so those are not used as ICP calibration data (they are its proper
    training set in [BK]'s sense).
  - **Window.** The last K passing frames up to t; no decision before K passing frames exist (t < K: the window is not
    full; [BK] defines combinations only for full windows, §VI-B p. 50). K = 5 (pre-registered, plan section 9);
    K = 9 (declared extra: the window of [BK]'s risk-coverage figure, Fig. 6 p. 54). [BK] evaluates K = 2 to 9 and
    recommends none (Tables III-IV, p. 54), so K = 5 is the plan's choice, not the paper's.
  - **Combination (selective classifier).** ECDF Sum and ECDF Min ([BK] eq. 22 p. 52; the only methods [BK] uses for
    its risk-coverage results, §VII-D p. 53, Tables III-IV): p_comb(j) = F(f(p_j over the window)), f = sum or min,
    F = the empirical CDF of f over all full windows of the true class in the ECDF-calibration tracks.
    **Interpretation of plan section 9:** "arithmetic mean" there is read as [BK]'s arithmetic-mean merging function
    (2 x mean, eq. 15, p. 51), which [BK] evaluates only for calibration (Table III, p. 54). It is reported only in the
    calibration check (g), not used as a selective classifier.
  - **Assurance evaluator.** credibility = the largest combined p-value, confidence = 1 - the second largest ([BK]
    eqs. 1-2, p. 47); k = a * credibility + b * confidence ([BK] eq. 7, p. 48). a, b by grid search over [-1, 1] in
    steps of 0.1 (441 pairs; [BK] gives no resolution) minimising the AURC ([BK] eq. 6, p. 48) over the selection-split
    frames with full windows; ties go to the first pair in (a, b) ascending order.
  - **Threshold from the risk-coverage curve** ([BK] p. 48: theta such that (r_hat, c_hat) = (r, c)). Per-track
    adaptation (declared: [BK] decides per frame only): a track commits at its first full-window frame with k > theta,
    with label = the class with the largest combined p-value there; otherwise it escalates. theta = the smallest value
    (maximum coverage) whose track-level selective error on the selection split is <= r, with r = alpha for each alpha
    in the frozen grid; if no theta meets r, B6 commits nothing at that alpha. No certificate (as in [BK]).

(f) **Test metrics** (per re-split, per alpha, 262 Test tracks), for B5, B5c and the four B6 variants (ECDF Sum and
ECDF Min, K = 5 and 9): coverage, error among committed signs, median commit (stop) frame and median ROI size at commit,
as in Stage 4 Part A (d). For B5/B5c also the Test accuracy gap: marginal, and accumulated over the 20% and 50%
earliest-stopping Test tracks ([R] Table 1, p. 8).

(g) **Stage 6 checks** (plan section 10), at the primary alpha 1%, PASS if true in >= 4 of 5 re-splits:
  - B5 meets its own target: Test marginal accuracy gap <= alpha.
  - B5c meets its own target: Test accumulated gap over the 20% earliest, over the 50% earliest, and marginal, all <= alpha.
  - B6 combination improves calibration in the direction [BK] reports ("ECDF based methods consistently improve the
    calibration over the baseline", p. 53): ECE of ECDF Sum and of ECDF Min (K = 5) < ECE of single-frame ICP, on the
    Test frames with full windows. ECE as [BK] eq. 24 (p. 53): prediction set = classes with p > eps,
    eps_i = i/1000, i = 1..200, ECE = mean_i |acc(eps_i) + eps_i - 1|. The arithmetic-mean merge (2 x mean, eq. 15) and
    K = 9 are reported, not scored.
  - Also reported at 2%. A failed check is reported as failed.

(h) **Comparison table (plan section 9).** Columns: Ours (commit rule), B1, B2, B3, B4, B5, B5c, B6 (ECDF Sum, K = 5),
B6 (ECDF Min, K = 5); the K = 9 variants in a second table. Two alphas: **1% (primary, pre-registered)** and **2%**
(declared extra: the smallest alpha certified for the commit rule in all five re-splits, Stage 4). Numbers are the
**median (min to max) over the five re-splits**; a method certified in fewer than five re-splits shows "k/5 certified",
with coverage 0 for the others and error / frame / size medians over the certified re-splits only.
Rows, and where each number comes from:
  1. Type of promise: certified real error among committed signs (Ours, B2, B3, B4); certified per-frame error with
     frames assumed independent (B1); accuracy gap to the full-sequence answer, marginal (B5) or conditional on the
     accumulated halt time (B5c); none (B6). From the methods' definitions.
  2. Valid on linked frames: violation rate vs delta at the alpha of the column, from Stage 3 (3b real data; 3a at
     rho_sim 0.45) for Ours and B1. B2, B3, B4: "valid by construction (track-level certificate), not measured".
     B5, B5c: "valid for its own target under i.i.d. tracks, not measured". B6: "no promise".
  3. Sign coverage at the alpha (Test): Stage 4 (Ours, B2, B3, B4), Stage 6 (B5, B5c, B6). B1: "n/a - B1 certifies
     frames, not signs, and was not run on the Stage 4 splits" (author's decision: B1 numbers come from Stage 3 only).
  4. Error among committed signs (Test): as row 3.
  5. Smallest certifiable alpha at n = 784: Stage 4 certifiability curve (Ours, B2, B3, B4); B1: the smallest alpha
     at which B1 certified in at least half of the Stage 3b draws; B5, B5c, B6: "n/a (no certified real error)".
  6. Median commit frame (Test): Stage 4 (Ours, B2, B3, B4), Stage 6 (B5, B5c, B6); B1 n/a.
  7. Median sign size at commit (px, Test): Stages 4 and 5 sources as row 6; B1 n/a.
  8. Can escalate: yes (Ours, B1, B2, B3, B4, B6); no (B5, B5c: always stop by t_max).

(i) **Integrity.** Stage 6 changes no frozen value and no earlier result; Stage 4 numbers are read, not recomputed
(only reproduced as a check). B5, B5c and B6 are specified after F1 from the papers and design data only.

#### Design-set values (from the Stage 6 design script, design set only)
____

### Part B: results (fill from the printout after the run)
- Date: ____ / Stage: R8 Stage 6 / Run tag: runs/20261003_stage1/poc2/stage6
- Git commit of Part A: ____ | Git commit of the code: ____
- B5 / B5c at 1% and 2%: ____
- B6 (4 variants) at 1% and 2%: ____
- Stage 6 checks: ____
- Comparison table: ____
- Checks: ____ — ALL PASS: ____
- Decision: ____
- Deviation from the plan? ____
- Next step: Stage 7 only after the author reviews Stage 6.
