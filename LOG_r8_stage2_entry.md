## POC-2 (R8) Stage 2: design phase and D-gate (design set only)

### Part A: fixed definitions (committed BEFORE any Stage 2 number was computed)
Written and committed before `src/r8_stage2.py` was run on real data. The git commit of this Part A is the proof of order.
Data: the 261 design tracks only (seed-0 Fit split, `track_split == 0` in `splits/seed0.npz`). delta = 0.1, beta = 5%,
n = 784 for every plug-in p-value. "Commit fraction" c = committed tracks / 261 (tracks with no passing frame count as
not committed). Grid = the Stage 1 lambda grid of the variant (200 points, `{V}_grid` in `track_table_design.npz`).

(a) **D1.** D1 passes if the upper 95% Clopper-Pearson bound of the design-set track error at the last frame
(V1, commit everything) is below the design-set single-frame error.
  - Track error at the last frame: share of design tracks with at least one passing frame whose V1 top class after
    their last frame (`V1_wrong_last`) differs from the label. Tracks with no passing frame are left out of this
    share (they are rejected as unknown, not misread) and their count is printed beside it.
  - Upper bound: the upper end of the two-sided 95% Clopper-Pearson interval (Beta quantile 0.975). This is the
    stricter of the two usual readings of "upper 95%".
  - Single-frame error: POC-1 head top-1 error on the gate-passing frames of those same design tracks (the evidence
    the commit rule uses). Chosen over "all design frames, no gate" because it is like-for-like and stricter: the
    gate's help is not credited to accumulation. The all-frames error is printed beside it for reference only.

(b) **Selective error** e = errors among committed tracks / committed tracks. If no track commits, e is undefined and
that lambda is not eligible in any rule that compares e.

(c) **Certificate A threshold order.** Start at the design-set argmin of mean L at the primary alpha, then relax one
grid point at a time.
  - mean L = (errors + alpha * (261 - committed)) / 261 = alpha * (1 - c) + c * e.
  - The argmin is taken over the frozen ~30-point grid of the chosen variant. Ties go to the larger lambda.
  - "Relax" = the next smaller lambda in that grid, down to the smallest grid point.

(d) **Certificate B size and lateness.** gamma = 10%. s = the smallest multiple of 5 pixels (5, 10, ... up to the
largest design-set ROI size, rounded up) at which the Certificate B plug-in (i) is predicted certifiable at some
working-grid lambda. If no s passes, Certificate B is recorded as "not predicted certifiable at gamma = 10%" and s
falls back to the plan's rule: the design-set median commit size at the start threshold (c), over committed tracks,
rounded up to the next 5 pixels. gamma is not changed in either case.
  - Why not the plan's median rule as primary: a track is late unless it commits at size <= s, so with s at the
    median commit size about half of the committed tracks plus all uncommitted tracks are late (near 50%), against
    gamma = 10%. That rule would decide R6 before any data is seen. Declared deviation from plan Stage 2 text
    ("For s, take the design-set median commit size"); allowed by plan section 2, which lets (s, gamma) be chosen
    in the design phase, and it follows the R6 mitigation (choose (s, gamma) with the same safety margin).
  - late_i(s) = 1 unless track i commits at a frame whose ROI size is at most s; that is,
    late_i = 1 - committed_i * [size at commit <= s]. ROI size = min(x2 - x1, y2 - y1), original pixels (Stage 1).
  - "Rounded up to the next 5 pixels" = smallest multiple of 5 that is >= the value (exact multiples unchanged).

(e) **D2 plug-in.** For each variant, each alpha in {0.5, 1, 2, 3, 5}% and each lambda of the Stage 1 grid:
mu = alpha * (1 - c) + c * e (mu = alpha when nothing commits), p = `src.certify.hb_pvalue(mu, 784, alpha)`. An alpha
is predicted certifiable for that variant if some lambda has p <= delta/10 = 0.01 and c >= 0.70. D2 passes if at least
one (variant, alpha) pair is predicted certifiable.

(f) **Primary alpha** = the smallest alpha in the grid that is predicted certifiable for at least one variant.

(g) **Variant** (plan section 5 rule, applied at the primary alpha): among the variants predicted certifiable at the
primary alpha, the one with the highest design-set commit fraction among lambdas whose selective error is at most
alpha/2. Ties go to V1, then V2, then V3.

(h) **Working lambda grid (~30 points)**, chosen variant only. Useful range = Stage 1 grid points with c >= 0.70 and
e <= primary alpha. Take 30 points evenly spaced by grid index from the lowest to the highest useful index (all of
them if there are 30 or fewer). The grid is a subset of the Stage 1 grid, so every row of it is the hash-checked
Stage 1 table.

(i) **Certificate B plug-in preview (risk R6).** For each working-grid lambda at the primary alpha and (s, gamma):
p_error as in (e), p_late = `hb_pvalue(mean late_i(s), 784, gamma)`, p = max(p_error, p_late). Predicted certifiable if
some lambda has p <= 0.01 (same safety margin as D2). Certificate B order = ascending design-set p, ties to the
larger lambda.

(j) **Pass levels.** P1 and the "4 of 5 re-splits" counts of P2 to P4 are as written in plan section 8.
P2: c_min = predicted coverage - 10 points, where predicted coverage = the design-set commit fraction at the threshold
where a fixed-sequence walk over the design-set plug-in p-values (n = 784, level delta = 0.1, order (c)) stops. This
is the threshold Certificate A would certify if Cert looked exactly like the design set.

(k) **D3** (informational). Frame wrong = POC-1 head top-1 (stored `top3[:, 0]`, no gate) differs from the label.
Count the design tracks with >= 25 wrong frames; for each, print wrong frames, passing frames, the last-frame margin
of V1, V2 and V3, and whether the last-frame top class is right.

(l) **Error-vs-frame curve** (descriptive). For each position t = 1..30 in frame order, among design tracks with at
least one passing frame by t: error of the top class at t for V1, V2 and V3, the single-frame head error at t, and
the share of tracks with evidence by t.

(m) **Integrity.** Only the 261 design tracks are used. `track_table_pool.npz` is never opened (the loader refuses
that path). Full-release arrays (embeddings, stored scores, gate flags) are cut to design rows right after loading,
before any computation. The design commit table recomputed in Stage 2 must equal the stored Stage 1 table exactly,
or the stage stops.

### Part B: results (fill from the printout after the run)
- Date: ____ / Stage: R8 Stage 2 / Run tag: runs/20261003_stage1/poc2/stage2
- Git commit of Part A: ____ | Git commit of the code: ____
- Input hashes checked against stage1_summary.json: ____
- D1: track error at last frame ____ (upper CP bound ____) vs single-frame error ____ -> PASS / FAIL
- D2: predicted-certifiable (variant, alpha) pairs: ____ -> PASS / FAIL
- D3: tracks wrong on >= 25 frames: ____ ; their last-frame margins: ____
- Primary alpha: ____ | Variant: ____ | Working grid: ____ points, lambda ____ to ____
- Start threshold: ____ (c = ____, e = ____) | s = ____ px, gamma = 10%
- Certificate B preview: min p = ____ -> predicted certifiable YES / NO (risk R6)
- P2 c_min: ____
- Checks: ____ — ALL PASS: ____
- Tests: first run 3 Stage 2 test failures, caused by the fake-table helper making every track commit at lambda = 0; fixed in tests/test_r8_stage2.py only, src/r8_stage2.py unchanged; second run 197 passed, 1 known sklearn failure.
- Decision: ____
- Deviation from the plan? ____
- Next step: the author reviews the D-gate; F1 (frozen_r8.yaml and the pre-registration block) only after that.

### Amendment 1 to Part A (d) (2026-11-13, before any Stage 2 run)
The fallback to the median rule is removed: if no s passes the Certificate B plug-in, s = not certifiable at n = 784,
with no substitute value. Reason: the median rule makes about half the tracks late by construction, so a fallback s
would be a value known to fail gamma. Made before `src/r8_stage2.py` was run on any data. No Stage 2 number existed,
and no Cert or Test track has been touched (plan section 2).
