## POC-2 (R8) Stage 5: Certificate B status and the descriptive commit-by-size curve

### Part A: fixed definitions (committed BEFORE any Stage 5 number was computed)
Confirm phase (F1 commit 3de4cb9). Nothing in `configs/frozen_r8.yaml` changes. No new certification is run.

(a) **Certificate B, as frozen.** `frozen_r8.yaml`: s = "not certifiable at n = 784", gamma = 10%. Stage 5 reports that
status as frozen. s and gamma are not moved, and no joint certificate is run at any substitute s.

(b) **Descriptive curve (not scored, not a certificate).** For each re-split (the Stage 4 splits, seeds 0 to 4) and each
alpha at which the commit rule was certified in Stage 4 (2%, 3% and 5% in every re-split; not 0.5% or 1%), at the
Stage 4 certified threshold, over the 262 Test tracks:
  - committed by size s = the share of Test tracks with committed_i = 1 AND ROI size at commit <= s. This is
    1 - late_i(s) averaged over Test, with late_i(s) exactly as frozen (`frozen_r8.yaml` certificate_B.late_definition);
  - never committed = the share of Test tracks not committed (tracks with no passing frame included);
  - s = 5, 10, 15, ... px, one common range for every method, re-split and alpha: from 5 px up to the largest ROI
    size at commit among all curves, rounded up to the next multiple of 5 (beyond it every curve is flat);
  - ROI size = min(x2 - x1, y2 - y1) in original pixels (Stage 1 convention), at the commit frame.
The same curve for B2, B3 and B4 at their Stage 4 certified thresholds, at the same alphas. Where a baseline was not
certified at that alpha in that re-split (B2 at 2% in seeds 2 and 3), it is reported as "not certified", no curve.

(c) **Inputs and integrity.** `load_frozen()` first. `stage4_summary.json` is checked against the SHA-256 recorded in
Stage 4 Part B (e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc). The certified threshold of every
(method, re-split, alpha) is read from it; nothing is re-certified.
  - The Stage 4 splits were not saved as files (Stage 4 regenerates them from the seeds). Stage 5 regenerates them with
    the same committed code: `src/r8_stage4.py` must match the SHA-256 recorded in stage4_summary.json
    (fae67390a488b44c57f1386c9b256cb0fbbcf2444282f5724c2580aa3e6445c2), and the stage stops otherwise.
  - Reproduction check: at every certified (method, re-split, alpha), the Test coverage recomputed from the
    regenerated split must equal the Stage 4 value exactly (it is also the right-hand end of the curve). Any mismatch
    stops the stage.
  - All other inputs are hash-checked as in Stage 4 (the Stage 3b pool loader and the train_meta.csv cross-check).

(d) **Summary numbers per curve (descriptive).**
  - The committed share at s = 25, 50 and 75 px.
  - **D90 size: the smallest s at which at least 90% of Test signs are committed (Test description, not a
    certificate).** s runs over the multiples of 5 px of (b). If a curve never reaches 90%, D90 is reported as
    "not reached".
  - Reported for the commit rule at every alpha where it was certified, and for B2, B3 and B4 at each alpha where both
    the commit rule and that baseline were certified in that re-split.
  - D90 is never described with any word suggesting a certified size; it is a Test description only.

(e) **Outputs.** `runs/20261003_stage1/poc2/stage5/stage5_summary.json` with the hashes of what it read. Stage 5 changes
no frozen value.

### Part B: results (fill from the printout after the run)
- Date: ____ / Stage: R8 Stage 5 / Run tag: runs/20261003_stage1/poc2/stage5
- Git commit of Part A: ____ | Git commit of the code: ____
- Certificate B: not certifiable at n = 784 (frozen); gamma 10%
- Curves (commit rule vs B2/B3/B4) at 2%, 3%, 5%: ____
- Checks: ____ — ALL PASS: ____
- Decision: ____
- Deviation from the plan? ____
- Next step: Stage 6 only after the author reviews Stage 5.
