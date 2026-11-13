## POC-2 (R8) Stage 1: accumulation and track table

- Date: ____ / Stage: R8 Stage 1 / Run tag: runs/20261003_stage1/poc2/stage1
- Checkpoint hash: (stage1_summary.json -> inputs_sha256.checkpoint) ____
- Git commit of the code: ____
- What was done: per-frame log-likelihoods, gate, running scores for V1/V2/V3, tracks-by-lambdas commit table (200-point grid per variant, from design-set margins only).
- Result numbers (from stage1_summary.json):
  - tau: ____ | frames failing gate: ____% | tracks with no passing frame: ____
  - Frame head accuracy: ____% (POC-1: ____)
  - ROI vs frame index, median per-track Spearman: ____ (positive in ____% of tracks)
  - Track lengths: ____
- Checks: C1a ___ C1b ___ C1c ___ C2 ___ C3 ___ C4 ___ C5 ___ C6 ___
- Output file SHA-256 (files_sha256): design ____ / pool ____ / frame_evidence ____
- Decision: the commit table is written as two files. The 261 design tracks get labels and 'wrong'; the 1,046 pool tracks are label-free. Why: plan section 2 forbids knowing commit-rule outcomes on Cert/Test before F1.
- Deviation from the plan? Yes. Section 10 Stage 1 asks for 'committed, wrong' on all tracks; 'wrong' is computed on design tracks only, and on pool tracks only after F1 (Stage 4). This is stricter than the plan and changes no result.
- Deviation from the plan? Yes (detail). ROI size = min(x2 - x1, y2 - y1) in original pixels. The plan says "shorter side" without a convention. This convention is fixed here and frozen with s at F1.
- Next step: Stage 2 (design phase and D-gate), design set only.
