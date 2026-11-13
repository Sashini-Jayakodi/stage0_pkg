## POC-2 (R8) Stage 1: accumulation and track table

- Date: 2026-11-13 / Stage: R8 Stage 1 / Run tag: runs/20261003_stage1/poc2/stage1
- Checkpoint hash: (stage1_summary.json -> inputs_sha256.checkpoint) 86527359bc61a5fbc4d48b7744e049b5ca3ff81cb016fb237298d0c007140bcc
- Git commit of the code: a3e0b48ec41f313b2b7d7690ae66e2f409050cde
- What was done: per-frame log-likelihoods, gate, running scores for V1/V2/V3, tracks-by-lambdas commit table (200-point grid per variant, from design-set margins only).
- Result numbers (from stage1_summary.json):
  - tau: 43.8435 | frames failing gate: 4.836% | tracks with no passing frame: 5
  - Frame head accuracy: 92.599% (POC-1: 0.9259863806779056)
  - ROI vs frame index, median per-track Spearman: 0.998 (positive in 99.8% of tracks)
  - Track lengths: {29: 1, 30: 1306}
- Checks: C1a PASS C1b PASS C1c PASS C2 PASS C3 PASS C4 PASS C5 PASS C6 PASS — ALL PASS: True
- Output file SHA-256 (files_sha256): design ad6fa0c59f65697f107b0cb6b3dbef25b4b6680f8618fe38c50bd770d1f2b391 / pool 0afb7e89052c393d695a8cb7e5796e4e811b1ff9c3c6bd392bf3cd10742250db / frame_evidence f79033e8cee5ba73a6b22d1d33b91227ff7cf628d95dc616a28824ad68a0e960
- Decision: the commit table is written as two files. The 261 design tracks get labels and 'wrong'; the 1,046 pool tracks are label-free. Why: plan section 2 forbids knowing commit-rule outcomes on Cert/Test before F1.
- Deviation from the plan? Yes. Section 10 Stage 1 asks for 'committed, wrong' on all tracks; 'wrong' is computed on design tracks only, and on pool tracks only after F1 (Stage 4). This is stricter than the plan and changes no result.
- Deviation from the plan? Yes (detail). ROI size = min(x2 - x1, y2 - y1) in original pixels. The plan says "shorter side" without a convention. This convention is fixed here and frozen with s at F1.
- Environment note: `py -3.12 -m pytest tests -q` gave 169 passed, 1 failed before this run. The failure, `tests/test_stage3.py::test_ledoit_wolf_matches_sklearn`, is `ImportError: DLL load failed while importing _middle_term_computer: An Application Control policy has blocked this file` (scikit-learn 1.9.1), reproduced standalone with `py -3.12 -c "import sklearn.covariance"`. LOG.md Stage 4 entry (line 260) records 41 passed, 0 failed, 0 skipped under the same Python 3.12.10 during POC-1, so this test passed cleanly then. Cause not established: scikit-learn was not version-recorded in POC-1, so either a package upgrade or an Application Control policy change could explain it. The author approved running Stage 1 despite this failure because Stage 1 does not refit the head or use sklearn: it hash-checks `head.npz` against the frozen checkpoint and cross-checks C1a/C1b against stored POC-1 outputs only. The test file itself was not edited.
- Environment versions (`env_poc2_stage1.txt`, pip freeze, Python 3.12.10): numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.1, torch 2.14.1+cu132. Compared with POC-1: numpy 2.5.2 is the same as in the Python 3.12 entries of LOG.md (lines 116, 134, 163). scipy: LOG.md only records 1.17.1 (Stage 0 lock, Python 3.11.9, line 46) and 1.18.1 (first Python 3.13 run, line 62); the POC-1 Stage 7 and 8 result JSONs (Python 3.12.10) record scipy 1.18.1, so no change. scikit-learn: no version is recorded anywhere in POC-1 (LOG.md or run JSONs), so it cannot be shown whether the sklearn DLL block comes from a package upgrade or from an Application Control policy change.
- Next step: Stage 2 (design phase and D-gate), design set only.
