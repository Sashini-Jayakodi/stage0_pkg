# POC-2 (R8) Stage 1: evidence accumulation and the commit table

NEW FILES ONLY. Unzip into your existing `stage0_pkg` folder. Nothing here replaces any POC-1 file,
`LOG.md`, `configs/poc.yaml` or the master README.

    src/r8_stage1.py           the stage (reuses src.head.mahal_sq and src.train.sha256_file)
    tests/test_r8_stage1.py    12 synthetic-data tests (no GTSRB needed)
    LOG_r8_stage1_entry.md     paste into LOG.md after the run

## One-time git setup (before running anything)
    git tag poc1-final
    git checkout -b poc2
    git add src/r8_stage1.py tests/test_r8_stage1.py README_r8_stage1.md
    git commit -m "R8 Stage 1 code"

## Run
1. `py -3.12 -m pytest tests -q`  -> all old tests plus 12 new ones pass
2. `py -3.12 -m src.r8_stage1 --run runs/20261003_stage1 --config configs/poc.yaml`  (seconds, CPU only)
3. Output in `runs/20261003_stage1/poc2/stage1/`:
   `track_table_design.npz`, `track_table_pool.npz`, `frame_evidence.npz`, `stage1_summary.json`
4. It must end with `ALL PASS: True`. Send me the whole printout.

## What to expect
- Frame head accuracy equals your POC-1 number exactly (about 92.6%).
- tau = 43.8435 (your POC-1 seed-0 value; seed-0 Fit is the R8 design set).
- Track lengths `{29: 1, 30: 1306}`.

## What it refuses to do
- Run if the checkpoint, `train_release.npz`, `head.npz` or `seed0.npz` changed since POC-1.
- Write labels or errors for the 1,046 pool tracks (plan section 2). Those are first computed in Stage 4, after the freeze.
