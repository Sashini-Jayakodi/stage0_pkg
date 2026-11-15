## POC-3 (R9) Stage 8: final tables (plan R9 section 9 Stage 8)

No new analysis and no new numbers. Every value in the tables is read by code from a committed summary whose SHA-256
(or git blob ID) is checked against its record first; values are only selected, formatted, or summarised as median
(min-max; count) over the five re-splits with the Stage 6 runner's helpers.

- Date: 2026-11-15 / Stage: R9 Stage 8 / Output: `docs/poc3-results-tables.md`
- Code commit: fd60e28. `src/r9_stage8_tables.py` SHA-256
  3740264e458f536234bc04e5caa32a5274eebf4b5fc563345046a52c0f127242 (blob 7631873bb1983a20858e4cf7185487df0ac71ee1);
  `tests/test_r9_stage8_tables.py` SHA-256 c058b2070d3169dbfd01569c5f15c406757a781fbf8dcc978783176320f1415e (blob
  87ddf20d75fb7292a59a222ef58784e04921fb97); `docs/poc3-results-tables.md` SHA-256
  ae889e2e8765b1efabb917aea6e3db780c0897105c53cc411b0229feeca1ed07 (blob 3c6fed2750650a38453c089a82d9e2b39834033a).
- Command: `py -3.12 -m src.r9_stage8_tables`. It calls `check_frozen()` first (4 files: plan R9, engine, engine tests,
  `configs/frozen_r9_btsc.yaml`; PASS), then hash-checks every source below.

### What was built
`src/r9_stage8_tables.py` writes `docs/poc3-results-tables.md` with:
1. GTSRB comparison table at alpha 1% (exploratory, post-hoc), plan section 8 rows plus a "Source / test" row.
2. BelgiumTSC comparison table at the primary alpha 3% with the Stage 6 Part B statement on why it does not separate
   the methods (copied from `LOG_r9_stage6_entry.md`), then the 1% table labelled "pre-registered, reported, not the
   primary alpha". Both are built by `src.r9_stage6.comparison_table`.
3. RQ1 validity: GTSRB real data (POC-2 Stage 3b, HB) with the POC-1 rho and design effect; simulation worst cases per
   cluster setting (R9 Stage 2); BelgiumTSC (R9 Stage 6, exact binomial / HB) with the design-phase design effect.
4. Exact binomial against Hoeffding-Bentkus under all three HB starts, GTSRB and BelgiumTSC: certified re-splits and
   Test coverage per score and alpha, and the smallest certifiable alpha per re-split.
5. S2 against B6 (cost of the certificate), GTSRB and BelgiumTSC, every alpha. BelgiumTSC rows are the recorded
   Stage 6 cost table; GTSRB rows are built by the same function, `src.r9_stage6.cost_table`.
6. Ringel B5 and B5c: gap against real error per re-split, POC-2 GTSRB and R9 Stage 6 BelgiumTSC.
7. Limitations: plan R9 section 12 copied, plus four POC-3 additions recorded in the Part B entries (no
   exact-binomial start at 0.5%, S2 resolution on BelgiumTSC, flat primary-alpha table, DITS unavailable), with their
   numbers read from the summaries.
Each table names its source file and the line of the cited JSON key, and the LOG line where the values were recorded.

### Sources (checked before reading)
| File | Check |
|---|---|
| `runs/20261003_stage1/poc3/stage2/stage2_summary.json` | SHA-256 8c152579...729f (R9 Stage 2 Part B) |
| `runs/20261003_stage1/poc3/stage3/stage3_summary.json` | SHA-256 aad2c03f...52e2 (R9 Stage 3 Part B) |
| `runs/btsc/stage6/stage6_summary.json` | SHA-256 71be7957...185f (R9 Stage 6 Part B) |
| `runs/btsc/stage1/stage1_summary.json` | SHA-256 77ba6a70...a0dd (R9 Stage 1 Part B) |
| `runs/20261003_stage1/poc2/stage3/stage3b_summary.json` | SHA-256 7591a438...7367 (POC-2 Stage 3 Part B) |
| `runs/20261003_stage1/poc2/stage4/stage4_summary.json` | SHA-256 e66e45f2...d2cc (POC-2 Stage 4 Part B) |
| `runs/20261003_stage1/poc2/stage6/stage6_summary.json` | SHA-256 59ef05a4...c25c (POC-2 Stage 6 Part B) |
| `runs/20261003_stage1/splits/stage5_results.json` | git blob 4060ae3f...52aa as committed in a3e0b48; LOG.md leaves its SHA-256 blank |
| `LOG_r9_stage6_entry.md` (statement copied into table 2) | git blob 34242a04...655c as committed in 4e41ee5 |
Full hashes are in the source table at the top of `docs/poc3-results-tables.md` and in the script.

### The mixed-record note (table 1, GTSRB)
R9 Stage 3 ran only the layer on S1 to S4. The GTSRB baseline columns are therefore the POC-2 records on the same five
re-splits, not R9 runs: B1 from the POC-2 Stage 3b resampling with Hoeffding-Bentkus; B2 from POC-2 Stage 4 with
Hoeffding-Bentkus and the POC-2 grid; B5 and B5c from POC-2 Stage 6; B6 from POC-2 Stage 6, calibrated on Cert thirds
(POC-2 Stage 6 Amendment 1) and not on the design set as S2 is. The columns are therefore not like-for-like in test
and calibration. Each baseline header says "POC-2 record" (with HB or Cert-thirds calibration where it applies), every
column has a "Source / test" row, and the note under the table states the differences, including that the 1% result
for S1 to S4 is confounded with the changed test, grid and start against POC-2 (R9 Stage 3 Part B).

### Choices made in the script (none changes a recorded number)
- GTSRB comparison alpha 1%: the smallest alpha at which every score certified in all 5 re-splits (R9 Stage 3).
- B6 on GTSRB: the K = 5 variants (ecdf_sum_K5, ecdf_min_K5), matching S2-sum and S2-min; the K = 9 extras are not
  shown.
- B5 and B5c on GTSRB: the "certified" row shows the recorded POC-2 own-gap-target check (at 1%), since neither
  certifies a real error.
- Simulation (table 3b): the setting with the largest recorded violation rate per cluster setting, for the sign level
  and B1 under the exact binomial and the primary HB start; ties are counted. No design effect is recorded for the
  simulation, so none is shown.
- The optional adverse-cells table (plan Stage 8) is not built.

### Checks
- `check_frozen()` PASS; all nine sources pass their hash check.
- `tests/test_r9_stage8_tables.py`: 22 passed after the final change. The tests recompute every table's cells from
  the raw JSON with independent code; check that the BelgiumTSC 3% table equals the comparison table stored in
  `stage6_summary.json` and that the cost function reproduces the stored Stage 6 cost table; check that the document
  on disk equals a fresh build; and refuse a changed JSON source, a changed copied LOG entry and a missing source.
- Full suite: 530 passed before the table 1 header change (the change touched table 1 rendering only; the 22 Stage 8
  tests were re-run after it).

### Deviations
- No Part A was committed before the code. Stage 8 computes no new number; the choices above are recorded here
  instead.
- Next step: Stage 9 (idea text and gates), only when the author says so.
