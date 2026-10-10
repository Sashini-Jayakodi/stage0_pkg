# Stage 4: reshuffles and Fit-side calibration

This package REPLACES the earlier Stage 4 files `src/splits.py` and `tests/test_stage4.py` (updated for the decision below). It does not touch `configs/poc.yaml`. `LOG.md` is delivered separately as a merged final version.

| File | Goes to |
|---|---|
| `src/splits.py` (updated) | `src/` |
| `tests/test_stage4.py` (updated, 16 tests) | `tests/` |
| `configs/poc_stage4_snippet.yaml` | optional: paste its `splits:` block at the end of `configs/poc.yaml` (same numbers as the code defaults) |
| `LOG_stage4_entry.md` | same text as the Stage 4 entry already inside the merged `LOG.md` |

## What changed since the first Stage 4 delivery
Decision 2026-10-10: the PER-TRACK MEAN of in-stratum error rates is the primary base error (it is what the Stage 6 certificate averages). The code now computes the exploratory alpha "rel_0.5base" from it, and keeps the pooled version as "rel_0.5base_pooled". Splits, tau, c*, easy-mass fractions, base errors, grids and the absolute-alpha orders are unchanged. **Re-run once** so the stored orders and file hashes match the decision (seconds).

## Run
```powershell
cd "D:\FYP Review\POC Implementation\stage0_package VI\stage0_pkg"
py -3.12 -m pytest tests -q                       # expect 48 passed
py -3.12 -m src.splits --run runs/20261003_stage1
```
It overwrites `runs/20261003_stage1/splits/seed0..4.npz` and `stage4_summary.json`. Then:
1. Compare the printout with the tables in the Stage 4 LOG entry (c*, easy mass, rejection rates and base errors must be identical; the first run printed the per-track mean second, now it is first).
2. Paste the five `files_sha256` values from `stage4_summary.json` into the Stage 4 entry.

## What it does (per seed 0-4)
1. Splits the 1,307 **tracks** into Fit / Cert / Test = 261 / 784 / 262, about 20/60/20 within every class, at least 1 track per class per split.
2. Gate: one random frame per Fit track, sort the 261 OOD scores, tau = the 249th smallest (beta = 5%).
3. Cut-point c*: on gated-in Fit frames, the distinct log eps value whose "easy" side (log eps <= c*) holds mass (weight 1/n_t per frame) closest to 50%.
4. Fit base error per stratum (per-track mean = primary, pooled = sensitivity), a 50-point margin grid per stratum, and the Fit-based testing order (start at the argmin of estimated E[L], then looser).
5. Repeats 3-4 for eps3 = eps(y1,y2) + eps(y1,y3), the pre-registered first fallback.

## What it deliberately does NOT do
It computes **no error rate on Cert or Test**. Cert gets only frame/track counts, easy-mass fractions and gate-rejection rates. This keeps the pre-registration commit ahead of any Cert result.

## Decisions (all recorded in `stage4_summary.json` and the LOG)
1. **Base error:** per-track mean is primary (decided 2026-10-10, after seeing Fit numbers; disclosed in the pre-registration block). Pooled 1/n_t rate is a sensitivity. On the real data they differ by 1-4 pp, which is why this had to be decided.
2. **Margin accept rule:** accept when margin >= threshold. Grid index 0 = loosest.
3. **Test order:** start = argmin of Fit E[L] (loosest on ties), then step looser to index 0. Alphas stored: 0.01, 0.02, 0.03, 0.05, 0.5 x per-track base, 0.5 x pooled base. None is chosen here; for G1(iii) only 1%, 2%, 3% count.
4. **eps3** is added in the log domain (`logaddexp`).

## Before Stage 5
Read, accept or edit, and **commit** the pre-registration block at the top of `LOG.md`, then paste the commit hash into it and into the Stage 4 entry. Its seven items: (1) G1 aggregation rules, (2) bootstrap B and percentile, (3) eps3 as first fallback, (4) G2 target bands, (5) the 6.3 ordering rule, (6) the base-error definition, (7) disclosure of the Stage 4 Fit numbers seen. Stage 5 starts only after that commit.

## Fit-only diagnostic (added 2026-10-10, before Stage 5)

New files: `src/fitdiag.py`, `tests/test_fitdiag.py`. Run AFTER Stage 4 and AFTER committing the LOG.md addenda A and B:

```powershell
py -3.12 -m pytest tests -q                                  # expect 48 passed
py -3.12 -m src.fitdiag --run runs/20261003_stage1           # seconds; writes splits/fit_diagnostic.json
```

It checks the Stage 4 file hashes first, recomputes the splits and c* and refuses to continue if they differ from Stage 4.

- **A. Gap noise:** track-cluster bootstrap (B = 2000, Fit tracks resampled) of the hard-minus-easy base-error gap, for eps and eps3, per-track mean (primary) and pooled (sensitivity). Then a ROUGH read-across to Cert (Cert has 784 tracks vs 261, so noise about sqrt(261/784) smaller) and a rough chance that G1(i) passes in >= 4 of 5 reshuffles. That chance assumes the true gap equals the Fit gap, so treat it as an order of magnitude, not a forecast.
- **B. Error by confusability slice:** 8 slices of equal track-weighted mass by log eps (class pairs never split). Shows where Fit errors sit and whether the median cut is too coarse.
- **Uses Fit frames only.** A unit test scrambles every non-Fit label and checks the output does not move.
- **It is information, not tuning** (Addendum B). Send back the printed output and `splits/fit_diagnostic.json`.
