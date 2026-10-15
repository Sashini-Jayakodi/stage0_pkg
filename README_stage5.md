# Stage 5: G1(i), G1(ii), rho and rank-of-error

This is the first stage that reads **Cert labels**. It applies the rules in the LOG.md pre-registration block (frozen at commit `45a4d41`) and Addenda A and B exactly as written. Nothing is tunable from the command line except `--boot`, which is for smoke tests only.

| File | Goes to |
|---|---|
| `src/stage5.py` | `src/` (new) |
| `src/util.py` | `src/` (new; torch-free `sha256_file`, see below) |
| `src/splits.py`, `src/fitdiag.py` | `src/` (replace; **one import line changed**, logic identical) |
| `tests/test_stage5.py`, `tests/test_util.py` | `tests/` (new) |
| `tests/test_stage1.py`, `tests/test_stage3.py`, `tests/test_stage4.py` | `tests/` (replace; they now skip cleanly when torch cannot load) |
| `README.md` | replace (test count, file list) |
| `LOG_stage5_entry.md` | numbers blank; paste into `LOG.md` **after** the real run. It already records the code change below. |

## Why `src/util.py` exists (Windows blocks torch on this machine)

Windows Smart App Control / Application Control blocks torch's unsigned DLLs (`WinError 4551`). Stages 3 to 6 are numpy/scipy only, but they imported `sha256_file` from `src/train.py`, which imports torch at the top, so torch's block stopped them too. `src/util.py` holds the same function with no heavy imports, and `splits.py`, `fitdiag.py` and `stage5.py` now import it from there. `train.py` and `head.py` are **unchanged**. Stage 4 outputs were not regenerated: Stage 5 recomputes the splits, τ and c* and refuses to run unless they equal the saved Stage 4 files and their hashes.

Not fixed by this: anything that really needs torch (Stage 1 training, Stage 2 embeddings, `src.head`, Stage 7 scenario images, Stage 8 latency). Those need a working torch, WSL2, Colab, or an exported model.

## Run

```powershell
cd "D:\FYP Review\POC Implementation\stage0_package VI\stage0_pkg"
py -3.12 -m pytest tests -q                       # torch works: 63 passed | torch blocked: 48 passed, 2 skipped (Stage 1 and Stage 3 files skip themselves)
git status                                        # commit anything pending BEFORE the run
py -3.12 -m src.stage5 --run runs/20261003_stage1
```

Takes under a minute. Needs `src.splits` to have been run. It refuses to run if any Stage 4 file hash changed, and it asserts that the recomputed split, gate τ and c* equal the Stage 4 values. Writes `runs/20261003_stage1/splits/stage5_results.json`.

**Do not re-run to "see if it changes".** The result is deterministic (fixed seeds). A re-run only matters if you changed code, and then the log must say so.

## What it computes

1. **G1(i).** Gated-in Cert frames, labelled by the head's top-1, no margin threshold. Strata from the seed's Fit c*. Hard-minus-easy gap, track-cluster bootstrap over Cert tracks (B = 2000), lower bound = 2.5th percentile. Pass if lower bound > 0 in at least 4 of 5 reshuffles.
   - **Primary: per-track mean** (decision 2026-10-10). Pooled rate = sensitivity.
   - If the two definitions give different verdicts, the script prints a CAVEAT line (Addendum A). Report both; per-track stays primary.
   - Also repeated with B doubled and a different random stream: the "B->2B stable" column must say True (plan check).
   - Run for ε (primary) and ε₃ (pre-registered first fallback), side by side.
2. **G1(ii).** Hard-group frame errors and distinct tracks with at least one hard error, per reshuffle. Pass if the **median** over the 5 meets ≥ 30 errors **and** ≥ 10 tracks.
3. **ρ (ICC).** One-way ANOVA estimator with the unequal-cluster-size correction, track = cluster, with a track bootstrap CI. Design effect = 1 + (30 − 1)ρ (plan formula) and, per stratum, 1 + (m̄ − 1)ρ with the real mean frames per track m̄. Effective sample size = frames / design effect.
   - Over **all 39,209 frames** (dataset-level, split-free, descriptive; it uses all labels like the overall accuracy does) and over gated-in Cert frames per reshuffle and per stratum.
4. **Rank of error.** For each wrong gated-in Cert frame: is the true class 2nd, 3rd or 4th-or-lower in the posterior? (Only the top 3 are stored, so "4th or lower" is one bucket.)
5. **Fragment counts** (no errors, no labels): gated-in frames per track in each stratum on Fit and on Cert, as histogram bins, mean, median, and the share of tracks with ≤ 2 frames. This documents the fragment-track effect seen in the Fit diagnostic.

## What it does not do

- **No Test labels are read.** A unit test scrambles every non-Cert prediction and requires identical output, and also requires that scrambling Cert predictions *does* change it.
- No margin threshold, no certification, no G1(iii). That is Stage 6.
- No rule is changed: c*, δ, β, strata, seeds, base-error definition and the aggregation rules are used as logged.

## Checks it reports (plan)

- Bootstrap stable when B is doubled.
- ICC in [−0.05, 1] (dataset-level and Cert).
- Error count reconciles with Stage 3: the all-frame head error equals `1 − head_accuracy` in `head_summary.json` to 1e-9 (printed as `reconciles_with_stage3_head_accuracy: True`). If it prints False, stop and tell me.

## How to log it

Paste the printed output into `LOG_stage5_entry.md`, then the entry into `LOG.md`. Write the numbers exactly as printed. The expected outcome (G1(i) fails for ε, probably ε₃) is already logged before Stage 5 (LOG.md, Fit-only diagnostic entry), so the entry does not need a prediction line. Whatever the result is, the decision tree is the plan's §9; there is no R7 unless you declare one in `LOG.md` and commit it first.

## What to send back

- The full printed output.
- `runs/20261003_stage1/splits/stage5_results.json` is optional; the print has everything the log needs.
