# FYP POC: project README (Stages 0-5)

One folder, files are added stage by stage. Run everything from the project folder with `py -3.12`.

```
configs/poc.yaml              every constant (single source of truth); you edit it, packages never overwrite it
configs/poc_stage4_snippet.yaml   optional `splits:` block for poc.yaml (same numbers as the code defaults)
scripts/check_zips.py         Stage 0: counts images/tracks in the zips (stdlib only)
scripts/inspect_dups.py       Stage 0: are the pHash near-duplicates real? writes exclusion lists
scripts/verify_exact_copies.py  Stage 0: byte-level proof of the exact copies
scripts/stage1_diagnostics.py Stage 1: read-only accuracy by size / frame position / class pair
scripts/stage3_pair_ranks.py  Stage 3: where do real confusions fall in the eps ranking (descriptive)
src/data.py, src/stage0.py    Stage 0: metadata, 112x112 cache, assertions, pHash
src/train.py                  Stage 1: ResNet-18 fine-tune, run ONCE
src/embed.py                  Stage 2: embeddings from the frozen checkpoint
src/head.py                   Stage 3: tied-Gaussian head, eps table, confusability atlas
src/splits.py                 Stage 4: reshuffles, gate, cut-point c*, Fit-side calibration
src/fitdiag.py                Stage 4 add-on: Fit-only diagnostic (gap noise, error by eps slice); information only
src/util.py                   torch-free sha256_file (CPU-only stages import this, so a blocked torch cannot stop them)
src/stage5.py                 Stage 5: G1(i), G1(ii), rho/ICC, rank-of-error, eps3, fragment counts (first stage that reads Cert labels)
tests/test_stage0.py ... test_stage5.py   synthetic-data tests
data/raw/                     the zips and GT-final_test.csv (read-only, never modified)
data/cache/                   decoded 112x112 arrays + metadata tables
runs/20261003_stage1/         checkpoint, embeddings/, head_emb/, splits/
LOG.md                        decision and result log (pre-registration block at the top)
LOG_stageN_entry.md           the same entries as separate files
README_stage3.md, README_stage4.md, README_stage5.md   per-stage run notes
```

## Environment
Python 3.12 (`py -3.12`), torch 2.14.1+cu132 and torchvision 0.29.1+cu132 (the RTX 5050 needs a CUDA 12.8+ build). Python 3.11 was abandoned because Windows Application Control blocked torch's DLLs. The Stage 0 cache hashes are identical under 3.11, 3.12 and 3.13.

## Tests
`py -3.12 -m pytest tests -q` gives 63 passed with the current test files when torch loads (the Stage 3 Ledoit-Wolf test is skipped if scikit-learn is not installed). If Windows blocks torch (WinError 4551), the Stage 1 and Stage 3 test files skip themselves and you see 48 passed, 2 skipped; Stages 4 to 6 still run. If you see fewer, one of your test files is an older copy.

## Stage 0: data
1. Put `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip` and the labelled `GT-final_test.csv` in `data/raw/`.
2. `py -3.12 scripts/check_zips.py` (expect train 39,209 images / 1,307 tracks / frames {30: 1306, 29: 1} / 43 classes; test 12,630 images; every line PASS).
3. `py -3.12 -m src.stage0` builds the cache and runs the assertions.
4. If it prints the >1% near-duplicate warning: `py -3.12 scripts/inspect_dups.py`, then `py -3.12 scripts/verify_exact_copies.py` (add `--max-mad 5` for the gray zone).

## Stage 1: fine-tune ResNet-18 once
1. Pipeline check, about a minute, random weights: `py -3.12 -m src.train --smoke --no-pretrained`
2. Real run (downloads ImageNet weights once): `py -3.12 -m src.train --tag 20261003_stage1`
3. Crashed? `py -3.12 -m src.train --tag 20261003_stage1 --resume`
4. Optional read-only diagnostics: `py -3.12 scripts/stage1_diagnostics.py --run runs/20261003_stage1`

Sanity number on the training release, by the PLAN's rule: below 90% = bug, stop; above 99.5% = suspect leakage and investigate, do not weaken the model. (The "95-99%" range in earlier versions of this README was only a guess, not a plan rule. The real run gave 91.9%, which passes.) The validation accuracy in the log is optimistic and only for early stopping; never quote it. Always label the number "backbone fine-tuned on the official test release", never "test accuracy".

The checkpoint `runs/20261003_stage1/best.pt` is FROZEN (SHA-256 in LOG.md and `embed.expected_checkpoint_sha256` in poc.yaml). Do not re-run Stage 1.

## Stage 2: embeddings
`py -3.12 -m src.embed --run runs/20261003_stage1` refuses to run if the checkpoint hash differs from the frozen one. It must reproduce the Stage 1 accuracy exactly. Output: `runs/20261003_stage1/embeddings/`. See LOG_stage2_entry.md.

## Stage 3: head, epsilon table, atlas
See `README_stage3.md`.

## Stage 4: reshuffles and Fit-side calibration
See `README_stage4.md`. After Stage 4, write and COMMIT the pre-registration block at the top of LOG.md before Stage 5.

## Rules that stay frozen
- Never change c*'s rule, delta = 0.1, beta = 5%, the strata, the reshuffles (seeds 0-4) or the severities after seeing data.
- No re-tuning of the backbone or head after seeing results. A new design would be a separate revision (R7).
- eps is used only as a RANK, never as an error probability.
- No Cert or Test error is looked at before the pre-registration commit.
- Back up `runs/20261003_stage1` (checkpoint, embeddings, head_emb, splits) after each stage.
