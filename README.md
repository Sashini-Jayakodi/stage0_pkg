# Stage 0 package

```
configs/poc.yaml        constants + expected GTSRB counts (single source of truth)
scripts/check_zips.py   FIRST STEP: counts images/tracks in your zips (stdlib only, seconds)
src/data.py             metadata, 112x112 cache, assertions, preview strips, pHash
src/stage0.py           full Stage 0 run (needs train zip, test zip, labelled GT csv)
scripts/inspect_dups.py checks whether pHash near-duplicates are real (pixel-level) + writes exclusion lists
tests/test_stage0.py    synthetic-data tests (pip install pytest pandas pillow scipy pyyaml tqdm matplotlib numpy)
data/raw/               put the zips here
```

## Step 1: check your two zips
1. Copy `GTSRB_Final_Training_Images.zip` and `GTSRB_Final_Test_Images.zip` into `data/raw/`.
2. Also copy `GT-final_test.csv` (labelled test CSV) into `data/raw/`.
3. `python scripts/check_zips.py`
4. Expected: train 39,209 images / 1,307 tracks / frames {30: 1306, 29: 1} / 43 classes; test 12,630 images.
   Every line should say PASS (now includes 4 checks on the CSV). Send me the full printout.

## Step 2 (later): full Stage 0
Needs the labelled `GT-final_test.csv` in `data/raw/` (a GT zip containing it also works), then
`python -m src.stage0`.

## Run the tests
`python -m pytest tests -q`

## Step 3: near-duplicate diagnosis (after `python -m src.stage0` printed the >1% warning)
`python scripts/inspect_dups.py` then open `data/cache/near_dup_check.png` and send me the printout.

## Step 4: independent proof of the exact copies
`py -3.11 scripts/verify_exact_copies.py` compares the raw bytes of each flagged pair in the original zips (no cache, no hashing, no resizing).

## Stage 1: fine-tune ResNet-18 once (plan section 5)
Files: `src/train.py`, `train:` block in `configs/poc.yaml`, `tests/test_stage1.py`, `LOG_stage1_entry.md` (paste into your LOG.md).

1. Install PyTorch (not in your lock file yet).
   - NVIDIA GPU: open https://pytorch.org/get-started/locally/ , pick Windows / Pip / Python / your CUDA version, run the command it shows with `py -3.11 -m` in front.
   - No GPU: `py -3.11 -m pip install torch torchvision` (CPU only; a full run will be slow).
   - Check: `py -3.11 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
2. Pipeline check (about a minute, random weights, no download): `py -3.11 -m src.train --smoke --no-pretrained`
3. Real run (downloads ImageNet ResNet-18 weights once, so needs internet): `py -3.11 -m src.train --tag 20261003_stage1`
4. Crashed or closed the window? `py -3.11 -m src.train --tag 20261003_stage1 --resume`
5. Then `py -3.11 -m pip freeze > requirements-lock.txt` again (now includes torch) and send me the printed summary plus `runs/<tag>/results.json` and `curves.png`.

Expected sanity number: about 95-99% on the training release. Below 90% = bug, stop. Above 99.5% = tell me, do not weaken the model.
The validation accuracy in the log is optimistic (the test release has no track IDs) and is only for early stopping; never quote it.
Colab: upload the folder, set `train.run_dir` to a Google Drive path so checkpoints survive a disconnect.
