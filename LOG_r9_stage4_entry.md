## POC-3 (R9) Stage 4: BelgiumTSC model and cache

### Part A: fixed definitions (committed before any Stage 4 number exists)

Plan: `docs/poc3-implementation-plan-r9.md` (git blob 0abe7fa51af3b499cd79218e46c83790dbfa2e99), sections 2, 3, 7 and
9 Stage 4. BelgiumTSC is the fresh dataset (Stage 1: F-A1 and F-A2 passed). This stage builds the model, the image
cache, the design/pool split, the embeddings and the head. It computes no certificate, no Cert or Test number and no
label-dependent number on pool poles (hard rule 3; plan section 2). delta = 0.1 is not used in this stage. Checks in
(j) are V-checks: a failure is a bug, the run stops and the failure is logged.

#### (a) Inputs and hash checks (hard rule 2)
- `runs/btsc/stage1/stage1_summary.json` must have SHA-256
  77ba6a7042a471fa3320059d85a5ec967a8d30a613355c05c74815d03277a0dd (Stage 1 Part B). Every other Stage 1 value used
  below is read from this file after the check.
- `runs/btsc/stage1/downloads_manifest.json` must have SHA-256
  06c90353f75a8c4133991eb32dad2281d922622cecc0591a1b9b5bc819ae4ca3, equal to `inputs_sha256.manifest_sha256` in the
  Stage 1 summary.
- Zips: `data/btsc_raw/zips/BelgiumTSC_Training.zip` SHA-256
  e5f6a33c18716917d69989488b9f6af573821a7126ca1390af7e75403af1ee61 and `BelgiumTSC_Testing.zip` SHA-256
  1811447452f02ede83c0237f2852fa8bd278f39e40ddcb1ba01c2ed212e735fa, each equal to the manifest and to
  `inputs_sha256` in the Stage 1 summary.
- Extracted trees `data/btsc_raw/extracted/<release>`: checked against their hash-checked zip with
  `src.r9_stage1_audit.verify_extracted` (same file set, CRC-32 of every file equal to its zip member). Reused by
  import; Stage 1 code is not changed.
- GT files: the SHA-256 of every `GT-*.csv` (62 per release) must equal `gt_sha256` in the Stage 1 summary; each
  `Readme.txt` must equal `readme_sha256`.
- Removed units: the 65 (pole, class) pairs in `removed_units` of the Stage 1 summary, read as stored, never
  recomputed. All 65 must be found among the testing sign units; if not, stop.
- Training release: used in full (all 4,575 images, 62 classes). No unit is removed from it (Stage 1 removed testing
  units only).
- ImageNet weights: the local torchvision file `resnet18-f37072fd.pth` (torch hub cache), loaded through
  `ResNet18_Weights.IMAGENET1K_V1`. No download. Its SHA-256 is recorded in Part B.
- Every later script reads Stage 4 outputs only after checking them against `stage4_summary.json` -> `files_sha256`.

#### (b) Views, de-duplication and view order
- One view = one image file, keyed by (class folder, file name). The file name alone is not a key: the same
  `PPPPP_VVVVV.ppm` can occur in two class folders when one pole carries two classes.
- Duplicated GT rows (training 16, testing 14 images with more than one row, Stage 1): one row is kept per image. If
  the duplicated rows of an image are identical in every column, the first is kept. If they differ in any column, the
  run stops (no rule is chosen after seeing the difference; an Amendment is written instead).
- After de-duplication: training 4,575 views, testing 2,520 views (equal to the zip image counts in Stage 1).
- Pole = PPPPP, view number = VVVVV, class = class folder (equal to ClassId, Stage 1 check). Sign unit = (pole, class).
- View order within a unit: ascending VVVVV (annotation order; Stage 1 Amendment 2). No camera index exists.
- Sign size per view: min(Roi.X2 - Roi.X1, Roi.Y2 - Roi.Y1) in original pixels, the POC-2 convention (R8 Stage 1
  deviation 2). Stored per view for Stage 5; not used here.
- Testing views of the 65 removed units are not cached and not embedded (decision 4 below); they are dropped from the
  view table right after the removal list is read.

#### (c) Testing-release split: design 20% / pool 80%, by pole
- Unit of assignment: pole. Every kept (pole, class) unit of a pole goes to the same side. Analysis unit stays
  (pole, class). A pole counts if it keeps at least one unit after the removal: 663 poles, 783 units (Stage 1).
- Design size: round(0.2 x 663) = 133 poles exactly; pool 530 poles.
- Stratum of a pole: its rarest class, that is, the class among the pole's kept units with the fewest kept testing
  units (counted over all 783 units); ties go to the smallest class ID. Single-class poles fall in their own class.
  This uses class counts only (dataset structure, as Stage 1 did), no prediction and no error.
- Allocation (largest remainder): each stratum s with n_s poles has quota 0.2 x n_s; it first gets floor(quota) design
  poles; the remaining design places (133 minus the sum of floors) go to the strata with the largest fractional
  remainder, ties broken by one seeded random order of the strata. Within a stratum, the design poles are the first
  ones in a seeded permutation of its pole numbers sorted ascending.
- Seed: 0 (the project seed in `configs/poc.yaml`), one `numpy.random.default_rng(0)` stream, used first for the
  stratum tie order and then for the within-stratum permutations in ascending stratum class ID.
- Strata with one or two poles get a design pole only through the remainder step. Classes with very few units can
  therefore sit on one side only; this is accepted and reported, not corrected.
- The design pole list is written to `runs/btsc/stage4/split_poles.json` and is fixed for good (plan Stage 4). It is
  never redrawn.
- Part B reports: poles and units per side; classes present per side; units per class per side; views per side.

#### (d) Training-release split for early stopping (by pole)
- The recipe stops early on a validation set (POC-1 Stage 1). No testing-release view is used for training, early
  stopping or any choice.
- Validation = 10% of training poles (POC-1 `val_fraction` 0.10), by pole, with the same rarest-class stratum and
  largest-remainder rule as (c) over the 62 training classes, seed 0. Training release: 1,364 poles, 1,545 units
  (Stage 1); validation size round(0.1 x 1,364) = 136 poles.
- Every one of the 62 classes must keep at least one unit in the training part (the head needs every class mean). If
  not, the run stops before training and an Amendment is written.
- Unlike POC-1 (split by image, optimistic), this validation set shares no pole with the training part.

#### (e) Image cache
- 112 x 112 uint8 RGB, Pillow BILINEAR resize (antialiased), as `src.data.load_resized` in POC-1. Input region: the
  whole image, no ROI crop (decision 1 below; POC-1 recipe unchanged). The ROI is used only for the sign size in (b).
- Files: `runs/btsc/stage4/cache/train_112.npy` (4,575 views) and `test_112.npy` (views of the 783 kept units only),
  each with a view table `train_views.csv` / `test_views.csv` (class folder, file, pole, view, class, width, height,
  ROI, size, and side or train/val part) in the same row order. Read with mmap.

#### (f) Model and training recipe (POC-1 Stage 1, unchanged except the number of classes)
- torchvision ResNet-18, ImageNet weights IMAGENET1K_V1, last layer replaced by a new 512 -> 62 linear layer.
- Optimiser AdamW; lr 0.0002 for pretrained layers and 0.001 for the new layer; weight decay 0.0001.
- Schedule: 1 epoch linear warm-up, then cosine decay to zero over the epoch budget.
- Epochs at most 20; early stopping with patience 5 on validation accuracy; best epoch = highest validation accuracy,
  ties to the lower validation loss.
- Batch size 128; mixed precision on CUDA for training only; evaluation in float32.
- Loss: plain cross-entropy, no label smoothing.
- Augmentation (light, as `configs/poc.yaml` train.aug): random crop area [0.85, 1.0], aspect [0.9, 1.1], rotation
  +-10 degrees, contrast +-20%, brightness off; no flips, blur, fog, rain, noise, perspective or shear. Applied with
  the POC-1 `src.train.augment` function (imported, not changed).
- ImageNet normalisation; seed 0; per-epoch generator seed 0 x 1000 + epoch, as POC-1.
- Trained once. No value above is tuned; nothing is changed after seeing a validation or design number.
- POC-1 code is imported, not edited (`src.train.augment`, `normalize`, `evaluate`, `sha256_file`). New code:
  `src/r9_stage4_*.py` with a 62-class model builder. Settings in a new `configs/r9_btsc_stage4.yaml`.
- Outputs: `runs/btsc/stage4/model/best.pt` (SHA-256 recorded in Part B and in `stage4_summary.json`),
  `history.csv`, `curves.png`, `results.json`.
- GPU runs are not bit-reproducible run to run; the checkpoint hash identifies the model used from here on.

#### (g) Embeddings
- From `best.pt` (hash-checked first), eval mode, float32, no augmentation, batch 512: 512-d penultimate feature
  (avgpool hook), its l2-normalised version and the 62 logits, as `src.embed` in POC-1.
- Sets: training part, training validation part, testing design poles, testing pool poles.
- Pool file `test_pool.npz` holds no class labels (as POC-2 R8 Stage 1 deviation 1): emb, raw, logits, pole, unit,
  view, size only. Design file `test_design.npz` holds labels. Training files hold labels.

#### (h) Tied-Gaussian head
- Same settings as POC-1 Stage 3 and POC-2: features = the l2-normalised 512-d embeddings; no PCA (none was used in
  POC-1 or POC-2); one class mean per class; one pooled within-class covariance with Ledoit-Wolf shrinkage (formula of
  `src.head.ledoit_wolf`); whitening W and precision from its eigendecomposition; class priors = class frequencies in
  the fit set. The POC-1 code fixes K = 43, so a 62-class copy is written in new code and tested against
  `src.head.fit_head` on synthetic 43-class data (equal to 1e-12).
- Fit set: the training part of the training release only, not the validation poles (decision 2 below; as POC-1,
  which fitted the head on the fine-tune-train frames).
- Per view: top-3 labels, normalised log-posteriors, margin (top-1 minus top-2) and s = sqrt(min squared Mahalanobis
  distance), as `src.head.score`.

#### (i) Pre-F3 guard (hard rule 3)
- A new module `src/r9_btsc_guard.py`: while `configs/frozen_r9_btsc.yaml` is absent from `FROZEN_BLOBS` in
  `src/r9_frozen.py`, any request for testing-release class labels, errors or accuracies of pool poles raises an
  assertion and stops. Only design poles pass.
- Stage 4 code reads testing labels only through this guard, except class counts per unit for the stratification in
  (c), which use no prediction.
- Label-free checks on pool views are allowed (norms, fc consistency, row counts).
- Tests: a pool pole is refused; a design pole passes; a test adds a fake frozen entry and checks the guard then lets
  the pool through (synthetic root only).

#### (j) V-checks (must pass; a failure is a bug)
- V4.1 Counts after de-duplication: training 4,575 views, 62 classes, 1,364 poles, 1,545 units; testing 2,520 views,
  53 class IDs, 701 poles, 848 units; after removing the 65 units, 663 poles and 783 units. All equal Stage 1.
- V4.2 Split: 133 design poles and 530 pool poles; design and pool are disjoint and together give all 663 poles; every
  unit lies on its pole's side; no removed unit lies on either side; training and validation poles are disjoint and
  every class is in the training part.
- V4.3 Hash checks: every input in (a) is checked before use and the script stops on a difference; tamper tests change
  a byte in each input type (zip, extracted file, GT csv, Stage 1 summary, checkpoint, Stage 4 output) and check that
  the code refuses it.
- V4.4 Guard: (i) holds in code and in tests.
- V4.5 Embeddings (POC-1 Stage 2 checks): all values finite; max |norm - 1| below 1e-5; fc(raw feature) equals the
  logits within 1e-3; argmax of the logits equals the `src.train.evaluate` predictions exactly for every set.
- V4.6 Cache: row count and order equal the view tables; the cache file hashes are recorded.
- Synthetic-data tests with brute-force references for the split rule (stratum, largest remainder, seed) and the
  de-duplication, plus the tamper tests. Full suite run before the real run.

#### (k) Reported in Part B (descriptive, no pass/fail)
- Best epoch, epochs run, validation accuracy (training-release poles not used for training).
- Head and softmax accuracy on the training part and on the validation part.
- Design-pole accuracy (head and softmax, per view and per unit at the last view in view order): reported only, no
  threshold and no action attached (decision 3 below).
- Split tables from (c) and (d); ImageNet weight file hash; environment versions.
- Nothing is reported on pool poles except row counts.

#### (l) Outputs
- `runs/btsc/stage4/`: `split_poles.json`, `train_views.csv`, `test_views.csv`, `cache/`, `model/`, `embeddings/`
  (`train_fit.npz`, `train_val.npz`, `test_design.npz`, `test_pool.npz`), `head/head.npz`, `stage4_summary.json` (all
  file SHA-256 values, code SHA-256, input hashes, checkpoint SHA-256), `stage4_printout.txt`.
- Large files (`*.npy`, `*.npz`, `*.pt`) stay git-ignored; their SHA-256 values are in `stage4_summary.json`.

#### Choices the plan leaves open (decided by the author before this Part A was committed)
1. Cache input region: the whole image resized to 112 x 112, POC-1 recipe unchanged; no ROI crop.
2. Head fit set: the training part only, as POC-1.
3. Design-pole accuracy: reported with no threshold.
4. The 65 removed units: not cached and not embedded.
5. Split seed: 0.

### Part B: results (filled from the printout after the run)
- Date: 2026-11-14 (steps run 21:03:45 to 21:08:53 UTC) / Stage: R9 Stage 4 / Run tag: runs/btsc/stage4
- Git commit of Part A: 1d9e00b | Git commit of the code: beeb114 (HEAD for all six steps). Code SHA-256 (working
  copy at run time, recorded in every step summary): src/r9_stage4.py
  d6ca35f8a42f9db26fa65205c8a4046ea190d09b06f10699dbb12f5cdc653150, src/r9_stage4_data.py
  18b853108207f97a11dc610d6bbc87f09716dba73bce1b96a3e64edcf19fde81, src/r9_stage4_model.py
  a0cd5a5220c2f6960ad5863ff273b91691b6196e23197075b776ed6bbd53ebb2, src/r9_btsc_guard.py
  d77d87365674560ff8013fc84a6ae51c5c1c93866c90cb3f72483c1ce8664d8d, configs/r9_btsc_stage4.yaml
  f92bcc7403953873620c398ffdfdb93bf3c55c2a12db6698793dee502786a7a8.
- Run commands, in order: `py -3.12 -m src.r9_stage4 split`, `cache`, `train`, `embed`, `head`, `vchecks`. Each step
  began with the freeze check (PASS, 3 files by git blob ID). Environment: Python 3.12.10, torch 2.14.1+cu132,
  torchvision 0.29.1+cu132, numpy 2.5.2, NVIDIA GeForce RTX 5050 Laptop GPU.
- Input hash checks: Stage 1 summary, manifest, both zips, both extracted trees (file set and CRC-32), all GT files and
  both readmes matched the recorded values in the split, cache and vchecks steps.
- Counts after de-duplication (V4.1): training 4,575 views, 62 classes, 1,364 poles, 1,545 units; testing 2,520 views,
  53 classes, 701 poles, 848 units; testing after removing the 65 Stage 1 units: 2,325 views, 53 classes, 663 poles,
  783 units. Duplicated GT rows dropped: training 16, testing 14 (all identical, so no stop).
- Split (V4.2), testing release:

  | Side | Poles | Units | Views | Classes |
  |---|---|---|---|---|
  | Design | 133 | 157 | 465 | 40 |
  | Pool | 530 | 626 | 1,860 | 53 |

  - 53 strata; 13 strata on one side only, all in the pool (stratum classes 0, 2, 3, 27, 43, 46 with 2 poles each;
    5, 12, 20, 25, 44, 49, 51 with 1 pole each).
  - 13 classes in the pool only (units): 0 (2), 2 (2), 3 (2), 5 (1), 12 (1), 20 (1), 25 (1), 27 (3), 43 (2), 44 (1),
    46 (2), 49 (1), 51 (1). The design set holds 40 of the 53 testing classes. This follows from the rule: strata of
    one or two poles have a quota of 0.2 or 0.4 and got no design place in the remainder step.
  - Units per class (design / pool), counts only: 0: 0/2, 1: 2/6, 2: 0/2, 3: 0/2, 4: 1/3, 5: 0/1, 6: 1/1, 7: 6/22,
    8: 1/3, 10: 2/8, 12: 0/1, 13: 2/11, 14: 1/4, 16: 1/3, 17: 12/45, 18: 7/29, 19: 10/42, 20: 0/1, 21: 3/12,
    22: 4/16, 23: 1/4, 24: 1/4, 25: 0/1, 27: 0/3, 28: 3/14, 29: 2/7, 30: 2/10, 31: 7/23, 32: 28/100, 34: 1/2,
    35: 10/41, 37: 2/8, 38: 10/46, 39: 7/24, 40: 3/13, 41: 1/3, 42: 1/2, 43: 0/2, 44: 0/1, 45: 5/22, 46: 0/2,
    47: 2/8, 49: 0/1, 51: 0/1, 53: 1/7, 54: 3/13, 55: 1/4, 56: 2/8, 57: 3/10, 58: 1/2, 59: 1/5, 60: 1/3, 61: 5/18.
- Training release split: fit part 1,228 poles, 1,384 units, 4,091 views, 62 classes; validation 136 poles, 161 units,
  484 views, 48 classes.
- Pole lists (`split_poles.json`, SHA-256 d0d39881bd394579502aa1b63b684a0460445a67f097e708d481d7062b78c44e); SHA-256 of
  each list as JSON: design a5da932ccef54c7dfa11721c79bdd38ad3f650eb209361d9b545555dc4722df5, pool
  2edd782dd226f257244252c07b80bf09392d2ff174766e12b4fd225de4e5b31d, training validation
  b20d1e18ca1cd236fb11df08fbedb5e1b0364afbf1e078cb78bf5ff1f0f04eb0, training fit
  bad15b27b1145d1212f3456b77edda0a803913bf487d8837a6707e4c68b3b272.
- ImageNet weight file `resnet18-f37072fd.pth` (local torch hub cache, no download) SHA-256
  f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec.
- Training: 18 epochs run, early stop (no validation improvement for 5 epochs); best epoch 13, validation accuracy
  0.9876 (478 of 484 views), validation loss 0.0712; 29.9 s on the GPU, mixed precision for training.
- Checkpoint `model/best.pt` SHA-256 83fdd0d1975f0b05f06ef45f8bfa45dc31846570f61db1be8da0299b06c5b76b.
- Embedding checks (V4.5): max |norm - 1| 1.0e-08 / 8.7e-09 / 8.4e-09 / 9.8e-09 and max fc consistency error 2.4e-06 /
  2.3e-06 / 2.2e-06 / 2.2e-06 for training fit (4,091 rows) / training validation (484) / design (465) / pool (1,860);
  logits argmax equals the evaluation path in all four sets. The pool file holds no labels.
- Head: fitted on the 4,091 training-part views; Ledoit-Wolf shrinkage 0.0791, covariance condition number 84.1.
  Training part: head 1.0000, softmax 1.0000, agreement 1.0000. Training validation: head 0.9814, softmax 0.9876,
  agreement 0.9938.
- Design-pole accuracy (decision 3, no threshold): per view 465 views, head 0.9699 (451 correct), softmax 0.9699,
  agreement 0.9871; per unit at the last view, 157 units, head 0.9554 (150 correct), softmax 0.9554.
- Pool poles: row count only (1,860 views); no pool label, error or accuracy was read or computed.
- Large files (git-ignored by `*.npy`, `*.npz`, `runs/**/*.pt`; SHA-256 here and in `stage4_summary.json`):
  - `cache/train_112.npy` 13a21060763323a72fed468b49d35ecf7cb51ede954edb5ab0f4f67fa44df5bc
  - `cache/test_112.npy` 94231770ef0cf99007eda8cc3ea73291c5fe9c1f3fae9b4eb1aa7c3bdb36c466
  - `model/best.pt` 83fdd0d1975f0b05f06ef45f8bfa45dc31846570f61db1be8da0299b06c5b76b
  - `embeddings/train_fit.npz` 74661849a135485e6097014cf8b5edb7e165418354828644371cc8ecd5e5488d
  - `embeddings/train_val.npz` 3969f2104621b3aaa99aab9ddb5dc11838f7111d9dd1cb32b3ca8944cd57c749
  - `embeddings/test_design.npz` 10085b80c20fee107cc1f398521c6bf481806d29ee00b6b0ecf0b8741c0b52fd
  - `embeddings/test_pool.npz` 47fa0f5cdeab2456ff325fc1cb707ef08b7ef3b0e08ea6c39c47d25dc4eb3183
  - `head/head.npz` a3b9d94e0351e1da9eb5bc233eff69c2acb6a0a3a27e379f31cd2dcfc7f45483
  - `head/scores_train_fit.npz` aa5829575364a979cfb01e69d890b3199810d9b46eb50f789c800caa0843df04
  - `head/scores_train_val.npz` e56c2d1ff5454bc14cdbf747ee255d66928d8c4e3429425e01b157af12928cd8
  - `head/scores_test_design.npz` 10f5483002d95ebb9d78e90771f3fbc56dd1f4ddc6f05c1c2efb6da73a6dcaef
  - `head/scores_test_pool.npz` a5a01131b5f11dea529e3ac75be661058204d2f78cce8e14f7de799b1fc8b9ee
- Step summaries (committed): split 475465d8c7de98366c555d42e4b5fdcbe7d3d3c74c0beae468d33d5f30b27bfd, cache
  925c6ebe5a1c75c56311523fd5d716bbdf695b65e6a4b04e7ee2fe425f288214, train
  9aee27951591f6e8ee10a80d3e2badd4b7df789259885104b084bccd08609e7e, embed
  6abd9867331e0f66580bfaff722d1e290d5cd14df4139d2b4f572856cb023307, head
  624eb57d276f1c32489c652e1e7b723517f5af6b060c831ed5be34c6e822b5e5.
- Output: stage4_summary.json SHA-256 1c66a0119d39473f818f1add56c8a878d32ddfcc91aef4b0f5a9bd76568a3a31.
- Checks: V4.1 PASS | V4.2 PASS | V4.3 PASS | V4.4 PASS | V4.5 PASS | V4.6 PASS | ALL PASS: True.
- Facts about the run:
  - Every step header and step summary records `worktree clean False`. The working tree held only the untracked
    Stage 4 outputs (the runner creates `stage4_printout.txt` before it checks git status); no tracked file was
    changed and HEAD was beeb114 for all six steps. The author decided not to change the code after the split step and
    to record this here.
  - The train step printed a PyTorch UserWarning that a non-writable NumPy array (the read-only mmap cache) was turned
    into a tensor. Nothing writes to that tensor. The warning went to stderr, so it is not in `stage4_printout.txt`.
  - Before the run, the first full test suite run had one failure, the POC-1 test
    `tests/test_stage3.py::test_ledoit_wolf_matches_sklearn` (ImportError); it passed alone and the rerun of the full
    suite gave 435 passed. This matches the intermittent Windows Application Control block on scikit-learn recorded in
    `docs/HANDOFF.md`; no POC-3 code imports scikit-learn.
  - V4.6 checks the cache shape against the view tables and re-decodes three rows per cache (first, middle, last).
- Deviation from the plan? Yes: (1) plan Stage 4 says "Cache crops at 112x112"; the released images are cached whole,
  without a further ROI crop (decision 1). (2) The split is by pole with (pole, class) as the analysis unit (Stage 1
  Amendment 2); 13 small classes are in the pool only. (3) Early stopping uses a by-pole 10% validation split of the
  training release (Part A (d)), not a split by image as in POC-1. (4) Part A (f) named `results.json` as a training
  output; the training values are in `train_summary.json` instead, and no `last.pt` is written.
- Next step: Stage 5 (BelgiumTSC design phase and freeze F3), only when the author says so.
