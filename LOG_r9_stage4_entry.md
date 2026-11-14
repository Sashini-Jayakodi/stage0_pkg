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

### Part B: results (fill from the printout after the run)
- Date: ____ / Stage: R9 Stage 4 / Run tag: runs/btsc/stage4
- Git commit of Part A: ____ | Git commit of the code: ____ | Run command: ____
- Input hash checks: ____
- Counts (V4.1): ____
- Split (V4.2): design ____ poles / ____ units / ____ classes; pool ____ poles / ____ units / ____ classes; views
  ____ / ____
- Training validation split: ____ poles / ____ units in validation; classes in training part ____
- ImageNet weight file SHA-256: ____
- Training: best epoch ____ of ____ run; validation accuracy ____
- Checkpoint SHA-256: ____
- Embedding checks (V4.5): ____
- Head: shrinkage ____; head / softmax accuracy, training part ____ / ____, validation ____ / ____
- Design-pole accuracy (decision 3, no threshold): ____
- Output: stage4_summary.json SHA-256 ____
- Checks: V4.1 ____ | V4.2 ____ | V4.3 ____ | V4.4 ____ | V4.5 ____ | V4.6 ____
- Deviation from the plan? ____
- Next step: Stage 5 (BelgiumTSC design phase and freeze F3), only when the author says so.
