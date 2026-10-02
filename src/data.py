"""Stage 0: GTSRB ingestion, track index, 112x112 uint8 cache.

Reads straight from the official zips (no extraction needed).
Tracks are keyed on (class, prefix); the prefix alone is NOT unique.
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# --- file-name patterns inside the official zips -------------------------------
# Train image:  .../<class folder>/<track>_<frame>.ppm   e.g. 00007/00012_00003.ppm
TRAIN_IMG_RE = re.compile(r"(?:^|/)(\d{5})/(\d{5})_(\d{5})\.ppm$")
# Per-class label file: .../<class folder>/GT-000NN.csv
TRAIN_CSV_RE = re.compile(r"(?:^|/)(\d{5})/GT-(\d{5})\.csv$")
# Test image: flat folder, no track information at all
TEST_IMG_RE = re.compile(r"(?:^|/)Final_Test/Images/([^/]+\.ppm)$")
# The label CSVs are semicolon-separated with these columns; we rename to snake_case.
CSV_COLS = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]
RENAME = {"Filename": "filename", "Width": "width", "Height": "height",
          "Roi.X1": "roi_x1", "Roi.Y1": "roi_y1", "Roi.X2": "roi_x2",
          "Roi.Y2": "roi_y2", "ClassId": "class_id"}


# ---------------------------------------------------------------- metadata
def build_train_meta(zip_path):
    """One row per training image, sorted by (class, prefix, frame)."""
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # Map (class folder, filename) -> full path inside the zip, so each CSV row
        # can later be matched to its image file.
        img_path = {}
        for n in names:
            m = TRAIN_IMG_RE.search(n)
            if m:
                img_path[(m.group(1), n.rsplit("/", 1)[1])] = n
        csvs = [(n, TRAIN_CSV_RE.search(n)) for n in names]
        csvs = [(n, m) for n, m in csvs if m]
        if not csvs:
            raise ValueError("no per-class GT-000NN.csv files found in training zip")
        for n, m in sorted(csvs):
            folder = m.group(1)
            with zf.open(n) as f:
                df = pd.read_csv(f, sep=";")
            df = df[CSV_COLS].rename(columns=RENAME)
            df["folder"] = folder
            rows.append(df)
    meta = pd.concat(rows, ignore_index=True)
    # filename "00012_00003.ppm" -> track prefix "00012", frame 3
    meta["track_prefix"] = meta["filename"].str.slice(0, 5)
    meta["frame_idx"] = meta["filename"].str.slice(6, 11).astype(int)
    meta["zip_path"] = [img_path.get((f, fn)) for f, fn in zip(meta["folder"], meta["filename"])]
    # Fail loudly if a CSV row points at an image that is not in the zip.
    if meta["zip_path"].isna().any():
        raise ValueError("CSV rows with no matching image in zip: "
                         f"{int(meta['zip_path'].isna().sum())}")
    # class_folder is kept so validate_train can check the CSV class equals the folder.
    meta["class_folder"] = meta["folder"].astype(int)
    # Track ID = class + prefix. The prefix repeats across classes, so using it alone
    # would merge different physical signs into one "track" (a silent, serious bug).
    meta["track_id"] = (meta["class_id"].astype(int).map("{:02d}".format)
                        + "_" + meta["track_prefix"])
    meta = meta.drop(columns=["folder"]).sort_values(
        ["class_id", "track_prefix", "frame_idx"]).reset_index(drop=True)
    return meta


def build_test_meta(images_zip, gt_path):
    """One row per test image, joined to GT on filename. No track column.

    gt_path may be the bare labelled CSV (GT-final_test.csv) or the GT zip that
    contains it. NOTE: the image zip also holds GT-final_test.test.csv, which has NO
    ClassId column (unlabelled); we never read that one.
    """
    gt_path = str(gt_path)
    if gt_path.lower().endswith(".csv"):
        gt = pd.read_csv(gt_path, sep=";")
    else:
        with zipfile.ZipFile(gt_path) as zf:
            gt_names = [n for n in zf.namelist() if n.endswith("GT-final_test.csv")]
            if not gt_names:
                raise ValueError("GT-final_test.csv not found in GT zip")
            with zf.open(gt_names[0]) as f:
                gt = pd.read_csv(f, sep=";")
    gt = gt[CSV_COLS].rename(columns=RENAME)
    with zipfile.ZipFile(images_zip) as zf:
        paths = {}
        for n in zf.namelist():
            m = TEST_IMG_RE.search(n)
            if m:
                paths[m.group(1)] = n
    # Join on filename; any unmatched row means a mis-join, so stop immediately.
    gt["zip_path"] = gt["filename"].map(paths)
    if gt["zip_path"].isna().any():
        raise ValueError("test GT rows with no image: " f"{int(gt['zip_path'].isna().sum())}")
    if gt["class_id"].isna().any():
        raise ValueError("missing labels after join")
    return gt.sort_values("filename").reset_index(drop=True)


# ------------------------------------------------------------------- cache
def load_resized(zf, name, size):
    """Decode a PPM from the zip, resize the WHOLE image (no ROI crop).
    Pillow's BILINEAR resize is antialiased when downscaling."""
    with zf.open(name) as f:
        im = Image.open(io.BytesIO(f.read())).convert("RGB")
    return np.asarray(im.resize((size, size), Image.BILINEAR))


def build_cache(zip_path, meta, size, desc="decode"):
    """Decode every image once into one big uint8 array (N, size, size, 3).
    Why: reading 39k PPMs every training epoch is slow; later stages read this cache."""
    from tqdm import tqdm
    out = np.empty((len(meta), size, size, 3), dtype=np.uint8)
    with zipfile.ZipFile(zip_path) as zf:
        for i, name in enumerate(tqdm(meta["zip_path"], desc=desc)):
            out[i] = load_resized(zf, name, size)
    return out


# ------------------------------------------------------------- assertions
def validate_train(meta, exp):
    """Raise AssertionError on any violated Stage 0 check (plan section 4)."""
    assert len(meta) == exp["train_images"], f"train images {len(meta)} != {exp['train_images']}"
    n_tracks = meta["track_id"].nunique()
    assert n_tracks == exp["train_tracks"], f"tracks {n_tracks} != {exp['train_tracks']}"
    # Frame-count histogram guards against merged or split tracks.
    sizes = meta.groupby("track_id").size().value_counts().to_dict()
    assert sizes == {int(k): v for k, v in exp["train_frame_counts"].items()}, f"frame counts {sizes}"
    assert set(meta["class_id"]) == set(range(exp["n_classes"])), "train class IDs not 0..42"
    # A physical sign has exactly one class.
    assert meta.groupby("track_id")["class_id"].nunique().max() == 1, "a track has >1 class"
    assert (meta["class_id"] == meta["class_folder"]).all(), "CSV ClassId != folder"
    # Frames must run 0..n-1 with no holes.
    for tid, g in meta.groupby("track_id"):
        f = g["frame_idx"].to_numpy()
        assert (np.sort(f) == np.arange(len(f))).all(), f"non-contiguous frames in track {tid}"


def validate_test(meta, train_meta, exp):
    assert len(meta) == exp["test_images"], f"test images {len(meta)} != {exp['test_images']}"
    assert set(meta["class_id"]) == set(range(exp["n_classes"])), "test class IDs not 0..42"
    # Same label space in both releases, otherwise class IDs are mismatched.
    assert set(meta["class_id"]) == set(train_meta["class_id"]), "class sets differ across releases"
    assert not meta["class_id"].isna().any(), "missing labels"


# ------------------------------------------------------------------ extras
def save_track_strips(images, meta, out_png, n=5, seed=0, size=112):
    """5 random tracks as 30-frame strips (sign should grow frame by frame).
    Visual sanity check that tracks really are one sign approaching the camera."""
    import matplotlib
    matplotlib.use("Agg")   # no display needed; write straight to a PNG
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    tids = rng.choice(meta["track_id"].unique(), n, replace=False)
    fig, axes = plt.subplots(n, 1, figsize=(15, 1.6 * n))
    for ax, tid in zip(np.atleast_1d(axes), tids):
        idx = meta.index[meta["track_id"] == tid].to_numpy()
        strip = np.concatenate([images[i] for i in idx], axis=1)
        ax.imshow(strip)
        ax.set_axis_off()
        ax.set_title(f"track {tid} (class {meta.at[idx[0], 'class_id']}), {len(idx)} frames",
                     fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def phash64(images):
    """64-bit pHash per image (32x32 gray -> DCT -> top-left 8x8 vs median).
    Used only to find the SAME picture appearing in both releases (leakage check)."""
    from scipy.fft import dctn
    g = images.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    out = np.empty((len(g), 64), dtype=bool)
    for i, im in enumerate(g):
        small = np.asarray(Image.fromarray(im).resize((32, 32), Image.BILINEAR), dtype=np.float32)
        low = dctn(small, norm="ortho")[:8, :8].ravel()
        out[i] = low > np.median(low[1:])   # skip the DC term; it dominates the median
    return out


def cross_release_near_dups(h_train, h_test, max_ham=2, chunk=2000):
    """Return (train_idx, test_idx, hamming) for pairs with Hamming <= max_ham.
    Hamming distance via a +-1 matrix product (fast); chunked to keep memory small."""
    a = h_train.astype(np.float32) * 2 - 1
    b = h_test.astype(np.float32) * 2 - 1
    res = []
    for s in range(0, len(a), chunk):
        ham = (64 - a[s:s + chunk] @ b.T) / 2
        ti, ji = np.nonzero(ham <= max_ham)
        res.append(np.stack([ti + s, ji, ham[ti, ji]], axis=1))
    return np.concatenate(res) if res else np.zeros((0, 3))
