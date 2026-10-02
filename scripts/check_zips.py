"""PRE-FLIGHT CHECK: are your zips the right GTSRB releases?

Why this exists: the plan (R11 in the risk register) says we already used the
wrong release once (torchvision gave 26,640 images / 888 tracks). So before any
heavy work we count images and tracks and compare to the known numbers.

It only reads the zip's file LIST (no image decoding, standard library only),
so it runs in seconds and needs no pip installs.

Usage (from the project folder):
    python scripts/check_zips.py
    python scripts/check_zips.py --train path/to/train.zip --test path/to/test.zip
Exit code is 0 if everything matches, 1 otherwise.
"""
import argparse
import glob
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# Training images live at  .../Images/<class 5-digit>/<track 5-digit>_<frame 5-digit>.ppm
# A "track" = one physical sign filmed as the car approaches (30 frames).
# The track prefix is only unique INSIDE a class, so we key tracks on (class, prefix).
TRAIN_IMG = re.compile(r"(?:^|/)(\d{5})/(\d{5})_(\d{5})\.ppm$")
# Per-class label file inside the training zip: .../<class>/GT-000NN.csv
TRAIN_CSV = re.compile(r"(?:^|/)(\d{5})/GT-(\d{5})\.csv$")
# Test images are a flat list: .../Final_Test/Images/00000.ppm ... 12629.ppm
TEST_IMG = re.compile(r"(?:^|/)([^/]+)\.ppm$", re.I)

# Known numbers for the official releases (from the plan, section 1).
EXPECTED = {
    "train_images": 39209,
    "train_tracks": 1307,
    "train_frame_counts": {30: 1306, 29: 1},
    "n_classes": 43,
    "test_images": 12630,
}


def inspect_train(zip_path):
    """Count images, classes, tracks and frames-per-track from the zip index."""
    tracks = defaultdict(list)          # (class, prefix) -> [frame indices]
    classes, n_csv = set(), 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = TRAIN_IMG.search(name)
            if m:
                cls, prefix, frame = m.group(1), m.group(2), int(m.group(3))
                classes.add(int(cls))
                tracks[(cls, prefix)].append(frame)
            elif TRAIN_CSV.search(name):
                n_csv += 1
    frame_counts = Counter(len(f) for f in tracks.values())
    # Frames inside a track must be 0,1,2,... with no gaps (the plan asserts this).
    gaps = [k for k, f in tracks.items() if sorted(f) != list(range(len(f)))]
    return {
        "images": sum(len(f) for f in tracks.values()),
        "tracks": len(tracks),
        "classes": sorted(classes),
        "frame_counts": dict(frame_counts),
        "tracks_with_frame_gaps": len(gaps),
        "per_class_csvs": n_csv,
        # How many tracks would we count if we (wrongly) keyed on prefix alone?
        "tracks_if_prefix_only": len({p for (_, p) in tracks}),
    }


def inspect_test(zip_path):
    """Count test images and check the numbering is 00000..12629 with no gaps."""
    stems, csvs, in_class_folders = [], [], 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                csvs.append(name)
            elif TEST_IMG.search(name):
                if TRAIN_IMG.search(name):      # looks like a training-layout file
                    in_class_folders += 1
                stems.append(TEST_IMG.search(name).group(1))
    nums = sorted(int(s) for s in stems if s.isdigit())
    return {
        "images": len(stems),
        "csv_files_inside": csvs,
        "contiguous_numbering": nums == list(range(len(nums))) and len(nums) == len(stems),
        "files_that_look_like_training": in_class_folders,
    }


def verdict(train, test):
    """Compare facts to expected numbers. Returns list of (label, ok, detail)."""
    E = EXPECTED
    r = []
    if train:
        r += [
            ("train: 39,209 images", train["images"] == E["train_images"], train["images"]),
            ("train: 1,307 tracks (class, prefix)", train["tracks"] == E["train_tracks"], train["tracks"]),
            ("train: frames/track = {30:1306, 29:1}",
             train["frame_counts"] == E["train_frame_counts"], train["frame_counts"]),
            ("train: classes 0..42", train["classes"] == list(range(E["n_classes"])), len(train["classes"])),
            ("train: no gaps in frame numbering", train["tracks_with_frame_gaps"] == 0,
             train["tracks_with_frame_gaps"]),
            ("train: 43 per-class label CSVs", train["per_class_csvs"] == E["n_classes"], train["per_class_csvs"]),
        ]
    if test:
        r += [
            ("test: 12,630 images", test["images"] == E["test_images"], test["images"]),
            ("test: numbering contiguous", test["contiguous_numbering"], test["contiguous_numbering"]),
            ("test: not training-layout files", test["files_that_look_like_training"] == 0,
             test["files_that_look_like_training"]),
        ]
    return r


def inspect_gt(csv_path, test_zip_path=None):
    """Check the labelled test CSV: 12,630 rows, ClassId 0..42, and (if the test zip is
    given) that its filenames are exactly the images inside the zip (no mis-join)."""
    import csv
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    cols = list(rows[0].keys()) if rows else []
    names = [r["Filename"] for r in rows]
    out = {"rows": len(rows), "has_classid": "ClassId" in cols,
           "classes": sorted({int(r["ClassId"]) for r in rows}) if "ClassId" in cols else [],
           "duplicate_filenames": len(names) - len(set(names)), "matches_zip": None}
    if test_zip_path:
        with zipfile.ZipFile(test_zip_path) as zf:
            zip_names = {n.rsplit("/", 1)[-1] for n in zf.namelist() if n.lower().endswith(".ppm")}
        out["matches_zip"] = set(names) == zip_names
    return out


def find(explicit, pattern):
    """Use the given path, else look in data/raw/ and the current folder."""
    if explicit:
        return explicit
    for folder in ("data/raw", "."):
        hits = glob.glob(str(Path(folder) / pattern))
        if hits:
            return hits[0]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--train", help="GTSRB_Final_Training_Images.zip")
    ap.add_argument("--test", help="GTSRB_Final_Test_Images.zip")
    ap.add_argument("--gt", help="GT-final_test.csv (labelled test CSV)")
    a = ap.parse_args(argv)

    tr_path = find(a.train, "*Training_Images*.zip")
    te_path = find(a.test, "*Test_Images*.zip")
    train = test = None
    if tr_path:
        print(f"Training zip: {tr_path}")
        train = inspect_train(tr_path)
        print(f"  images={train['images']}  tracks={train['tracks']}  classes={len(train['classes'])}")
        print(f"  frames-per-track histogram: {train['frame_counts']}")
        print(f"  tracks if keyed on prefix alone (WRONG way): {train['tracks_if_prefix_only']}")
    else:
        print("Training zip not found (looked in data/raw/ and here).")
    if te_path:
        print(f"Test zip: {te_path}")
        test = inspect_test(te_path)
        print(f"  images={test['images']}  csv files inside: {test['csv_files_inside'] or 'none'}")
    else:
        print("Test zip not found (looked in data/raw/ and here).")

    # Labelled test CSV (optional): auto-found as data/raw/GT-final_test.csv.
    gt_path = find(a.gt, "GT-final_test.csv")
    gt = None
    if gt_path:
        print(f"Test labels CSV: {gt_path}")
        gt = inspect_gt(gt_path, te_path)
        print(f"  rows={gt['rows']}  classes={len(gt['classes'])}  has ClassId={gt['has_classid']}")

    print("\nChecks")
    results = verdict(train, test)
    if gt:
        results += [
            ("gt: 12,630 rows with ClassId", gt["rows"] == EXPECTED["test_images"] and gt["has_classid"], gt["rows"]),
            ("gt: classes 0..42", gt["classes"] == list(range(EXPECTED["n_classes"])), len(gt["classes"])),
            ("gt: no duplicate filenames", gt["duplicate_filenames"] == 0, gt["duplicate_filenames"]),
        ]
        if gt["matches_zip"] is not None:
            results.append(("gt: filenames == images in test zip", gt["matches_zip"], gt["matches_zip"]))
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}   (got: {detail})")
    # The test zip has images only. Labels come from the separate GT zip.
    # The test image zip only holds an UNLABELLED csv (GT-final_test.test.csv, no ClassId).
    if test and not gt:
        print("\nNOTE: no labelled GT-final_test.csv found. Put it in data/raw/ "
              "(the CSV inside the image zip has no labels).")
    if train and (train["images"] == 26640 or train["tracks"] == 888):
        print("\nWARNING: 26,640 images / 888 tracks is the torchvision split, the WRONG release.")
    ok_all = bool(results) and all(ok for _, ok, _ in results)
    print("\nOVERALL:", "PASS - right releases" if ok_all else "FAIL - see above")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
