"""Stage 0 tests on tiny synthetic zips shaped like the official GTSRB releases.

Why synthetic: we can prove the parsing/assertion LOGIC is right without the 300 MB
downloads. Passing here does NOT prove your real zips are correct; scripts/check_zips.py
and the real Stage 0 run do that.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import data as D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_zips as C  # noqa: E402
import inspect_dups as ID  # noqa: E402
import verify_exact_copies as V  # noqa: E402

N_CLASSES = 3
EXP = {"train_images": 0, "train_tracks": 0, "train_frame_counts": {}, "test_images": 0,
       "n_classes": N_CLASSES}
HDR = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId\n"


def ppm(w, h, val):
    """Make a flat-colour PPM image in memory."""
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), val, np.uint8)).save(buf, format="PPM")
    return buf.getvalue()


@pytest.fixture
def zips(tmp_path):
    """3 classes; classes 0, 1 and 2 ALL have a track prefix 00000 (collision test).
    Class 0: tracks 00000 (4 frames), 00001 (3 frames). Class 1: 00000 (4). Class 2: 00000 (4)."""
    layout = {0: {"00000": 4, "00001": 3}, 1: {"00000": 4}, 2: {"00000": 4}}
    tr = tmp_path / "train.zip"
    with zipfile.ZipFile(tr, "w") as zf:
        for c, tracks in layout.items():
            rows = ""
            for pre, n in tracks.items():
                for f in range(n):
                    fn = f"{pre}_{f:05d}.ppm"
                    zf.writestr(f"GTSRB/Final_Training/Images/{c:05d}/{fn}", ppm(30 + f, 40 + f, c * 80))
                    rows += f"{fn};{30+f};{40+f};2;2;27;37;{c}\n"
            zf.writestr(f"GTSRB/Final_Training/Images/{c:05d}/GT-{c:05d}.csv", HDR + rows)
    te, gt = tmp_path / "test.zip", tmp_path / "gt.zip"
    gt_rows = ""
    with zipfile.ZipFile(te, "w") as zf:
        for i in range(6):
            c = i % N_CLASSES
            zf.writestr(f"GTSRB/Final_Test/Images/{i:05d}.ppm", ppm(35, 35, 50 * c))
            gt_rows += f"{i:05d}.ppm;35;35;3;3;30;30;{c}\n"
    with zipfile.ZipFile(gt, "w") as zf:
        zf.writestr("GT-final_test.csv", HDR + gt_rows)
    return tr, te, gt


def test_track_ids_are_class_scoped(zips):
    # Same prefix in different classes must NOT merge into one track.
    meta = D.build_train_meta(zips[0])
    assert len(meta) == 15
    assert meta["track_id"].nunique() == 4  # NOT 2: prefix 00000 recurs across classes
    assert set(meta["track_id"]) == {"00_00000", "00_00001", "01_00000", "02_00000"}
    assert meta.groupby("track_id").size().to_dict() == {
        "00_00000": 4, "00_00001": 3, "01_00000": 4, "02_00000": 4}


def test_validate_train_passes_and_catches_errors(zips):
    # Good data passes; wrong counts and a missing frame must both be rejected.
    meta = D.build_train_meta(zips[0])
    exp = dict(EXP, train_images=15, train_tracks=4, train_frame_counts={4: 3, 3: 1})
    D.validate_train(meta, exp)
    with pytest.raises(AssertionError):
        D.validate_train(meta, dict(exp, train_tracks=3))
    bad = meta.drop(index=1).reset_index(drop=True)  # gap in frame indices
    with pytest.raises(AssertionError):
        D.validate_train(bad, dict(exp, train_images=14, train_frame_counts={4: 2, 3: 2}))


def test_test_meta_join_and_validation(zips):
    # Test labels join on filename; the test table has no track column by design.
    tr = D.build_train_meta(zips[0])
    te = D.build_test_meta(zips[1], zips[2])
    assert len(te) == 6 and "track_id" not in te.columns
    D.validate_test(te, tr, dict(EXP, test_images=6))


def test_bare_gt_csv_matches_zip_variant(zips, tmp_path):
    # Using the bare CSV must give the same table as reading it from the GT zip,
    # and check_zips must confirm its filenames match the test image zip.
    with zipfile.ZipFile(zips[2]) as zf:
        csv_path = tmp_path / "GT-final_test.csv"
        csv_path.write_bytes(zf.read("GT-final_test.csv"))
    a = D.build_test_meta(zips[1], csv_path)
    b = D.build_test_meta(zips[1], zips[2])
    assert a.equals(b)
    g = C.inspect_gt(csv_path, zips[1])
    assert g["rows"] == 6 and g["matches_zip"] and g["classes"] == [0, 1, 2]


def test_cache_shape_dtype_and_whole_image_resize(zips):
    # Cache must be uint8 (N,112,112,3); flat colours stay flat (no border artefacts).
    meta = D.build_train_meta(zips[0])
    img = D.build_cache(zips[0], meta, 112)
    assert img.shape == (15, 112, 112, 3) and img.dtype == np.uint8
    c = meta["class_id"].to_numpy()
    assert (img[0] == c[0] * 80).all() and (img[-1] == c[-1] * 80).all()


def test_phash_and_dup_detection():
    # An image copied into the other set must be found; random others must not match.
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (5, 112, 112, 3), dtype=np.uint8)
    b = np.concatenate([a[2:3], rng.integers(0, 255, (3, 112, 112, 3), dtype=np.uint8)])
    pairs = D.cross_release_near_dups(D.phash64(a), D.phash64(b), max_ham=0)
    assert [(2, 0)] == [(int(i), int(j)) for i, j, _ in pairs]


def test_check_zips_counts_match_loader(zips):
    # The stdlib pre-flight counter must agree with the pandas loader.
    t = C.inspect_train(zips[0])
    assert t["images"] == 15 and t["tracks"] == 4 and t["classes"] == [0, 1, 2]
    assert t["per_class_csvs"] == 3 and t["frame_counts"] == {4: 3, 3: 1}
    # Keying on prefix alone would give 2 tracks: the bug the (class, prefix) key avoids.
    assert t["tracks_if_prefix_only"] == 2
    s = C.inspect_test(zips[1])
    assert s["images"] == 6 and s["contiguous_numbering"] and s["csv_files_inside"] == []


def test_check_zips_verdict_flags_wrong_release(zips):
    # Our tiny fixture is NOT the real release, so the real-number checks must FAIL.
    res = C.verdict(C.inspect_train(zips[0]), C.inspect_test(zips[1]))
    assert res and not all(ok for _, ok, _ in res)


def test_inspect_dups_separates_real_copy_from_lookalike():
    # A true duplicate has ~0 pixel distance; an unrelated image does not.
    rng = np.random.default_rng(1)
    tr = rng.integers(0, 255, (6, 112, 112, 3), dtype=np.uint8)
    te = np.concatenate([tr[3:4], rng.integers(0, 255, (2, 112, 112, 3), dtype=np.uint8)])
    pairs = np.array([[3, 0, 0], [1, 1, 2]])
    df = ID.score_pairs(tr, te, pairs, np.zeros(6, int), np.zeros(3, int))
    assert df.loc[0, "mad"] == 0.0 and df.loc[1, "mad"] > 20
    assert df.loc[0, "corr"] > 0.99 and abs(df.loc[1, "corr"]) < 0.3
    same, diff = ID.random_baseline(tr, te, np.zeros(6, int), np.zeros(3, int), n=50)
    assert len(same) == 50 and np.median(same) > 20


def test_verify_exact_copies_compares_raw_bytes(tmp_path):
    # Same bytes in both zips -> identical; a different file in the same slot -> not identical.
    same, other = ppm(20, 20, 7), ppm(20, 20, 9)
    tr, te = tmp_path / "tr.zip", tmp_path / "te.zip"
    with zipfile.ZipFile(tr, "w") as z:
        z.writestr("GTSRB/Final_Training/Images/00014/00001_00000.ppm", same)
        z.writestr("GTSRB/Final_Training/Images/00014/00001_00001.ppm", other)
    with zipfile.ZipFile(te, "w") as z:
        z.writestr("GTSRB/Final_Test/Images/00000.ppm", same)
        z.writestr("GTSRB/Final_Test/Images/00001.ppm", same)
    import pandas as pd
    tm = pd.DataFrame({"class_folder": [14, 14], "filename": ["00001_00000.ppm", "00001_00001.ppm"]})
    em = pd.DataFrame({"filename": ["00000.ppm", "00001.ppm"]})
    pairs = pd.DataFrame({"train_idx": [0, 1], "test_idx": [0, 1], "mad": [0.0, 3.0]})
    res = V.compare_pairs(tr, te, tm, em, pairs)
    assert res["identical"].tolist() == [True, False]
