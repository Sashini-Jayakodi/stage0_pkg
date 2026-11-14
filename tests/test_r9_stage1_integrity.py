"""Tests for R9 Stage 1, Amendment 1 conditions (a) and (b). Synthetic zips only."""
import json
import zipfile

import pytest

from src.r9_stage1_download import sha256_file
from src.r9_stage1_integrity import count_images, extract, integrity


def make_zip(path, members, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def write_manifest(path, zip_path, sha=None):
    rec = {"name": zip_path.name, "sha256": sha or sha256_file(zip_path)}
    path.write_text(json.dumps({"files": [rec]}))
    return path


def test_good_zip_passes(tmp_path):
    z = make_zip(tmp_path / "Good.zip", {"00000/01_00.ppm": b"P6 abc", "00000/GT.csv": b"x;y"})
    m = write_manifest(tmp_path / "m.json", z)
    r = integrity(z, m)
    assert r["result"] == "PASS" and r["first_bad_member"] is None and r["members"] == 2


def test_corrupted_zip_fails_testzip(tmp_path):
    payload = b"A" * 200
    z = make_zip(tmp_path / "Bad.zip", {"00000/01_00.ppm": payload})
    raw = bytearray(z.read_bytes())
    i = raw.find(payload)
    raw[i + 10] ^= 0xFF  # change one stored byte so its CRC no longer matches
    z.write_bytes(bytes(raw))
    m = write_manifest(tmp_path / "m.json", z)  # manifest of the corrupted file, so only the CRC can catch it
    r = integrity(z, m)
    assert r["result"] == "FAIL" and r["first_bad_member"] == "00000/01_00.ppm"


def test_escaping_member_is_refused(tmp_path):
    z = make_zip(tmp_path / "Evil.zip", {"ok/01_00.ppm": b"x", "../escape.ppm": b"y"})
    m = write_manifest(tmp_path / "m.json", z)
    with pytest.raises(SystemExit, match="unsafe member path"):
        extract(z, m, tmp_path / "out")
    assert not (tmp_path / "escape.ppm").exists()
    assert not (tmp_path / "out" / "Evil" / "ok").exists()  # refused before anything was written


def test_wrong_sha_is_refused(tmp_path):
    z = make_zip(tmp_path / "Good.zip", {"a.ppm": b"x"})
    m = write_manifest(tmp_path / "m.json", z, sha="0" * 64)
    with pytest.raises(SystemExit, match="differs from manifest"):
        integrity(z, m)
    with pytest.raises(SystemExit, match="differs from manifest"):
        extract(z, m, tmp_path / "out")


def test_changed_zip_after_record_is_refused(tmp_path):
    z = make_zip(tmp_path / "Good.zip", {"a.ppm": b"x"})
    m = write_manifest(tmp_path / "m.json", z)
    make_zip(z, {"a.ppm": b"changed"})  # tamper after the manifest was written
    with pytest.raises(SystemExit, match="differs from manifest"):
        integrity(z, m)


def test_extract_needs_new_folder_and_counts_ppm_only(tmp_path):
    z = make_zip(tmp_path / "Rel.zip", {"00000/a.ppm": b"1", "00001/b.PPM": b"2",
                                         "00001/GT.csv": b"c", "Readme.txt": b"r"})
    m = write_manifest(tmp_path / "m.json", z)
    dest = extract(z, m, tmp_path / "out")
    assert count_images(dest) == 2  # brute force: two .ppm files, any case
    with pytest.raises(SystemExit, match="new empty folder"):
        extract(z, m, tmp_path / "out")
