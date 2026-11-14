"""Tests for the POC-3 freeze guard (src/r9_frozen.py): real files pass, tampered or missing files are refused,
line-ending-only differences pass, and the Python blob ID equals git's."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import r9_frozen as FZ  # noqa: E402


def copy_frozen(tmp):
    for rel in FZ.FROZEN_BLOBS:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp / rel)
    return tmp


def test_real_files_pass():
    seen = FZ.check_frozen()
    assert seen == FZ.FROZEN_BLOBS


def test_recorded_ids_are_the_ones_in_the_log():
    log = (ROOT / "LOG.md").read_text(encoding="utf-8")
    for want in FZ.FROZEN_BLOBS.values():
        assert want in log


def test_one_changed_byte_is_refused(tmp_path):
    root = copy_frozen(tmp_path)
    p = root / FZ.ENGINE
    b = bytearray(p.read_bytes())
    i = b.index(b"DELTA = 0.1")
    b[i + 10] = ord("2")                                   # DELTA = 0.2
    p.write_bytes(bytes(b))
    with pytest.raises(AssertionError, match="src/r9_certify.py differs"):
        FZ.check_frozen(root)


def test_appended_newline_is_refused(tmp_path):
    root = copy_frozen(tmp_path)
    p = root / "docs/poc3-implementation-plan-r9.md"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(AssertionError, match="plan-r9.md differs"):
        FZ.check_frozen(root)


def test_missing_file_is_refused(tmp_path):
    root = copy_frozen(tmp_path)
    (root / "tests/test_r9_certify.py").unlink()
    with pytest.raises(AssertionError, match="missing"):
        FZ.check_frozen(root)


@pytest.mark.parametrize("eol", [b"\n", b"\r\n"])
def test_line_endings_alone_pass(tmp_path, eol):
    root = copy_frozen(tmp_path)
    for rel in FZ.FROZEN_BLOBS:
        p = root / rel
        lf = p.read_bytes().replace(b"\r\n", b"\n")
        p.write_bytes(lf.replace(b"\n", eol))
    assert FZ.check_frozen(root) == FZ.FROZEN_BLOBS


def test_blob_id_matches_git_definition_on_a_small_file(tmp_path):
    # git's documented value for the content "hello\n"
    p = tmp_path / "h.txt"
    p.write_bytes(b"hello\r\n")
    assert FZ.blob_id(p) == "ce013625030ba8dba906f756967f9e9ca394464a"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_blob_id_equals_git_index_blob():
    # the blob git stores in the index for each frozen file (independent of how hash-object filters the working copy)
    for rel in FZ.FROZEN_BLOBS:
        out = subprocess.run(["git", "ls-files", "-s", rel], cwd=ROOT, capture_output=True, text=True, check=True)
        assert out.stdout.split()[1] == FZ.blob_id(ROOT / rel), rel


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_blob_id_equals_git_hash_object():
    for rel in FZ.FROZEN_BLOBS:
        out = subprocess.run(["git", "hash-object", rel], cwd=ROOT, capture_output=True, text=True, check=True)
        assert out.stdout.strip() == FZ.blob_id(ROOT / rel), rel


def test_guard_module_reads_no_data():
    src = (ROOT / "src" / "r9_frozen.py").read_text(encoding="utf-8")
    for bad in ("np.load", "btsc_raw", "runs/btsc", "track_table_pool", "torch"):
        assert bad not in src
