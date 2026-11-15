"""POC-3 (R9) freeze guard. Every r9 script that uses the certification engine calls check_frozen() first.

Frozen files are compared by git blob ID, not SHA-256: the working copy on Windows has CRLF line endings while the
committed blob has LF, so a byte hash would change with the machine's git settings. The blob ID is computed here in
Python, after CRLF -> LF, exactly as git computes it for a normalised text file (LOG_r9_stage3_entry.md Part A (b)).
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "src/r9_certify.py"

# Recorded in LOG.md: plan R9 in the Stage 0 entry, engine and engine tests in the Stage 2 entry (freeze 99da0b5),
# the BelgiumTSC frozen config in the F3 pre-registration block (Stage 5). Its presence here closes F3 and opens the
# pool labels in src/r9_btsc_guard.py.
FROZEN_BLOBS = {
    "docs/poc3-implementation-plan-r9.md": "0abe7fa51af3b499cd79218e46c83790dbfa2e99",
    ENGINE: "158a9c9e32ab8ec1f568f75b5c35c9992711db74",
    "tests/test_r9_certify.py": "ecb6e04429883ddbf298b22b803cd3bc70c959ef",
    "configs/frozen_r9_btsc.yaml": "f4d37e5d73ef230e91689be5303e5eac51b15450",
}


def blob_id(path):
    """git blob ID of the file with CRLF turned into LF (what git stores for a text file under autocrlf)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_frozen(root=ROOT, blobs=None):
    """Stop if any frozen file is missing or differs from its recorded blob ID; return the IDs for the summary JSON.
    root and blobs exist only for synthetic tests."""
    root = Path(root)
    blobs = FROZEN_BLOBS if blobs is None else blobs
    seen = {}
    for rel, want in blobs.items():
        p = root / rel
        assert p.is_file(), f"frozen file {rel} is missing; stop (plan section 2)"
        seen[rel] = blob_id(p)
        assert seen[rel] == want, f"{rel} differs from its frozen blob ID; stop (plan section 2)"
    if root == ROOT and ENGINE in blobs:
        # the module actually imported must be the file that was just checked (no shadow copy on sys.path)
        from src import r9_certify
        assert Path(r9_certify.__file__).resolve() == (ROOT / ENGINE).resolve(), "imported engine is not the checked file"
    return seen
