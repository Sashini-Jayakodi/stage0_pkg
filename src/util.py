"""Tiny helpers with NO heavy imports (no torch). Stages 3-6 are numpy/scipy only, so they must not need torch just to hash a file.
Why this file exists: on some Windows machines Smart App Control / Application Control blocks torch's unsigned DLLs
(WinError 4551). src/train.py imports torch at the top, so importing its sha256_file would drag torch in and block CPU-only stages.
This is a byte-for-byte copy of the function in src/train.py; train.py itself is unchanged."""
import hashlib


def sha256_file(path):
    """Fingerprint of a file; stored next to results so a checkpoint or output can be verified later."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
