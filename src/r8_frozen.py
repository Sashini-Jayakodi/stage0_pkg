"""POC-2 (R8) freeze guard. Every confirm-phase script (Stage 3 onward) calls load_frozen() first.

It refuses to go on if configs/frozen_r8.yaml or the plan file differ from the SHA-256 values recorded at F1
(LOG.md F1 record, commit 3de4cb9). After F1 nothing frozen may change (plan section 2), so a mismatch means stop.
"""
from pathlib import Path

import yaml

from src.util import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path("configs/frozen_r8.yaml")
PLAN = Path("docs/poc2-implementation-plan-r8.md")
FROZEN_SHA256 = "4f7a79156a4486b6b5f97515457513e6df2a2bfeed2a41b6fa05936bd1141d24"   # F1 record in LOG.md
PLAN_SHA256 = "b0ad6292138732f40639753b6e5b76d050cfc30da137cc2b1d8f0c0e85174bd2"     # frozen_r8.yaml provenance


def load_frozen(root=ROOT, frozen_sha=FROZEN_SHA256, plan_sha=PLAN_SHA256):
    """Check both hashes, then return the parsed frozen config. The sha arguments exist only for synthetic tests."""
    root = Path(root)
    assert sha256_file(root / FROZEN) == frozen_sha, "configs/frozen_r8.yaml differs from the F1 hash; stop (plan section 2)"
    assert sha256_file(root / PLAN) == plan_sha, "the plan file differs from the F1 hash; stop (plan section 2)"
    cfg = yaml.safe_load((root / FROZEN).read_text())
    assert cfg.get("r8_frozen") is True, "frozen_r8.yaml is not marked frozen"
    return cfg
