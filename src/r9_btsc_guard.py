"""POC-3 (R9) pre-F3 guard for BelgiumTSC testing labels (hard rule 3; plan R9 section 2; Stage 4 Part A (i)).

Until configs/frozen_r9_btsc.yaml is listed in FROZEN_BLOBS (src/r9_frozen.py, added at F3), no class label of a pool
pole may leave this module, so no error, accuracy or other label-dependent number can be computed on pool poles.
Design poles pass. Every r9 BelgiumTSC script reads testing labels only through GuardedTestLabels.
"""
import numpy as np

from src import r9_frozen

F3_CONFIG = "configs/frozen_r9_btsc.yaml"


def f3_closed(blobs=None):
    """True once the BelgiumTSC frozen config is a frozen file. blobs exists only for synthetic tests."""
    return F3_CONFIG in (r9_frozen.FROZEN_BLOBS if blobs is None else blobs)


def assert_design_only(poles, design_poles, blobs=None):
    """Stop if any requested pole is not a design pole while F3 is open."""
    if f3_closed(blobs):
        return
    pool = sorted(set(int(p) for p in np.atleast_1d(poles)) - set(int(p) for p in design_poles))
    assert not pool, (f"labels of {len(pool)} pool pole(s) requested before F3; refused "
                      "(hard rule 3, plan R9 section 2)")


class GuardedTestLabels:
    """Holds the testing-release labels and hands out only design-pole labels before F3."""

    def __init__(self, pole, y, design_poles, blobs=None):
        self._pole = np.asarray(pole, dtype=np.int64).copy()
        self._y = np.asarray(y, dtype=np.int64).copy()
        assert len(self._pole) == len(self._y), "pole and label arrays differ in length"
        self._design = frozenset(int(p) for p in design_poles)
        self._blobs = blobs

    def __len__(self):
        return len(self._y)

    def labels(self, rows):
        """Labels of the given row positions; refused if any row belongs to a pool pole before F3."""
        rows = np.asarray(rows, dtype=np.int64)
        assert_design_only(self._pole[rows], self._design, self._blobs)
        return self._y[rows].copy()

    def design_rows(self):
        """Row positions of design-pole views (label-free: uses pole numbers only)."""
        return np.flatnonzero(np.isin(self._pole, sorted(self._design)))
