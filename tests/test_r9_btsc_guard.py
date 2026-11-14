"""Tests for the pre-F3 guard on BelgiumTSC testing labels (src/r9_btsc_guard.py): pool labels are refused before
F3, design labels pass, and only a frozen BelgiumTSC config opens the pool."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import r9_btsc_guard as G  # noqa: E402
from src import r9_frozen as FZ  # noqa: E402

POLE = np.array([10, 10, 11, 12, 12, 13])
Y = np.array([3, 3, 5, 7, 8, 1])
DESIGN = [10, 13]


def test_f3_is_open_in_the_real_repo():
    assert G.F3_CONFIG not in FZ.FROZEN_BLOBS
    assert not G.f3_closed()


def test_design_labels_pass():
    g = G.GuardedTestLabels(POLE, Y, DESIGN)
    rows = g.design_rows()
    assert rows.tolist() == [0, 1, 5]
    assert g.labels(rows).tolist() == [3, 3, 1]


def test_pool_label_refused():
    g = G.GuardedTestLabels(POLE, Y, DESIGN)
    for r in (2, 3, 4):
        with pytest.raises(AssertionError, match="pool pole"):
            g.labels([r])


def test_mixed_request_refused():
    g = G.GuardedTestLabels(POLE, Y, DESIGN)
    with pytest.raises(AssertionError):
        g.labels([0, 2])


def test_all_rows_refused_before_f3():
    g = G.GuardedTestLabels(POLE, Y, DESIGN)
    with pytest.raises(AssertionError):
        g.labels(np.arange(len(Y)))


def test_returned_labels_are_a_copy():
    g = G.GuardedTestLabels(POLE, Y, DESIGN)
    out = g.labels([0])
    out[0] = 99
    assert g.labels([0]).tolist() == [3]


def test_pool_opens_only_with_frozen_config_entry():
    fake = dict(FZ.FROZEN_BLOBS)
    g = G.GuardedTestLabels(POLE, Y, DESIGN, blobs=fake)
    with pytest.raises(AssertionError):
        g.labels([2])
    fake[G.F3_CONFIG] = "0" * 40
    g = G.GuardedTestLabels(POLE, Y, DESIGN, blobs=fake)
    assert g.labels([2, 3]).tolist() == [5, 7]


def test_assert_design_only_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        design = set(rng.choice(20, 5, replace=False).tolist())
        req = rng.choice(20, rng.integers(1, 6)).tolist()
        should_pass = all(p in design for p in req)
        if should_pass:
            G.assert_design_only(req, design)
        else:
            with pytest.raises(AssertionError):
                G.assert_design_only(req, design)


def test_length_mismatch_refused():
    with pytest.raises(AssertionError):
        G.GuardedTestLabels([1, 2], [1], [1])
