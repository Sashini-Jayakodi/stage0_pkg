"""Synthetic tests for src/d1_grid_check.py: level definitions, blend against a brute-force loop, shared severity-1
draw across L1-L6, threshold metrics against a loop, and a tamper test."""
import numpy as np
import pytest

from src import d0_corrupt_compat as DC
from src import d1_grid_check as G
from src import r9_stage3_scores as SC


def test_level_spec():
    assert [G.level_spec(lv) for lv in G.LEVELS] == [(1, 0.1), (1, 0.2), (1, 0.3), (1, 0.5), (1, 0.75), (1, 1.0),
                                                      (2, None), (3, None), (4, None), (5, None)]
    with pytest.raises(AssertionError):
        G.level_spec(11)


def test_blend_matches_pixel_loop():
    rng = np.random.default_rng(0)
    a = (rng.random((5, 6, 3)) * 255).astype(np.uint8)
    b = (rng.random((5, 6, 3)) * 255).astype(np.uint8)
    for t in G.BLEND_T:
        got = G.blend(a, b, t)
        for idx in np.ndindex(a.shape):
            v = (1 - t) * float(a[idx]) + t * float(b[idx])
            assert got[idx] == min(255, max(0, int(np.rint(v))))
    assert np.array_equal(G.blend(a, b, 1.0), b)


@pytest.mark.parametrize("name", DC.CORRUPTIONS)
def test_blend_levels_share_one_severity1_draw_and_l6_is_severity1(name):
    rng = np.random.default_rng(1)
    frames = (rng.random((3, 48, 48, 3)) * 255).astype(np.uint8)
    seed = [20261117, 42]
    sev1 = DC.corrupt_track(frames, name, 1, seed)
    for lv in range(1, 7):
        t = G.BLEND_T[lv - 1]
        want = np.stack([G.blend(f, c, t) for f, c in zip(frames, sev1)])
        assert np.array_equal(G.corrupt_track_level(frames, name, lv, seed), want)
    assert np.array_equal(G.corrupt_track_level(frames, name, 6, seed), sev1)
    assert np.array_equal(G.corrupt_track_level(frames, name, 8, seed), DC.corrupt_track(frames, name, 3, seed))


def test_at_threshold_matches_loop():
    rng = np.random.default_rng(2)
    seqs, y = [], rng.integers(0, 3, 40)
    for _ in range(40):
        conf = rng.normal(1, 1, 8)
        conf[: rng.integers(0, 3)] = -np.inf
        seqs.append((conf, rng.integers(0, 3, 8)))
    grid = np.array([-np.inf, 0.5, 1.5, 9.0])
    assert G.at_threshold(seqs, y, grid, -1) == (None, None)
    for j in range(4):
        c, e = G.at_threshold(seqs, y, grid, j)
        com, wr = 0, 0
        for (conf, lab), yy in zip(seqs, y):
            ok = np.flatnonzero(np.isfinite(conf) & (conf >= grid[j]))
            if len(ok):
                com += 1
                wr += int(lab[ok[0]] != yy)
        assert c == com / 40 and e == (wr / com if com else None)


def test_worker_reads_only_given_rows(tmp_path):
    X = np.zeros((10, 40, 40, 3), np.uint8)
    X[5:] = 255                                                    # rows a design-only run never indexes
    np.save(tmp_path / "x.npy", X)
    G._worker_init(str(tmp_path / "x.npy"))
    out = G._worker(("contrast", 7, [(0, np.array([0, 1])), (1, np.array([2, 3, 4]))]))
    assert out.shape == (5, 40, 40, 3) and out.max() < 255


def test_tampered_input_is_refused(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    h = G.sha256_file(f)
    f.write_text("{ }")
    with pytest.raises(AssertionError, match="differs from its recorded SHA-256"):
        G.D0.verify_sha(f, h)
