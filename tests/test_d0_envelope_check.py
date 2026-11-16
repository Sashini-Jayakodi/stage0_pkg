"""Synthetic tests for src/d0_envelope_check.py and src/d0_corrupt_compat.py: brute-force references, the fog patch
against the unmodified package formula, determinism of the per-track seed, and a tamper test."""
import numpy as np
import pytest

from src import d0_corrupt_compat as DC
from src import d0_envelope_check as D
from src import r9_certify as C
from src import r9_stage3_scores as SC


def random_seqs(rng, n_tracks, length=12, K=4):
    """Running (conf, lab) sequences with leading -inf (no value yet), as the real scores give."""
    seqs = []
    for _ in range(n_tracks):
        conf = rng.normal(2, 2, length)
        conf[: rng.integers(0, 4)] = -np.inf
        lab = rng.integers(0, K, length)
        lab[~np.isfinite(conf)] = -1
        seqs.append((conf, lab))
    return seqs, rng.integers(0, K, n_tracks)


def brute_frame_table(seqs, y, grid):
    A = np.zeros((len(seqs), len(grid)), int); W = np.zeros_like(A); NF = np.zeros(len(seqs), int)
    for i, (conf, lab) in enumerate(seqs):
        NF[i] = len(conf)
        for j, g in enumerate(grid):
            for t in range(len(conf)):
                if np.isfinite(conf[t]) and conf[t] >= g:
                    A[i, j] += 1
                    W[i, j] += int(lab[t] != y[i])
    return A, W, NF


def test_frame_table_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        seqs, y = random_seqs(rng, 15)
        grid = np.concatenate([[-np.inf], np.sort(rng.normal(2, 2, 6))])
        for got, want in zip(D.frame_table(seqs, y, grid), brute_frame_table(seqs, y, grid)):
            assert np.array_equal(got, want)


def test_frame_level_never_commits_a_frame_without_value():
    seqs = [(np.array([-np.inf, -np.inf, 1.0]), np.array([-1, -1, 2]))]
    A, W, NF = D.frame_table(seqs, np.array([2]), np.array([-np.inf, 0.5, 5.0]))
    assert A.tolist() == [[1, 1, 0]] and W.tolist() == [[0, 0, 0]] and NF.tolist() == [3]


def test_certify_both_matches_direct_engine_calls():
    rng = np.random.default_rng(2)
    seqs, y = random_seqs(rng, 400, length=10)
    y = np.array([lab[np.isfinite(c)][-1] if np.isfinite(c).any() else 0 for c, lab in seqs])   # mostly right
    flip = rng.random(400) < 0.01
    y[flip] = (y[flip] + 1) % 4
    grid = np.concatenate([[-np.inf], np.quantile(np.concatenate([c[np.isfinite(c)] for c, _ in seqs]),
                                                  [0.1, 0.3, 0.5, 0.7, 0.9])])
    tab = SC.track_table(seqs, y, grid)
    A, W, NF = D.frame_table(seqs, y, grid)
    f_s, f_f = tab["committed"].mean(0), A.sum(0) / NF.sum()
    cert = rng.random(400) < 0.75
    for al in (0.02, 0.05):
        j_s, j_f = D.certify_both(tab, A, W, NF, cert, al, f_s, f_f, 100, 1000)
        com = tab["committed"][cert]
        assert j_s == C.certify((com & tab["wrong"][cert]).sum(0), com.sum(0), int(cert.sum()), al, f_s, test="binom")
        # frame level by explicit loops over frames
        k = [sum(W[i, j] for i in np.flatnonzero(cert)) for j in range(len(grid))]
        n_c = [sum(A[i, j] for i in np.flatnonzero(cert)) for j in range(len(grid))]
        assert j_f == C.certify(np.array(k), np.array(n_c), int(sum(NF[cert])), al, f_f, test="binom")


def test_pool_at_matches_brute_force():
    rng = np.random.default_rng(3)
    seqs, y = random_seqs(rng, 30)
    grid = np.array([-np.inf, 1.0, 3.0])
    tab = SC.track_table(seqs, y, grid)
    A, W, NF = D.frame_table(seqs, y, grid)
    assert D.pool_at(tab, A, W, NF, -1) is None
    for j in range(3):
        p = D.pool_at(tab, A, W, NF, j)
        com = [i for i, (c, _) in enumerate(seqs) if (np.isfinite(c) & (c >= grid[j])).any()]
        wrong = [i for i in com if seqs[i][1][np.flatnonzero(np.isfinite(seqs[i][0]) & (seqs[i][0] >= grid[j]))[0]] != y[i]]
        assert p["n_c"] == len(com) and p["k"] == len(wrong) and p["commit_rate"] == len(com) / 30
        fr = [(t, i) for i, (c, _) in enumerate(seqs) for t in range(len(c)) if np.isfinite(c[t]) and c[t] >= grid[j]]
        fw = sum(seqs[i][1][t] != y[i] for t, i in fr)
        assert p["frame_risk"] == (fw / len(fr) if fr else None)
        assert p["frame_commit_rate"] == len(fr) / sum(len(c) for c, _ in seqs)


@pytest.mark.parametrize("cert, want", [([True] * 5, 5), ([False] * 5, 0), ([True, True, False, True, True], 2),
                                        ([False, True, True, True, True], 0), ([True, False, False, False, False], 1)])
def test_boundary_is_last_severity_before_first_failure(cert, want):
    assert D.boundary(cert) == want


def test_fog_patch_equals_unmodified_package_on_three_images():
    rng = np.random.default_rng(4)
    imgs = [(rng.random((112, 112, 3)) * 255).astype(np.uint8) for _ in range(3)]
    assert DC.reference_check(imgs) == [True, True, True]
    assert "float_" not in np.__dict__                         # the alias is removed again afterwards


@pytest.mark.parametrize("name", DC.CORRUPTIONS)
def test_corrupt_track_is_deterministic_per_seed_and_keeps_shape(name):
    rng = np.random.default_rng(5)
    frames = (rng.random((3, 64, 64, 3)) * 255).astype(np.uint8)
    a = DC.corrupt_track(frames, name, 3, [D.D0_SEED, 7])
    np.random.seed(123)                                        # other global draws in between do not matter
    b = DC.corrupt_track(frames, name, 3, [D.D0_SEED, 7])
    assert a.dtype == np.uint8 and a.shape == frames.shape and np.array_equal(a, b)
    assert not np.array_equal(a, frames)


def test_worker_output_independent_of_chunking(tmp_path):
    rng = np.random.default_rng(6)
    X = (rng.random((12, 40, 40, 3)) * 255).astype(np.uint8)
    np.save(tmp_path / "x.npy", X)
    D._worker_init(str(tmp_path / "x.npy"))
    items = [(0, np.array([2, 0, 1])), (5, np.array([3, 4, 5])), (9, np.array([6, 7, 8]))]
    whole = D._worker(("gaussian_noise", 2, items))
    split = np.concatenate([D._worker(("gaussian_noise", 2, items[:1])), D._worker(("gaussian_noise", 2, items[1:]))])
    assert np.array_equal(whole, split)


def test_tampered_input_is_refused(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    h = D.sha256_file(f)
    assert D.verify_sha(f, h) == h
    f.write_text("{ }")
    with pytest.raises(AssertionError, match="differs from its recorded SHA-256"):
        D.verify_sha(f, h)
