"""Tests for the R9 Stage 2 population curves (synthetic only)."""
import ast
from pathlib import Path

import numpy as np
import pytest

import src.r9_stage2_popcurves as P
from src import r8_stage3_synth as R8S


def test_fixed30_draw_equals_r8_sample():
    # The POC-2 Stage 3a model is reused unchanged: for T = 30 the r9 draw must be the r8 draw.
    r = R8S.solve_r(0.45)
    X9, T = P.draw(200, "fixed-30", r, R8S.M, np.random.default_rng(5))
    X8 = R8S.sample(200, r, np.random.default_rng(5))
    assert np.array_equal(X9, X8) and (T == 30).all()


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.45])
def test_model_functions_match_r8_at_default_p(rho):
    assert P.m_of(R8S.P_FRAME) == pytest.approx(R8S.M, abs=1e-15)
    assert P.solve_r(rho, R8S.P_FRAME) == pytest.approx(R8S.solve_r(rho), abs=1e-9)


def brute_commit(X, T, lam):
    """Per-unit loop: first t < T_i with |S_t| >= lam; wrong iff S_t < 0 there."""
    com, wr = np.zeros(len(X), bool), np.zeros(len(X), bool)
    for i in range(len(X)):
        s = 0.0
        for t in range(T[i]):
            s += X[i, t]
            if abs(s) >= lam:
                com[i], wr[i] = True, s < 0
                break
    return com, wr


@pytest.mark.parametrize("cluster", ["fixed-3", "view-mix", "fixed-30"])
def test_commit_matches_brute_force(cluster):
    X, T = P.draw(400, cluster, P.solve_r(0.3, 0.074), P.m_of(0.074), np.random.default_rng(1))
    grid, _ = P.make_grid(P.commit_score(X))
    c, w = R8S.track_table(X, grid)
    for j in (0, 7, 15, 29):
        cb, wb = brute_commit(X, T, grid[j])
        assert np.array_equal(c[:, j], cb) and np.array_equal(w[:, j], wb)


def test_view_mix_counts_and_padding():
    X, T = P.draw(20_000, "view-mix", 0.2, 1.4, np.random.default_rng(2))
    assert set(np.unique(T)) <= {1, 2, 3, 4}
    assert (X[np.arange(4)[None, :] >= T[:, None]] == 0).all()
    assert np.mean(T == 3) == pytest.approx(816 / 848, abs=0.01)


def test_grid_loosest_first_and_fractions():
    C = P.commit_score(P.draw(20_000, "fixed-3", 0.2, 1.4, np.random.default_rng(3))[0])
    grid, f = P.make_grid(C)
    assert grid[0] == 0 and np.all(np.diff(grid) > 0) and len(grid) == 30
    for j in (0, 10, 29):
        assert np.mean(C >= grid[j]) == pytest.approx(f[j], abs=1e-3)


def test_eligibility_against_loop():
    cv = {"pop_commit_fraction": np.linspace(1.0, 0.3, 30), "true_risk": np.linspace(0.05, 0.001, 30)}
    for n, al in ((470, 0.01), (784, 0.02)):
        e = P.eligibility(cv, n, al)
        idx = [j for j in range(30) if cv["pop_commit_fraction"][j] * n * al >= 4]
        assert e["eligible_idx"] == idx
        assert e["n_above"] == sum(cv["true_risk"][j] > al for j in idx)
        assert e["n_below"] == sum(cv["true_risk"][j] < al for j in idx)


def test_small_population_runs_and_calibrates():
    cv = P.pair_curves(0, 0.45, "fixed-30", 0.074, 99, n_pop=60_000)
    assert cv["calibration"]["view_error"] == pytest.approx(0.074, abs=0.003)
    assert cv["calibration"]["wrong_icc"] == pytest.approx(0.45, abs=0.03)
    assert np.all(np.diff(cv["pop_commit_fraction"]) <= 0)


def test_design_phase_guard():
    # Hard rule 3: no torch, no data loaders, no pool-file guard, no path under data/ or runs/btsc.
    src = Path(P.__file__).read_text()
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert not any(m and (m.startswith("torch") or m in {"src.r8_stage2", "src.data", "src.embed"}) for m in mods)
    assert "data/" not in src and "runs/btsc" not in src and "open_npz" not in src
