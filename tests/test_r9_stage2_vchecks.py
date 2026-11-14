"""Tests for the R9 Stage 2 V-check runner (synthetic only; small runs)."""
import ast
import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

import src.r9_stage2_vchecks as V
from src import r9_stage2_popcurves as P

CURVES = Path("runs/20261003_stage1/poc3/stage2")


def test_v1_v3_v4_static_pass():
    assert V.v1()["pass"] and V.v1()["n_match"] == 40
    assert V.v3()["pass"]
    assert V.v4_static()["pass"]


def test_exact_binom_reference():
    for n, k, a in ((50, 2, Fraction(3, 100)), (300, 4, Fraction(2, 100))):
        assert float(V.exact_binom(n, k, a)) == pytest.approx(binom.cdf(k, n, float(a)), rel=1e-12)


def test_frame_totals_skip_padded_views():
    X, T = P.draw(500, "view-mix", 0.3, 1.4, np.random.default_rng(4))
    grid = np.array([0.0, 0.5, 1.0])
    A, W, N = V.frame_totals(X, T, grid)
    assert A[0] == T.sum() == N                                        # at lambda 0 every valid view, no padded one
    mask = np.arange(4)[None, :] < T[:, None]
    for g, lam in enumerate(grid):                                     # brute force over valid views
        acc = [(abs(X[i, t]) >= lam, X[i, t] < 0) for i in range(500) for t in range(T[i])]
        assert A[g] == sum(a for a, _ in acc) and W[g] == sum(a and w for a, w in acc)


def test_small_run_writes_summary(tmp_path):
    s = V.main(["--runs", "20", "--n-pop", "50000", "--out", str(tmp_path)])
    assert (tmp_path / "stage2_summary.json").exists()
    assert len([x for p in s["pairs"] for x in p["settings"]]) == 36
    assert all(p["grid_matches_curve_file"] for p in s["pairs"])       # same pilot seeds as the committed curves
    assert s["V4"]["v2_runs_zero_nc_certified"] == 0


def test_violation_definition_matches_true_risk():
    # mean-loss form (POC-2) > alpha  <=>  R = P(wrong | committed) > alpha
    rng = np.random.default_rng(5)
    c, R, a = rng.random(30) * 0.7 + 0.3, rng.random(30) * 0.04, 0.02
    mu = a * (1 - c) + R * c
    cert = np.arange(30)
    assert np.array_equal(V.E.violations(cert, mu, a), R > a + 1e-12 / c)


def test_changed_curve_file_is_refused(tmp_path):
    for lv in (0, 1):
        shutil.copy(CURVES / f"population_curves_L{lv}.json", tmp_path)
    d = json.loads((tmp_path / "population_curves_L1.json").read_text())
    d["seconds"] = 0.0
    (tmp_path / "population_curves_L1.json").write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="STOP"):
        V.main(["--runs", "2", "--n-pop", "50000", "--curves-dir", str(tmp_path), "--out", str(tmp_path / "o")])


def test_engine_change_during_run_blocks_summary(tmp_path, monkeypatch):
    real = V.sha256_file
    calls = {"n": 0}

    def fake(path):
        if Path(path).name == "r9_certify.py":
            calls["n"] += 1
            return real(path) if calls["n"] == 1 else "0" * 64
        return real(path)

    monkeypatch.setattr(V, "sha256_file", fake)
    with pytest.raises(SystemExit, match="changed during the run"):
        V.main(["--runs", "2", "--n-pop", "50000", "--out", str(tmp_path)])
    assert not (tmp_path / "stage2_summary.json").exists()


def test_design_phase_guard():
    for mod in (V, P):
        src = Path(mod.__file__).read_text()
        tree = ast.parse(src)
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not any(m.startswith("torch") or m == "src.r8_stage2" for m in mods)
        assert "data/" not in src and "runs/btsc" not in src
