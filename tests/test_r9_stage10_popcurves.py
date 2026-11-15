"""Stage 10 step 2: delta_eff, the fixed-T draw, the boundary diagnostic and a small population run (synthetic)."""
import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import r9_stage10_popcurves as Q
from src import r9_stage2_popcurves as P
from src import r9_stage2_vchecks as V


# ------------------------------------------------------------------ brute-force normal CDF and quantile (no scipy)
def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_inv(p, lo=-10.0, hi=10.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if phi(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("rho,m", [(0.1, 5), (0.2, 10), (0.6, 15), (0.45, 30), (0.3, 3)])
def test_delta_eff_against_brute_force(rho, m):
    ref = phi(phi_inv(0.1) / math.sqrt(1 + (m - 1) * rho))
    assert abs(Q.delta_eff(rho, m) - ref) < 1e-9


def test_delta_eff_independent_frames_is_delta():
    assert abs(Q.delta_eff(0.0, 15) - 0.1) < 1e-12
    assert abs(Q.delta_eff(0.6, 1) - 0.1) < 1e-12


def test_predictions_match_part_a_table():
    rows = Q.check_predictions()
    assert len(rows) == 9 and all(r["ok"] for r in rows)
    assert [(r["rho"], r["m"]) for r in rows] == Q.PAIRS


def test_changed_prediction_is_refused(monkeypatch):
    monkeypatch.setitem(Q.PREDICTED_PCT, (0.2, 10), 22.20)     # 0.012 pp away from 22.1876
    with pytest.raises(SystemExit):
        Q.check_predictions()


@pytest.mark.parametrize("cluster,T", [("fixed-3", 3), ("fixed-30", 30)])
@pytest.mark.parametrize("rho", [0.0, 0.45])
def test_fixed_draw_reproduces_stage2_draw(cluster, T, rho):
    r, mu = P.solve_r(rho, Q.P_FRAME), P.m_of(Q.P_FRAME)
    X2, T2 = P.draw(500, cluster, r, mu, np.random.default_rng(7))
    X10 = Q.draw_fixed(500, T, r, mu, np.random.default_rng(7))
    assert np.array_equal(T2, np.full(500, T))
    assert np.array_equal(X2, X10)


def test_boundary_against_loop():
    rng = np.random.default_rng(3)
    for _ in range(300):
        G = 30
        accept = np.sort(rng.uniform(0.05, 1.0, G))[::-1]
        accept[0] = 1.0
        risk = rng.uniform(0.0, 0.06, G)
        n, alpha = int(rng.integers(500, 12000)), float(rng.choice([0.01, 0.02]))
        got = Q.boundary(accept, risk, n, alpha)
        start = max([j for j in range(G) if accept[j] * n * alpha >= 4], default=-1)
        first = next((j for j in range(start, -1, -1) if risk[j] > alpha), None)
        assert got["start"] == start and got["first_above_idx"] == first
        if first is None:
            assert got["R_b"] is None and not got["boundary"]
        else:
            assert got["R_b"] == risk[first] and got["boundary"] == (risk[first] <= 2 * alpha)


def test_boundary_hand_case():
    accept = np.array([1.0, 0.8, 0.6, 0.4])
    risk = np.array([0.074, 0.025, 0.008, 0.002])
    b = Q.boundary(accept, risk, 1000, 0.01)          # start: 0.4 * 1000 * 0.01 = 4 -> index 3
    assert b["start"] == 3 and b["first_above_idx"] == 1 and abs(b["R_b"] - 0.025) < 1e-15 and not b["boundary"]
    b = Q.boundary(accept, risk, 1000, 0.02)          # 2 x alpha = 4%: 2.5% is in the subset
    assert b["first_above_idx"] == 1 and b["boundary"]


def test_small_population_run(tmp_path):
    s = Q.main(["--n-pop", "20000", "--out", str(tmp_path)])
    data = json.loads((tmp_path / Q.CURVES_NAME).read_text())
    assert len(data["curves"]) == 9 and len(data["settings"]) == 18
    for cv in s["curves"]:
        assert len(cv["b1_grid"]) == P.G and cv["b1_grid"][0] == 0 and cv["sign_grid"][0] == 0
        assert cv["b1_pop_accept"][0] == 1.0 and cv["sign_pop_commit"][0] == 1.0
        assert abs(cv["calibration"]["view_error"] - Q.P_FRAME) < 0.01
        assert abs(cv["calibration"]["wrong_icc"] - cv["rho"]) < 0.05
    assert data["config"]["seed_key"] == "[20261115, 10, pair, stream]"


def test_b1_grid_is_stage2_rule():
    r, mu, sgrid, bgrid = Q.grids(4, 0.2, 10)
    Xp = Q.draw_fixed(P.N_PILOT, 10, r, mu, Q.rng_for(4, 0))
    assert np.array_equal(bgrid, P.make_grid(np.abs(Xp).ravel())[0])
    assert np.array_equal(bgrid, V.b1_grid(Xp, np.full(P.N_PILOT, 10)))


def test_design_phase_guard():
    src = Path(Q.__file__).read_text()
    tree = ast.parse(src)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m.startswith("torch") or m == "src.r8_stage2" for m in mods)
    assert "data/" not in src and "runs/btsc" not in src
