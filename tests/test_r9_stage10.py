"""Stage 10 runner: reproduction of the Stage 2 run loop, agreement rule, retrospective table, tamper tests."""
import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import r8_stage3_engine as E
from src import r9_stage10 as S
from src import r9_stage10_popcurves as Q
from src import r9_stage2_popcurves as P
from src import r9_stage2_vchecks as V


# ------------------------------------------------------------------ reproduction of Stage 2 (fixed-3 and fixed-30)
@pytest.mark.parametrize("pair,rho,cluster,T", [(3, 0.3, "fixed-3", 3), (8, 0.45, "fixed-30", 30)])
def test_run_loop_reproduces_stage2(pair, rho, cluster, T):
    """With Stage 2's seeds, sizes and grids, the Stage 10 loop gives exactly Stage 2's rows (level 0)."""
    runs, n_pop = 15, 20_000
    r, mu = P.solve_r(rho, Q.P_FRAME), P.m_of(Q.P_FRAME)
    ref = V.run_pair(0, pair, rho, cluster, Q.P_FRAME, runs, n_pop, {"grid": [0.0], "true_risk": [None]})
    Xp, Tp = P.draw(P.N_PILOT, cluster, r, mu, P.rng_for(0, pair, 0))
    sgrid, _ = P.make_grid(P.commit_score(Xp))
    bgrid = V.b1_grid(Xp, Tp)
    assert np.array_equal(sgrid, ref["sign_grid"]) and np.array_equal(bgrid, ref["b1_grid"])
    pop = V.population(0, pair, cluster, r, mu, sgrid, bgrid, n_pop)
    cert, _, no_start, zero = S.simulate(P.rng_for(0, pair, 2), T, r, mu, sgrid, bgrid,
                                         V.N_DESIGN[cluster], P.N_CERT[cluster], runs, P.ALPHAS)
    assert zero == ref["zero_nc_certified"]
    for st in ref["settings"]:
        a = st["alpha"]
        mus = {"sign": a * (1 - pop["sign_c"]) + pop["sign_w"], "B1": a * (1 - pop["b1_c"]) + pop["b1_w"]}
        for u in S.UNITS:
            for t in S.TESTS:
                mine = E.rate_row(cert[(a, u, t)], mus[u], a)
                want = st["rows"][f"{u}_{t}"]
                for key in ("certified_share", "violations", "violation_rate", "cp95", "mean_true_risk_certified"):
                    assert mine[key] == want[key], (a, u, t, key)
                assert no_start[(a, u, t)] == want["runs_no_eligible_start"]


def test_mean_loss_equals_stage2_form():
    rng = np.random.default_rng(1)
    com = rng.integers(1, 10_000, 30).astype(float)
    wr = np.floor(com * rng.uniform(0, 0.1, 30))
    n = 20_000
    for a in (0.01, 0.02):
        stage2 = a * (1 - com / n) + wr / n
        assert np.allclose(S.mean_loss(a, com / n, wr / com), stage2, rtol=0, atol=1e-15)


# ------------------------------------------------------------------ agreement rule and summary
def test_agreement_and_summary_against_loop():
    rng = np.random.default_rng(5)
    rows = []
    for _ in range(40):
        pred, obs = rng.uniform(0.05, 0.4), rng.uniform(0.0, 0.45)
        lo = max(0.0, obs - rng.uniform(0, 0.03))
        rows.append({"predicted": pred, "observed": obs, "agree": S.agree(lo, pred), "lo": lo})
    for r in rows:
        assert r["agree"] == (r["lo"] <= r["predicted"])
    s = S.summarise(rows)
    n_agree = 0
    abs_sum = signed_sum = 0.0
    for r in rows:
        n_agree += r["lo"] <= r["predicted"]
        abs_sum += abs(100 * r["observed"] - 100 * r["predicted"])
        signed_sum += 100 * r["observed"] - 100 * r["predicted"]
    assert s["n"] == 40 and s["n_agree"] == n_agree
    assert abs(s["mean_abs_gap_pp"] - abs_sum / 40) < 1e-9 and abs(s["mean_signed_gap_pp"] - signed_sum / 40) < 1e-9
    assert S.summarise([])["n"] == 0


def test_retrospective_predictions_hand_values():
    assert abs(100 * S.de_to_delta_eff(13.9) - 36.5521) < 1e-3
    assert abs(100 * S.de_to_delta_eff(2.24) - 19.5923) < 1e-3
    assert abs(100 * S.de_to_delta_eff(1.25) - 12.5845) < 1e-3 and abs(100 * S.de_to_delta_eff(2.74) - 21.9402) < 1e-3
    assert abs(S.STAGE2_M["view-mix"] - 7550 / 2518) < 1e-12
    for rho, m in Q.PAIRS:
        assert abs(S.de_to_delta_eff(Q.design_effect(rho, m)) - Q.delta_eff(rho, m)) < 1e-15


def test_retrospective_rows():
    rows, sha = S.retrospective()
    assert len(rows) == 36 * 2 + 5 + 5 * 2          # all 36 Stage 2 settings x 2 rules; 3b 5 alphas; Stage 6 5 x 2
    g = [r for r in rows if r["source"].startswith("GTSRB") and r["alpha"] == 0.01][0]
    assert g["observed"] == 0.2
    b = [r for r in rows if r["source"].startswith("Belgium") and r["alpha"] == 0.005 and r["rule"] == "binom"][0]
    assert b["observed"] == 0.14
    assert set(sha) == {"stage2_summary.json", "stage3b_summary.json", "stage6_summary.json"}


# ------------------------------------------------------------------ tamper tests
def test_changed_curve_file_is_refused(tmp_path):
    d = json.loads(S.CURVES_PATH.read_text())
    d["config"]["n_pop"] = 1
    bad = tmp_path / Q.CURVES_NAME
    bad.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="STOP"):
        S.main(["--runs", "1", "--n-pop", "20000", "--curves", str(bad), "--out", str(tmp_path / "o")])


@pytest.mark.parametrize("key", ["stage2", "gtsrb_3b", "btsc_6"])
def test_changed_retrospective_source_is_refused(tmp_path, monkeypatch, key):
    path, sha = S.RETRO[key]
    copy = tmp_path / path.name
    shutil.copy(path, copy)
    with open(copy, "ab") as f:
        f.write(b" ")
    monkeypatch.setitem(S.RETRO, key, (copy, sha))
    with pytest.raises(SystemExit, match="STOP"):
        S.retrospective()


def test_engine_change_during_run_blocks_summary(tmp_path, monkeypatch):
    real = S.sha256_file
    calls = {"n": 0}

    def fake(path):
        if Path(path).name == "r9_certify.py":
            calls["n"] += 1
            return real(path) if calls["n"] == 1 else "0" * 64
        return real(path)

    monkeypatch.setattr(S, "sha256_file", fake)
    with pytest.raises(SystemExit, match="changed during the run"):
        S.main(["--runs", "2", "--n-pop", "20000", "--out", str(tmp_path)])
    assert not (tmp_path / "stage10_summary.json").exists()


def test_small_run(tmp_path):
    s = S.main(["--runs", "3", "--n-pop", "20000", "--out", str(tmp_path)])
    data = json.loads((tmp_path / "stage10_summary.json").read_text())
    sets = [x for pr in data["pairs"] for x in pr["settings"]]
    assert len(sets) == 18 and data["zero_nc_certified"] == 0
    assert all(x["boundary"] for x in sets)                  # committed boundary subset: 18 of 18
    assert s["summary"]["all"]["n"] == 18 and s["summary"]["boundary"]["n"] == 18
    for x in sets:
        assert abs(x["predicted"] - Q.delta_eff(x["rho"], x["m"])) < 1e-15
        assert x["agree"] == (x["cp95"][0] <= x["predicted"])


def test_no_dataset_path():
    for mod in (S, Q):
        src = Path(mod.__file__).read_text()
        tree = ast.parse(src)
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not any(m.startswith("torch") or m == "src.r8_stage2" for m in mods)
        assert "data/" not in src
