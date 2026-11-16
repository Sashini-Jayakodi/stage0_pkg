"""r13: synthetic tests (LOG_r13_entry.md Part A (f)). No BelgiumTSC or GTSRB image, embedding or track table is read;
the tamper tests read only committed summary files."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import r11_forecast_ucb as F
from src import r13_forecast as R
from src import r9_certify as C


def synth(seed, nd=150, m=8, G=12, equal=True):
    rng = np.random.default_rng(seed)
    sizes = np.full(nd, m) if equal else rng.integers(1, 7, nd)
    acc = rng.binomial(sizes[:, None], np.linspace(0.95, 0.4, G)[None, :])
    p = rng.beta(0.3, 30, nd)
    wr = rng.binomial(acc, np.clip(p[:, None] * np.linspace(1, 0.3, G)[None, :], 0, 1))
    return sizes, acc, wr


# ------------------------------------------------------------------ bootstrap pieces
@pytest.mark.parametrize("equal", [True, False])
def test_icc_batch_matches_r11_icc_loop(equal):
    sizes, _, wr = synth(1, equal=equal)
    idx = np.random.default_rng(2).integers(0, len(sizes), (40, len(sizes)))
    got = R.icc_batch(sizes[idx], wr[idx, 3])
    for b in range(40):
        k = wr[idx[b], 3]
        want = 0.0 if k.sum() == 0 else F.icc_anova([(int(n), int(c)) for n, c in zip(sizes[idx[b]], k)])
        assert abs(got[b] - want) < 1e-12


def test_icc_batch_zero_error_rows_are_zero():
    n = np.full((3, 10), 5)
    k = np.zeros((3, 10), int)
    k[1, 2] = 1
    out = R.icc_batch(n, k)
    assert out[0] == 0.0 and out[2] == 0.0 and np.isfinite(out[1])


def test_boot_upper_against_loop():
    sizes, _, wr = synth(3, equal=False)
    idx = np.random.default_rng(4).integers(0, len(sizes), (1000, len(sizes)))
    bound, vals = R.boot_upper(sizes, wr[:, 2], idx)
    loop = []
    for b in range(1000):
        k = wr[idx[b], 2]
        v = 0.0 if k.sum() == 0 else F.icc_anova([(int(n), int(c)) for n, c in zip(sizes[idx[b]], k)])
        loop.append(max(v, 0.0))
    assert np.allclose(vals, loop, rtol=0, atol=1e-12)
    assert abs(bound - np.percentile(loop, 80)) < 1e-12


# ------------------------------------------------------------------ the forecast
@pytest.mark.parametrize("seed,alpha,test", [(5, 0.01, "binom"), (6, 0.02, "hb"), (7, 0.005, "binom"), (8, 0.01, "hb")])
def test_r11_rule_equals_r11_run_forecast_for_equal_sizes(seed, alpha, test):
    sizes, acc, wr = synth(seed, nd=261, m=10, G=20)
    rec = R.forecast(acc, wr, sizes, 10, alpha, test, 784, "r11")
    start = C.start_for(test, acc.sum(0) / (261 * 10), 784 * 10, alpha, 261 * 10)
    ref = F.run_forecast(acc, wr, 10, alpha, test, start, 261, 784)
    assert rec["j_star"] == ref["j_star"] and rec["forecast"] == ref["forecast"] and rec["N_g"] == ref["N_g"]
    if ref["rho_I"] is not None:
        assert rec["rho_point"] == ref["rho_I"]


def test_r13_equals_r11_when_bound_is_the_point(monkeypatch):
    sizes, acc, wr = synth(9, nd=261, m=10, G=20)
    monkeypatch.setattr(R, "boot_upper", lambda s, k, idx, q=80.0: (F.icc_anova([(int(a), int(b)) for a, b in zip(s, k)]), None))
    for a in (0.005, 0.01, 0.02):
        for t in ("binom", "hb"):
            r11 = R.forecast(acc, wr, sizes, 10, a, t, 784, "r11")
            r13 = R.forecast(acc, wr, sizes, 10, a, t, 784, "r13", np.zeros((2, 261), int))
            assert r13["forecast"] == r11["forecast"]


def test_r13_uses_the_bootstrap_bound():
    """The r13 rule feeds the 80th-percentile bootstrap bound (not the point rho_I) into the model."""
    sizes, acc, wr = synth(10, nd=261, m=10, G=20)
    idx = np.random.default_rng(11).integers(0, 261, (1000, 261))
    for a in (0.01, 0.02):
        r11 = R.forecast(acc, wr, sizes, 10, a, "binom", 784, "r11")
        r13 = R.forecast(acc, wr, sizes, 10, a, "binom", 784, "r13", idx)
        if r11["j_star"] >= 0 and r11["case"] != "zero_errors":
            assert r13["rho_in"] == R.boot_upper(sizes, wr[:, r11["j_star"]], idx)[0]


def test_unequal_sizes_use_real_sizes_for_f_and_rho():
    sizes, acc, wr = synth(12, nd=133, m=4, G=15, equal=False)
    rec = R.forecast(acc, wr, sizes, 4, 0.01, "binom", 397, "r11")
    if rec["j_star"] >= 0:
        js = rec["j_star"]
        assert abs(rec["f"] - acc[:, js].sum() / sizes.sum()) < 1e-15
        if wr[:, js].sum() > 0:
            assert rec["rho_point"] == F.icc_anova([(int(n), int(k)) for n, k in zip(sizes, wr[:, js])])


def test_m_star_hand_value():
    assert abs(R.m_star([1, 3, 3, 4]) - 35 / 11) < 1e-15 and R.m_star([5, 5, 5]) == 5.0


# ------------------------------------------------------------------ criteria (Part A (d)), edges inclusive
def test_crit_btsc_edges():
    assert R.crit_btsc(0.12, 0.14)["pass"] and R.crit_btsc(0.19, 0.14)["pass"]
    assert not R.crit_btsc(0.1199, 0.14)["pass"] and not R.crit_btsc(0.1901, 0.14)["pass"]
    assert R.crit_btsc(0.066, 0.086)["pass"] and R.crit_btsc(0.136, 0.086)["pass"]


def test_crit_sim_hand_cases():
    o = 0.10
    f = np.r_[np.full(80, 0.08), np.full(20, 0.05)]          # 80% at exactly O - 2 pp
    c = R.crit_sim(f, o)
    assert c["share_not_below"] == 0.8 and c["i"] and abs(c["mean_gap_pp"] - (-2.6)) < 1e-9 and c["ii"] and c["both"]
    f2 = np.r_[np.full(79, 0.08), np.full(21, 0.05)]
    assert not R.crit_sim(f2, o)["i"]
    assert R.crit_sim(np.full(10, 0.15), o)["ii"] and not R.crit_sim(np.full(10, 0.1501), o)["ii"]


def test_heuristic_table():
    R.check_heuristics()


def test_seed_base_is_new():
    assert R.SEED not in (20261114, 20261115, 20261116, 20261117, 20261118, 20261119)


def test_sim_pair_smoke_on_a_test_only_pair_index():
    """Pair index 99 is not one of the run's pairs (0..3), so no run seed is touched."""
    st, cal = R.sim_pair(99, 0.25, 6, 3, 50_000)
    assert len(st) == 3 and all(set(s["rows"]) == {"binom", "hb"} for s in st)
    for s in st:
        for t in ("binom", "hb"):
            for r in ("r11", "r13"):
                assert 0.0 <= s["rows"][t][r]["mean"] <= 1.0


# ------------------------------------------------------------------ tamper tests (hard rule 2)
@pytest.mark.parametrize("which,src", [("stage6", "runs/btsc/stage6/stage6_summary.json"),
                                       ("r12", "runs/20261003_stage1/r12/r12_summary.json")])
def test_tampered_inputs_refused(tmp_path, which, src):
    bad = tmp_path / Path(src).name
    shutil.copy(src, bad)
    with open(bad, "ab") as fh:
        fh.write(b" ")
    with pytest.raises(SystemExit) as e:
        R.main([f"--{which}", str(bad), "--out", str(tmp_path / "out")])
    assert "SHA-256" in str(e.value) and not (tmp_path / "out").exists()
