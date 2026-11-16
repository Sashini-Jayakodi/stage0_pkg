"""r12b post hoc diagnostic: synthetic checks of its pieces and a tamper test."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from src import r11_forecast_ucb as F
from src import r12b_diagnostic as D


def test_oracle_forecast_is_r11_steps_4_to_6():
    for risk, f, rho, a, t in [(0.0103, 0.82, 0.19, 0.01, "hb"), (0.0053, 0.77, 0.12, 0.005, "binom"),
                               (0.02, 0.8, 0.0, 0.01, "binom"), (0.01, 0.8, 1.0, 0.01, "hb")]:
        o = D.oracle_forecast(risk, f, rho, a, t)
        ng = int(round(30 * f))
        pmf, clip = F.clipped_pmf(30, risk * f, rho if rho > 0 else 0.0, ng)
        assert o["N_g"] == ng and o["clipped"] == clip
        assert o["forecast"] == F.total_cdf(pmf, 784, F.k_max(784 * ng, 784 * 30, a, t))


def test_bb_pmf_limits_and_sums():
    for n in (0, 1, 7, 30):
        for rho in (0.0, 0.3, 0.999, 1.0):
            q = D.bb_pmf(n, 0.01, rho)
            assert len(q) == n + 1 and abs(q.sum() - 1) < 1e-12
    assert np.allclose(D.bb_pmf(10, 0.02, 1.0), np.r_[0.98, np.zeros(9), 0.02])
    assert np.allclose(D.bb_pmf(10, 0.02, 0.9999), D.bb_pmf(10, 0.02, 1.0), atol=1e-3)


def test_mixture_expected_against_loop():
    rng = np.random.default_rng(2)
    ns = rng.integers(0, 31, 200)
    got = D.mixture_expected(ns, 0.01, 0.2, 30)
    want = np.zeros(31)
    for n in ns:
        q = D.bb_pmf(int(n), 0.01, 0.2)
        want[: len(q)] += q
    assert np.allclose(got, want, rtol=0, atol=1e-12) and abs(got.sum() - 200) < 1e-9


def test_chi_square_merging_against_hand():
    obs = [900, 80, 15, 3, 1, 1]
    exp = [895.0, 85.0, 14.0, 4.0, 1.5, 0.5]
    cs = D.chi_square(obs, exp, 2)
    # top bins merged until expected >= 5: [895], [85], [14], [4 + 1.5 + 0.5 = 6]
    assert cs["bins_lower_edges"] == [0, 1, 2, 3] and cs["exp"] == [895.0, 85.0, 14.0, 6.0] and cs["obs"] == [900, 80, 15, 5]
    stat = (5 ** 2 / 895) + (5 ** 2 / 85) + (1 / 14) + (1 / 6)
    assert abs(cs["stat"] - stat) < 1e-12 and cs["df"] == 1 and abs(cs["p"] - chi2.sf(stat, 1)) < 1e-15


def test_chi_square_merges_small_middle_bins():
    """Small expected counts in the middle (3, 2 and 1, 1) must be merged with neighbours, not kept."""
    obs = [495, 4, 3, 9, 2, 0, 7]
    exp = [500.0, 3.0, 2.0, 8.0, 1.0, 1.0, 6.0]
    cs = D.chi_square(obs, exp, 2)
    # top down: [6] -> group at 6; 1 + 1 + 8 = 10 -> group at 3; 2 + 3 = 5 -> group at 1; 500 -> group at 0
    assert cs["bins_lower_edges"] == [0, 1, 3, 6]
    assert cs["exp"] == [500.0, 5.0, 10.0, 6.0] and cs["obs"] == [495, 7, 11, 7]
    stat = 25 / 500 + 4 / 5 + 1 / 10 + 1 / 6
    assert abs(cs["stat"] - stat) < 1e-12 and cs["df"] == 1


def test_chi_square_bottom_leftover_joins_group_above():
    cs = D.chi_square([2, 49, 3, 6], [3.0, 50.0, 2.0, 4.0], 0)
    assert cs["bins_lower_edges"] == [0, 2] and cs["exp"] == [53.0, 6.0] and cs["obs"] == [51, 9]


def test_chi_square_every_group_reaches_minimum_random():
    rng = np.random.default_rng(13)
    for _ in range(300):
        n = int(rng.integers(2, 25))
        exp = rng.exponential(3.0, n)
        if exp.sum() < 5:
            continue
        obs = rng.poisson(exp)
        cs = D.chi_square(obs, exp, 0)
        assert all(e >= 5 - 1e-12 for e in cs["exp"])
        assert abs(sum(cs["exp"]) - exp.sum()) < 1e-9 and sum(cs["obs"]) == obs.sum()
        assert cs["bins_lower_edges"][0] == 0 and cs["bins_lower_edges"] == sorted(set(cs["bins_lower_edges"]))


def test_categories_against_hand():
    k = np.array([0, 0, 1, 3, 12, 13, 25, 26])
    exp = np.full(27, 0.5)
    c = D.categories(k, exp, 25)
    assert c["k=0"][0] == 2 and c["k>=1"][0] == 6 and c["k>=N_g/2"][0] == 3 and c["k=N_g"][0] == 1
    assert c["k>N_g (observed only; the clipped model puts this mass at N_g)"][0] == 1
    assert c["k>=N_g/2"][1] == 0.5 * (27 - 13)


def test_mc_real_acceptance_matches_exact_when_acceptance_is_constant():
    """All pool tracks accept 25 of 30: the Monte Carlo must agree with the exact clipped-free forecast."""
    rng = np.random.default_rng(9)
    A_pool = np.full(500, 25)
    risk, rho = 0.009, 0.15
    mc = D.mc_forecast_real_acceptance(A_pool, risk, rho, 0.01, "binom", rng, n_sets=20_000)
    pmf = D.bb_pmf(25, risk, rho)
    ex = F.total_cdf(pmf, 784, F.k_max(784 * 25, 784 * 30, 0.01, "binom"))
    assert abs(mc - ex) <= 5 * np.sqrt(ex * (1 - ex) / 20_000) + 1e-6


def test_tampered_r12_summary_refused(tmp_path):
    src = Path("runs/20261003_stage1/r12/r12_summary.json")
    bad = tmp_path / "r12_summary.json"
    shutil.copy(src, bad)
    with open(bad, "ab") as fh:
        fh.write(b" ")
    with pytest.raises(SystemExit) as e:
        D.main(["--r12", str(bad), "--out", str(tmp_path / "out")])
    assert "SHA-256" in str(e.value) and not (tmp_path / "out").exists()
