"""r11 forecast: brute-force references for every piece, reproduction of Stage 10, tamper tests (LOG_r11_entry.md (f))."""
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from src import r11_forecast_ucb as F
from src import r9_certify as C
from src import r9_stage10 as S10
from src import r9_stage10_popcurves as Q


# ------------------------------------------------------------------ CP upper bound
def cp_upper_bisect(w, n, conf):
    """Upper bound p solves P(X <= w | n, p) = 1 - conf; P(X <= w) falls in p, so bisect."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom.cdf(w, n, mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@pytest.mark.parametrize("w,n", [(0, 10), (1, 50), (3, 784), (17, 2000), (0, 3920), (9, 10)])
def test_cp_upper_matches_bisection(w, n):
    assert abs(F.cp_upper([w], [n])[0] - cp_upper_bisect(w, n, 0.80)) < 1e-9


def test_cp_upper_edges():
    U = F.cp_upper([0, 5, 3], [0, 5, 10])
    assert U[0] == 1.0 and U[1] == 1.0 and 0.3 < U[2] < 1.0


# ------------------------------------------------------------------ ICC
def icc_bruteforce(groups):
    """Explicit one-way ANOVA on the 0/1 values themselves."""
    vals = [np.r_[np.ones(k), np.zeros(n - k)] for n, k in groups]
    allv = np.concatenate(vals)
    N, G = len(allv), len(vals)
    grand = allv.mean()
    ssb = sum(len(v) * (v.mean() - grand) ** 2 for v in vals)
    ssw = sum(((v - v.mean()) ** 2).sum() for v in vals)
    msb, msw = ssb / (G - 1), ssw / (N - G)
    n0 = (N - sum(len(v) ** 2 for v in vals) / N) / (G - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


@pytest.mark.parametrize("seed,equal", [(0, True), (1, True), (2, False), (3, False)])
def test_icc_matches_bruteforce(seed, equal):
    rng = np.random.default_rng(seed)
    ns = np.full(40, 10) if equal else rng.integers(2, 15, 40)
    groups = [(int(n), int(rng.binomial(n, rng.beta(0.3, 3)))) for n in ns]
    assert abs(F.icc_anova(groups) - icc_bruteforce(groups)) < 1e-12


# ------------------------------------------------------------------ j*, j_true, k_max
def test_jstar_and_jtrue_against_loops():
    rng = np.random.default_rng(4)
    for _ in range(300):
        U = rng.uniform(0, 0.03, 30)
        mu = rng.uniform(0, 0.03, 30)
        start = int(rng.integers(-1, 30))
        a = 0.01
        want = next((j for j in range(start, -1, -1) if U[j] > a), -1) if start >= 0 else -1
        assert F.j_star(U, start, a) == want
        want_t = next((j for j in range(start, -1, -1) if mu[j] > a + 1e-12), -1) if start >= 0 else -1
        assert F.j_true(mu, start, a) == want_t


@pytest.mark.parametrize("n_c,n,alpha,test", [(784, 784, 0.01, "binom"), (3920, 3920, 0.02, "binom"),
                                              (3136, 3920, 0.01, "hb"), (9408, 11760, 0.02, "hb"),
                                              (100, 100, 0.01, "binom"), (0, 100, 0.01, "binom")])
def test_kmax_against_scan(n_c, n, alpha, test):
    want = -1
    for k in range(n_c + 1):
        if C.pvalues(np.array([k]), np.array([n_c]), n, alpha, test)[0] <= C.DELTA:
            want = k
    assert F.k_max(n_c, n, alpha, test) == want


def test_kmax_matches_section6_table():
    """Plan R9 section 6: n_c 784 tolerates 3 errors at 1%, 10 at 2%; 300 tolerates 0 at 1%."""
    assert F.k_max(784, 784, 0.01, "binom") == 3 and F.k_max(784, 784, 0.02, "binom") == 10
    assert F.k_max(300, 300, 0.01, "binom") == 0 and F.k_max(250, 250, 0.005, "binom") == -1


# ------------------------------------------------------------------ clipped pmf and the exact total
def test_clipped_pmf_sums_and_clip_mass():
    for m, p, rho, ng in [(5, 0.01, 0.2, 4), (10, 0.02, 0.0, 10), (15, 0.008, 0.5, 11), (8, 0.3, 0.6, 2)]:
        pmf, clip = F.clipped_pmf(m, p, rho, ng)
        assert len(pmf) == ng + 1 and abs(pmf.sum() - 1) < 1e-12 and clip >= 0
        if ng == m:
            assert clip == 0.0


def test_total_cdf_enumeration_tiny():
    pmf = np.array([0.6, 0.25, 0.1, 0.05])
    for n in (1, 2, 3, 4):
        dist = {}
        for tup in itertools.product(range(4), repeat=n):
            dist[sum(tup)] = dist.get(sum(tup), 0.0) + float(np.prod(pmf[list(tup)]))
        for k in range(-1, 3 * n + 1):
            want = sum(v for s, v in dist.items() if s <= k)
            assert abs(F.total_cdf(pmf, n, k) - want) < 1e-12


def test_total_cdf_binomial_closed_form():
    """rho 0 and N_g = m (no clipping): the total is Binomial(n m, p)."""
    m, p, n = 10, 0.012, 784
    pmf, clip = F.clipped_pmf(m, p, 0.0, m)
    assert clip == 0.0
    for k in (0, 50, 80, 94, 120):
        assert abs(F.total_cdf(pmf, n, k) - binom.cdf(k, n * m, p)) < 1e-10


def test_total_cdf_against_sampling():
    """The r10 sampling model (beta then binomial, clipped at N_g), 200,000 Cert sets."""
    rng = np.random.default_rng(11)
    m, p, rho, ng, n = 10, 0.008, 0.25, 8, 60
    s = 1 / rho - 1
    tot = np.zeros(200_000)
    for c in range(0, 200_000, 20_000):
        K = rng.binomial(m, rng.beta(p * s, (1 - p) * s, size=(20_000, n)))
        tot[c:c + 20_000] = np.minimum(K, ng).sum(1)
    pmf, _ = F.clipped_pmf(m, p, rho, ng)
    for k in (0, 1, 3, 5, 8):
        emp = float(np.mean(tot <= k))
        ex = F.total_cdf(pmf, n, k)
        se = np.sqrt(ex * (1 - ex) / len(tot))
        assert abs(emp - ex) <= 5 * se + 1e-6, (k, emp, ex)


# ------------------------------------------------------------------ run forecast
def test_run_forecast_hand_case():
    """One pilot built by hand: check every step against direct formulas."""
    m, nd, nc, a = 5, 40, 784, 0.01
    G = 3
    acc = np.full((nd, G), 4)
    wr = np.zeros((nd, G), int)
    wr[0, :] = [2, 1, 0]
    wr[1, 0] = 1
    rec2 = F.run_forecast(acc, wr, m, a, "binom", 2, nd, nc)      # j* = 2 has W = 0: binomial model, rho_I 0
    assert rec2["j_star"] == 2 and rec2["rho_I"] == 0.0 and rec2["binomial_fallback"] is True
    rec = F.run_forecast(acc, wr, m, a, "binom", 1, nd, nc)
    U = F.cp_upper(wr.sum(0), acc.sum(0))
    js = max(j for j in range(2) if U[j] > a)
    assert wr[:, js].sum() > 0
    assert rec["j_star"] == js
    f = acc[:, js].sum() / (nd * m)
    assert abs(rec["p"] - U[js] * f) < 1e-15 and rec["N_g"] == round(m * f)
    rho = F.icc_anova([(m, int(k)) for k in wr[:, js]])
    assert abs(rec["rho_I"] - rho) < 1e-15
    pmf, clip = F.clipped_pmf(m, U[js] * f, max(rho, 0.0) if rho > 0 else 0.0, rec["N_g"])
    km = F.k_max(nc * rec["N_g"], nc * m, a, "binom")
    assert abs(rec["forecast"] - F.total_cdf(pmf, nc, km)) < 1e-15 and abs(rec["clipped"] - clip) < 1e-15


def test_run_forecast_zero_cases():
    acc = np.full((30, 4), 5)
    wr = np.zeros((30, 4), int)
    assert F.run_forecast(acc, wr, 5, 0.01, "binom", -1, 30, 784)["forecast"] == 0.0     # no start
    rec = F.run_forecast(acc, wr, 5, 0.5, "binom", 3, 30, 784)                          # U < alpha everywhere
    assert rec["j_star"] == -1 and rec["forecast"] == 0.0
    rec = F.run_forecast(acc, wr, 5, 0.01, "binom", 3, 30, 784)                         # W = 0: binomial model
    assert rec["j_star"] == 3 and rec["rho_I"] == 0.0 and rec["binomial_fallback"] is True


# ------------------------------------------------------------------ criteria (Part A (e))
def test_criteria_with_ties_and_edges():
    def mk(o, f, h):
        return {"rows": {"binom": {"observed": o, "forecast_mean": f, "heuristic": h,
                                   "forecast_inside_cp": False, "heuristic_inside_cp": False}}}
    st = [mk(0.10, 0.13, 0.25)] * 6 + [mk(0.10, 0.08, 0.12)] + [mk(0.10, 0.07, 0.25)]
    s = F.set_summary(st, "binom")
    # gaps 3, 3, 3, 3, 3, 3, 2, 3 -> MAE 2.875; tie at row 7 (|-2| vs |+2|) is not closer; row 8 is 3 below
    assert abs(s["mae_forecast_pp"] - 2.875) < 1e-9
    assert s["n_forecast_closer"] == 7 and s["n_forecast_not_below_obs_minus_2"] == 7
    assert F.criteria(s) == {"a_mae_le_3": True, "b_closer_ge_7_of_8": True, "c_not_below_ge_7_of_8": True, "all": True}
    st2 = st[:6] + [mk(0.10, 0.07, 0.25)] * 2
    c2 = F.criteria(F.set_summary(st2, "binom"))
    assert c2["c_not_below_ge_7_of_8"] is False and c2["all"] is False


def test_held_heuristic_table():
    rows = F.check_held_heuristic()
    assert len(rows) == 4 and all(abs(100 * r["delta_eff"] - r["table_pct"]) <= 0.005 for r in rows)


def test_seed_keys_are_new():
    """Part A (c): base 20261118 differs from every earlier base (r9: 20261114/5, r10: 20261116/7)."""
    assert F.SEED_NEW not in (20261114, 20261115, 20261116, 20261117) and Q.SEED == 20261115


# ------------------------------------------------------------------ reproduction of Stage 10 and of the curve builder
def test_b1_curves_equal_stage10_builder():
    pair, (rho, m) = 4, Q.PAIRS[4]
    mine = F.b1_curves(rho, m, lambda s: Q.rng_for(pair, s), n_pop=60_000)
    ref = Q.pair_curves(pair, rho, m, n_pop=60_000)
    assert np.array_equal(mine["b1_grid"], ref["b1_grid"])
    assert np.array_equal(mine["b1_pop_accept"], ref["b1_pop_accept"])
    assert np.array_equal(mine["b1_true_risk"], ref["b1_true_risk"], equal_nan=True)
    assert mine["calibration"] == ref["calibration"]


def test_simulate_reproduces_stage10_b1():
    curves = json.loads(Path(F.CURVES_PATH).read_text())
    for pair in (0, 7):
        cv = curves["curves"][pair]
        m = cv["m"]
        runs = 4
        acc = np.asarray(cv["b1_pop_accept"], float)
        risk = np.array([np.nan if v is None else v for v in cv["b1_true_risk"]])
        mu_true = {a: S10.mean_loss(a, acc, risk) for a in Q.ALPHAS}
        mine, fc = F.simulate(Q.rng_for(pair, 2), m, cv["r_latent"], cv["mu"], np.asarray(cv["b1_grid"]),
                              Q.N_DESIGN, Q.N_CERT, runs, Q.ALPHAS, mu_true)
        ref, *_ = S10.simulate(Q.rng_for(pair, 2), m, cv["r_latent"], cv["mu"], np.asarray(cv["sign_grid"]),
                               np.asarray(cv["b1_grid"]), Q.N_DESIGN, Q.N_CERT, runs, Q.ALPHAS)
        for a in Q.ALPHAS:
            for t in F.TESTS:
                assert np.array_equal(mine[(a, t)], ref[(a, "B1", t)])
                assert len(fc[(a, t)]) == runs and all(0.0 <= r["forecast"] <= 1.0 for r in fc[(a, t)])


# ------------------------------------------------------------------ tamper tests (hard rule 2)
@pytest.mark.parametrize("which", ["curves", "stage10"])
def test_tampered_inputs_refused(tmp_path, which):
    src = Path(F.CURVES_PATH if which == "curves" else F.STAGE10_PATH)
    bad = tmp_path / src.name
    shutil.copy(src, bad)
    with open(bad, "ab") as fh:
        fh.write(b" ")
    args = ["--out", str(tmp_path / "out"), f"--{which}", str(bad)]
    with pytest.raises(SystemExit) as e:
        F.main(args)
    assert "SHA-256" in str(e.value)
    assert not (tmp_path / "out").exists()


def test_main_smoke(tmp_path):
    """End to end with 2 runs and a small population: summary written, check skipped (runs != 2,000)."""
    s = F.main(["--runs", "2", "--n-pop", "50000", "--out", str(tmp_path)])
    assert (tmp_path / "r11_summary.json").exists() and (tmp_path / "r11_report.md").exists()
    assert len(s["settings"]["development"]) == 18 and len(s["settings"]["replication"]) == 18
    assert len(s["settings"]["held-out"]) == 8
    assert set(s["held_out_criteria"]) == {"a_mae_le_3", "b_closer_ge_7_of_8", "c_not_below_ge_7_of_8", "all"}
