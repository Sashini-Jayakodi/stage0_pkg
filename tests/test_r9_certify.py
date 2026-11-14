"""Tests for the R9 certification engine v2 (src/r9_certify.py), with brute-force references."""
import ast
import inspect
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import src.r9_certify as C
from src import r8_stage3_engine as E
from src.certify import hb_pvalue


def brute_binom(k, n, a):
    return sum(math.comb(n, i) * a ** i * (1 - a) ** (n - i) for i in range(k + 1))


def test_binom_pvalue_matches_explicit_sum():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 900))
        k = int(rng.integers(0, min(n, 40) + 1))
        a = float(rng.choice([0.005, 0.01, 0.02, 0.03, 0.05, 0.1]))
        assert float(C.binom_pvalue(k, n, a)) == pytest.approx(brute_binom(k, n, a), rel=1e-9, abs=1e-300)


@pytest.mark.parametrize("test", ["binom", "hb"])
def test_zero_committed_gives_p_one(test):
    p = C.pvalues(np.zeros(5), np.zeros(5), 470, 0.01, test)
    assert np.all(p == 1.0)


def test_hb_counts_equal_poc2_track_mu_route():
    # The comparator must be exactly POC-2's: mean shifted loss over all Cert units into src.certify.hb_pvalue.
    rng = np.random.default_rng(1)
    for _ in range(20):
        n, G, a = 470, 30, 0.02
        com = rng.random((n, G)) < np.linspace(1.0, 0.3, G)
        wr = com & (rng.random((n, G)) < 0.02)
        mu, nn = E.track_mu(com, wr, np.ones((1, n)), a)
        ref = np.where(com.sum(0) > 0, hb_pvalue(mu[0], n, a), 1.0)
        got = C.hb_pvalue_counts(wr.sum(0), com.sum(0), n, a)
        assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_power_start_matches_loop():
    rng = np.random.default_rng(2)
    for _ in range(300):
        f = np.sort(rng.random(30))[::-1]
        n, a = int(rng.choice([157, 470, 784, 23520])), float(rng.choice([0.01, 0.02]))
        ref = -1
        for j in range(30):
            if f[j] * n * a >= 4:
                ref = j
        assert C.power_start(f, n, a) == ref


def test_walk_matches_loop():
    rng = np.random.default_rng(3)
    for _ in range(500):
        p = rng.random(30) * 0.2
        s = int(rng.integers(-1, 30))
        ref, j = -1, s
        while j >= 0 and p[j] <= 0.1:
            ref, j = j, j - 1
        assert C.walk(p, s) == ref


def test_walk_boundary_is_inclusive():
    assert C.walk(np.array([0.1, 0.1]), 1) == 0
    assert C.walk(np.array([0.1, 0.1000001]), 1) == -1


def test_start_ignores_cert_outcomes():
    # power_start sees only design fractions, n and alpha; changing Cert counts changes p, never the start.
    assert list(inspect.signature(C.power_start).parameters) == ["f_design", "n", "alpha", "min_expected"]
    f, n, a = np.linspace(1.0, 0.3, 30), 470, 0.02
    s = C.power_start(f, n, a)
    n_c = np.round(f * n).astype(int)
    assert C.certify(np.zeros(30), n_c, n, a, f) == 0                  # all pass: walk runs from s down to 0
    k_bad = np.zeros(30)
    k_bad[s] = n_c[s]                                                  # only the start fails
    assert C.certify(k_bad, n_c, n, a, f) == -1                        # so the walk stops at the same start
    assert C.power_start(f, n, a) == s


def test_hb_start_hand_case_exact_arithmetic():
    # Amendment 1. n = 470, alpha = 2%, design fractions 1, 0.9, 0.7, 0.5, 0.3. Zero-error HB p-values in exact
    # arithmetic (r8_stage3_engine.exact_hb): 7.5e-5, 0.00171, 0.0395, 0.231, 0.618, so the HB start is index 2,
    # while the binomial power-aware start is index 3 (0.5 * 470 * 0.02 = 4.7 >= 4; 0.3 * 470 * 0.02 = 2.82 < 4).
    a = Fraction(1, 50)
    fs = [Fraction(1), Fraction(9, 10), Fraction(7, 10), Fraction(1, 2), Fraction(3, 10)]
    ref = max(j for j, f in enumerate(fs) if E.exact_hb(a * (1 - f), 470, a)[0] <= 0.1)
    assert ref == 2
    f = [float(x) for x in fs]
    assert C.hb_start(f, 470, 0.02) == 2 and C.start_for("hb", f, 470, 0.02) == 2
    assert C.power_start(f, 470, 0.02) == 3 and C.start_for("binom", f, 470, 0.02) == 3


def test_hb_start_matches_loop():
    rng = np.random.default_rng(6)
    for _ in range(200):
        f = np.sort(rng.random(30))[::-1]
        n, a = int(rng.choice([157, 470, 784, 23520])), float(rng.choice([0.01, 0.02]))
        ref = -1
        for j in range(30):
            if float(hb_pvalue(a * (1 - f[j]), n, a)) <= 0.1:
                ref = j
        assert C.hb_start(f, n, a) == ref


def test_hb_start_ignores_cert_outcomes():
    # hb_start sees only design fractions, n, alpha and delta; changing Cert counts changes p, never the start.
    assert list(inspect.signature(C.hb_start).parameters) == ["f_design", "n", "alpha", "delta"]
    f, n, a = np.linspace(1.0, 0.3, 30), 470, 0.02
    s = C.hb_start(f, n, a)
    assert s >= 0
    n_c = np.round(f * n).astype(int)
    assert C.certify(np.zeros(30), n_c, n, a, f, "hb") == 0           # zero errors: the walk runs from s to 0
    k_bad = np.zeros(30)
    k_bad[s] = n_c[s]                                                 # only the start fails
    assert C.certify(k_bad, n_c, n, a, f, "hb") == -1                 # so the walk stops at the same start
    k_other = np.zeros(30)
    k_other[s + 1:] = n_c[s + 1:]                                     # errors beyond the start change nothing
    assert C.certify(k_other, n_c, n, a, f, "hb") == 0
    assert C.hb_start(f, n, a) == s


def test_no_eligible_start_certifies_nothing():
    f = np.full(30, 0.5)
    assert C.power_start(f, 470, 0.01) == -1
    assert C.certify(np.zeros(30), np.full(30, 470), 470, 0.01, f) == -1


def test_max_errors_and_min_n_against_brute_force():
    for n in (45, 114, 250, 470, 784):
        for a in (0.005, 0.01, 0.02, 0.05):
            ks = [k for k in range(n + 1) if brute_binom(k, n, a) <= 0.1]
            assert C.max_errors(n, a) == (max(ks) if ks else None)
    for a in (0.005, 0.01, 0.02, 0.03, 0.05):
        assert C.min_n_zero_errors(a) == math.ceil(math.log(0.1) / math.log(1 - a))


def test_engine_has_no_io_and_no_data_imports():
    src = Path(C.__file__).read_text()
    tree = ast.parse(src)
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert mods <= {"numpy", "scipy.stats", "src.certify"}
    assert "open(" not in src and "Path" not in src
