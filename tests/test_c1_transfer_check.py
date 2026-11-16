"""Synthetic tests for src/c1_transfer_check.py: brute-force references and a tamper test."""
import math

import pytest
from scipy.stats import beta

from src import c1_transfer_check as C1
from src.util import sha256_file


def brute_bound(k, ng, T, n, a):
    kN, wN = math.floor(k / T * n + 0.5), math.floor(ng / T * n + 0.5)
    up = 1.0 if kN >= n else beta.ppf(1 - a, kN + 1, n - kN)
    lo = 0.0 if wN <= 0 else beta.ppf(a, wN, n - wN + 1)
    return (up / lo if lo > 0 else float("inf")), kN, wN


@pytest.mark.parametrize("k,ng", [(0, 500), (4, 300), (11, 160), (30, 520), (2, 0), (7, 1046)])
def test_transfer_bound_brute_force(k, ng):
    U, kN, wN, _, _ = C1.transfer_bound(k, ng)
    Ub, kb, wb = brute_bound(k, ng, 1046, 784, 0.025)
    assert (kN, wN) == (kb, wb)
    assert U == pytest.approx(Ub, rel=1e-12) if math.isfinite(Ub) else U == Ub


def test_zero_errors_upper_closed_form():
    _, _, _, up, _ = C1.transfer_bound(0, 500)
    assert up == pytest.approx(1 - 0.025 ** (1 / 784), abs=1e-12)


@pytest.mark.parametrize("alpha,wN,nc", [(0.01, 600, 900), (0.02, 240, 960), (0.05, 120, 1000), (0.01, 120, 1000),
                                         (0.03, 400, 980)])
def test_max_pooled_error_brute_force(alpha, wN, nc):
    lo = beta.ppf(0.025, wN, 784 - wN + 1)
    acc = [x for x in range(785) if (1.0 if x >= 784 else beta.ppf(0.975, x + 1, 784 - x)) < alpha * lo]
    xm, em = C1.max_pooled_error(alpha, wN, nc)
    if not acc:
        assert xm is None and em is None
        return
    assert xm == max(acc)
    # e just below e_max maps to k_N <= x_max; at e_max it maps to x_max + 1
    ec = nc / 1046
    assert math.floor((em - 1e-9) * ec * 784 + 0.5) == xm
    assert math.floor(em * ec * 784 + 0.5) == xm + 1


def test_transfer_accepts_iff_kN_le_xmax():
    for k in range(0, 40, 3):
        for ng in (150, 300, 500):
            for alpha in C1.ALPHAS:
                U, kN, wN, _, _ = C1.transfer_bound(k, ng)
                xm, _ = C1.max_pooled_error(alpha, wN, 950)
                assert (U < alpha) == (xm is not None and kN <= xm)


def test_verify_sha_refuses_changed_file(tmp_path):
    p = tmp_path / "c0_summary.json"
    p.write_text("{}")
    h = sha256_file(p)
    assert C1.A0.verify_sha(p, h) == h
    p.write_text("{ }")
    with pytest.raises(AssertionError):
        C1.A0.verify_sha(p, h)
