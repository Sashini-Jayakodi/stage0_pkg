"""Synthetic tests for src/a0_silent_error_check.py: brute-force references and a tamper test."""
import math

import numpy as np
import pytest
from scipy.stats import beta, binom

from src import a0_silent_error_check as A
from src.util import sha256_file


def brute_consensus(rows, top, margin, K):
    out = []
    for r in rows:
        best = None
        for k in range(K):
            n = sum(1 for i in r if top[i] == k)
            msum = sum(margin[i] for i in r if top[i] == k)
            key = (n, msum, -k)
            if n and (best is None or key > best[0]):
                best = (key, k)
        out.append(best[1])
    return np.array(out)


def test_consensus_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        K, n = 4, 60
        top = rng.integers(0, K, n)
        margin = rng.integers(0, 3, n).astype(float)     # integer margins force exact ties too
        cuts = np.sort(rng.choice(np.arange(1, n), 9, replace=False))
        rows = np.split(np.arange(n), cuts)
        assert (A.consensus(rows, top, margin, K) == brute_consensus(rows, top, margin, K)).all()


def test_tie_broken_by_margin_then_id():
    top = np.array([0, 0, 1, 1]); rows = [np.arange(4)]
    assert A.consensus(rows, top, np.array([1., 1., 3., 0.]), 2)[0] == 1
    assert A.consensus(rows, top, np.array([1., 1., 2., 0.]), 2)[0] == 0


def test_decompose_brute_force_and_identity():
    rng = np.random.default_rng(1)
    n = 500
    c = rng.random(n) < 0.7
    y = rng.integers(0, 3, n); yhat = rng.integers(0, 3, n)
    lab = np.where(c, rng.integers(0, 3, n), -1)
    e, v, s = A.decompose(c, lab, yhat, y)
    for i in range(n):
        assert e[i] == (c[i] and lab[i] != y[i])
        assert v[i] == (c[i] and lab[i] != yhat[i])
        assert s[i] == (c[i] and lab[i] == yhat[i] and yhat[i] != y[i])
    ok = yhat == y
    assert (e[ok] == v[ok]).all() and (e <= (v | s)).all()


def test_committed_labels_first_frame_at_or_above_threshold():
    seqs = [(np.array([-np.inf, 0.5, 2.0, 1.0]), np.array([-1, 3, 4, 5])),
            (np.array([0.1, 0.2]), np.array([1, 2]))]
    c, lab = A.committed_labels(seqs, 1.5)
    assert c.tolist() == [True, False] and lab.tolist() == [4, -1]
    c, lab = A.committed_labels(seqs, -np.inf)
    assert lab.tolist() == [3, 1]


def test_cp_bounds():
    assert A.cp_upper(0, 50, 0.1) == pytest.approx(1 - 0.1 ** (1 / 50), abs=1e-12)
    assert A.cp_lower(0, 50, 0.1) == 0.0 and A.cp_upper(50, 50, 0.1) == 1.0
    assert A.cp_lower(50, 50, 0.1) == pytest.approx(0.1 ** (1 / 50), abs=1e-12)
    # coverage identity: P(X <= x | p = upper) = a
    u = A.cp_upper(3, 40, 0.05)
    assert binom.cdf(3, 40, u) == pytest.approx(0.05, abs=1e-9)


def brute_n_audit(q, gap, delta, n_max):
    if gap <= q:
        return None
    for n in range(1, n_max + 1):
        x = math.ceil(q * n - 1e-12)
        ub = 1.0 if x >= n else beta.ppf(1 - delta / 3, x + 1, n - x)
        if ub < gap:
            return n
    return None


@pytest.mark.parametrize("q,gap", [(0.0, 0.01), (0.001, 0.008), (0.002, 0.004), (0.005, 0.005), (0.0, 0.0005)])
def test_n_audit_brute_force(q, gap):
    got, note = A.n_audit(q, gap, 0.1, n_max=3000)
    assert got == brute_n_audit(q, gap, 0.1, 3000)
    if got is None:
        assert note


def brute_n_full(Ec, err, alpha, delta, n_max):
    for n in range(1, n_max + 1):
        nc = math.floor(Ec * n + 0.5)
        k = math.floor(err * nc + 0.5)
        if nc > 0 and binom.cdf(k, nc, alpha) <= delta:
            return n
    return None


@pytest.mark.parametrize("Ec,err,alpha", [(0.9, 0.0, 0.01), (0.8, 0.004, 0.01), (0.95, 0.01, 0.02), (0.5, 0.02, 0.05)])
def test_n_full_brute_force(Ec, err, alpha):
    assert A.n_full(Ec, err, alpha, 0.1, n_max=20000)[0] == brute_n_full(Ec, err, alpha, 0.1, 20000)


def test_n_full_impossible_when_error_at_alpha():
    n, note = A.n_full(0.9, 0.05, 0.05, 0.1, n_max=2000)
    assert n is None and note


def test_gap_formula():
    g, cN, vN = A.gap_value(0.02, 0.9, 0.003, N=784, delta=0.1)
    assert (cN, vN) == (706, 2)
    ref = 0.02 * beta.ppf(0.1 / 3, 706, 784 - 706 + 1) - beta.ppf(1 - 0.1 / 3, 3, 782)
    assert g == pytest.approx(ref, abs=1e-12)


def test_verify_sha_refuses_changed_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    h = sha256_file(p)
    assert A.verify_sha(p, h) == h
    p.write_text("{ }")
    with pytest.raises(AssertionError):
        A.verify_sha(p, h)
