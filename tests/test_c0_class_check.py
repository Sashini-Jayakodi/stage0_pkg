"""Synthetic tests for src/c0_class_check.py: brute-force references and a tamper test."""
import math

import numpy as np
import pytest
from scipy.stats import binom

from src import c0_class_check as C0
from src.util import sha256_file


def test_groupings_partition_and_class_6_in_others():
    for t, groups in C0.GROUPINGS.items():
        ids = sorted(c for _, cs in groups for c in cs)
        assert ids == list(range(43)), t
        others = groups[-1][1]
        assert 6 in others and not ({0, 1, 2, 3, 4, 5, 7, 8, 12, 13, 14, 17} & set(others))
    assert len(C0.GROUPINGS["Grouping 1 (plan)"]) == 6 and len(C0.GROUPINGS["Grouping 2 (merged)"]) == 3


def test_expected_names_match_head():
    from src.head import NAMES
    for c, n in C0.EXPECTED_NAMES.items():
        assert NAMES[c] == n


def test_group_index_brute_force():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 43, 500)
    for groups in C0.GROUPINGS.values():
        g = C0.group_index(y, groups)
        for i in range(len(y)):
            assert y[i] in groups[g[i]][1]


def test_group_index_refuses_bad_partition():
    with pytest.raises(AssertionError):
        C0.group_index(np.array([0]), [("a", (0, 1)), ("b", (1,))])


def brute_min(err, alpha, level, cap):
    if err >= alpha:
        return None, None
    rej = [binom.cdf(math.floor(err * m + 0.5), m, alpha) <= level for m in range(1, cap + 1)]
    if not any(rej):
        return None, None
    first = rej.index(True) + 1
    if not rej[-1]:
        return first, None
    last_bad = max((m for m in range(1, cap + 1) if not rej[m - 1]), default=0)
    return first, last_bad + 1


@pytest.mark.parametrize("err,alpha,level", [(0.0, 0.01, 0.05), (0.0, 0.05, 0.1 / 12), (0.004, 0.01, 0.05),
                                             (0.01, 0.02, 0.05), (0.02, 0.05, 0.1 / 6), (0.015, 0.02, 0.05),
                                             (0.03, 0.03, 0.05), (0.04, 0.03, 0.05)])
def test_min_committed_brute_force(err, alpha, level):
    cap = 6000
    mf, ma, note = C0.min_committed(err, alpha, level, cap)
    assert (mf, ma) == brute_min(err, alpha, level, cap)
    if ma is None:
        assert note


def test_zero_error_closed_form():
    for alpha in C0.ALPHAS:
        for level in (0.05, 0.1 / 6, 0.1 / 12):
            mf, ma, _ = C0.min_committed(0.0, alpha, level, 100_000)
            assert mf == ma == math.ceil(math.log(level) / math.log(1 - alpha))


def test_first_can_be_unstable():
    # with a positive error, rounding to k = 0 at small m can reject before the stable point
    mf, ma, _ = C0.min_committed(0.004, 0.01, 0.05, 20000)
    assert mf is not None and ma is not None and mf <= ma


def brute_rows(groups, y_pool, y_des, com, wr, alpha, n_cal):
    out = []
    for name, cs in groups:
        inn = [i for i in range(len(y_pool)) if y_pool[i] in cs]
        ng = sum(1 for i in inn if com[i])
        kg = sum(1 for i in inn if com[i] and wr[i])
        out.append({"n_pool": len(inn), "n_design": sum(1 for v in y_des if v in cs), "n_committed": ng, "errors": kg,
                    "exp_committed_784": n_cal * ng / len(y_pool)})
    return out


def test_group_rows_brute_force():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 43, 400)
    yd = rng.integers(0, 43, 100)
    com = rng.random(400) < 0.8
    wr = com & (rng.random(400) < 0.03)
    for groups in C0.GROUPINGS.values():
        rows, tot = C0.group_rows(groups, C0.group_index(y, groups), C0.group_index(yd, groups), com, wr, 0.02,
                                  len(y), 784, 0.1, 5000)
        ref = brute_rows(groups, y, yd, com, wr, 0.02, 784)
        assert tot["n_committed"] == com.sum() and tot["errors"] == wr.sum()
        for r, b in zip(rows, ref):
            for k, v in b.items():
                assert r[k] == pytest.approx(v)
            G = len(groups)
            assert r["a_level"] == pytest.approx(0.1 / (2 * G)) and r["b_level"] == 0.05
            m0b = C0.min_committed(0.0, 0.02, 0.05, 5000)[0]
            assert r["b_m0"] == m0b and r["b_feasible_zero"] == (r["exp_committed_784"] >= m0b)
            if r["n_committed"]:
                eg = r["errors"] / r["n_committed"]
                assert r["group_error"] == pytest.approx(eg)
                ma = C0.min_committed(eg, 0.02, 0.1 / (2 * G), 5000)[1]
                assert r["a_m_all"] == ma
                assert r["a_feasible_obs"] == (ma is not None and r["exp_committed_784"] >= ma)
                pooled = tot["errors"] / tot["n_committed"]
                if pooled:
                    assert r["ratio_to_pooled"] == pytest.approx(eg / pooled)


def test_group_without_committed_tracks():
    y = np.array([14, 14, 0, 0, 1])
    com = np.array([False, False, True, True, True])
    rows, _ = C0.group_rows(C0.GROUPINGS["Grouping 1 (plan)"], C0.group_index(y, C0.GROUPINGS["Grouping 1 (plan)"]),
                            np.array([], int), com, np.zeros(5, bool), 0.05, 5, 784, 0.1, 2000)
    stop = rows[0]
    assert stop["n_committed"] == 0 and stop["group_error"] is None and not stop["b_feasible_obs"]


def test_verify_sha_refuses_changed_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    h = sha256_file(p)
    assert C0.A0.verify_sha(p, h) == h
    p.write_text("{ }")
    with pytest.raises(AssertionError):
        C0.A0.verify_sha(p, h)
