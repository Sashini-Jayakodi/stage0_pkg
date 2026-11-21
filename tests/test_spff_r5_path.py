import itertools
import numpy as np
import pytest
from scipy.stats import binom

from src import r9_certify as R9
from src.spff_r5_path import (
    binomial_lower_tail, exact_path, pilot_confidence_interval,
    selected_threshold, validate,
)


def example():
    sig = np.array([
        [[0, 0], [1, 0], [1, 0]],
        [[1, 1], [1, 1], [1, 0]],
        [[1, 0], [0, 0], [0, 0]],
    ], dtype=int)
    return sig, np.array([0.50, 0.20, 0.30])


def reference(sig, q, n, alpha, delta, start=None):
    """Enumerate with frozen R9; do not reuse R5 selection or excess helpers."""
    g = sig.shape[1]
    start = g - 1 if start is None else start
    out = np.zeros(g + 1)
    for seq in itertools.product(range(len(q)), repeat=n):
        mass = np.prod(q[list(seq)])
        counts = sig[list(seq)].sum(axis=0)
        p = R9.binom_pvalue(counts[:, 1], counts[:, 0], alpha)
        chosen = R9.walk(p, start, delta)
        out[chosen + 1] += mass
    a, w = q @ sig[:, :, 0], q @ sig[:, :, 1]
    excess = np.maximum(w - alpha * a, 0)
    return out, float(out[1:] @ excess), float(
        out[1:] @ ((a > 0) & (w > alpha * a))
    )


@pytest.mark.parametrize("n", [0, 2, 4, 6])
@pytest.mark.parametrize("start", [None, -1, 0, 1, 2])
def test_exact_matches_direct_enumeration(n, start):
    sig, q = example()
    got = exact_path(sig, q, n, 0.30, 0.30, start=start)
    out, sev, failure = reference(sig, q, n, 0.30, 0.30, start=start)
    assert np.allclose(got["selected_probabilities"], out, atol=1e-12, rtol=0)
    assert abs(got["severity"] - sev) < 1e-12
    assert abs(got["failure_probability"] - failure) < 1e-12
    assert abs(out.sum() - 1) < 1e-12


def test_acceptance_counts_change_by_threshold():
    sig, q = example()
    got = exact_path(sig, q, 8, 0.30, 0.30)
    assert got["states"] > 1
    assert np.any(got["acceptance"] != got["acceptance"][0])
    assert np.any(got["wrong"] > 0)


def test_simultaneous_pilot_interval_in_constructed_case():
    sig, q = example()
    truth = exact_path(sig, q, 5, 0.30, 0.30)["severity"]
    ci = pilot_confidence_interval(sig, [100, 40, 60], 5, 0.30, 0.30)
    assert ci["lower"] <= truth <= ci["upper"]
    assert ci["confidence"] == 0.95


def test_state_limit_and_signature_validation():
    sig, q = example()
    with pytest.raises(ValueError):
        exact_path(sig, np.array([0.7, 0.7, -0.4]), 2, 0.30)
    with pytest.raises(RuntimeError, match="state limit"):
        exact_path(sig, q, 6, 0.30, 0.30, max_states=1)
    with pytest.raises(ValueError):
        pilot_confidence_interval(sig, [20, 30], 2, 0.30)


def test_large_binomial_tail_matches_frozen_r9():
    expected = float(R9.binom_pvalue(400, 1100, .3))
    assert expected == binom.cdf(400, 1100, .3)
    assert binomial_lower_tail(400, 1100, .3) == expected
    assert expected == pytest.approx(.9999974349948432)
    for n, k in [(0, 0), (100000, 0), (100000, 30000), (100000, 100000)]:
        assert binomial_lower_tail(k, n, .3) == float(R9.binom_pvalue(k, n, .3))


@pytest.mark.parametrize("alpha", [.01, .3, .5, .9])
def test_exact_and_adjacent_pvalue_boundaries_match_r9(alpha):
    for n in range(1, 41):
        for k in range(n + 1):
            p = float(R9.binom_pvalue(k, n, alpha))
            assert binomial_lower_tail(k, n, alpha) == p
            for delta in (np.nextafter(p, 0.), p, np.nextafter(p, 1.)):
                if 0 < delta < 1:
                    assert selected_threshold([[n, k]], alpha, delta) == R9.walk(
                        np.array([p]), 0, delta
                    )


@pytest.mark.parametrize("start", [-1, 0, 1, 2])
def test_variable_acceptance_nonmonotone_path_matches_r9(start):
    sig = np.array([
        [[1, 1], [0, 0], [1, 1]],
        [[0, 0], [1, 1], [1, 0]],
        [[1, 0], [1, 0], [0, 0]],
        [[1, 0], [1, 1], [1, 0]],
    ])
    q = np.array([.1, .2, .3, .4])
    out, sev, fail = reference(sig, q, 4, .3, .3, start)
    got = exact_path(sig, q, 4, .3, .3, start=start)
    np.testing.assert_allclose(got["selected_probabilities"], out, atol=1e-12, rtol=0)
    assert got["severity"] == pytest.approx(sev, abs=1e-12)
    assert got["failure_probability"] == pytest.approx(fail, abs=1e-12)


def test_walk_stops_at_first_failure_despite_later_pass():
    counts = np.array([[8, 0], [6, 2], [8, 0]])
    p = R9.binom_pvalue(counts[:, 1], counts[:, 0], .3)
    assert (p <= .3).tolist() == [True, False, True]
    assert selected_threshold(counts, .3, .3) == R9.walk(p, 2, .3) == 2
    assert selected_threshold(counts, .3, .3, start=1) == -1


@pytest.mark.parametrize("start", [None, -1, 0, 1, 2])
def test_zero_accepted_signs_never_certify(start):
    counts = np.zeros((3, 2), dtype=int)
    assert binomial_lower_tail(0, 0, .3) == 1.
    assert selected_threshold(counts, .3, .3, start=start) == -1
    got = exact_path(counts[None], [1.], 4, .3, .3, start=start)
    np.testing.assert_array_equal(got["selected_probabilities"], [1., 0., 0., 0.])
    assert got["severity"] == got["failure_probability"] == 0.


@pytest.mark.parametrize("start", [None, -1, 0, 1, 2])
def test_interval_uses_registered_start(start):
    sig, q = example()
    expected, sev, _ = reference(sig, q, 8, .3, .3, start=start)
    ci = pilot_confidence_interval(sig, [100, 40, 60], 8, .3, .3, start=start)
    np.testing.assert_allclose(ci["selected_probabilities"], expected, atol=1e-12, rtol=0)
    assert ci["estimate"] == pytest.approx(sev, abs=1e-12)
    assert ci["lower"] <= sev <= ci["upper"]


def test_interval_interior_start_changes_target():
    sig, _ = example()
    default = pilot_confidence_interval(sig, [100, 40, 60], 8, .3, .3)
    interior = pilot_confidence_interval(sig, [100, 40, 60], 8, .3, .3, start=0)
    assert default["estimate"] == pytest.approx(.001771875)
    assert interior["estimate"] == pytest.approx(.0029420415)
    assert default["estimate"] != interior["estimate"]


@pytest.mark.parametrize("pair", [
    [.9, .8], [1.9, 1.1], [-.9, 0.], [0, 1], [2, 0],
    [np.nan, 0], [1, np.nan], [np.inf, 0], [1, -np.inf],
    ["1", "0"], [1+0j, 0], [None, 0],
])
def test_raw_signature_indicators_rejected(pair):
    sig = [[pair]]
    with pytest.raises(ValueError):
        validate(sig, [1.])
    with pytest.raises(ValueError):
        exact_path(sig, [1.], 2, .3, .3)
    with pytest.raises(ValueError):
        pilot_confidence_interval(sig, [10], 2, .3, .3)


@pytest.mark.parametrize("counts", [
    [100.9, 40.9, 60.9], [-.9, 40, 60], [-1, 40, 60],
    [np.nan, 40, 60], [np.inf, 40, 60], [100, -np.inf, 60],
    ["100", "40", "60"], [None, 40, 60], [True, False, True],
    [1+0j, 40, 60], [2**63, 40, 60], [float(2**63), 40, 60],
    [0, 0, 0], [[100, 40, 60]], [100, 40],
])
def test_invalid_pilot_counts_rejected(counts):
    sig, _ = example()
    with pytest.raises(ValueError):
        pilot_confidence_interval(sig, counts, 2, .3, .3)


@pytest.mark.parametrize("counts", [
    [[8.5, 0]], [[-.9, 0]], [[-1, 0]], [[0, 1]],
    [[np.nan, 0]], [[np.inf, 0]], [["8", "0"]], [[1+0j, 0]],
    [[True, False]], [[2**63, 0]], [], [8, 0], [[8, 0, 0]],
])
def test_invalid_selection_counts_rejected(counts):
    with pytest.raises(ValueError):
        selected_threshold(counts, .3, .3)


@pytest.mark.parametrize("n", [
    2.0, 2.5, np.float64(2), True, np.bool_(False), "2", None,
    np.nan, np.inf, -np.inf, 2+0j, [2], np.array(2), -1, 100001,
])
def test_calibration_size_requires_integer_type(n):
    sig, q = example()
    with pytest.raises(ValueError):
        exact_path(sig, q, n, .3, .3)
    with pytest.raises(ValueError):
        pilot_confidence_interval(sig, [100, 40, 60], n, .3, .3)


@pytest.mark.parametrize("n", [0, 2, np.int32(2), np.int64(2), np.uint64(2)])
def test_calibration_size_accepts_python_and_numpy_integers(n):
    sig, q = example()
    got = exact_path(sig, q, n, .3, .3)
    expected, sev, _ = reference(sig, q, int(n), .3, .3)
    np.testing.assert_allclose(got["selected_probabilities"], expected, atol=1e-12, rtol=0)
    assert got["severity"] == pytest.approx(sev)


@pytest.mark.parametrize("start", [-2, 3, .5, 1., np.nan, True, "1"])
def test_invalid_starts_rejected_in_all_interfaces(start):
    sig, q = example()
    with pytest.raises(ValueError):
        selected_threshold([[8, 0]] * 3, .3, .3, start=start)
    with pytest.raises(ValueError):
        exact_path(sig, q, 0, .3, .3, start=start)
    with pytest.raises(ValueError):
        pilot_confidence_interval(sig, [100, 40, 60], 2, .3, .3, start=start)


def test_integer_valued_real_counts_and_binary_booleans_are_lossless():
    sig, q = example()
    x, _ = validate(sig.astype(bool), q)
    np.testing.assert_array_equal(x, sig)
    ci = pilot_confidence_interval(sig.astype(float), [100., 40., 60.], 2, .3, .3)
    assert ci["pilot_signs"] == 200
    assert selected_threshold([[8., 0.]], .3, .3) == 0
