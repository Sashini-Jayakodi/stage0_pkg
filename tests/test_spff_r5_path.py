import itertools
import numpy as np
import pytest

from src.spff_r5_path import (
    exact_path, pilot_confidence_interval, selected_threshold,
    excess_per_threshold
)


def example():
    sig = np.array([
        [[0, 0], [1, 0], [1, 0]],
        [[1, 1], [1, 1], [1, 0]],
        [[1, 0], [0, 0], [0, 0]],
    ], dtype=int)
    return sig, np.array([0.50, 0.20, 0.30])


def reference(sig, q, n, alpha, delta):
    g = sig.shape[1]
    out = np.zeros(g + 1)
    for seq in itertools.product(range(len(q)), repeat=n):
        mass = np.prod(q[list(seq)])
        counts = sig[list(seq)].sum(axis=0)
        chosen = selected_threshold(counts, alpha, delta)
        out[chosen + 1] += mass
    excess, a, w = excess_per_threshold(sig, q, alpha)
    return out, float(out[1:] @ excess), float(
        out[1:] @ ((a > 0) & (w > alpha * a))
    )


@pytest.mark.parametrize("n", [2, 4, 6])
def test_exact_matches_direct_enumeration(n):
    sig, q = example()
    got = exact_path(sig, q, n, 0.30, 0.30)
    out, sev, failure = reference(sig, q, n, 0.30, 0.30)
    assert np.allclose(got["selected_probabilities"], out, atol=1e-12)
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
