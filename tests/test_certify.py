"""Tests for the Stage 6 engine (src/certify.py). Hand-computed values are exact-arithmetic numbers, not output of the code under test."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import certify as C  # noqa: E402


def test_hb_pvalue_matches_exact_hand_values():
    # (mu_hat, n, alpha, p): zero-error cases are (1-alpha)^n; the last two pick the Hoeffding and the Bentkus term respectively
    for mu, n, a, ref in [(0.0, 10, 0.1, 0.3486784401), (0.0, 50, 0.05, 0.07694497527671333), (0.0, 392, 0.0077, 0.048311732434177586),
                          (0.05, 20, 0.1, 0.7159606844069272), (0.02, 100, 0.1, 0.005286744607654344)]:
        assert abs(float(C.hb_pvalue(mu, n, a)) - ref) / ref < 1e-9


def test_zero_error_p_is_one_minus_alpha_to_the_n_and_floor_inverts_it():
    for n in (10, 100, 392, 784):
        for a in (0.005, 0.02, 0.05):
            assert abs(float(C.hb_pvalue(0.0, n, a)) - (1 - a) ** n) < 1e-12 * max(1, (1 - a) ** n) + 1e-15
    f = float(C.floor_alpha(392, 0.05))
    assert 0.0075 < f < 0.0077 and abs((1 - f) ** 392 - 0.05) < 1e-12
    assert np.isnan(float(C.floor_alpha(0, 0.05)))


def test_pvalue_is_monotone_in_mu_one_at_or_above_alpha_and_vectorised():
    mu = np.linspace(0, 0.06, 61)
    p = C.hb_pvalue(mu, 300, 0.02)
    assert (np.diff(p) >= -1e-12).all() and (p <= 1 + 1e-12).all()
    assert p[-1] > 0.99                                              # mu_hat far above alpha: nothing to reject
    assert abs(float(C.hb_pvalue(0.02, 300, 0.02)) - min(1.0, math.e * 0.5)) < 0.6   # at the boundary the p-value is large (no false confidence)
    assert float(C.hb_pvalue(0.02, 300, 0.02)) > 0.3
    assert np.isclose(C.hb_pvalue(mu, 300, 0.02)[5], float(C.hb_pvalue(mu[5], 300, 0.02)))


def test_h1_is_binary_kl_with_zero_log_zero():
    assert abs(float(C.h1(0.0, 0.1)) + math.log(0.9)) < 1e-14
    assert abs(float(C.h1(0.3, 0.3))) < 1e-15
    assert abs(float(C.h1(0.2, 0.1)) - (0.2 * math.log(2) + 0.8 * math.log(0.8 / 0.9))) < 1e-12


def test_shifted_losses_zero_when_clean_alpha_when_nothing_accepted_and_hand_case():
    n_tk = np.array([30.0, 10.0, 5.0])
    assert (C.shifted_losses(n_tk, n_tk, np.zeros(3), 0.02) == 0.0).all()               # exact zero: the floor depends on it
    assert np.allclose(C.shifted_losses(n_tk, np.zeros(3), np.zeros(3), 0.02), 0.02)
    # track 0: 20 of 30 accepted, 3 of them wrong -> 3/30 + 0.02 * 10/30
    assert np.isclose(C.shifted_losses(n_tk, np.array([20.0, 10, 5]), np.array([3.0, 0, 0]), 0.02)[0], 0.1 + 0.02 / 3)


def test_track_sums_handles_arbitrary_ids_and_tracks_are_averaged_equally():
    track = np.array([7, 7, 7, 7, 900, 900, 5])                                          # sizes 4, 2, 1 (ids not 0..n-1)
    acc = np.array([1, 1, 0, 1, 1, 1, 1], bool)
    wrong = np.array([1, 0, 1, 0, 0, 1, 0], bool)
    n_tk, a, e = C.track_sums(track, acc, wrong)
    order = np.argsort(np.unique(track))                                                 # unique() sorts: 5, 7, 900
    assert n_tk.tolist() == [1, 4, 2] and a.tolist() == [1, 3, 2] and e.tolist() == [0, 1, 1]
    L = C.shifted_losses(n_tk, a, e, 0.1)
    assert np.isclose(L.mean(), np.mean([0.0, 1 / 4 + 0.1 * 1 / 4, 1 / 2]))             # mean over TRACKS, not frames
    assert order.tolist() == [0, 1, 2]


def _stack(n_tracks, frames, err_rate, rng, mu_w=1.0, mu_c=20.0):
    track = np.repeat(np.arange(n_tracks), frames)
    wrong = rng.random(len(track)) < err_rate
    margin = np.where(wrong, rng.exponential(mu_w, len(track)), rng.exponential(mu_c, len(track)))
    return track, margin, wrong


def test_certify_sequence_floor_behaviour_zero_errors_certifies_only_above_the_floor():
    track, margin, wrong = np.repeat(np.arange(392), 30), np.full(392 * 30, 5.0), np.zeros(392 * 30, bool)
    grid, order = np.array([0.0, 1.0]), [1, 0]
    assert C.certify_sequence(track, margin, wrong, grid, order, 0.0077, 0.05)["certified"]
    assert not C.certify_sequence(track, margin, wrong, grid, order, 0.0075, 0.05)["certified"]


def test_certify_sequence_stops_at_first_failure_and_returns_the_last_pass():
    rng = np.random.default_rng(0)
    track, margin, wrong = _stack(600, 30, 0.05, rng)
    grid = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    r = C.certify_sequence(track, margin, wrong, grid, [5, 4, 3, 2, 1, 0], 0.03, 0.05)
    ps = [t["p"] for t in r["tests"]]
    first_fail = next((i for i, p in enumerate(ps) if p > 0.05), None)
    assert first_fail is not None and len(ps) == first_fail + 1                          # nothing is tested after the first failure
    if first_fail > 0:
        assert r["certified"] and r["grid_index"] == r["tests"][first_fail - 1]["grid_index"] and r["p_at_certified"] <= 0.05
    # a later threshold that would pass is never reached: put a failing step first
    r2 = C.certify_sequence(track, margin, wrong, grid, [0, 5, 4], 0.03, 0.05)           # loosest first fails (5% errors > 3%)
    assert not r2["certified"] and len(r2["tests"]) == 1


def test_certify_sequence_empty_input_and_unsorted_track_ids():
    r = C.certify_sequence(np.array([], int), np.array([]), np.array([], bool), np.array([0.0]), [0], 0.02, 0.05)
    assert not r["certified"] and r["n_tracks"] == 0
    rng = np.random.default_rng(1)
    track, margin, wrong = _stack(300, 30, 0.01, rng)
    perm = rng.permutation(len(track))
    a = C.certify_sequence(track, margin, wrong, np.array([0.0, 2.0]), [1, 0], 0.05, 0.05)
    b = C.certify_sequence(track[perm] * 3 + 11, margin[perm], wrong[perm], np.array([0.0, 2.0]), [1, 0], 0.05, 0.05)
    assert a["certified"] == b["certified"] and a["threshold"] == b["threshold"] and a["n_tracks"] == b["n_tracks"] == 300


def test_risk_stats_hand_example_and_edge_cases():
    track = np.array([0, 0, 0, 0, 1, 1])
    acc = np.array([1, 1, 0, 0, 1, 1], bool)
    wrong = np.array([1, 0, 0, 0, 0, 0], bool)
    r = C.risk_stats(track, acc, wrong, alpha=0.1)
    assert r["n_tracks"] == 2 and r["accepted"] == 4 and r["errors_accepted"] == 1
    assert np.isclose(r["frame_selective_error"], 0.25)
    # track 0: error share 1/4, accepted share 2/4; track 1: 0 and 1 -> ratio (1/8) / (3/4) = 1/6
    assert np.isclose(r["cert_risk"], (0.25 / 2) / (0.75)) and np.isclose(r["mean_L"], 0.125 + 0.1 * (1 - 0.75))
    assert r["violates"] == (r["mean_L"] > 0.1)
    z = C.risk_stats(track, np.zeros(6, bool), wrong, alpha=0.1)
    assert z["accepted"] == 0 and np.isnan(z["cert_risk"]) and z["coverage"] == 0
    e = C.risk_stats(np.array([], int), np.array([], bool), np.array([], bool), alpha=0.1)
    assert e["n_tracks"] == 0 and e["violates"] is None


def test_accept_mask_requires_gate_certified_group_and_margin():
    group = np.array([0, 0, 1, 1, 0])
    margin = np.array([5.0, 1.0, 9.0, 9.0, 5.0])
    gated = np.array([1, 1, 1, 1, 0], bool)
    res = {0: {"certified": True, "threshold": 2.0}, 1: {"certified": False, "threshold": None}}
    assert C.accept_mask(group, margin, gated, res).tolist() == [True, False, False, False, False]
