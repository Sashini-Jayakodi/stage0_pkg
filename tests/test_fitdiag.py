"""Tests for the Fit-only diagnostic (src/fitdiag.py) on synthetic data."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import fitdiag as F  # noqa: E402
from src import splits as S  # noqa: E402
from test_stage4 import _fake_run  # noqa: E402


def test_slices_keep_equal_scores_together_and_are_monotone():
    rng = np.random.default_rng(0)
    score = np.repeat(rng.normal(size=60), rng.integers(1, 9, 60))        # 60 "pairs", many frames each
    w = np.ones(len(score))
    sl = F.slice_by_mass(score, w, 8)
    for v in np.unique(score):
        assert len(set(sl[score == v])) == 1                              # a pair is never split
    order = np.argsort(score, kind="stable")
    assert (np.diff(sl[order]) >= 0).all() and sl.min() == 0 and sl.max() == 7


def test_stratum_errors_match_brute_force():
    rng = np.random.default_rng(1)
    n = rng.integers(0, 6, 15).astype(float)
    e = np.minimum(n, rng.integers(0, 4, 15)).astype(float)
    n_t = np.full(15, 30.0)
    pt, pooled = F.stratum_errors(np.ones((1, 15)), n, e, n_t)
    has = n > 0
    assert abs(pt[0] - np.mean(e[has] / n[has])) < 1e-12
    assert abs(pooled[0] - e.sum() / n.sum()) < 1e-12                     # constant n_t -> plain frame error rate
    W = rng.multinomial(15, np.full(15, 1 / 15), size=3).astype(float)
    pt2, _ = F.stratum_errors(W, n, e, n_t)
    for b in range(3):
        m = has & (W[b] > 0)
        assert abs(pt2[b] - np.sum(W[b][m] * e[m] / n[m]) / W[b][m].sum()) < 1e-12


def test_bootstrap_detects_planted_gap_and_covers_zero_without_one():
    rng = np.random.default_rng(2)
    n_tracks, per = 260, 10
    track = np.repeat(np.arange(n_tracks), per)
    score = np.repeat(rng.normal(size=n_tracks), per)                     # one score per track -> simple strata
    c = np.median(score)
    for gap_p, expect_positive in ((0.25, True), (0.0, False)):
        p = np.where(score > c, 0.05 + gap_p, 0.05)
        wrong = rng.random(len(track)) < p
        n_t = np.bincount(track).astype(float)
        out = F.bootstrap_gap(track, wrong, np.ones(len(track), bool), score, c, np.arange(n_tracks), n_t, 500, np.random.default_rng(3))
        lo, hi = out["per_track"]["ci95"]
        if expect_positive:
            assert lo > 0 and out["per_track"]["frac_boot_gap_positive"] > 0.99
        else:
            assert lo < 0 < hi


def test_cert_readacross_is_scaled_by_track_counts():
    r = F.cert_readacross(0.01, 0.02, 261, 784)
    assert abs(r["se_cert_rough"] - 0.02 * (261 / 784) ** 0.5) < 1e-12
    assert abs(r["gap_needed_rough"] - 1.96 * r["se_cert_rough"]) < 1e-12
    assert 0 < r["rough_pass_probability"] < 1


def test_output_does_not_depend_on_cert_or_test_labels(tmp_path):
    # The diagnostic must use Fit labels only. Scramble every non-Fit frame's prediction: results must be identical.
    run, cfgp, _ = _fake_run(tmp_path, seed=3)
    cfg = {**S.DEFAULTS, "sizes": [69, 206, 69], "margin_points": 20}
    data = S.load_inputs(run, __import__("yaml").safe_load(open(cfgp)))
    base, split = F.diagnose_seed(0, data, cfg, 200)
    d2 = dict(data); d2["top3"] = data["top3"].copy()
    other = split[data["track"]] != 0
    d2["top3"][other, 0] = (d2["top3"][other, 0] + 7) % S.K
    again, _ = F.diagnose_seed(0, d2, cfg, 200)
    assert json.dumps(base, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_end_to_end_runs_writes_json_and_checks_stage4_files(tmp_path):
    run, cfgp, _ = _fake_run(tmp_path, seed=4)
    S.main(["--run", str(run), "--config", str(cfgp)])
    out = F.main(["--run", str(run), "--config", str(cfgp), "--boot", "100"])
    assert (run / "splits" / "fit_diagnostic.json").exists() and len(out["seeds"]) == 5
    r = out["seeds"][0]["eps2"]
    assert len(r["slices"]) == 8 and set(r["gap"]) == {"per_track", "pooled"}
    # a tampered Stage 4 file must be refused
    f = run / "splits" / "seed0.npz"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(AssertionError):
        F.main(["--run", str(run), "--config", str(cfgp), "--boot", "50"])


def test_rough_g1i_is_monotone_in_the_gap():
    def mk(gap):
        return [{"eps2": {"gap": {"per_track": {"gap": gap, "se_cert_rough": 0.01}}}} for _ in range(5)]
    lo, hi = F.rough_g1i(mk(0.0), "eps2"), F.rough_g1i(mk(0.04), "eps2")
    assert lo["p_at_least_4_of_5"] < 0.01 and hi["p_at_least_4_of_5"] > 0.95
    assert 0 <= lo["p_single_reshuffle"] <= hi["p_single_reshuffle"] <= 1
