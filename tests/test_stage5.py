"""Tests for Stage 5 (src/stage5.py) on synthetic data."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import splits as S  # noqa: E402
from src import stage5 as T  # noqa: E402
from test_stage4 import _fake_run  # noqa: E402

C5 = {**T.DEFAULTS5, "boot": 150}


def _cfg():
    return {**S.DEFAULTS, "sizes": [69, 206, 69], "margin_points": 20}


def _data(tmp_path, seed=3):
    run, cfgp, _ = _fake_run(tmp_path, seed=seed)
    return run, cfgp, S.load_inputs(run, yaml.safe_load(open(cfgp)))


# ------------------------------------------------------------------ constants guard
def test_preregistered_constants_have_not_drifted():
    # LOG.md pre-registration: B = 2000, hard group >= 30 errors over >= 10 tracks, pass in >= 4 of 5, design effect uses m = 30.
    d = T.DEFAULTS5
    assert (d["boot"], d["min_hard_errors"], d["min_hard_tracks"], d["pass_seeds"], d["frames_per_track"]) == (2000, 30, 10, 4, 30)


# ------------------------------------------------------------------ rho
def _brute_icc(clusters):
    """Textbook one-way ANOVA ICC from a list of 0/1 arrays, written independently of icc_anova."""
    k = len(clusters); N = sum(len(c) for c in clusters)
    grand = np.concatenate(clusters).mean()
    ssb = sum(len(c) * (c.mean() - grand) ** 2 for c in clusters)
    ssw = sum(((c - c.mean()) ** 2).sum() for c in clusters)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - sum(len(c) ** 2 for c in clusters) / N) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def test_icc_matches_brute_force_including_unequal_sizes_and_bootstrap_multiplicities():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 12, 40)
    clusters = [(rng.random(m) < rng.choice([0.05, 0.5])).astype(float) for m in sizes]
    n = np.array([len(c) for c in clusters], float); e = np.array([c.sum() for c in clusters])
    assert abs(T.icc_anova(np.ones(40), n, e)[0] - _brute_icc(clusters)) < 1e-10
    mult = rng.multinomial(40, np.full(40, 1 / 40))                         # a track drawn twice is two clusters
    dup = [c for c, m in zip(clusters, mult) for _ in range(m)]
    assert abs(T.icc_anova(mult.astype(float), n, e)[0] - _brute_icc(dup)) < 1e-10


def test_icc_recovers_planted_value_and_is_near_zero_for_independent_frames():
    rng = np.random.default_rng(1)
    T_, m = 3000, 30
    for rho_true, tol in ((0.3, 0.03), (0.0, 0.02)):
        if rho_true:
            a, b = 0.1 * (1 / rho_true - 1) / 1, 0.9 * (1 / rho_true - 1) / 1       # Beta(a,b): mean .1, rho = 1/(a+b+1)
            p = rng.beta(a, b, T_)
        else:
            p = np.full(T_, 0.1)
        e = rng.binomial(m, p).astype(float)
        est = T.icc_anova(np.ones(T_), np.full(T_, float(m)), e)[0]
        assert abs(est - rho_true) < tol


def test_rho_block_reports_design_effect_and_degenerate_cases_do_not_crash():
    rng = np.random.default_rng(2)
    track = np.repeat(np.arange(80), 10)
    wrong = rng.random(800) < 0.1
    r = T.rho_block(track, wrong, np.ones(800, bool), 80, 100, np.random.default_rng(0))
    assert r["frames"] == 800 and r["tracks"] == 80
    assert abs(r["design_effect_actual"] - (1 + 9 * r["rho"])) < 1e-12
    assert abs(r["design_effect_m30"] - (1 + 29 * r["rho"])) < 1e-12
    assert r["rho_ci95"][0] <= r["rho"] <= r["rho_ci95"][1] or r["rho_ci95"][0] > r["rho"] - 0.05
    z = T.rho_block(track, np.zeros(800, bool), np.ones(800, bool), 80, 50, np.random.default_rng(0))   # no errors at all
    assert np.isnan(z["rho"]) and "design_effect_m30" in z


# ------------------------------------------------------------------ rank of error, fragments
def test_rank_of_error_counts_second_third_and_beyond():
    top3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = np.array([2, 3, 9, 2])
    r = T.rank_of_error(top3, y, np.arange(4))
    assert (r["rank2"], r["rank3"], r["rank4plus"], r["errors"]) == (2, 1, 1, 4)
    assert abs(r["share_rank2"] - 0.5) < 1e-12


def test_fragment_counts_bins_add_up_and_need_no_labels():
    track = np.repeat(np.arange(6), [1, 2, 4, 8, 15, 30])
    sel = np.ones(len(track), bool)
    f, cnt = T.fragment_counts(track, sel, 6, np.arange(6))
    assert f["tracks_with_frames"] == 6 and sum(f["bins"].values()) == 6
    assert f["bins"] == {"1": 1, "2": 1, "3-5": 1, "6-10": 1, "11-20": 1, "21-30": 1}
    assert abs(f["share_tracks_le2_frames"] - 2 / 6) < 1e-12 and list(cnt) == [1, 2, 4, 8, 15, 30]


# ------------------------------------------------------------------ verdict logic (pre-registered rules)
def _mk(passes_pt, passes_po, errs, tracks, name="eps2"):
    seeds = []
    for i in range(5):
        gap = {k: {"passes": bool(p[i]), "gap": 0.01, "stable_when_B_doubled": True} for k, p in (("per_track", passes_pt), ("pooled", passes_po))}
        seeds.append({name: {"gap": gap, "g1ii": {"hard_errors": errs[i], "hard_tracks_with_error": tracks[i]}}})
    return seeds


def test_g1i_needs_four_of_five_and_flags_definition_disagreement():
    a = T.aggregate([{**s, "eps3": s["eps2"]} for s in _mk([1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [50] * 5, [20] * 5)], C5)["eps2"]
    assert a["g1i_per_track_pass"] and not a["g1i_pooled_pass"] and a["g1i_definitions_disagree"]
    b = T.aggregate([{**s, "eps3": s["eps2"]} for s in _mk([1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [50] * 5, [20] * 5)], C5)["eps2"]
    assert not b["g1i_per_track_pass"] and not b["g1i_definitions_disagree"] and b["g1i_per_track_seeds_passing"] == 3


def test_g1ii_uses_the_median_over_reshuffles_for_both_counts():
    def run(errs, trk):
        return T.aggregate([{**s, "eps3": s["eps2"]} for s in _mk([0] * 5, [0] * 5, errs, trk)], C5)["eps2"]["g1ii_pass"]
    assert run([100, 100, 100, 0, 0], [20] * 5)                      # median 100 errors
    assert not run([10, 10, 10, 100, 100], [20] * 5)                 # median 10 errors
    assert not run([100] * 5, [5, 5, 5, 40, 40])                     # median 5 tracks
    assert run([30, 30, 30, 5, 5], [10, 10, 10, 1, 1])               # exactly at the thresholds


# ------------------------------------------------------------------ data-driven behaviour
def _planted(data, strength, seed=0):
    """Errors whose probability rises with log eps of the top-2 pair (strength = extra error probability for the hard half)."""
    rng = np.random.default_rng(seed)
    d = dict(data); d["top3"] = data["top3"].copy()
    hard = data["log_eps12"] > np.median(data["log_eps12"])
    wrong = rng.random(len(hard)) < (0.04 + strength * hard)
    d["top3"][:, 0] = np.where(wrong, (data["y"] + 1) % S.K, data["y"])
    return d


def test_pipeline_detects_a_planted_gap_and_not_a_null(tmp_path):
    _, _, data = _data(tmp_path)
    cfg = _cfg()
    strong = T.aggregate([T.analyse_seed(s, _planted(data, 0.30), cfg, C5)[0] for s in range(5)], C5)
    assert strong["eps2"]["g1i_per_track_pass"] and strong["eps2"]["g1i_pooled_pass"]
    null = [T.analyse_seed(s, _planted(data, 0.0, seed=s), cfg, C5)[0] for s in range(5)]
    assert T.aggregate(null, C5)["eps2"]["g1i_per_track_seeds_passing"] <= 2


def test_output_does_not_depend_on_fit_or_test_labels(tmp_path):
    # Stage 5 may read CERT labels only. Scramble the prediction of every non-Cert frame: the result must be identical.
    _, _, data = _data(tmp_path, seed=5)
    cfg = _cfg()
    base, split = T.analyse_seed(0, data, cfg, C5)
    d2 = dict(data); d2["top3"] = data["top3"].copy()
    other = split[data["track"]] != 1
    d2["top3"][other, 0] = (d2["top3"][other, 0] + 7) % S.K
    again, _ = T.analyse_seed(0, d2, cfg, C5)
    assert json.dumps(base, sort_keys=True, default=T._js) == json.dumps(again, sort_keys=True, default=T._js)
    # and it does depend on Cert labels (so the test above is not vacuous)
    d3 = dict(data); d3["top3"] = data["top3"].copy()
    cert = split[data["track"]] == 1
    d3["top3"][cert, 0] = (d3["top3"][cert, 0] + 7) % S.K
    changed, _ = T.analyse_seed(0, d3, cfg, C5)
    assert json.dumps(base, sort_keys=True, default=T._js) != json.dumps(changed, sort_keys=True, default=T._js)


def test_strata_partition_cert_gated_frames_and_counts_reconcile(tmp_path):
    _, _, data = _data(tmp_path, seed=6)
    r, _ = T.analyse_seed(1, data, _cfg(), C5)
    for nm in ("eps2", "eps3"):
        s = r[nm]["strata"]
        assert s["easy"]["cert_frames"] + s["hard"]["cert_frames"] == r["cert_frames_gated_in"]
        assert s["easy"]["errors"] + s["hard"]["errors"] == r["cert_errors_all_gated_in"]
        assert s["easy"]["rank_of_error"]["errors"] == s["easy"]["errors"]
        assert s["hard"]["tracks_with_error"] <= s["hard"]["errors"]
    ro = r["rank_of_error_cert_all"]
    assert ro["rank2"] + ro["rank3"] + ro["rank4plus"] == ro["errors"] == r["cert_errors_all_gated_in"]


def test_bootstrap_lower_bound_is_stable_when_b_is_doubled_for_a_clear_effect(tmp_path):
    _, _, data = _data(tmp_path, seed=7)
    r, _ = T.analyse_seed(0, _planted(data, 0.35), _cfg(), {**C5, "boot": 400})
    g = r["eps2"]["gap"]["per_track"]
    assert g["passes"] and g["passes_2B"] and g["stable_when_B_doubled"]
    assert abs(g["lower_bound"] - g["lower_bound_2B"]) < 0.03


# ------------------------------------------------------------------ end to end
def test_end_to_end_writes_results_checks_stage4_hashes_and_reconciles_accuracy(tmp_path):
    run, cfgp, _ = _fake_run(tmp_path, seed=8)
    S.main(["--run", str(run), "--config", str(cfgp)])
    out = T.main(["--run", str(run), "--config", str(cfgp), "--boot", "60"])
    res = json.loads((run / "splits" / "stage5_results.json").read_text())
    assert len(res["seeds"]) == 5 and set(res["aggregate"]) == {"eps2", "eps3"} and res["bootstrap_draws"] == 60
    assert out["files_sha256"]["stage5_results.json"]
    # reconcile with Stage 3 head accuracy: fake summary has none -> no key; add the true value -> True; a wrong one -> False
    d = np.load(run / "head_emb" / "scores_train_release.npz")
    acc = float((d["top3"][:, 0] == d["y"]).mean())
    hp = run / "head_emb" / "head_summary.json"
    hs = json.loads(hp.read_text())
    for val, expect in ((acc, True), (acc - 0.01, False)):
        hs["train_release"] = {"head_accuracy": val}; hp.write_text(json.dumps(hs))
        assert T.main(["--run", str(run), "--config", str(cfgp), "--boot", "30"])["checks"]["reconciles_with_stage3_head_accuracy"] is expect
    # a tampered Stage 4 file must be refused
    f = run / "splits" / "seed0.npz"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(AssertionError):
        T.main(["--run", str(run), "--config", str(cfgp), "--boot", "30"])
