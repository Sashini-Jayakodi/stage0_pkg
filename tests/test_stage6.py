"""Tests for Stage 6 on real-data plumbing (src/stage6.py), using synthetic data. They check plumbing and the no-Test-labels rule, not real numbers."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import certify as C  # noqa: E402
from src import splits as S  # noqa: E402
from src import stage6 as T  # noqa: E402
from src.util import sha256_file  # noqa: E402
from test_stage4 import _fake_run  # noqa: E402

ALPHAS = [0.05, 0.1]                                   # the fake run has 69/206/69 tracks, so the 1-3% floor is out of reach
C6 = {**T.DEFAULTS6, "boot": 60, "g1iii_alphas": ALPHAS, "scan_step": 0.02, "scan_max": 0.10}


def _cfg():
    return {**S.DEFAULTS, "sizes": [69, 206, 69], "margin_points": 20, "alpha_grid": ALPHAS}


def _setup(tmp_path, seed=3, separable=True):
    run, cfgp, cfg = _fake_run(tmp_path, seed=seed)
    cfg["splits"]["alpha_grid"] = ALPHAS
    cfg["stage6"] = {"g1iii_alphas": ALPHAS, "scan_step": 0.02, "scan_max": 0.10}
    cfgp.write_text(yaml.safe_dump(cfg))
    data = S.load_inputs(run, yaml.safe_load(open(cfgp)))
    if separable:                                       # errors get low margins, correct frames high ones: a certifiable world
        rng = np.random.default_rng(seed)
        wrong = data["top3"][:, 0] != data["y"]
        data["margin"] = np.where(wrong, rng.exponential(1.0, len(wrong)), rng.exponential(20.0, len(wrong)))
    return run, cfgp, data


def _decide(data, seed=0):
    return T.decide_seed(seed, data, _cfg(), C6)


def test_stallkamp_subsets_partition_the_43_classes():
    m = T.stallkamp_map()
    assert len(m) == 43 and set(m) == set(range(6))
    assert m[0] == m[8] == 0 and m[9] == m[16] == 1 and m[6] == m[42] == 2 and m[33] == m[40] == 3 and m[11] == m[31] == 4 and m[12] == m[17] == 5
    assert [len(v) for v in T.STALLKAMP.values()] == [8, 4, 4, 8, 15, 4]


def test_preregistered_constants_have_not_drifted():
    d = T.DEFAULTS6
    assert (d["pass_seeds"], d["g1iii_alphas"], d["boot"], d["h1_alpha"]) == (4, [0.01, 0.02, 0.03], 2000, 0.02)
    assert S.DEFAULTS["delta"] == 0.1 and S.DEFAULTS["alpha_grid"] == [0.01, 0.02, 0.03, 0.05]


def test_guard_refuses_without_a_passing_synthetic_validation_for_this_engine(tmp_path):
    cert = ROOT / "src" / "certify.py"
    with pytest.raises(AssertionError):
        T.require_synthetic(tmp_path, cert)                                   # no file
    d = tmp_path / "stage6"; d.mkdir()
    f = d / "synthetic_validation.json"
    f.write_text(json.dumps({"all_pass": False, "certify_sha256": sha256_file(cert)}))
    with pytest.raises(AssertionError):
        T.require_synthetic(tmp_path, cert)                                   # a check failed
    f.write_text(json.dumps({"all_pass": True, "certify_sha256": "0" * 64}))
    with pytest.raises(AssertionError):
        T.require_synthetic(tmp_path, cert)                                   # engine changed since the validation
    f.write_text(json.dumps({"all_pass": True, "certify_sha256": sha256_file(cert)}))
    v, h = T.require_synthetic(tmp_path, cert)
    assert v["all_pass"] and h == sha256_file(f)


def test_schemes_have_the_right_groups_and_levels(tmp_path):
    _, _, data = _setup(tmp_path)
    out, ctx, schemes, masks, cals = _decide(data)
    assert {k: (v["K"], round(v["delta_k"], 10)) for k, v in schemes.items()} == {
        "eps2": (2, 0.05), "eps3": (2, 0.05), "marginal": (1, 0.1), "stallkamp6": (6, round(0.1 / 6, 10)), "mondrian43": (43, round(0.1 / 43, 10))}
    assert (schemes["marginal"]["group"] == 0).all() and (schemes["mondrian43"]["group"] == ctx["pred"]).all()
    assert set(np.unique(schemes["stallkamp6"]["group"])) <= set(range(6))
    # eps strata follow the Fit cut-point rule
    assert ((data["log_eps12"] > schemes["eps2"]["c_star"]) == (schemes["eps2"]["group"] == 1)).all()


def test_decisions_do_not_depend_on_test_data_but_the_test_columns_do(tmp_path):
    _, _, data = _setup(tmp_path, seed=5)
    cfg = _cfg()
    out, ctx, schemes, masks, _ = _decide(data)
    base = json.dumps(T.strip_test(T.evaluate_seed(out, ctx, schemes, masks, data, C6, 40)), sort_keys=True, default=T._js)
    # scramble the prediction and margin of every TEST frame; Fit/Cert decisions must be identical
    d2 = dict(data); d2["top3"] = data["top3"].copy(); d2["margin"] = data["margin"].copy()
    test_f = ctx["split"][data["track"]] == 2
    rng = np.random.default_rng(0)
    d2["top3"][test_f, 0] = (d2["top3"][test_f, 0] + 7) % S.K
    d2["margin"][test_f] = rng.random(test_f.sum()) * 9
    out2, ctx2, schemes2, masks2, _ = _decide(d2)
    ev2 = T.evaluate_seed(out2, ctx2, schemes2, masks2, d2, C6, 40)
    assert json.dumps(T.strip_test(ev2), sort_keys=True, default=T._js) == base
    # ... and the Test columns are not vacuous: they do change when Test predictions change
    out3, ctx3, schemes3, masks3, _ = _decide(data)
    ev3 = T.evaluate_seed(out3, ctx3, schemes3, masks3, data, C6, 40)
    assert json.dumps(ev3["schemes"]["eps2"]["alphas"]["abs_0.1"]["test"], sort_keys=True, default=T._js) != \
        json.dumps(ev2["schemes"]["eps2"]["alphas"]["abs_0.1"]["test"], sort_keys=True, default=T._js)


def test_decision_labels_never_include_test_labels(tmp_path):
    _, _, data = _setup(tmp_path)
    ctx = T.seed_context(0, data, _cfg())
    assert not ctx["wrong_dec"][ctx["fs"] == 2].any() and ctx["wrong_dec"][ctx["fs"] == 1].any()


def test_a_certifiable_world_certifies_and_certificates_are_internally_consistent(tmp_path):
    _, _, data = _setup(tmp_path)
    results = []
    for seed in range(5):
        out, ctx, schemes, masks, _ = _decide(data, seed)
        results.append(T.evaluate_seed(out, ctx, schemes, masks, data, C6, 40))
    assert T.check_certificates(results) == []
    agg = T.aggregate(results, C6)
    x = agg["schemes"]["eps2"]["alphas"]["abs_0.1"]
    assert x["group_seed_certified"] >= 5                                   # the separable world certifies at alpha = 10%
    assert agg["schemes"]["mondrian43"]["alphas"]["abs_0.05"]["group_seed_certified"] == 0   # ~5 Cert tracks per class: below the floor, as expected
    for r in results:
        g = r["schemes"]["eps2"]["alphas"]["abs_0.1"]["groups"]
        for lab, v in g.items():
            assert abs(v["floor_alpha"] - float(C.floor_alpha(v["n_tracks"], 0.05))) < 1e-15
            if v["certified"]:
                assert v["alpha"] >= v["floor_alpha"] and "test" in v and "cert" in v
    assert set(agg["g1iii"]) == {"eps2", "eps3"} and agg["g1iii"]["eps2"]["seeds_passing"] >= 0


def test_a_world_without_signal_certifies_nothing_at_an_alpha_below_the_base_error(tmp_path):
    _, _, data = _setup(tmp_path, seed=9, separable=False)                 # margin carries no information about errors (8% errors)
    n_cert = 0
    for seed in range(5):
        out, *_ = _decide(data, seed)
        n_cert += sum(g["certified"] for g in out["schemes"]["eps2"]["alphas"]["abs_0.05"]["groups"].values())
    assert n_cert == 0


def test_marginal_scheme_equals_a_direct_k1_call(tmp_path):
    _, _, data = _setup(tmp_path)
    out, ctx, schemes, masks, cals = _decide(data)
    g = out["schemes"]["marginal"]["alphas"]["abs_0.1"]["groups"]["all"]
    rows = np.flatnonzero(ctx["cert_f"])
    a, order = cals["marginal"][0]["orders"]["abs_0.1"]
    r = C.certify_sequence(ctx["track"][rows], ctx["m"][rows], ctx["wrong_dec"][rows], cals["marginal"][0]["grid"], order, a, 0.1)
    assert g["certified"] == r["certified"] and g["threshold"] == r["threshold"] and g["n_tracks"] == r["n_tracks"]


def test_uncertified_group_escalates_every_frame_and_gate_rejects_escalate(tmp_path):
    _, _, data = _setup(tmp_path, separable=False)
    out, ctx, schemes, masks, _ = _decide(data)
    acc = masks[("eps2", "abs_0.05")]
    assert not acc.any()                                                    # nothing certified -> everything escalated
    for k, v in masks.items():
        assert not (v & ~ctx["gated"]).any()                                # nothing outside the gate is ever accepted


def test_boot_group_risk_collapses_when_all_tracks_are_identical_and_flags_a_clear_violation():
    track = np.repeat(np.arange(40), 10)
    acc = np.ones(400, bool)
    wrong = np.tile(np.array([1] + [0] * 9, bool), 40)                      # 10% error in every track
    r = T.boot_group_risk(track, acc, wrong, np.arange(400), 0.05, 200, np.random.default_rng(0))
    assert np.isclose(r["point"], 0.1) and np.allclose(r["ci95"], [0.1, 0.1]) and r["share_draws_above_alpha"] == 1.0
    assert T.boot_group_risk(track, acc, wrong, np.array([], int), 0.05, 10, np.random.default_rng(0)) is None


def test_strip_test_removes_only_test_keys():
    o = {"a": 1, "test": {"x": 1}, "b": [{"test": 2, "c": 3}]}
    assert T.strip_test(o) == {"a": 1, "b": [{"c": 3}]}


def test_end_to_end_refuses_without_synthetic_then_runs_and_writes_hashes(tmp_path):
    run, cfgp, _ = _setup(tmp_path, seed=8)
    S.main(["--run", str(run), "--config", str(cfgp)])
    with pytest.raises(AssertionError):
        T.main(["--run", str(run), "--config", str(cfgp), "--boot", "30"])           # no synthetic validation yet
    d = run / "stage6"; d.mkdir()
    (d / "synthetic_validation.json").write_text(json.dumps({"all_pass": True, "certify_sha256": sha256_file(ROOT / "src" / "certify.py")}))
    out = T.main(["--run", str(run), "--config", str(cfgp), "--boot", "30"])
    res = json.loads((d / "stage6_results.json").read_text())
    assert len(res["seeds"]) == 5 and res["checks"]["stage4_split_tau_cstar_grids_el_reproduced"] and res["checks"]["certificates_consistent"]
    assert out["files_sha256"]["stage6_results.json"] == sha256_file(d / "stage6_results.json")
    assert set(res["aggregate"]["schemes"]) == set(T.SCHEME_ORDER) and set(res["aggregate"]["g1iii"]) == {"eps2", "eps3"}
    # a tampered Stage 4 file is refused
    f = run / "splits" / "seed0.npz"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(AssertionError):
        T.main(["--run", str(run), "--config", str(cfgp), "--boot", "30"])


def test_mu_star_is_the_boundary_of_the_hb_test_and_implied_risk_is_consistent():
    for n, a in ((392, 0.02), (784, 0.03), (200, 0.05)):
        ms = T.mu_star(n, a, 0.05)
        assert float(C.hb_pvalue(ms, n, a)) <= 0.05 < float(C.hb_pvalue(ms + 1e-4, n, a))
        for cov in (1.0, 0.9):
            r = T.implied_max_risk(ms, a, cov)
            assert r is None or abs(cov * r + a * (1 - cov) - ms) < 1e-12
    assert T.mu_star(200, 0.01, 0.05) is None and T.implied_max_risk(None, 0.01, 0.9) is None   # below the floor: 1 - 0.05^(1/200) = 1.49%
    assert abs(T.mu_star(392, 0.0077, 0.05) - 0.0) < 1e-12 or T.mu_star(392, 0.0077, 0.05) < 1e-4
