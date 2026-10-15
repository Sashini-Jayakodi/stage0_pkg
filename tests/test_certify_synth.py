"""Tests for the synthetic validation harness (src/certify_synth.py). Small runs only: the full 1,000-run validation is the CLI."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import certify as C  # noqa: E402
from src import certify_synth as Z  # noqa: E402
from src.util import sha256_file  # noqa: E402


def _small(**kw):
    return {**Z.CFG, "runs": 12, "null_runs": 400, "ordering_runs": 12, "population_tracks": 20000, **kw}


def test_constants_have_not_drifted():
    c = Z.CFG
    assert (c["frames"], c["n_fit"], c["n_cert"], c["alpha"], c["delta_k"], c["runs"], c["rhos"]) == (30, 261, 784, 0.02, 0.05, 1000, [0.0, 0.1, 0.3])


def test_beta_design_has_the_requested_mean_and_icc():
    rng = np.random.default_rng(0)
    m, w = Z.make_tracks(20000, 30, Z.beta_design(0.08, 0.3), rng, 1.0, 20.0)
    assert abs(w.mean() - 0.08) < 0.004
    k = w.sum(1).astype(float)
    p = w.mean()
    rho = (k.var(ddof=1) - 30 * p * (1 - p)) / (30 * 29 * p * (1 - p))               # moment estimator of the ICC for equal-size clusters
    assert abs(rho - 0.3) < 0.03


def test_cluster_design_whole_track_errors_with_confident_margins():
    rng = np.random.default_rng(1)
    d = Z.cluster_design({"q_normal": 0.0, "pi": 0.05, "q_conf": 1.0})
    m, w = Z.make_tracks(4000, 30, d, rng, 1.0, 20.0)
    bad = w.all(1)
    assert 0.03 < bad.mean() < 0.07 and (w.any(1) == bad).all()                     # only confusable tracks err, and every frame of them
    assert abs(m[bad].mean() - 20.0) < 1.5 and abs(m[~bad].mean() - 20.0) < 1.0


def test_true_risk_formula_matches_a_large_population_for_every_design():
    rng = np.random.default_rng(2)
    for d in (Z.beta_design(0.08, 0.0), Z.beta_design(0.08, 0.3), Z.cluster_design(Z.CFG["cluster"])):
        m, w = Z.make_tracks(60000, 30, d, rng, 1.0, 20.0)
        for lam in (0.0, 1.0, 3.0):
            acc = m >= lam
            assert abs((acc & w).mean() / acc.mean() - float(Z.true_risk(lam, d, 1.0, 20.0))) < 0.004


def test_exact_checks_pass():
    assert Z.check_hand()["pass"] and Z.check_floor(_small())["pass"]


def test_null_check_passes_and_a_deliberately_anticonservative_p_value_would_fail(monkeypatch):
    assert Z.check_null(_small())["pass"]
    monkeypatch.setattr(C, "hb_pvalue", lambda mu, n, a: np.exp(-n * 4 * C.h1(np.minimum(mu, a), a)))   # 4x too confident
    assert not Z.check_null(_small())["pass"]


def test_ordering_check_distinguishes_fit_ordered_from_strict_to_loose():
    r = Z.check_ordering(_small(ordering_runs=15))
    assert r["certified_share_fit_ordered"] >= 0.9 and r["certified_share_strict_to_loose"] <= 0.05


def test_validity_check_flags_a_broken_engine(monkeypatch):
    # a certifier that always certifies the loosest threshold must be caught by the track-level validity check
    def always_loose(track, margin, wrong, grid, order, alpha, delta_k):
        return {"certified": True, "grid_index": 0, "threshold": float(grid[0]), "n_tracks": 1, "n_frames": 1, "p_at_certified": 0.0, "tests": []}
    monkeypatch.setattr(C, "certify_sequence", always_loose)
    v = Z.check_validity(_small(runs=20, rhos=[0.0]))
    assert not v["track_level_valid"]


def test_cli_writes_json_with_engine_hash(tmp_path):
    res = Z.main(["--run", str(tmp_path), "--runs", "6"])
    f = tmp_path / "stage6" / "synthetic_validation.json"
    j = json.loads(f.read_text())
    assert j["certify_sha256"] == sha256_file(ROOT / "src" / "certify.py") and "all_pass" in j and j["constants"]["runs"] == 6
    assert set(j) >= {"1_null", "2_hand_values", "3_floor", "4_track_level_validity", "5_naive_breaks", "6_ordering_fix", "population_cross_check"}
    assert res["1_null"]["pass"] and res["2_hand_values"]["pass"] and res["3_floor"]["pass"]
