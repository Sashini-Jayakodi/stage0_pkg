"""Stage 4 tests on synthetic data (no real GTSRB). They check the plumbing and the maths of the split,
gate, cut-point and Fit-side calibration. They cannot say anything about the real numbers."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import splits as S  # noqa: E402
from src.train import sha256_file  # noqa: E402

K = S.K


def test_allocate_counts_exact_targets_and_at_least_one_per_split():
    rng = np.random.default_rng(0)
    for trial in range(20):
        # random class sizes (>=8 tracks per class) that add up to 1,307 like the real data
        T = np.full(K, 8) + np.bincount(rng.integers(0, K, 1307 - 8 * K), minlength=K)
        assert T.sum() == 1307 and T.min() >= 8
        c = S.allocate_counts(T, [0.2, 0.6, 0.2], [261, 784, 262], np.random.default_rng(trial))
        assert (c.sum(0) == [261, 784, 262]).all() and (c.sum(1) == T).all() and (c >= 1).all()


def test_allocate_counts_close_to_20_60_20_per_class():
    T = np.full(K, 30); T[:10] = 31; T[10:12] = 32          # 1,304 tracks
    targets = [261, 782, 261]
    c = S.allocate_counts(T, [0.2, 0.6, 0.2], targets, np.random.default_rng(1))
    assert (np.abs(c - T[:, None] * np.array([0.2, 0.6, 0.2])) < 2.5).all()


def test_make_split_disjoint_covers_all_tracks_and_is_seeded():
    rng = np.random.default_rng(3)
    tc = np.repeat(np.arange(K), 8)                                   # 344 tracks
    rng.shuffle(tc)
    sp, counts = S.make_split(tc, [69, 206, 69], [0.2, 0.6, 0.2], seed=0)
    assert sp.shape == tc.shape and set(np.unique(sp)) == {0, 1, 2}   # every track in exactly one split
    assert (np.bincount(sp) == [69, 206, 69]).all()
    for k in range(3):
        assert set(tc[sp == k]) == set(range(K))                      # every class in every split
    sp2, _ = S.make_split(tc, [69, 206, 69], [0.2, 0.6, 0.2], seed=0)
    sp3, _ = S.make_split(tc, [69, 206, 69], [0.2, 0.6, 0.2], seed=1)
    assert (sp == sp2).all() and not (sp == sp3).all()
    with pytest.raises(AssertionError):
        S.make_split(tc, [10, 10, 10], [0.2, 0.6, 0.2], seed=0)       # sizes that do not add up are refused


def test_conformal_rank_values():
    assert S.conformal_rank(261, 0.05) == 249
    assert S.conformal_rank(19, 0.05) == 19      # (20)(0.95) = 19.000000000000004 in floats; must not become 20
    assert S.conformal_rank(99, 0.1) == 90
    assert S.conformal_rank(9, 0.1) == 9


def test_one_frame_per_track_picks_a_frame_of_that_track():
    track = np.repeat(np.arange(6), [3, 1, 5, 2, 4, 7])
    tracks = np.array([4, 1, 2])
    rows = S.one_frame_per_track(track, tracks, np.random.default_rng(0))
    assert (track[rows] == tracks).all() and len(set(rows)) == 3
    seen = {S.one_frame_per_track(track, np.array([2]), np.random.default_rng(i))[0] for i in range(60)}
    assert seen == set(np.flatnonzero(track == 2))                    # every frame of the track can be drawn


def test_fit_gate_threshold_is_the_kth_smallest():
    n_tracks, per = 40, 5
    track = np.repeat(np.arange(n_tracks), per)
    s = np.random.default_rng(0).random(len(track))
    split = np.zeros(n_tracks, dtype=int); split[20:] = 1
    tau, rows, k = S.fit_gate(s, track, split, seed=0, beta=0.05)
    assert len(rows) == 20 and (split[track[rows]] == 0).all() and k == 20
    assert tau == np.sort(s[rows])[k - 1] == s[rows].max()


def test_choose_cut_hits_closest_to_half_and_keeps_pairs_together():
    score = np.array([1., 1., 2., 2., 3., 3., 4., 4.])
    w = np.ones(8)
    c, frac = S.choose_cut(score, w)
    assert c == 2.0 and frac == 0.5
    # weights matter: heavy mass on the lowest value pushes the cut down
    w2 = np.array([5., 5., 1., 1., 1., 1., 1., 1.])
    c2, frac2 = S.choose_cut(score, w2)
    assert c2 == 1.0 and abs(frac2 - 10 / 16) < 1e-12
    # equal-score frames can never be separated: the easy side is always "score <= c*"
    c3, _ = S.choose_cut(np.array([5.] * 10), np.ones(10))
    assert c3 == 5.0
    # tie between two cut values equally far from 0.5 -> the lowest wins
    c4, f4 = S.choose_cut(np.array([1., 2., 3., 4.]), np.array([4., 2., 2., 2.]))   # fractions .4 / .6 / .8 / 1
    assert c4 == 1.0 and abs(f4 - 0.4) < 1e-12


def test_margin_grid_ascending_unique_and_spans_accept_all_to_p98():
    m = np.random.default_rng(0).random(2000) * 10
    g = S.margin_grid(m, 50, 0.98)
    assert (np.diff(g) > 0).all() and g[0] == m.min() and abs(g[-1] - np.quantile(m, 0.98)) < 1e-12 and len(g) == 50
    assert ((m >= g[0]).all())                                         # index 0 accepts everything
    g2 = S.margin_grid(np.zeros(100), 50, 0.98)
    assert len(g2) == 1                                               # duplicates collapse


def _brute_el(track, rows, m, wrong, grid, alpha):
    out = []
    tracks = sorted(set(track[rows]))
    for t in grid:
        vals = []
        for tk in tracks:
            r = [i for i in rows if track[i] == tk]
            loss = sum((1.0 if (m[i] >= t and wrong[i]) else 0.0) - alpha * (1.0 if m[i] >= t else 0.0) for i in r) / len(r)
            vals.append(loss + alpha)
        out.append(np.mean(vals))
    return np.array(out)


def test_el_curve_matches_brute_force():
    rng = np.random.default_rng(5)
    track = np.repeat(np.arange(12), rng.integers(2, 9, 12))
    m = rng.random(len(track)) * 5
    wrong = rng.random(len(track)) < 0.2
    rows = np.flatnonzero(rng.random(len(track)) < 0.7)               # a sub-sample: some tracks lose frames
    grid = S.margin_grid(m[rows], 12, 0.98)
    for alpha in (0.0, 0.03, 0.2):
        got = S.el_curve(track, rows, m, wrong, grid, alpha, 12)
        assert np.allclose(got, _brute_el(track, rows, m, wrong, grid, alpha))
    # accept-all with alpha: E[L] = mean per-track error rate (the alpha terms cancel)
    got = S.el_curve(track, rows, m, wrong, grid[:1], 0.1, 12)[0]
    per = [np.mean(wrong[[i for i in rows if track[i] == t]]) for t in sorted(set(track[rows]))]
    assert abs(got - np.mean(per)) < 1e-12


def test_order_from_curve_starts_at_argmin_and_relaxes_looser():
    el = np.array([0.30, 0.20, 0.10, 0.15, 0.25])
    start, order = S.order_from_curve(el)
    assert start == 2 and list(order) == [2, 1, 0]
    start, order = S.order_from_curve(np.array([0.2, 0.2, 0.2]))      # ties -> loosest minimiser
    assert start == 0 and list(order) == [0]


def test_track_weighted_error_two_definitions():
    track = np.array([0, 0, 0, 0, 1, 1])
    n_t = np.array([4., 2.])
    wrong = np.array([1, 0, 0, 0, 1, 1], dtype=bool)
    rows = np.arange(6)
    lit, ptm = S.track_weighted_error(track, rows, wrong, n_t)
    assert abs(lit - (0.25 + 0.5 + 0.5) / (4 * 0.25 + 2 * 0.5)) < 1e-12
    assert abs(ptm - (0.25 + 1.0) / 2) < 1e-12


# ---------------------------------------------------------------- end to end on synthetic Stage 2/3 files
def _fake_run(tmp_path, per_class_tracks=8, frames=10, seed=0):
    rng = np.random.default_rng(seed)
    n_tracks = K * per_class_tracks
    tc = np.repeat(np.arange(K), per_class_tracks)
    track = np.repeat(np.arange(n_tracks), frames).astype(np.int32)
    y = tc[track]
    n = len(y)
    log_eps = -rng.random((K, K)) * 30
    log_eps = (log_eps + log_eps.T) / 2
    np.fill_diagonal(log_eps, np.nan)
    wrong = rng.random(n) < 0.08
    p1 = np.where(wrong, (y + 1 + rng.integers(0, K - 1, n)) % K, y)
    p2 = (p1 + 1 + rng.integers(0, K - 1, n)) % K
    p3 = (p2 + 1 + rng.integers(0, K - 1, n)) % K
    p3 = np.where(p3 == p1, (p3 + 1) % K, p3); p3 = np.where(p3 == p2, (p3 + 1) % K, p3)
    p3 = np.where(p3 == p1, (p3 + 1) % K, p3)               # top-3 classes must be distinct (as in the real head)
    top3 = np.stack([p1, p2, p3], 1).astype(np.int32)
    run = tmp_path / "run"
    (run / "embeddings").mkdir(parents=True); (run / "head_emb").mkdir()
    np.savez(run / "embeddings" / "train_release.npz", emb=np.zeros((n, 4), np.float32), y=y, track=track,
             track_names=np.array([f"{k // per_class_tracks:02d}_{k % per_class_tracks:05d}" for k in range(n_tracks)]))
    np.savez(run / "head_emb" / "head.npz", log_eps=log_eps)
    np.savez(run / "head_emb" / "scores_train_release.npz", top3=top3, margin=rng.random(n) * 8 * (1 + 0.5 * ~wrong),
             s=rng.random(n) * 3, log_eps12=log_eps[p1, p2].astype(np.float32), y=y, track=track)
    es = {"checkpoint_sha256": "ab" * 32, "files_sha256": {"train_release.npz": sha256_file(run / "embeddings" / "train_release.npz")}}
    (run / "embeddings" / "embeddings_summary.json").write_text(json.dumps(es))
    hs = {"files_sha256": {f: sha256_file(run / "head_emb" / f) for f in ("head.npz", "scores_train_release.npz")}}
    (run / "head_emb" / "head_summary.json").write_text(json.dumps(hs))
    sizes = [69, 206, 69]
    cfg = {"embed": {"expected_checkpoint_sha256": "ab" * 32}, "splits": {"sizes": sizes, "margin_points": 20}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    return run, tmp_path / "c.yaml", cfg


def _all_keys(d):
    if isinstance(d, dict):
        for k, v in d.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(d, list):
        for v in d:
            yield from _all_keys(v)


def test_end_to_end_synthetic_run(tmp_path):
    run, cfgp, _ = _fake_run(tmp_path)
    summ = S.main(["--run", str(run), "--config", str(cfgp)])
    out = run / "splits"
    assert (out / "stage4_summary.json").exists()
    assert len(summ["seeds"]) == 5
    for r in summ["seeds"]:
        sd = r["seed"]
        z = np.load(out / f"seed{sd}.npz")
        sp = z["track_split"]
        assert (np.bincount(sp) == [69, 206, 69]).all() and len(sp) == 344
        assert r["gate"]["conformal_rank"] == S.conformal_rank(69, 0.05) == 67
        for v in ("eps2", "eps3"):
            e = r[v]
            assert 0 < e["fit_easy_mass_fraction"] < 1 and 0 < e["cert_easy_mass_fraction"] < 1
            for lab in ("easy", "hard"):
                x = e[lab]
                assert x["fit_frames_gated_in"] > 0 and x["grid_points"] >= 2
                assert 0 <= x["fit_base_error_frame_weighted"] <= 1
                for o in x["orders"].values():
                    assert 0 <= o["start_index"] < x["grid_points"] and o["n_tested_at_most"] == o["start_index"] + 1
        # the gate keeps ~95% of Fit one-frame-per-track scores by construction
        s = np.load(run / "head_emb" / "scores_train_release.npz")["s"]
        track = np.load(run / "head_emb" / "scores_train_release.npz")["track"]
        rows = z["gate_rows_fit"]
        assert (s[rows] <= r["gate"]["tau"]).mean() >= 0.94
        assert (sp[track[rows]] == 0).all()
    # same seed twice -> identical files
    h1 = {k: v for k, v in summ["files_sha256"].items()}
    S.main(["--run", str(run), "--config", str(cfgp)])
    assert h1 == {f"seed{s}.npz": sha256_file(out / f"seed{s}.npz") for s in range(5)}


def test_summary_contains_no_cert_or_test_error_numbers(tmp_path):
    # The plan forbids looking at Cert/Test errors before the pre-registration commit.
    run, cfgp, _ = _fake_run(tmp_path, seed=1)
    summ = S.main(["--run", str(run), "--config", str(cfgp)])
    keys = set(_all_keys(summ))
    bad = [k for k in keys if ("error" in k.lower() or "acc" in k.lower()) and not k.startswith("fit_")]
    assert not bad, bad
    z = np.load(run / "splits" / "seed0.npz")
    assert not [k for k in z.files if "cert" in k.lower() or "test" in k.lower()]


def test_tampered_inputs_and_wrong_checkpoint_are_refused(tmp_path):
    run, cfgp, cfg = _fake_run(tmp_path)
    cfg["embed"]["expected_checkpoint_sha256"] = "cd" * 32
    cfgp.write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError):
        S.main(["--run", str(run), "--config", str(cfgp)])
    cfg["embed"]["expected_checkpoint_sha256"] = "ab" * 32
    cfgp.write_text(yaml.safe_dump(cfg))
    f = run / "head_emb" / "scores_train_release.npz"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(AssertionError):
        S.main(["--run", str(run), "--config", str(cfgp)])


def test_wrong_number_of_tracks_is_refused(tmp_path):
    run, cfgp, cfg = _fake_run(tmp_path)
    cfg["splits"]["sizes"] = [261, 784, 262]
    cfgp.write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError):
        S.main(["--run", str(run), "--config", str(cfgp)])


def test_primary_base_error_is_the_per_track_mean_and_drives_relative_alpha(tmp_path):
    # Decision 2026-10-10: per-track mean is primary; alpha "rel_0.5base" = 0.5 x it; the pooled version is a sensitivity.
    run, cfgp, _ = _fake_run(tmp_path, seed=2)
    summ = S.main(["--run", str(run), "--config", str(cfgp)])
    for r in summ["seeds"]:
        for v in ("eps2", "eps3"):
            for lab in ("easy", "hard"):
                x = r[v][lab]
                assert x["fit_base_error_primary"] == x["fit_base_error_per_track_mean"]
                assert abs(x["orders"]["rel_0.5base"]["alpha"] - 0.5 * x["fit_base_error_per_track_mean"]) < 1e-12
                assert abs(x["orders"]["rel_0.5base_pooled"]["alpha"] - 0.5 * x["fit_base_error_frame_weighted"]) < 1e-12
    z = np.load(run / "splits" / "seed0.npz")
    assert "eps2_easy_EL_rel_0.5base" in z.files and "eps2_easy_EL_rel_0.5base_pooled" in z.files
