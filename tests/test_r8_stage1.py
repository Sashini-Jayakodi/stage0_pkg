"""Tests for POC-2 Stage 1 (src/r8_stage1.py) on synthetic data. No real GTSRB is read.
Brute-force references are plain loops written independently of the vectorised code."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import r8_stage1 as R  # noqa: E402
from src.train import sha256_file  # noqa: E402

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ brute-force references
def brute_scores(ll, passed, log_prior, variant):
    n, K = ll.shape
    S = np.full((n, K), np.nan)
    for t in range(n):
        idx = [i for i in range(t + 1) if passed[i]]
        if not idx:
            continue
        if variant == "V3":
            idx = idx[-R.WINDOW:]
        tot = log_prior + sum(ll[i] for i in idx)
        S[t] = tot / len([i for i in range(t + 1) if passed[i]]) if variant == "V2" else tot
    return S


def brute_first_commit(M, lam):
    for t, m in enumerate(M):
        if m >= lam:
            return t
    return len(M)


# ------------------------------------------------------------------ unit tests of the rule
@pytest.mark.parametrize("variant", R.VARIANTS)
def test_running_scores_match_brute_force(variant):
    for n in (1, 3, 7, 29, 30):
        ll = rng.normal(size=(n, 5)) * 3
        passed = rng.random(n) < 0.7
        lp = np.log(rng.dirichlet(np.ones(5)))
        S, _ = R.running_scores(ll, passed, lp, variant)
        Sb = brute_scores(ll, passed, lp, variant)
        assert np.array_equal(np.isnan(S), np.isnan(Sb))
        assert np.allclose(S[~np.isnan(S)], Sb[~np.isnan(Sb)])


def test_no_commit_on_prior_alone():
    """Frames before the first passing frame carry no evidence: even lambda = 0 cannot commit there."""
    ll = rng.normal(size=(6, 4))
    passed = np.array([False, False, True, False, True, True])
    for v in R.VARIANTS:
        M, top = R.margins(R.running_scores(ll, passed, np.log(np.full(4, 0.25)), v)[0])
        assert np.isneginf(M[:2]).all() and (top[:2] == -1).all()
        assert R.first_commit(M, np.array([0.0]))[0] == 2


def test_track_with_no_passing_frame_never_commits():
    M, _ = R.margins(R.running_scores(rng.normal(size=(5, 4)), np.zeros(5, bool), np.zeros(4), "V1")[0])
    assert (R.first_commit(M, np.array([0.0, 1.0])) == 5).all()


def test_first_commit_matches_brute_force_for_every_lambda():
    for _ in range(50):
        M = rng.normal(size=int(rng.integers(1, 31))) * 5
        M[: int(rng.integers(0, 3))] = -np.inf
        grid = np.sort(np.concatenate([[0.0], rng.random(20) * 10]))
        assert list(R.first_commit(M, grid)) == [brute_first_commit(M, g) for g in grid]


def test_committed_label_can_change_with_lambda():
    """Risk R3: early weak evidence for class 0, later strong evidence for class 1. Small lambda commits early to 0, large
    lambda commits later to 1. A table built from one score per track would get one of these wrong."""
    ll = np.array([[0.0, -1.0, -9.0], [0.0, -1.0, -9.0], [-30.0, 0.0, -30.0], [-30.0, 0.0, -30.0]])
    M, top = R.margins(R.running_scores(ll, np.ones(4, bool), np.zeros(3), "V1")[0])
    idx = R.first_commit(M, np.array([0.5, 20.0]))
    assert top[idx[0]] == 0 and top[idx[1]] == 1 and idx[0] < idx[1]


def test_variable_track_lengths_and_frame_order():
    track = np.array([1, 0, 1, 0, 2, 1])
    frame = np.array([5, 3, 1, 0, 9, 2])
    rows = R.track_rows(track, frame)
    assert [list(frame[r]) for r in rows] == [[0, 3], [1, 2, 5], [9]]


# ------------------------------------------------------------------ end-to-end on a fake POC-1 run folder
CK = "8652735" + "a" * 53 + "0bcc"


def fake_run(tmp, n_tracks=12, K=R.K, d=8, seed=1):
    r = np.random.default_rng(seed)
    run, cache = tmp / "run", tmp / "cache"
    for p in (run / "embeddings", run / "head_emb", run / "splits", cache):
        p.mkdir(parents=True, exist_ok=True)
    mu = r.normal(size=(K, d)) * 2
    lens = [30] * (n_tracks - 1) + [29]                                   # one 29-frame track, as in GTSRB
    ys = r.integers(0, K, n_tracks)
    track = np.concatenate([np.full(L, t) for t, L in enumerate(lens)])
    frame = np.concatenate([np.arange(L) for L in lens])
    y = ys[track]
    noise = np.concatenate([np.linspace(1.5, 0.3, L) for L in lens])[:, None]    # early frames noisier, like small signs
    emb = (mu[y] + r.normal(size=(len(y), d)) * noise).astype(np.float32)
    head = {"mu": mu, "W": np.eye(d), "priors": np.full(K, 1.0 / K)}
    names = np.array([f"{ys[t]:02d}_{t:05d}" for t in range(n_tracks)])
    np.savez(run / "embeddings" / "train_release.npz", emb=emb, y=y, track=track, track_names=names, frame=frame)
    np.savez(run / "head_emb" / "head.npz", **head)
    from src.head import mahal_sq
    D2 = mahal_sq(head, emb)
    lp = np.log(head["priors"])[None] - 0.5 * D2
    top3 = np.argsort(-lp, 1)[:, :3]
    np.savez(run / "head_emb" / "scores_train_release.npz", top3=top3, s=np.sqrt(D2.min(1)).astype(np.float32), y=y)
    acc = float((top3[:, 0] == y).mean())
    split = np.array([0] * (n_tracks // 2) + [1] * (n_tracks - n_tracks // 2))   # first half = design
    np.savez(run / "splits" / "seed0.npz", track_split=split, track_names=names)
    tau = float(np.quantile(np.sqrt(D2.min(1)), 0.95))
    (run / "embeddings" / "embeddings_summary.json").write_text(json.dumps(
        {"checkpoint_sha256": CK, "files_sha256": {"train_release.npz": sha256_file(run / "embeddings" / "train_release.npz")}}))
    (run / "head_emb" / "head_summary.json").write_text(json.dumps(
        {"train_release": {"head_accuracy": acc},
         "files_sha256": {f: sha256_file(run / "head_emb" / f) for f in ("head.npz", "scores_train_release.npz")}}))
    (run / "splits" / "stage4_summary.json").write_text(json.dumps(
        {"seeds": [{"seed": 0, "gate": {"tau": tau}}], "files_sha256": {"seed0.npz": sha256_file(run / "splits" / "seed0.npz")}}))
    roi = 15 + 3 * frame + r.integers(0, 2, len(frame))                   # sign grows with frame index
    pd.DataFrame({"class_id": y, "track_id": names[track], "frame_idx": frame, "roi_x1": 5, "roi_y1": 5,
                  "roi_x2": 5 + roi, "roi_y2": 6 + roi}).to_csv(cache / "train_meta.csv", index=False)
    cfg = {"data": {"cache_dir": str(cache)}, "embed": {"expected_checkpoint_sha256": CK}}
    (tmp / "c.yaml").write_text(yaml.safe_dump(cfg))
    return run, tmp / "c.yaml", n_tracks


@pytest.fixture
def small_design(monkeypatch):
    monkeypatch.setattr(R, "N_GRID", 40)
    # the fake run has 6 design tracks, not 261
    orig = R.run_checks

    def rc(inp, B):
        res, ok = orig(inp, B)
        ok["C6_design_set_is_261_tracks"] = int(B["design"].sum()) == 6
        return res, ok
    monkeypatch.setattr(R, "run_checks", rc)


def test_end_to_end_all_checks_pass_and_pool_file_is_label_free(tmp_path, small_design):
    run, cfgp, n = fake_run(tmp_path)
    s = R.main(["--run", str(run), "--config", str(cfgp)])
    assert s["all_pass"], s["checks"]
    out = run / "poc2" / "stage1"
    des, pool = np.load(out / "track_table_design.npz"), np.load(out / "track_table_pool.npz")
    assert not any(k.endswith(("wrong", "wrong_last", "y_track")) for k in pool.files)
    assert len(des["track_names"]) == 6 and len(pool["track_names"]) == n - 6
    split = np.load(run / "splits" / "seed0.npz")
    design_names = set(split["track_names"][split["track_split"] == 0])
    assert set(des["track_names"]) == design_names and not set(pool["track_names"]) & design_names
    for v in R.VARIANTS:
        c, w = des[f"{v}_committed"], des[f"{v}_wrong"]
        assert (w <= c).all()                                             # wrong only where committed
        assert c.shape == (6, 40) and (c[:, 0] == (des["n_passing"] > 0)).all()   # lambda = 0 commits at the first passing frame
        assert np.isnan(des[f"{v}_size"][~c]).all() and (des[f"{v}_pos"][~c] == -1).all()


def test_tampered_inputs_are_refused(tmp_path, small_design):
    run, cfgp, _ = fake_run(tmp_path)
    p = run / "splits" / "seed0.npz"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(AssertionError, match="changed since Stage 4"):
        R.main(["--run", str(run), "--config", str(cfgp)])


def test_wrong_checkpoint_is_refused(tmp_path, small_design):
    run, cfgp, _ = fake_run(tmp_path)
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["embed"]["expected_checkpoint_sha256"] = "ff" * 32
    cfgp.write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError, match="frozen checkpoint"):
        R.main(["--run", str(run), "--config", str(cfgp)])


def test_misaligned_meta_is_refused(tmp_path, small_design):
    run, cfgp, _ = fake_run(tmp_path)
    mp = Path(yaml.safe_load(cfgp.read_text())["data"]["cache_dir"]) / "train_meta.csv"
    m = pd.read_csv(mp)
    m.loc[3, "frame_idx"] = 99
    m.to_csv(mp, index=False)
    with pytest.raises(AssertionError, match="do not line up"):
        R.main(["--run", str(run), "--config", str(cfgp)])
