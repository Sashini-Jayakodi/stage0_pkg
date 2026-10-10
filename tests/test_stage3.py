"""Stage 3 tests on synthetic Gaussian data (no real GTSRB)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import head as H  # noqa: E402
from src.train import sha256_file  # noqa: E402

D = 24


def make_data(n_per=40, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(H.K, D)) * sep / np.sqrt(D)
    y = np.repeat(np.arange(H.K), n_per)
    return means[y] + rng.normal(size=(len(y), D)) * 0.5, y, means


def test_ledoit_wolf_matches_sklearn():
    skl = pytest.importorskip("sklearn.covariance")
    X, y, _ = make_data()
    Xc = X - np.stack([X[y == k].mean(0) for k in range(H.K)])[y]
    cov, s = H.ledoit_wolf(Xc)
    ref = skl.LedoitWolf(assume_centered=True).fit(Xc)
    assert abs(s - ref.shrinkage_) < 1e-10 and np.allclose(cov, ref.covariance_, atol=1e-10)


def test_whitened_distance_equals_direct_mahalanobis_and_posterior_normalised():
    X, y, _ = make_data()
    h = H.fit_head(X, y)
    D2 = H.mahal_sq(h, X[:5])
    for r in range(5):
        for k in (0, 7, 42):
            d = X[r] - h["mu"][k]
            assert abs(D2[r, k] - d @ h["prec"] @ d) < 1e-6 * max(1, D2[r, k])
    s = H.score(h, X)
    assert (s["margin"] >= 0).all() and (s["lp3"][:, 0] >= s["lp3"][:, 1]).all()
    assert (s["top3"][:, 0] == y).mean() > 0.9
    assert np.allclose(s["s"], np.sqrt(H.mahal_sq(h, X).min(1)), atol=1e-4)


def test_eps_table_symmetric_log_finite_and_avoids_underflow():
    X, y, _ = make_data(sep=400.0)                      # absurdly well separated: plain eps underflows
    h = H.fit_head(X, y)
    delta, le = H.eps_table(h)
    off = ~np.eye(H.K, dtype=bool)
    assert np.allclose(delta[off], delta.T[off]) and np.allclose(le[off], le.T[off])
    assert np.isfinite(le[off]).all()
    from scipy.stats import norm
    assert (norm.cdf(-delta[off] / 2) == 0).any()        # the plain version really would be 0
    assert (np.diff(np.sort(le[off])) >= 0).all()


def test_pair_table_and_correlation_report():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(H.K), 20)
    pred = y.copy()
    flip = (y == 3) & (rng.random(len(y)) < .5)          # class 3 -> 4 confusion planted
    pred[flip] = 4
    le = -np.abs(np.arange(H.K)[:, None] - np.arange(H.K)[None, :]).astype(float)   # eps larger for neighbours
    np.fill_diagonal(le, np.nan)
    p = H.pair_table(le, y, pred)
    assert len(p) == 903 and p.loc[(p.i == 3) & (p.j == 4), "confusions"].iloc[0] > 0
    r = H.correlation_report(p)
    assert r["top10_empirical_in_top20_eps"] >= 1 and r["n_pairs_with_zero_confusion"] > 800


def test_stallkamp_order_covers_all_classes_once():
    assert sorted(H.stallkamp_order()) == list(range(43)) and len(H.NAMES) == 43


def test_end_to_end_on_synthetic_stage2_outputs(tmp_path):
    X, y, _ = make_data(n_per=40)
    Xn = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    run = tmp_path / "run"; (run / "embeddings").mkdir(parents=True)
    rng = np.random.default_rng(2)
    ytr = np.repeat(np.arange(H.K), 90)                    # 3 tracks x 30 frames per class
    _, _, means = make_data()
    Xtr = means[ytr] + rng.normal(size=(len(ytr), D)) * 0.9
    Xtr = (Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)).astype(np.float32)
    W = rng.normal(size=(D, H.K)).astype(np.float32)
    track = np.repeat(np.arange(H.K * 3), 30).astype(np.int32)
    np.savez(run / "embeddings" / "ft_train.npz", emb=Xn, raw=Xn, logits=Xn @ W, y=y)
    np.savez(run / "embeddings" / "train_release.npz", emb=Xtr, raw=Xtr, logits=Xtr @ W, y=ytr, track=track,
             track_names=np.array([f"{k // 3:02d}_{k % 3:05d}" for k in range(H.K * 3)]), frame=np.tile(np.arange(30), H.K * 3))
    summ = {"checkpoint_sha256": "ab" * 32,
            "files_sha256": {f: sha256_file(run / "embeddings" / f) for f in ("ft_train.npz", "train_release.npz")}}
    (run / "embeddings" / "embeddings_summary.json").write_text(json.dumps(summ))
    cache = tmp_path / "cache"; cache.mkdir()
    meta = pd.DataFrame({"class_id": ytr, "track_id": [f"{k // 3:02d}_{k % 3:05d}" for k in track], "frame_idx": np.tile(np.arange(30), H.K * 3)})
    meta.to_csv(cache / "train_meta.csv", index=False)
    np.save(cache / "train_16.npy", rng.integers(0, 255, (len(ytr), 16, 16, 3), dtype=np.uint8))
    cfg = {"data": {"cache_dir": str(cache), "image_size": 16}, "embed": {"expected_checkpoint_sha256": "ab" * 32}}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    res = H.main(["--run", str(run), "--config", str(tmp_path / "c.yaml")])
    out = run / "head_emb"
    for f in ("head.npz", "scores_train_release.npz", "scores_ft_train.npz", "head_summary.json", "pairs_903.csv",
              "top10_confusable_pairs.csv", "atlas.png", "top10_pairs.png"):
        assert (out / f).exists(), f
    s = np.load(out / "scores_train_release.npz")
    assert len(s["margin"]) == len(ytr) and np.isfinite(s["log_eps12"]).all()
    assert 0 <= res["train_release"]["head_accuracy"] <= 1
    # tampering with an input file must be caught
    (run / "embeddings" / "ft_train.npz").write_bytes((run / "embeddings" / "ft_train.npz").read_bytes() + b"x")
    with pytest.raises(AssertionError):
        H.main(["--run", str(run), "--config", str(tmp_path / "c.yaml")])
    # embeddings from a different checkpoint must be refused
    cfg["embed"]["expected_checkpoint_sha256"] = "cd" * 32
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(AssertionError):
        H.main(["--run", str(run), "--config", str(tmp_path / "c.yaml")])
