"""Stage 1 tests: logic checks on synthetic data (no real GTSRB, no downloads, CPU only).

They prove the split/augmentation/training-loop plumbing works. They do NOT prove the real
accuracy; the real run's sanity number does that.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

try:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
except OSError as e:                      # e.g. WinError 4551: Application Control blocked torch's DLLs (importorskip only catches ImportError)
    pytest.skip(f"torch cannot be loaded here ({type(e).__name__}); Stage 1 tests skipped", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train as T  # noqa: E402

AUG0 = {"crop_area_scale": [1.0, 1.0], "crop_aspect": [1.0, 1.0], "rotation_deg": 0,
        "contrast": 0.0, "brightness": 0.0}
AUG = {"crop_area_scale": [0.85, 1.0], "crop_aspect": [0.9, 1.1], "rotation_deg": 10,
       "contrast": 0.2, "brightness": 0.0}


def test_stratified_split_disjoint_covers_and_keeps_every_class():
    # Parts must not overlap, must cover everything, and every class (even a 2-image one) is in both.
    labels = np.array([0] * 50 + [1] * 10 + [2] * 2)
    tr, va = T.stratified_split(labels, 0.1, seed=0)
    assert not set(tr) & set(va)
    assert sorted(np.concatenate([tr, va])) == list(range(len(labels)))
    for c in (0, 1, 2):
        assert (labels[tr] == c).any() and (labels[va] == c).any()
    tr2, va2 = T.stratified_split(labels, 0.1, seed=0)          # deterministic for a given seed
    assert (tr == tr2).all() and (va == va2).all()


def test_augment_is_identity_at_zero_strength():
    # With all strengths off the warp must return the input (guards the affine maths).
    x = torch.rand(4, 3, 32, 32)
    y = T.augment(x, torch.Generator().manual_seed(0), AUG0)
    assert torch.allclose(x, y, atol=1e-4)


def test_augment_range_shape_and_seed_reproducibility():
    x = torch.rand(8, 3, 32, 32)
    a = T.augment(x, torch.Generator().manual_seed(1), AUG)
    b = T.augment(x, torch.Generator().manual_seed(1), AUG)
    c = T.augment(x, torch.Generator().manual_seed(2), AUG)
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert not torch.allclose(a, x)                            # it really changes the images


def test_augment_never_flips():
    # A left-heavy image must stay left-heavy under any draw (flips are forbidden by the plan).
    x = torch.zeros(64, 3, 32, 32)
    x[..., :16] = 1.0
    out = T.augment(x, torch.Generator().manual_seed(3), AUG)
    assert (out[..., :16].mean(dim=(1, 2, 3)) > out[..., 16:].mean(dim=(1, 2, 3))).all()


def test_model_head_and_param_groups():
    m = T.build_model(pretrained=False)
    assert m.fc.out_features == 43 and m(torch.zeros(2, 3, 112, 112)).shape == (2, 43)


def _fake_cache(root: Path, n_test=300, n_train=120):
    """Tiny cache: class c images are filled with a class-specific colour, so they are learnable."""
    cache = root / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)

    def make(n):
        y = np.arange(n) % T.N_CLASSES
        x = np.stack([np.full((112, 112, 3), 5 * c + 10, np.uint8) for c in y])
        x = np.clip(x.astype(int) + rng.integers(0, 8, x.shape), 0, 255).astype(np.uint8)
        return x, y

    xt, yt = make(n_test)
    xr, yr = make(n_train)
    np.save(cache / "test_112.npy", xt)
    np.save(cache / "train_112.npy", xr)
    pd.DataFrame({"class_id": yt}).to_csv(cache / "test_meta.csv", index=False)
    pd.DataFrame({"class_id": yr}).to_csv(cache / "train_meta.csv", index=False)
    pd.DataFrame({"test_idx": [0, 5, 7]}).to_csv(cache / "excl.csv", index=False)
    cfg = {"data": {"cache_dir": str(cache), "image_size": 112}, "seed": 0,
           "train": {"exclude_file": str(cache / "excl.csv"), "val_fraction": 0.1, "epochs": 2,
                     "patience": 5, "batch_size": 32, "lr_backbone": 2e-4, "lr_head": 1e-3,
                     "weight_decay": 1e-4, "warmup_epochs": 1, "amp": True,
                     "run_dir": str(root / "runs"), "aug": AUG}}
    p = root / "poc.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_end_to_end_smoke_run_and_resume_guard(tmp_path):
    cfg = _fake_cache(tmp_path)
    args = ["--config", str(cfg), "--smoke", "--no-pretrained", "--device", "cpu",
            "--smoke-n", "200", "--tag", "t"]
    res = T.main(args)
    run = tmp_path / "runs" / "t_smoke"
    for f in ("best.pt", "last.pt", "history.csv", "results.json", "curves.png", "config_used.yaml"):
        assert (run / f).exists(), f
    r = json.loads((run / "results.json").read_text())
    assert r["n_excluded"] == 3 and r["n_pool"] == 200
    assert r["n_train"] + r["n_val"] == 200
    assert "NOT test accuracy" in r["sanity_accuracy_label"]
    assert len(r["checkpoint_sha256"]) == 64 and res["epochs_run"] == 2
    # a second run in the same folder must be refused unless --resume is given
    with pytest.raises(SystemExit):
        T.main(args)
    # resume after finishing: nothing left to train, but it must not crash and keeps the history
    again = T.main(args + ["--resume"])
    assert again["epochs_run"] == 2


def test_bad_exclusion_list_is_rejected(tmp_path):
    cfg = _fake_cache(tmp_path)
    c = yaml.safe_load(cfg.read_text())
    pd.DataFrame({"test_idx": [1, 1]}).to_csv(c["train"]["exclude_file"], index=False)  # duplicate rows
    with pytest.raises(AssertionError):
        T.main(["--config", str(cfg), "--smoke", "--no-pretrained", "--device", "cpu", "--tag", "x"])


def test_diagnostics_script_runs_on_smoke_checkpoint(tmp_path):
    # Read-only diagnostics must run on a real checkpoint and write their tables.
    cfg = _fake_cache(tmp_path)
    T.main(["--config", str(cfg), "--smoke", "--no-pretrained", "--device", "cpu",
            "--smoke-n", "200", "--tag", "d"])
    cache = tmp_path / "cache"
    n = 120
    cls = np.arange(n) % T.N_CLASSES
    pd.DataFrame({"class_id": cls, "width": 20 + np.arange(n) % 60, "height": 25 + np.arange(n) % 50,
                  "frame_idx": np.arange(n) % 30, "track_id": [f"{c:02d}_{i // 30:05d}" for i, c in enumerate(cls)]}
                 ).to_csv(cache / "train_meta.csv", index=False)
    pd.DataFrame({"class_id": cls, "width": 30, "height": 30}).to_csv(cache / "test_meta.csv", index=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import stage1_diagnostics as D
    D.main(["--run", str(tmp_path / "runs" / "d_smoke"), "--config", str(cfg), "--device", "cpu"])
    out = json.loads((tmp_path / "runs" / "d_smoke" / "diagnostics.json").read_text())
    assert 0 <= out["overall"] <= 1 and out["tracks_total"] >= 1
    assert (tmp_path / "runs" / "d_smoke" / "diagnostics_by_size.csv").exists()


def test_stage2_embeddings_checks_and_hash_guard(tmp_path):
    # Full Stage 1 (tiny, real mode) -> Stage 2. Checks the plan's Stage 2 checks and the hash guard.
    from src import embed as E
    cfg = _fake_cache(tmp_path)
    c = yaml.safe_load(cfg.read_text())
    cache = tmp_path / "cache"
    n = 120
    cls = np.arange(n) % T.N_CLASSES
    pd.DataFrame({"class_id": cls, "width": 40, "height": 40, "frame_idx": np.arange(n) % 30,
                  "track_id": [f"{k:02d}_{i // 30:05d}" for i, k in enumerate(cls)]}).to_csv(cache / "train_meta.csv", index=False)
    T.main(["--config", str(cfg), "--no-pretrained", "--device", "cpu", "--epochs", "1", "--tag", "e"])
    run = tmp_path / "runs" / "e"
    sha = T.sha256_file(run / "best.pt")
    c["embed"] = {"expected_checkpoint_sha256": sha, "batch_size": 64, "norm_tol": 1e-5}
    cfg.write_text(yaml.safe_dump(c))
    s = E.main(["--run", str(run), "--config", str(cfg), "--device", "cpu"])
    z = np.load(run / "embeddings" / "train_release.npz")
    assert z["emb"].shape == (120, 512) and z["logits"].shape == (120, 43) and len(z["track"]) == 120
    assert np.abs(np.linalg.norm(z["emb"].astype(np.float64), axis=1) - 1).max() < 1e-5
    f = np.load(run / "embeddings" / "ft_train.npz")
    assert len(f["emb"]) == s["n_ft_train"] and (f["y"] == pd.read_csv(cache / "test_meta.csv")["class_id"].to_numpy()[f["test_row"]]).all()
    # a wrong frozen hash must stop everything
    c["embed"]["expected_checkpoint_sha256"] = "0" * 64
    cfg.write_text(yaml.safe_dump(c))
    with pytest.raises(AssertionError):
        E.main(["--run", str(run), "--config", str(cfg), "--device", "cpu"])
