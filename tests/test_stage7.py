"""Stage 7: selection rules (pure), Fit-only data access, smoke test (no torch), freeze rules, and an end-to-end calibration on a tiny fake run."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import scenarios as SC  # noqa: E402
from src import stage7 as S7  # noqa: E402
from src.util import sha256_file  # noqa: E402
try:
    SC._alb()
except RuntimeError as e:
    pytest.skip(f"albumentations/OpenCV not importable here: {e}", allow_module_level=True)

SIZE, NT, FR = 48, 12, 8                      # 12 tracks of 8 frames at 48 px; tracks 0-5 = Fit, 6-9 = Cert, 10-11 = Test
SPLIT = np.array([0] * 6 + [1] * 4 + [2] * 2)


# ---------------------------------------------------------------- pure selection logic
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def test_preregistered_constants_are_the_ones_in_the_plan():
    assert S7.BANDS == {"mild": (3.0, 8.0), "severe": (15.0, 30.0)}
    assert S7.MIDPOINT == {"mild": 5.5, "severe": 22.5}
    assert S7.DEFAULTS7["seed"] == 0                                        # Fit of reshuffle seed 0


def test_pick_level_takes_the_level_closest_to_the_midpoint_inside_the_band():
    drops = [0.5, 1.5, 2.9, 4.0, 6.0, 8.5, 12.0, 16.0, 21.0, 33.0]
    i, how = S7.pick_level(GRID, drops, "mild")
    assert GRID[i] == 0.5 and how["band_reached"]                           # 6.0 is nearer 5.5 than 4.0
    i, how = S7.pick_level(GRID, drops, "severe")
    assert GRID[i] == 1.5 and how["band_reached"]                           # 21.0 nearer 22.5 than 16.0; 33.0 is outside the band


def test_pick_level_ties_go_to_the_smaller_severity():
    drops = [0, 0, 0, 4.5, 0, 6.5, 0, 0, 0, 0]                              # 4.5 and 6.5 are both 1.0 from 5.5
    i, _ = S7.pick_level(GRID, drops, "mild")
    assert GRID[i] == 0.4


def test_pick_level_flags_a_missed_band():
    drops = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]              # never reaches 3 pp
    i, how = S7.pick_level(GRID, drops, "mild")
    assert not how["band_reached"] and GRID[i] == 2.0
    i, how = S7.pick_level(GRID, drops, "severe")                           # never reaches 15 pp -> strongest allowed level, flagged
    assert not how["band_reached"] and GRID[i] == GRID[-1] and "cap" in how["rule"]
    jump = [0, 1, 2, 2.5, 9, 9.5, 40, 41, 42, 43]                           # jumps over the severe band: closest to 22.5, flagged
    i, how = S7.pick_level(GRID, jump, "severe")
    assert not how["band_reached"] and GRID[i] == 0.75        # 9.5 pp is nearer 22.5 than 40 pp


def test_choose_levels_and_order_flag():
    sweep = {sc: {"grid": GRID, "drop_pp": [0.5, 1, 2, 3.5, 5.5, 9, 14, 20, 24, 28]} for sc in SC.SCENARIOS}
    ch = S7.choose_levels(sweep)
    for sc in SC.SCENARIOS:
        assert ch[sc]["mild"]["severity"] == 0.5 and ch[sc]["severe"]["severity"] == 1.5 and ch[sc]["ordered"]
    sweep["fog"]["drop_pp"] = [50, 0, 0, 0, 0, 0, 0, 0, 0, 0]               # degenerate: only the first level is in neither band
    ch = S7.choose_levels(sweep)
    assert not ch["fog"]["ordered"] and ch["rain"]["ordered"]


def _chosen(drops_by_sc):
    return {sc: {"mild": {"severity": 0.5, "fit_drop_pp": m, "band_reached": True, "rule": "r"},
                 "severe": {"severity": 2.0, "fit_drop_pp": v, "band_reached": True, "rule": "r"}, "ordered": True} for sc, (m, v) in drops_by_sc.items()}


def test_choose_cells_takes_the_two_largest_and_breaks_ties_in_scenario_order_then_severe_first():
    ch = _chosen({"rain": (5, 20), "fog": (6, 25), "night": (5, 20), "side_view": (4, 18), "motion_blur": (5, 24)})
    cells = S7.choose_cells(ch)
    assert [(c["scenario"], c["level"]) for c in cells] == [("fog", "severe"), ("motion_blur", "severe")]
    ch = _chosen({"rain": (5, 20), "fog": (5, 20), "night": (5, 20), "side_view": (5, 20), "motion_blur": (5, 20)})
    cells = S7.choose_cells(ch)
    assert [(c["scenario"], c["level"]) for c in cells] == [("rain", "severe"), ("fog", "severe")]
    ch = _chosen({"rain": (21, 5), "fog": (1, 1), "night": (1, 1), "side_view": (1, 1), "motion_blur": (1, 1)})
    cells = S7.choose_cells(ch)
    assert cells[0]["scenario"] == "rain" and cells[0]["level"] == "mild" and cells[0]["fit_drop_pp"] == 21


def test_drop_stats_matches_hand_computation():
    n_t = np.array([10., 10., 10., 10.])
    clean = np.array([10., 9., 10., 8.])
    corr = np.array([7., 9., 6., 8.])
    d, se, a_clean, a_corr = S7.drop_stats(n_t, clean, corr, 200, np.random.default_rng(0))
    assert abs(d - 100 * (37 - 30) / 40) < 1e-12 and abs(a_clean - 92.5) < 1e-12 and abs(a_corr - 75.0) < 1e-12
    assert se > 0
    d2, se2, *_ = S7.drop_stats(n_t, clean, clean, 50, np.random.default_rng(0))
    assert d2 == 0 and se2 == 0


# ---------------------------------------------------------------- a tiny fake run (Stage 1-4 files in the real layout)
def _fake_run(tmp_path, with_model=True):
    rng = np.random.default_rng(0)
    n = NT * FR
    track = np.repeat(np.arange(NT), FR).astype(np.int32)
    frame = np.tile(np.arange(FR), NT).astype(np.int16)
    y = track.astype(np.int64)                                              # track k has class k
    names = np.array([f"{k:02d}_{k:05d}" for k in range(NT)])
    imgs = np.empty((n, SIZE, SIZE, 3), np.uint8)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for i in range(n):
        disc = ((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2) < (SIZE * 0.35) ** 2
        im = np.full((SIZE, SIZE, 3), 100.0)
        im[disc] = np.array([30 + 17 * (y[i] % 12), 200 - 11 * (y[i] % 12), 90 + 5 * y[i]])
        imgs[i] = np.clip(im + rng.normal(0, 4, im.shape), 0, 255).astype(np.uint8)
    cache = tmp_path / "cache"; cache.mkdir()
    np.save(cache / f"train_{SIZE}.npy", imgs)
    import pandas as pd
    pd.DataFrame({"class_id": y, "frame_idx": frame.astype(int), "track_id": names[track]}).to_csv(cache / "train_meta.csv", index=False)
    run = tmp_path / "run"
    for d in ("embeddings", "head_emb", "splits"):
        (run / d).mkdir(parents=True)
    (run / "config_used.yaml").write_text(yaml.safe_dump({"image_size": SIZE}))
    np.savez(run / "splits" / "seed0.npz", track_names=names, track_split=SPLIT.astype(np.int8))
    (run / "splits" / "stage4_summary.json").write_text(json.dumps({"files_sha256": {"seed0.npz": sha256_file(run / "splits" / "seed0.npz")}}))
    cfg = {"data": {"cache_dir": str(cache)}, "embed": {"expected_checkpoint_sha256": "x"},
           "stage7": {"stride": 2, "boot": 20, "smoke_images": 4, "batch": 16, "det_check_frames": 10}}
    K = 43
    head = {"mu": rng.normal(size=(K, 512)), "W": np.eye(512), "prec": np.eye(512), "priors": np.full(K, 1.0 / K)}
    if with_model:
        torch = pytest.importorskip("torch")
        from src import embed as EM
        from src import train as TR
        torch.manual_seed(0)
        model = TR.build_model(pretrained=False)
        model.eval()
        torch.save({"model": model.state_dict()}, run / "best.pt")
        raw, _ = EM.extract(model, imgs, torch.device("cpu"), 16)
        emb = EM.l2_normalise(raw)
        head["mu"] = np.stack([emb[y == k].mean(0) if (y == k).any() else rng.normal(size=512) for k in range(K)])
        from src import head as HD
        top3 = HD.score(head, emb)["top3"]
        cfg["embed"]["expected_checkpoint_sha256"] = sha256_file(run / "best.pt")
    else:
        emb = rng.normal(size=(n, 512)).astype(np.float32)
        top3 = np.zeros((n, 3), np.int16)
    np.savez(run / "embeddings" / "train_release.npz", emb=emb, y=y, track=track, track_names=names, frame=frame)
    (run / "embeddings" / "embeddings_summary.json").write_text(json.dumps(
        {"checkpoint_sha256": cfg["embed"]["expected_checkpoint_sha256"], "files_sha256": {"train_release.npz": sha256_file(run / "embeddings" / "train_release.npz")}}))
    np.savez(run / "head_emb" / "head.npz", **head)
    np.savez(run / "head_emb" / "scores_train_release.npz", top3=top3, y=y, track=track, frame=frame)
    (run / "head_emb" / "head_summary.json").write_text(json.dumps(
        {"files_sha256": {f: sha256_file(run / "head_emb" / f) for f in ("head.npz", "scores_train_release.npz")}}))
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    return run, cfgp, cfg


def test_only_fit_rows_are_indexed_and_loaded(tmp_path):
    run, cfgp, cfg = _fake_run(tmp_path, with_model=False)
    fit = S7.fit_index(run, cfg, 2)
    assert set(fit["track_code"]) == set(range(6)) and (fit["frames"] % 2 == 0).all() and len(fit["rows"]) == 6 * 4
    loaded = S7.load_fit(run, cfg, {**S7.DEFAULTS7, **cfg["stage7"]})
    assert loaded["images"].shape == (24, SIZE, SIZE, 3)
    full = np.load(Path(cfg["data"]["cache_dir"]) / f"train_{SIZE}.npy")
    assert np.array_equal(loaded["images"], full[fit["rows"]])
    assert (SPLIT[loaded["track_code"]] == 0).all()                         # no Cert or Test frame anywhere


def test_misaligned_cache_or_changed_split_is_refused(tmp_path):
    run, cfgp, cfg = _fake_run(tmp_path, with_model=False)
    import pandas as pd
    mp = Path(cfg["data"]["cache_dir"]) / "train_meta.csv"
    m = pd.read_csv(mp); m.loc[0, "class_id"] = 99; m.to_csv(mp, index=False)
    with pytest.raises(AssertionError, match="do not line up"):
        S7.load_fit(run, cfg, {**S7.DEFAULTS7, **cfg["stage7"]})
    (tmp_path / "b").mkdir()
    run2, _, cfg2 = _fake_run(tmp_path / "b", with_model=False)
    sp = np.load(run2 / "splits" / "seed0.npz")
    np.savez(run2 / "splits" / "seed0.npz", track_names=sp["track_names"], track_split=np.zeros(NT, np.int8))
    with pytest.raises(AssertionError, match="changed since Stage 4"):
        S7.fit_index(run2, cfg2, 2)


def test_smoke_test_runs_without_torch_and_is_deterministic(tmp_path):
    run, cfgp, cfg = _fake_run(tmp_path, with_model=False)
    out = tmp_path / "smoke"
    info = S7.main(["--run", str(run), "--config", str(cfgp), "--smoke", "--out", str(out)])
    assert (out / "contact_sheet.png").exists() and info["deterministic_under_reordering"]
    assert set(info["classes"]) <= set(range(6))                            # Fit tracks only (class = track in the fake run)
    assert len(info["sha256_by_cell"]) == 15                                # 5 scenarios x 3 strengths
    again = S7.main(["--run", str(run), "--config", str(cfgp), "--smoke", "--out", str(tmp_path / "smoke2")])
    assert again["sha256_by_cell"] == info["sha256_by_cell"]               # same hashes on a re-run


def test_smoke_test_works_when_torch_is_blocked(tmp_path):
    # Windows Application Control can block torch's DLLs (OSError WinError 4551). albumentations imports torch too and only tolerates ImportError,
    # so scenarios._alb() hides torch from it. Run the real smoke command in a fresh interpreter whose torch import raises that OSError.
    (tmp_path / "r").mkdir()
    run, cfgp, cfg = _fake_run(tmp_path / "r", with_model=False)
    (tmp_path / "blk" / "torch").mkdir(parents=True)
    (tmp_path / "blk" / "torch" / "__init__.py").write_text("raise OSError('[WinError 4551] Application Control policy has blocked this file')")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path / "blk"), str(ROOT)])}
    out = tmp_path / "smoke_blocked"
    r = subprocess.run([sys.executable, "-m", "src.stage7", "--run", str(run), "--config", str(cfgp), "--smoke", "--out", str(out)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "smoke test OK" in r.stdout, (r.stdout[-500:], r.stderr[-800:])
    assert (out / "contact_sheet.png").exists()
    # and the pixels are the same as in an unblocked interpreter
    here = S7.main(["--run", str(run), "--config", str(cfgp), "--smoke", "--out", str(tmp_path / "smoke_open")])
    assert json.loads((out / "smoke.json").read_text())["sha256_by_cell"] == here["sha256_by_cell"]


# ---------------------------------------------------------------- freeze
def _results_for_freeze(run, out, drops=None, ordered=True):
    sweep = {sc: {"grid": GRID, "drop_pp": drops or [0.5, 1, 2, 3.5, 5.5, 9, 14, 20, 24, 28]} for sc in SC.SCENARIOS}
    ch = S7.choose_levels(sweep)
    if not ordered:
        ch["fog"]["ordered"] = False
    res = {"seed": 0, "stride": 3, "fit_tracks": 261, "fit_frames_used": 2600, "bands_pp": S7.BANDS, "midpoints_pp": S7.MIDPOINT, "clean_accuracy_pct": 99.1,
           "sweep": sweep, "chosen": ch, "recertification_cells": S7.choose_cells(ch), "checkpoint_sha256": "ab" * 32, "device": "cpu", "torch": "t",
           "versions": SC.versions(), "caps": SC.CAP, "scenarios_py_sha256": sha256_file(Path(SC.__file__)),
           "fit_sample_sha256": {f"{sc}_{k}": "00" * 32 for sc in SC.SCENARIOS for k in ("mild", "severe")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage7_results.json").write_text(json.dumps(res))
    return res


def test_freeze_writes_once_and_refuses_to_overwrite(tmp_path):
    out, fz = tmp_path / "s7", tmp_path / "configs" / "frozen_g2.yaml"
    _results_for_freeze(tmp_path, out)
    doc = S7.run_freeze(tmp_path / "runX", out, fz)
    saved = yaml.safe_load(fz.read_text())
    assert saved["g2_frozen"] and saved["scenarios"]["rain"]["mild"]["severity"] == 0.5 and len(saved["recertification_cells"]) == 2
    assert saved["stage7_results_sha256"] == sha256_file(out / "stage7_results.json") and saved["bands_pp"]["severe"] == [15.0, 30.0]
    assert saved["pins"]["albumentations"] == "2.0.8" and doc["checkpoint_sha256"] == "ab" * 32
    before = fz.read_text()
    with pytest.raises(AssertionError, match="already exists"):
        S7.run_freeze(tmp_path / "runX", out, fz)
    assert fz.read_text() == before                                         # untouched


def test_freeze_refuses_when_scenarios_changed_or_levels_unordered_or_no_results(tmp_path):
    out, fz = tmp_path / "s7", tmp_path / "frozen.yaml"
    with pytest.raises(AssertionError, match="calibration first"):
        S7.run_freeze(tmp_path, out, fz)
    res = _results_for_freeze(tmp_path, out)
    res["scenarios_py_sha256"] = "00" * 32
    (out / "stage7_results.json").write_text(json.dumps(res))
    with pytest.raises(AssertionError, match="scenarios.py changed"):
        S7.run_freeze(tmp_path, out, fz)
    _results_for_freeze(tmp_path, out, ordered=False)
    with pytest.raises(AssertionError, match="mild >= severe"):
        S7.run_freeze(tmp_path, out, fz)
    assert not fz.exists()


# ---------------------------------------------------------------- end to end (needs torch; random-weight resnet, so the drops themselves mean nothing here)
def test_calibration_end_to_end_on_a_tiny_fake_run(tmp_path):
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"torch cannot be loaded here ({type(e).__name__})")
    run, cfgp, cfg = _fake_run(tmp_path)
    out = tmp_path / "s7"
    res = S7.main(["--run", str(run), "--config", str(cfgp), "--device", "cpu", "--out", str(out)])
    assert (out / "stage7_results.json").exists() and (out / "chosen_levels_sheet.png").exists() and (out / "sweep_cache.jsonl").exists()
    assert res["clean_reproduction"]["max_embedding_difference"] < 1e-3 and res["clean_reproduction"]["label_agreement_with_stage3"] >= 0.999
    assert res["fit_frames_used"] == 6 * 4 and res["fit_tracks"] == 6 and all(v["identical"] for v in res["determinism_check"].values())
    assert set(res["determinism_check"]) == set(SC.SCENARIOS)
    for sc in SC.SCENARIOS:
        assert res["sweep"][sc]["grid"] == SC.grid_for(sc) and len(res["sweep"][sc]["drop_pp"]) == len(SC.grid_for(sc))
        assert set(res["chosen"][sc]) == {"mild", "severe", "ordered"}
    assert len(res["recertification_cells"]) == 2
    # tampering with the checkpoint is refused
    (run / "best.pt").write_bytes((run / "best.pt").read_bytes() + b"0")
    with pytest.raises(AssertionError, match="CHECKPOINT HASH MISMATCH"):
        S7.main(["--run", str(run), "--config", str(cfgp), "--device", "cpu", "--out", str(tmp_path / "s7b")])


def test_calibration_resume_uses_the_cache(tmp_path, monkeypatch):
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except (ImportError, OSError) as e:
        pytest.skip(f"torch cannot be loaded here ({type(e).__name__})")
    run, cfgp, cfg = _fake_run(tmp_path)
    out = tmp_path / "s7"
    a = S7.main(["--run", str(run), "--config", str(cfgp), "--device", "cpu", "--out", str(out)])
    calls = {"n": 0}
    real = SC.corrupt_batch

    def counting(*args, **kw):
        calls["n"] += 1
        return real(*args, **kw)
    monkeypatch.setattr(SC, "corrupt_batch", counting)
    b = S7.main(["--run", str(run), "--config", str(cfgp), "--device", "cpu", "--out", str(out)])
    assert b["sweep"] == a["sweep"] and b["chosen"] == a["chosen"]
    # only the determinism checks (5 x 2), the chosen-level fingerprints (10) and the sheet (10) are recomputed: no sweep level (that would be 100+)
    assert calls["n"] == 30, calls["n"]
    assert b["determinism_check"] == a["determinism_check"]
