"""Stage 7: scenario smoke test, Fit-only severity calibration and the G2 freeze (plan section 7). Touches FIT frames of reshuffle seed 0 only.

Three commands (run them in this order; the third one is the freeze and is meant to happen exactly once):

  py -3.12 -m src.stage7 --run runs/20261003_stage1 --smoke     no torch. 20 Fit images x 5 scenarios x 3 strengths -> contact_sheet.png, plus
                                                                a determinism check (re-run in another order, hashes must match).
  py -3.12 -m src.stage7 --run runs/20261003_stage1             needs torch. Calibration: for each scenario, head accuracy on Fit over the strength
                                                                grid -> mild / severe level -> two recertification cells -> stage7_results.json
                                                                and chosen_levels_sheet.png. Nothing is frozen yet.
  py -3.12 -m src.stage7 --run runs/20261003_stage1 --freeze    no torch. Writes configs/frozen_g2.yaml (refuses to overwrite). Then YOU commit it
                                                                and put the commit hash in LOG.md. G2 is closed from that moment.

Rules, fixed BEFORE any accuracy was measured (they are constants below, not options):
  * bands (LOG.md pre-registration): mild = a 3 to 8 pp drop in Fit head accuracy, severe = a 15 to 30 pp drop;
  * mild level = the grid level whose drop is closest to 5.5 pp among those inside the band (if none is inside: the closest overall, flagged);
  * severe level = the same with 22.5 pp inside 15 to 30; if no level reaches 15 pp, the strongest level allowed by the human-recognisable cap
    (scenarios.CAP) is used and flagged "band not reached" (plan: cap at the strongest human-recognisable level and log it);
  * ties go to the smaller severity; the cells are the 5 scenarios x {mild, severe} = 10;
  * the two recertification cells = the two of those 10 with the largest Fit error increase (ties: scenario order, then severe before mild);
  * drop = accuracy of the Gaussian head (top-1) on the clean frames minus on the corrupted frames, same frames, in percentage points;
    only every `stride`-th frame (frame index % stride == 0) of each Fit track is used, to keep the sweep fast (about 10 frames per track at stride 3).
Cert and Test frames are never loaded: the code indexes the image cache with Fit rows only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src import scenarios as SC
from src.util import sha256_file

BANDS = {"mild": (3.0, 8.0), "severe": (15.0, 30.0)}              # pp, pre-registered in LOG.md
MIDPOINT = {"mild": 5.5, "severe": 22.5}
DEFAULTS7 = {"seed": 0, "stride": 3, "batch": 256, "boot": 500, "smoke_images": 20, "smoke_levels": [1.0, 2.5, "cap"], "det_check_frames": 120,
             "repro_emb_tol": 1e-3, "repro_label_agreement": 0.999, "pinned_albumentations": "2.0.8"}


# ------------------------------------------------------------------ pure selection logic (unit-tested)
def pick_level(grid, drops, kind):
    """Index of the chosen grid level for 'mild' or 'severe' and a record of how it was chosen. grid ascending, drops in pp."""
    lo, hi = BANDS[kind]
    inside = [i for i, d in enumerate(drops) if lo <= d <= hi]
    if inside:
        i = min(inside, key=lambda j: (abs(drops[j] - MIDPOINT[kind]), grid[j]))
        return i, {"band_reached": True, "rule": "closest to the band midpoint among levels inside the band"}
    if kind == "severe" and max(drops) < lo:
        return len(grid) - 1, {"band_reached": False, "rule": "no level reaches the band: strongest level allowed by the recognisability cap"}
    i = min(range(len(grid)), key=lambda j: (abs(drops[j] - MIDPOINT[kind]), grid[j]))
    return i, {"band_reached": False, "rule": "no level inside the band: closest to the band midpoint"}


def choose_levels(sweep):
    """sweep: {scenario: {"grid": [...], "drop_pp": [...]}} -> {scenario: {"mild": {...}, "severe": {...}, "ordered": bool}}.
    'ordered' is False only if the two rules pick the same or reversed levels (a degenerate sweep); --freeze refuses such a result."""
    out = {}
    for sc, v in sweep.items():
        cell = {}
        for kind in ("mild", "severe"):
            i, how = pick_level(v["grid"], v["drop_pp"], kind)
            cell[kind] = {"severity": v["grid"][i], "fit_drop_pp": v["drop_pp"][i], **how}
        cell["ordered"] = bool(cell["mild"]["severity"] < cell["severe"]["severity"])
        out[sc] = cell
    return out


def choose_cells(chosen, n=2):
    """The n cells (of scenario x {mild, severe}) with the largest Fit error increase. Ties: scenario order, then severe before mild."""
    cells = [{"scenario": sc, "level": kind, "severity": c[kind]["severity"], "fit_drop_pp": c[kind]["fit_drop_pp"]}
             for sc, c in chosen.items() for kind in ("mild", "severe")]
    key = lambda x: (-x["fit_drop_pp"], SC.SCENARIOS.index(x["scenario"]), 0 if x["level"] == "severe" else 1)
    return sorted(cells, key=key)[:n]


def drop_stats(n_t, corr_clean_t, corr_t, B, rng):
    """Accuracy drop in pp (clean minus corrupted) from per-track counts, with a track-bootstrap standard error."""
    n_t, a, b = (np.asarray(x, dtype=float) for x in (n_t, corr_clean_t, corr_t))
    drop = 100.0 * (a.sum() - b.sum()) / n_t.sum()
    T = len(n_t)
    idx = rng.integers(0, T, (B, T))
    boot = 100.0 * (a[idx].sum(1) - b[idx].sum(1)) / n_t[idx].sum(1)
    return float(drop), float(boot.std(ddof=1)), float(100.0 * a.sum() / n_t.sum()), float(100.0 * b.sum() / n_t.sum())


# ------------------------------------------------------------------ Fit data (the ONLY images this stage ever loads)
def fit_index(run, cfg_all, stride):
    """Which cache rows are Fit frames of reshuffle seed 0 (every `stride`-th frame). Metadata only: no pixels are read here."""
    run = Path(run)
    sp_dir = run / "splits"
    summ = json.loads((sp_dir / "stage4_summary.json").read_text())
    f = f"seed{DEFAULTS7['seed']}.npz"
    assert sha256_file(sp_dir / f) == summ["files_sha256"][f], f"{f} changed since Stage 4"
    sp = np.load(sp_dir / f)
    es = json.loads((run / "embeddings" / "embeddings_summary.json").read_text())
    assert sha256_file(run / "embeddings" / "train_release.npz") == es["files_sha256"]["train_release.npz"], "train_release.npz changed since Stage 2"
    E = np.load(run / "embeddings" / "train_release.npz")
    track, names, frame, y = E["track"].astype(int), np.asarray(E["track_names"]), E["frame"].astype(int), E["y"].astype(int)
    assert (np.asarray(sp["track_names"]) == names).all(), "track order in the split file differs from the Stage 2 embeddings"
    track_split = sp["track_split"]
    rows = np.flatnonzero((track_split[track] == 0) & (frame % stride == 0))              # FIT frames only
    assert (track_split[track[rows]] == 0).all()
    return {"rows": rows, "y": y[rows], "track_names": names[track[rows]], "frames": frame[rows], "track_code": track[rows],
            "n_fit_tracks": int((track_split == 0).sum()), "checkpoint_sha256": es["checkpoint_sha256"]}


def load_fit(run, cfg_all, c7, images=True, stride=None):
    fit = fit_index(run, cfg_all, c7["stride"] if stride is None else stride)
    size = yaml.safe_load((Path(run) / "config_used.yaml").read_text())["image_size"]
    cache = Path(cfg_all["data"]["cache_dir"])
    import pandas as pd
    meta = pd.read_csv(cache / "train_meta.csv")
    r = fit["rows"]
    E = np.load(Path(run) / "embeddings" / "train_release.npz")
    assert len(meta) == len(E["y"]) and (meta["class_id"].to_numpy()[r] == fit["y"]).all() and (meta["frame_idx"].to_numpy()[r] == fit["frames"]).all() \
        and (meta["track_id"].to_numpy()[r] == fit["track_names"]).all(), "image cache rows do not line up with the Stage 2 embeddings"
    fit["image_size"] = size
    if images:
        X = np.load(cache / f"train_{size}.npy", mmap_mode="r")
        fit["images"] = np.ascontiguousarray(X[r])                                         # only the Fit rows are ever read from the cache
        fit["stored_emb"] = E["emb"][r]
    else:
        fit["_cache_path"] = cache / f"train_{size}.npy"
    return fit


# ------------------------------------------------------------------ smoke test (no torch)
def smoke_images(fit, n, seed):
    """n Fit images from n distinct classes (mid-track frames when available), chosen with a fixed seed."""
    rng = np.random.default_rng([seed, 7])
    classes = np.unique(fit["y"])
    pick = []
    for c in rng.permutation(classes)[:n]:
        idx = np.flatnonzero(fit["y"] == c)
        mid = idx[fit["frames"][idx] >= np.median(fit["frames"][idx])]
        pick.append(int(rng.choice(mid if len(mid) else idx)))
    return np.array(sorted(pick))


def levels_for(sc, spec):
    return [SC.CAP[sc] if s == "cap" else float(s) for s in spec]


def run_smoke(run, cfg_all, c7, out):
    fit = load_fit(run, cfg_all, c7, images=False, stride=1)
    idx = smoke_images(fit, c7["smoke_images"], c7["seed"])
    X = np.load(fit["_cache_path"], mmap_mode="r")
    imgs, names, frames = np.ascontiguousarray(X[fit["rows"][idx]]), fit["track_names"][idx], fit["frames"][idx]
    rows = [("clean", None, 0.0)]
    hashes = {}
    for sc in SC.SCENARIOS:
        for s in levels_for(sc, c7["smoke_levels"]):
            rows.append((f"{sc} s={s:g}", sc, s))
            a = SC.corrupt_batch(imgs, names, frames, sc, s)
            perm = np.random.default_rng(1).permutation(len(imgs))
            b = SC.corrupt_batch(imgs[perm], names[perm], frames[perm], sc, s)[np.argsort(perm)]       # another order: must be identical
            assert np.array_equal(a, b), f"{sc} s={s}: not deterministic"
            hashes[f"{sc}@{s:g}"] = SC.array_sha256(a)
    out.mkdir(parents=True, exist_ok=True)
    sheet = SC.contact_sheet(imgs, names, frames, rows, out / "contact_sheet.png")
    info = {"images": len(idx), "classes": sorted(int(c) for c in fit["y"][idx]), "deterministic_under_reordering": True, "sha256_by_cell": hashes,
            "versions": SC.versions(), "scenarios_py_sha256": sha256_file(Path(SC.__file__))}
    (out / "smoke.json").write_text(json.dumps(info, indent=2))
    print(f"smoke test OK: {len(idx)} Fit images, {len(rows) - 1} corrupted rows, identical when the order is shuffled. Contact sheet: {sheet}")
    print("Open it and look: every row should look like its name, and the sign should still be recognisable to you up to the strongest level shown.")
    return info


# ------------------------------------------------------------------ calibration (needs torch)
def build_pipeline(run, cfg_all, c7, device_arg):
    import torch
    from src import embed as EM
    from src import head as HD
    from src import train as TR
    run = Path(run)
    want = (cfg_all["embed"].get("expected_checkpoint_sha256") or "").lower()
    ck = run / "best.pt"
    sha = sha256_file(ck)
    assert sha.lower() == want, f"CHECKPOINT HASH MISMATCH: {sha} != {want}"
    hs = json.loads((run / "head_emb" / "head_summary.json").read_text())
    for f in ("head.npz", "scores_train_release.npz"):
        assert sha256_file(run / "head_emb" / f) == hs["files_sha256"][f], f"{f} changed since Stage 3"
    device = TR.pick_device(device_arg)
    model = TR.build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
    model.eval()
    H = np.load(run / "head_emb" / "head.npz")
    head = {k: H[k] for k in ("mu", "W", "prec", "priors")}
    stored = np.load(run / "head_emb" / "scores_train_release.npz")

    def predict(images):
        raw, _ = EM.extract(model, images, device, c7["batch"])
        emb = EM.l2_normalise(raw)
        return emb, HD.score(head, emb)["top3"][:, 0].astype(int)
    return predict, sha, stored, str(device), torch.__version__


def run_calibration(run, cfg_all, c7, device_arg, out):
    A, _ = SC._alb()
    assert A.__version__ == c7["pinned_albumentations"], \
        f"albumentations {A.__version__} is installed, the plan pins {c7['pinned_albumentations']}: py -3.12 -m pip install albumentations=={c7['pinned_albumentations']}"
    fit = load_fit(run, cfg_all, c7)
    predict, ck_sha, stored, device, torch_v = build_pipeline(run, cfg_all, c7, device_arg)
    assert ck_sha == fit["checkpoint_sha256"], "checkpoint differs from the one Stage 2 used"
    images, y, names, frames = fit["images"], fit["y"], fit["track_names"], fit["frames"]
    _, tl = np.unique(fit["track_code"], return_inverse=True)
    n_t = np.bincount(tl).astype(float)
    print(f"Fit seed {c7['seed']}: {fit['n_fit_tracks']} tracks, {len(y)} frames at stride {c7['stride']}, device {device}")

    # --- the clean frames through THIS pipeline must reproduce the Stage 2/3 numbers
    emb_clean, pred_clean = predict(images)
    emb_diff = float(np.abs(emb_clean - fit["stored_emb"]).max())
    stored_pred = stored["top3"][fit["rows"], 0].astype(int)
    agree = float((pred_clean == stored_pred).mean())
    assert emb_diff < c7["repro_emb_tol"] and agree >= c7["repro_label_agreement"], \
        f"clean frames do not reproduce Stage 2/3 (max embedding difference {emb_diff:.2e}, label agreement {agree:.4f})"
    corr_clean = np.bincount(tl, weights=(pred_clean == y).astype(float), minlength=len(n_t))
    acc_clean = 100.0 * corr_clean.sum() / n_t.sum()
    rng = np.random.default_rng([c7["seed"], 77])
    sweep = {}
    # determinism spot check, every run (also after a resume): a subset of frames in another order must hash-identical at each scenario's strongest level
    det = {}
    sel = np.random.default_rng(3).permutation(len(y))[:min(c7["det_check_frames"], len(y))]
    for sc in SC.SCENARIOS:
        s_cap = SC.CAP[sc]
        full = SC.corrupt_batch(images, names, frames, sc, s_cap)
        again = SC.corrupt_batch(images[sel], names[sel], frames[sel], sc, s_cap)
        assert np.array_equal(again, full[sel]), f"{sc}: corruption is not deterministic"
        det[sc] = {"severity": s_cap, "frames_rechecked": int(len(sel)), "identical": True}
    cache_file = out / "sweep_cache.jsonl"
    done = {}
    if cache_file.exists():                                                   # resume after a crash: results are deterministic
        for line in cache_file.read_text().splitlines():
            r = json.loads(line)
            if r["stride"] == c7["stride"] and r["n_frames"] == len(y) and r["pipeline"] == [ck_sha, device]:
                done[(r["scenario"], r["severity"])] = r
    out.mkdir(parents=True, exist_ok=True)
    for sc in SC.SCENARIOS:
        grid = SC.grid_for(sc)
        rec = {"grid": grid, "acc": [], "drop_pp": [], "drop_se_pp": []}
        for s in grid:
            if (sc, s) in done:
                r = done[(sc, s)]
                corr = np.array(r["correct_per_track"])
            else:
                imgs_c = SC.corrupt_batch(images, names, frames, sc, s)
                _, pred = predict(imgs_c)
                corr = np.bincount(tl, weights=(pred == y).astype(float), minlength=len(n_t))
                with open(cache_file, "a") as fh:
                    fh.write(json.dumps({"scenario": sc, "severity": s, "stride": c7["stride"], "n_frames": int(len(y)), "pipeline": [ck_sha, device],
                                         "correct_per_track": corr.tolist()}) + "\n")
            drop, se, _, acc_c = drop_stats(n_t, corr_clean, corr, c7["boot"], rng)
            rec["acc"].append(acc_c); rec["drop_pp"].append(drop); rec["drop_se_pp"].append(se)
            print(f"  {sc:<12} s={s:<5} accuracy {acc_c:6.2f}%  drop {drop:6.2f} pp (se {se:.2f})", flush=True)
        sweep[sc] = rec
    chosen = choose_levels(sweep)
    cells = choose_cells(chosen)
    # a fingerprint of the corrupted Fit frames at the chosen cells (first 100 rows): lets the full study check it regenerates the same images
    fp = {}
    for sc, c in chosen.items():
        for kind in ("mild", "severe"):
            fp[f"{sc}_{kind}"] = SC.array_sha256(SC.corrupt_batch(images[:100], names[:100], frames[:100], sc, c[kind]["severity"]))
    res = {"seed": c7["seed"], "stride": c7["stride"], "fit_tracks": fit["n_fit_tracks"], "fit_frames_used": int(len(y)), "bands_pp": BANDS, "midpoints_pp": MIDPOINT,
           "clean_accuracy_pct": acc_clean, "clean_reproduction": {"max_embedding_difference": emb_diff, "label_agreement_with_stage3": agree},
           "determinism_check": det, "sweep": sweep, "chosen": chosen, "recertification_cells": cells, "fit_sample_sha256": fp,
           "checkpoint_sha256": ck_sha, "device": device, "torch": torch_v, "versions": SC.versions(), "caps": SC.CAP,
           "scenarios_py_sha256": sha256_file(Path(SC.__file__)), "stage7_py_sha256": sha256_file(Path(__file__)),
           "notes": ["Fit frames of reshuffle seed 0 only; Cert and Test images were never loaded.",
                     "drop = clean minus corrupted Gaussian-head top-1 accuracy on the same frames; se = track bootstrap.",
                     "Nothing is frozen until --freeze is run and configs/frozen_g2.yaml is committed."]}
    (out / "stage7_results.json").write_text(json.dumps(res, indent=2))
    idx = smoke_images(fit, c7["smoke_images"], c7["seed"])
    rows = [("clean", None, 0.0)] + [(f"{sc} {k} s={chosen[sc][k]['severity']:g}", sc, chosen[sc][k]["severity"]) for sc in SC.SCENARIOS for k in ("mild", "severe")]
    SC.contact_sheet(images[idx], names[idx], frames[idx], rows, out / "chosen_levels_sheet.png")
    report(res)
    return res


def report(res):
    print("\n=== Stage 7: Fit-only severity calibration ===")
    print(f"clean Fit accuracy (Gaussian head, {res['fit_frames_used']} frames): {res['clean_accuracy_pct']:.2f}% | reproduces Stage 2/3: max |embedding diff| "
          f"{res['clean_reproduction']['max_embedding_difference']:.1e}, label agreement {100 * res['clean_reproduction']['label_agreement_with_stage3']:.2f}%")
    print(f"bands (pre-registered): mild {BANDS['mild']} pp, severe {BANDS['severe']} pp\n")
    print("  scenario      level   severity  Fit drop (pp)  se     inside band  how")
    for sc in SC.SCENARIOS:
        for k in ("mild", "severe"):
            c = res["chosen"][sc][k]
            print(f"  {sc:<12}  {k:<7} {c['severity']:<8g}  {c['fit_drop_pp']:>8.2f}      "
                  f"{res['sweep'][sc]['drop_se_pp'][res['sweep'][sc]['grid'].index(c['severity'])]:.2f}   {'yes' if c['band_reached'] else 'NO ':<11}  {c['rule']}")
    print("\nrecertification cells (largest Fit error increase):")
    for c in res["recertification_cells"]:
        print(f"  {c['scenario']} / {c['level']} (severity {c['severity']:g}): +{c['fit_drop_pp']:.2f} pp")
    miss = [(sc, k) for sc in SC.SCENARIOS for k in ("mild", "severe") if not res["chosen"][sc][k]["band_reached"]]
    if miss:
        print(f"\nBAND NOT REACHED for {miss}: logged as such (plan section 7); do not change any rule.")
    bad = [sc for sc in SC.SCENARIOS if not res["chosen"][sc]["ordered"]]
    if bad:
        print(f"\nWARNING: mild level is not below severe level for {bad}. --freeze will refuse. Send me stage7_results.json; do not edit anything.")
    print("\nNEXT: open chosen_levels_sheet.png and check that every row is still recognisable to a human. Then run --freeze.")


# ------------------------------------------------------------------ freeze (no torch)
def run_freeze(run, out, frozen_path):
    frozen_path = Path(frozen_path)
    assert not frozen_path.exists(), f"{frozen_path} already exists: G2 is frozen once and never edited (plan section 7)"
    rf = out / "stage7_results.json"
    assert rf.exists(), "run the calibration first"
    res = json.loads(rf.read_text())
    assert res["scenarios_py_sha256"] == sha256_file(Path(SC.__file__)), "src/scenarios.py changed after the calibration: rerun it"
    assert res["fit_frames_used"] > 0 and len(res["recertification_cells"]) == 2
    assert all(c["ordered"] for c in res["chosen"].values()), "a scenario has mild >= severe: send me stage7_results.json, do not freeze"
    doc = {"g2_frozen": True, "frozen_on": __import__("datetime").date.today().isoformat(), "run": str(Path(run).name), "checkpoint_sha256": res["checkpoint_sha256"],
           "calibrated_on": {"split": "Fit", "reshuffle_seed": res["seed"], "frame_stride": res["stride"], "frames": res["fit_frames_used"],
                             "tracks": res["fit_tracks"], "clean_accuracy_pct": res["clean_accuracy_pct"]},
           "bands_pp": res["bands_pp"], "selection_rule": "closest to the band midpoint among grid levels inside the band; severe falls back to the recognisability cap; "
                                                          "recertification cells = the two of the ten with the largest Fit error increase",
           "scenarios": {sc: {kind: {"severity": c[kind]["severity"], "fit_drop_pp": c[kind]["fit_drop_pp"], "band_reached": c[kind]["band_reached"],
                                     "nominal_parameters": SC.nominal(sc, c[kind]["severity"]), "fit_sample_sha256": res["fit_sample_sha256"][f"{sc}_{kind}"]}
                              for kind in ("mild", "severe")} for sc, c in res["chosen"].items()},
           "recertification_cells": res["recertification_cells"],
           "seeding": "per track (track name, scenario, severity) and per frame (track name, frame index, scenario, severity), SHA-256 of a text key; see src/scenarios.py",
           "pins": {"albumentations": "2.0.8", "imagecorruptions": "1.1.2 (fog and motion_blur vendored in src/scenarios.py)"},
           "versions_at_calibration": res["versions"], "scenarios_py_sha256": res["scenarios_py_sha256"], "stage7_results_sha256": sha256_file(rf)}
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"wrote {frozen_path}  (sha256 {sha256_file(frozen_path)})")
    print("NOW: git add configs/frozen_g2.yaml && git commit -m \"G2 freeze\", then put `git rev-parse HEAD` in LOG.md. Never edit this file again.")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--stride", type=int, help="use every n-th frame of each Fit track (default 3; the value used is logged)")
    ap.add_argument("--smoke", action="store_true", help="contact sheet + determinism check only (no torch)")
    ap.add_argument("--freeze", action="store_true", help="write configs/frozen_g2.yaml from stage7_results.json (once)")
    ap.add_argument("--frozen", default="configs/frozen_g2.yaml")
    ap.add_argument("--out", help="default <run>/stage7")
    a = ap.parse_args(argv)
    cfg_all = yaml.safe_load(open(a.config))
    c7 = {**DEFAULTS7, **cfg_all.get("stage7", {})}
    if a.stride:
        c7["stride"] = a.stride
    out = Path(a.out) if a.out else Path(a.run) / "stage7"
    if a.smoke:
        return run_smoke(a.run, cfg_all, c7, out)
    if a.freeze:
        return run_freeze(a.run, out, a.frozen)
    out.mkdir(parents=True, exist_ok=True)
    return run_calibration(a.run, cfg_all, c7, a.device, out)


if __name__ == "__main__":
    main()
