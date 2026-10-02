"""Stage 2: run the FROZEN Stage 1 backbone and save embeddings (plan section 6, Stage 2).

Two sets are embedded, both in float32, eval mode, no augmentation:
  ft_train       the fine-tune-TRAIN frames from the official test release (~11.3k; Stage 3 fits the head on these)
  train_release  all 39,209 frames of the GTSRB training release (everything later is built on these)
For each frame we save: the 512-d l2-normalised embedding, the raw (un-normalised) 512-d feature
(plan Stage 3 mitigation 2 may need it, so we save it now instead of re-running), the 43 logits,
and class / track / frame indices.

Safety rules built in:
  * the checkpoint SHA-256 must equal the value frozen in configs/poc.yaml, otherwise we stop;
  * the fine-tune split is RECOMPUTED with the same code and seed as Stage 1 and must give the same counts;
  * argmax of the logits must reproduce Stage 1's predictions and its logged accuracy exactly.

Usage:  py -3.12 -m src.embed --run runs/20261003_stage1
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src import train as T


def extract(model, images, device, batch):
    """One forward pass per batch. A hook on avgpool captures the 512-d penultimate feature,
    and the normal forward gives the logits, so both come from the same computation."""
    model.eval()
    grab = {}
    h = model.avgpool.register_forward_hook(lambda m, i, o: grab.__setitem__("f", o.flatten(1)))
    feats, logits = [], []
    try:
        with torch.no_grad():
            for s in range(0, len(images), batch):
                x = torch.from_numpy(np.array(images[s:s + batch])).to(device).permute(0, 3, 1, 2).float().div_(255)
                out = model(T.normalize(x))
                feats.append(grab["f"].cpu().numpy()); logits.append(out.cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(feats).astype(np.float32), np.concatenate(logits).astype(np.float32)


def l2_normalise(raw):
    """Normalise in float64 (accurate), store as float32."""
    n = np.linalg.norm(raw.astype(np.float64), axis=1, keepdims=True)
    assert (n > 0).all(), "a feature vector has zero norm"
    return (raw / n).astype(np.float32)


def check_block(name, raw, emb, logits, y, fc_w, fc_b, tol):
    """The plan's Stage 2 checks, one set at a time. Returns a dict of the numbers for the log."""
    assert len(raw) == len(emb) == len(logits) == len(y), f"{name}: row counts differ"
    assert np.isfinite(raw).all() and np.isfinite(emb).all() and np.isfinite(logits).all(), f"{name}: NaN/inf found"
    norm_err = float(np.abs(np.linalg.norm(emb.astype(np.float64), axis=1) - 1).max())
    assert norm_err < tol, f"{name}: max |norm-1| = {norm_err:.2e} exceeds {tol}"
    # hook sanity: the captured feature really is the input of the final layer
    lg = raw.astype(np.float64) @ fc_w.astype(np.float64).T + fc_b.astype(np.float64)
    fc_err = float(np.abs(lg - logits).max())
    assert fc_err < 1e-3, f"{name}: fc(feature) differs from logits by {fc_err:.2e}; hook captured the wrong tensor"
    return {"rows": int(len(y)), "max_norm_error": norm_err, "max_fc_consistency_error": fc_err}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Stage 1 run folder, e.g. runs/20261003_stage1")
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", help="output folder (default <run>/embeddings)")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    ec = cfg["embed"]
    run = Path(a.run)
    used = yaml.safe_load((run / "config_used.yaml").read_text())          # exactly what Stage 1 used
    res = json.loads((run / "results.json").read_text())
    assert not res["smoke"], "this run is a smoke test, not the real Stage 1 run"
    tc, seed, size = used["train"], used["seed"], used["image_size"]
    cache = Path(cfg["data"]["cache_dir"])
    out = Path(a.out) if a.out else run / "embeddings"
    device = T.pick_device(a.device)

    # --- 1. the backbone must be the frozen one -------------------------------------------------
    ck_path = run / "best.pt"
    sha = T.sha256_file(ck_path)
    want = (ec.get("expected_checkpoint_sha256") or "").lower()
    assert want, "configs/poc.yaml embed.expected_checkpoint_sha256 is empty: freeze the checkpoint hash first"
    assert sha.lower() == want, f"CHECKPOINT HASH MISMATCH\n  found    {sha}\n  expected {want}"
    assert sha == res["checkpoint_sha256"], "best.pt does not match the hash stored in results.json"
    print(f"checkpoint hash OK: {sha}")
    model = T.build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(ck_path, map_location=device, weights_only=False)["model"])
    model.eval()
    fc_w, fc_b = model.fc.weight.detach().cpu().numpy(), model.fc.bias.detach().cpu().numpy()

    # --- 2. recompute the Stage 1 fine-tune split (same code, same seed) --------------------------
    tmeta = pd.read_csv(cache / "test_meta.csv")
    excl = pd.read_csv(tc["exclude_file"])["test_idx"].to_numpy()
    keep = np.setdiff1d(np.arange(len(tmeta)), excl)
    lab_pool = tmeta["class_id"].to_numpy()[keep]
    tr_idx, va_idx = T.stratified_split(lab_pool, tc["val_fraction"], seed)
    assert (len(keep), len(tr_idx), len(va_idx)) == (res["n_pool"], res["n_train"], res["n_val"]), \
        "recomputed split differs from the Stage 1 counts"
    ft_rows = keep[tr_idx]                                    # row positions in the test release
    Xte = np.load(cache / f"test_{size}.npy", mmap_mode="r")
    Xft = np.ascontiguousarray(Xte[ft_rows])
    y_ft = tmeta["class_id"].to_numpy()[ft_rows]

    rmeta = pd.read_csv(cache / "train_meta.csv")
    Xtr = np.load(cache / f"train_{size}.npy", mmap_mode="r")
    y_tr = rmeta["class_id"].to_numpy()
    assert len(Xtr) == len(rmeta), "train cache and train_meta.csv differ in length"

    # --- 3. embed ---------------------------------------------------------------------------------
    bs = ec["batch_size"]
    sets = {}
    for name, X, y in (("ft_train", Xft, y_ft), ("train_release", Xtr, y_tr)):
        raw, logits = extract(model, X, device, bs)
        emb = l2_normalise(raw)
        info = check_block(name, raw, emb, logits, y, fc_w, fc_b, ec["norm_tol"])
        pred = logits.argmax(1)
        # reproduce Stage 1: a second pass through the SAME evaluation function must agree exactly
        acc_ref, _, pred_ref = T.evaluate(model, X, y, device, batch=bs)
        n_diff = int((pred != pred_ref).sum())
        assert n_diff == 0, f"{name}: {n_diff} predictions differ from the Stage 1 evaluation path"
        info["accuracy"] = float((pred == y).mean())
        sets[name] = (raw, emb, logits, y, info)
        print(f"{name}: {info['rows']} rows, max|norm-1| {info['max_norm_error']:.1e}, "
              f"accuracy {info['accuracy']:.4f}")
    # the logged Stage 1 number must come back exactly
    acc_tr = sets["train_release"][4]["accuracy"]
    assert acc_tr == res["sanity_accuracy_training_release"], \
        f"train-release accuracy {acc_tr!r} != Stage 1 logged {res['sanity_accuracy_training_release']!r}"
    print(f"Stage 1 accuracy reproduced exactly: {acc_tr:.4f}")

    # --- 4. save ------------------------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    raw, emb, logits, y, _ = sets["ft_train"]
    np.savez(out / "ft_train.npz", emb=emb, raw=raw, logits=logits, y=y,
             test_row=ft_rows.astype(np.int64), split_pool_index=tr_idx.astype(np.int64))
    raw, emb, logits, y, _ = sets["train_release"]
    track_codes, track_names = pd.factorize(rmeta["track_id"])
    np.savez(out / "train_release.npz", emb=emb, raw=raw, logits=logits, y=y,
             track=track_codes.astype(np.int32), track_names=np.asarray(track_names, dtype=str),
             frame=rmeta["frame_idx"].to_numpy().astype(np.int16))
    np.save(out / "val_pool_index.npy", va_idx.astype(np.int64))      # which pool rows were validation
    summary = {
        "checkpoint_sha256": sha, "n_ft_train": int(len(ft_rows)), "n_train_release": int(len(rmeta)),
        "checks": {k: v[4] for k, v in sets.items()},
        "stage1_accuracy_reproduced_exactly": True,
        "files_sha256": {f: T.sha256_file(out / f) for f in ("ft_train.npz", "train_release.npz")},
        "device": str(device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
    }
    (out / "embeddings_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Stage 2 summary ===")
    print(json.dumps({k: summary[k] for k in ("n_ft_train", "n_train_release", "files_sha256")}, indent=2))
    print(f"accuracy on the frames the head will be fitted on (ft_train): {sets['ft_train'][4]['accuracy']:.4f}")
    print("All Stage 2 checks passed. Saved to", out)
    return summary


if __name__ == "__main__":
    main()
