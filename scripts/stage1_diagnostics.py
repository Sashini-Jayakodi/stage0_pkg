"""Stage 1 read-only diagnostics: WHERE does the fine-tuned backbone make its errors on the
GTSRB training release? Nothing here changes the model or any frozen rule; it only explains
the sanity number (plan section 5) and checks the train-vs-test-release size shift.

Usage: py -3.12 scripts/stage1_diagnostics.py --run runs/20261003_stage1
Writes <run>/diagnostics.json and diagnostics_*.csv and prints short tables.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train as T  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    cache, size = Path(cfg["data"]["cache_dir"]), cfg["data"]["image_size"]
    run = Path(a.run)
    dev = T.pick_device(a.device)

    m = T.build_model(pretrained=False).to(dev)
    m.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=False)["model"])
    meta = pd.read_csv(cache / "train_meta.csv")
    X = np.load(cache / f"train_{size}.npy", mmap_mode="r")
    y = meta["class_id"].to_numpy()
    _, _, pred = T.evaluate(m, X, y, dev)               # same fp32 path as the Stage 1 sanity number
    meta["correct"] = pred == y
    meta["pred"] = pred
    out = {"overall": float(meta["correct"].mean())}

    # 1. Accuracy by ORIGINAL image size (small = far away). Core question for a distance project.
    meta["min_side"] = meta[["width", "height"]].min(axis=1)
    bins = [0, 20, 25, 30, 40, 50, 70, 1000]
    meta["size_bin"] = pd.cut(meta["min_side"], bins)
    by_size = meta.groupby("size_bin", observed=True)["correct"].agg(["mean", "size"])
    # 2. Accuracy by position in the track (frame 0 = farthest, 29 = closest in GTSRB).
    meta["frame_bin"] = pd.cut(meta["frame_idx"], [-1, 4, 9, 14, 19, 24, 29])
    by_frame = meta.groupby("frame_bin", observed=True)["correct"].agg(["mean", "size"])
    # 3. Per class and the most frequent confusions.
    by_class = meta.groupby("class_id")["correct"].agg(["mean", "size"]).sort_values("mean")
    wrong = meta[~meta["correct"]]
    conf = wrong.groupby(["class_id", "pred"]).size().sort_values(ascending=False).head(15)
    # 4. Sensitivity: drop the one training track that has byte-identical copies in the test release.
    keep = meta["track_id"] != "14_00023"
    out["overall_without_track_14_00023"] = float(meta.loc[keep, "correct"].mean())
    # 5. Size shift between the two releases (what the backbone was fine-tuned on vs evaluated on).
    te = pd.read_csv(cache / "test_meta.csv")
    te_side = te[["width", "height"]].min(axis=1)
    shift = pd.DataFrame({
        "train_release": pd.cut(meta["min_side"], bins).value_counts(normalize=True).sort_index(),
        "test_release": pd.cut(te_side, bins).value_counts(normalize=True).sort_index()})
    # 6. How many errors / tracks have errors: input to the G1(ii) feasibility (>=30 errors, >=10 tracks).
    out["errors"] = int((~meta["correct"]).sum())
    out["tracks_with_errors"] = int(wrong["track_id"].nunique())
    out["tracks_total"] = int(meta["track_id"].nunique())

    pd.set_option("display.width", 120)
    print(f"overall {out['overall']:.4f} | without track 14_00023 {out['overall_without_track_14_00023']:.4f}")
    print(f"errors {out['errors']} in {out['tracks_with_errors']} of {out['tracks_total']} tracks")
    print("\nAccuracy by original image size (shorter side, px):\n", by_size.round(4))
    print("\nAccuracy by frame position in track (0 = far):\n", by_frame.round(4))
    print("\nWorst 10 classes:\n", by_class.head(10).round(4))
    print("\nTop confusions (true, predicted) -> count:\n", conf)
    print("\nShare of images by size, train release vs test release:\n", shift.round(3))
    by_size.to_csv(run / "diagnostics_by_size.csv"); by_frame.to_csv(run / "diagnostics_by_frame.csv")
    by_class.to_csv(run / "diagnostics_by_class.csv"); shift.to_csv(run / "diagnostics_size_shift.csv")
    (run / "diagnostics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
