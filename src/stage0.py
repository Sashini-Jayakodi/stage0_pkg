"""Stage 0 CLI:  python -m src.stage0 [--config configs/poc.yaml] [--skip-phash]

Order of work (each step is a gate; a failure stops the run):
  1. find the three zips          2. build metadata tables
  3. run the count assertions     4. decode + cache 112x112 images
  5. write preview strips         6. (optional) cross-release duplicate check
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from . import data as D


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--skip-phash", action="store_true")
    ap.add_argument("--no-validate", action="store_true",
                    help="only for synthetic smoke tests; never for real data")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    d = cfg["data"]
    raw, cache, size = Path(d["raw_dir"]), Path(d["cache_dir"]), d["image_size"]
    cache.mkdir(parents=True, exist_ok=True)

    # Step 1: all three files must exist (the test labels are in the separate GT zip).
    tr_zip, te_zip, gt_zip = raw / d["train_zip"], raw / d["test_zip"], raw / d["test_gt"]
    for p in (tr_zip, te_zip, gt_zip):
        if not p.exists():
            raise SystemExit(f"missing {p}. Download the official GTSRB zips into {raw}/")

    # Steps 2-3: metadata first, assertions BEFORE the slow image decoding,
    # so a wrong release fails in seconds rather than after an hour.
    train_meta = D.build_train_meta(tr_zip)
    test_meta = D.build_test_meta(te_zip, gt_zip)
    if not a.no_validate:
        D.validate_train(train_meta, d["expected"])
        D.validate_test(test_meta, train_meta, d["expected"])
        print("metadata assertions passed")

    # Step 4: decode once; every later stage reads these .npy files.
    train_img = D.build_cache(tr_zip, train_meta, size, "train")
    test_img = D.build_cache(te_zip, test_meta, size, "test")
    np.save(cache / f"train_{size}.npy", train_img)
    np.save(cache / f"test_{size}.npy", test_img)
    # zip_path is dropped: it is only needed during decoding.
    train_meta.drop(columns=["zip_path"]).to_csv(cache / "train_meta.csv", index=False)
    test_meta.drop(columns=["zip_path"]).to_csv(cache / "test_meta.csv", index=False)
    # Step 5: eyeball check that tracks look like one sign growing in size.
    D.save_track_strips(train_img, train_meta, cache / "track_strips.png", seed=cfg["seed"])

    summary = {"train_images": len(train_meta), "train_tracks": int(train_meta.track_id.nunique()),
               "test_images": len(test_meta), "image_size": size}
    # Step 6: how many test images also appear (near-identically) in the train release?
    # The plan says: >1% means flag it to the supervisor.
    if cfg["phash"]["enabled"] and not a.skip_phash:
        ht, he = D.phash64(train_img), D.phash64(test_img)
        dups = D.cross_release_near_dups(ht, he, cfg["phash"]["near_dup_hamming"])
        np.save(cache / "near_dup_pairs.npy", dups)
        n_test_dup = len(set(dups[:, 1].astype(int))) if len(dups) else 0
        summary["near_dup_pairs"] = int(len(dups))
        summary["test_images_with_near_dup"] = n_test_dup
        summary["test_dup_fraction"] = n_test_dup / len(test_meta)
        if summary["test_dup_fraction"] > 0.01:
            print("WARNING: >1% near-duplicates across releases. Flag to supervisor.")
    (cache / "stage0_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Eyeball data/cache/track_strips.png: signs should grow along each strip.")


if __name__ == "__main__":
    main()
