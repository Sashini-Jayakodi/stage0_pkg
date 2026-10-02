"""Are the pHash 'near-duplicates' REAL duplicates, or just look-alike images?

Why this exists: Stage 0 flagged 449 (train, test) pairs whose 64-bit pHash differs by
<= 2 bits. pHash keeps only very coarse layout (32x32 gray, 8x8 low frequencies), so two
DIFFERENT photos of the same sign class, or two very dark frames, can collide. A real
duplicate (same picture in both releases) would match almost pixel-for-pixel.

So we compare actual pixels (mean absolute difference, MAD, on a 0-255 scale) and put the
matched pairs next to two baselines: random same-class pairs and random different-class
pairs. If matched pairs look like random same-class pairs, they are false alarms.

Usage (after python -m src.stage0):   python scripts/inspect_dups.py
Outputs in data/cache/: near_dup_pairs_scored.csv, near_dup_check.png, near_dup_grayzone.png, finetune_exclude_final.csv,
                         finetune_exclude_exact.csv, finetune_exclude_all_phash.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def mad(a, b):
    """Mean absolute pixel difference (0-255). Identical images -> ~0."""
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def gray_corr(a, b):
    """Pearson correlation of grayscale pixels. Unlike MAD it ignores overall brightness,
    so dark true copies and dark look-alikes are not confused by their small pixel range."""
    x = a.astype(np.float32).mean(axis=2).ravel()
    y = b.astype(np.float32).mean(axis=2).ravel()
    if x.std() < 1e-6 or y.std() < 1e-6:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def score_pairs(train_img, test_img, pairs, train_cls, test_cls):
    """One row per matched pair with pixel distance and class agreement."""
    rows = []
    for ti, ei, h in pairs:
        ti, ei = int(ti), int(ei)
        rows.append(dict(
            train_idx=ti, test_idx=ei, hamming=int(h),
            train_class=int(train_cls[ti]), test_class=int(test_cls[ei]),
            same_class=bool(train_cls[ti] == test_cls[ei]),
            mad=mad(train_img[ti], test_img[ei]),
            corr=gray_corr(train_img[ti], test_img[ei]),
            test_brightness=float(test_img[ei].mean())))   # dark images collide in pHash
    return pd.DataFrame(rows)


def random_baseline(train_img, test_img, train_cls, test_cls, n=3000, seed=0):
    """MAD of random (test, train) pairs: same class vs different class.
    This is the 'what does an unrelated-but-similar pair look like' yardstick."""
    rng = np.random.default_rng(seed)
    by_class = {c: np.flatnonzero(train_cls == c) for c in np.unique(train_cls)}
    same, diff = [], []
    for e in rng.integers(0, len(test_img), n):
        c = int(test_cls[e])
        same.append(mad(train_img[int(rng.choice(by_class[c]))], test_img[e]))
        diff.append(mad(train_img[int(rng.integers(0, len(train_img)))], test_img[e]))
    return np.array(same), np.array(diff)


def save_sheet(train_img, test_img, df, out_png, k=10, seed=0, groups=None):
    """Top-k closest-by-pixels pairs (top rows) and k random matched pairs (bottom rows).
    Each cell = train frame | test frame, so you can judge 'same picture?' by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if groups is None:
        closest = df.nsmallest(k, "mad")
        rand = df.sample(min(k, len(df)), random_state=seed)
        groups = [("closest by pixels", closest), ("random matched pairs", rand)]
    fig, axes = plt.subplots(len(groups), k, figsize=(2.0 * k, 2.3 * len(groups)), squeeze=False)
    for r, (title, g) in enumerate(groups):
        for c in range(k):
            ax = axes[r, c]
            ax.set_axis_off()
            if c < len(g):
                row = g.iloc[c]
                ax.imshow(np.concatenate([train_img[int(row.train_idx)],
                                          test_img[int(row.test_idx)]], axis=1))
                ax.set_title(f"MAD {row.mad:.1f}\nc{row.train_class}/c{row.test_class}", fontsize=7)
        axes[r, 0].text(0, -0.35, title, transform=axes[r, 0].transAxes, fontsize=9, va="bottom")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--cache-dir", help="override cache dir from the config")
    ap.add_argument("--exact-mad", type=float, default=2.0,
                    help="MAD <= this counts as an exact pixel copy (same pipeline => ~0)")
    ap.add_argument("--near-mad", type=float, default=5.0,
                    help="same-class pairs with MAD <= this are excluded from fine-tuning (final list)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    cache = Path(a.cache_dir or cfg["data"]["cache_dir"])
    size = cfg["data"]["image_size"]

    # mmap: the train cache is ~1.5 GB; we only touch a few thousand images.
    train_img = np.load(cache / f"train_{size}.npy", mmap_mode="r")
    test_img = np.load(cache / f"test_{size}.npy", mmap_mode="r")
    train_meta = pd.read_csv(cache / "train_meta.csv")
    train_cls = train_meta["class_id"].to_numpy()
    test_cls = pd.read_csv(cache / "test_meta.csv")["class_id"].to_numpy()
    pairs = np.load(cache / "near_dup_pairs.npy")
    if len(pairs) == 0:
        print("No pHash pairs were found. Nothing to inspect.")
        return

    df = score_pairs(train_img, test_img, pairs, train_cls, test_cls)
    # Which training track does each match come from? (many frames of ONE track matching
    # ONE test image would point to a real shared physical sign.)
    df["train_track"] = train_meta["track_id"].to_numpy()[df["train_idx"].to_numpy()]
    same, diff = random_baseline(train_img, test_img, train_cls, test_cls)
    # Informational only: closer than 99% of random same-class pairs. NOT a duplicate test,
    # because dark images have small MAD against anything (the baseline is not brightness-matched).
    thr = float(np.percentile(same, 1))
    # The real test: the same picture, resized by the same pipeline, differs by ~0 pixels.
    df["exact_copy"] = df["mad"] <= a.exact_mad
    df.to_csv(cache / "near_dup_pairs_scored.csv", index=False)
    save_sheet(train_img, test_img, df, cache / "near_dup_check.png")
    # Gray zone: same class, a little above "exact". Same scene (neighbouring frame of the same
    # physical sign) or just a look-alike? Judge by eye before choosing an exclusion rule.
    gray = df[(df["mad"] > a.exact_mad) & (df["mad"] <= 8.0) & df["same_class"]]
    if len(gray):
        save_sheet(train_img, test_img, df, cache / "near_dup_grayzone.png", groups=[(
            f"gray zone: same class, {a.exact_mad} < MAD <= 8 (random 10)",
            gray.sample(min(10, len(gray)), random_state=0))])

    # Sample of overall test brightness, to test the 'dark images collide' idea.
    rng = np.random.default_rng(0)
    all_bright = float(np.median([test_img[int(i)].mean() for i in rng.integers(0, len(test_img), 2000)]))

    n_te = df["test_idx"].nunique()
    print(f"pHash pairs: {len(df)}  | unique test images: {n_te}  | unique train frames: {df['train_idx'].nunique()}")
    print(f"Hamming distance histogram: {df['hamming'].value_counts().sort_index().to_dict()}")
    print(f"Same class in the pair: {df['same_class'].mean():.1%}")
    q = lambda x: np.round(np.percentile(x, [1, 10, 50, 90]), 1).tolist()
    print("\nPixel distance MAD (0-255), percentiles [1, 10, 50, 90]:")
    print(f"  matched pairs            {q(df['mad'])}")
    print(f"  random same-class pairs  {q(same)}")
    print(f"  random diff-class pairs  {q(diff)}")
    print(f"Median brightness: matched test images {df['test_brightness'].median():.0f}"
          f"  vs typical test image {all_bright:.0f}")
    print(f"\nLooser than random same-class pairs (MAD <= {thr:.1f}): {int((df['mad'] <= thr).sum())} pairs "
          "(informational; not a duplicate test)")
    print("\nHow many matched pairs are (near-)exact pixel copies?")
    for t in (0.5, a.exact_mad, 5.0):
        s_ = df[df["mad"] <= t]
        print(f"  MAD <= {t:>4}: {len(s_):3d} pairs | {s_['test_idx'].nunique():3d} test images | "
              f"same class {int(s_['same_class'].sum())}, different class {int((~s_['same_class']).sum())}")
    ex2 = df[df["mad"] <= a.exact_mad]
    print("\nExact copies by class:", ex2["train_class"].value_counts().to_dict())
    print("Exact copies by training track (top 5):", ex2["train_track"].value_counts().head(5).to_dict())
    near = df[df["same_class"] & (df["mad"] <= 5.0)]
    print("Same-class pairs with MAD <= 5, by class:", near["train_class"].value_counts().to_dict())
    for c in (0.99, 0.95):
        sc = df[df["same_class"] & (df["corr"] >= c)]
        print(f"Same-class pairs with grayscale correlation >= {c}: {len(sc)} pairs | "
              f"{sc['test_idx'].nunique()} test images")
    cols = ["train_idx", "train_track", "test_idx", "hamming", "train_class", "test_class",
            "mad", "corr", "test_brightness"]
    print("\nClosest 15 pairs by pixels (black/blank frames would show brightness near 0):")
    print(df.nsmallest(15, "mad")[cols].round(1).to_string(index=False))

    # Exclusion lists for Stage 1 (indices into test_meta.csv rows). You decide which to use.
    ex = df[df["exact_copy"]]
    pd.Series(sorted(ex["test_idx"].unique()), name="test_idx").to_csv(
        cache / "finetune_exclude_exact.csv", index=False)
    pd.Series(sorted(df["test_idx"].unique()), name="test_idx").to_csv(
        cache / "finetune_exclude_all_phash.csv", index=False)
    # FINAL list consumed by Stage 1: any test image with a same-class pHash partner at
    # MAD <= near_mad. Covers exact copies, near-copies and (harmlessly) a few look-alikes.
    # Threshold chosen AFTER seeing the counts; recorded in LOG.md. Impact is tiny (~0.2%).
    fin = df[df["same_class"] & (df["mad"] <= a.near_mad)]
    pd.Series(sorted(fin["test_idx"].unique()), name="test_idx").to_csv(
        cache / "finetune_exclude_final.csv", index=False)
    print(f"\nFINAL fine-tune exclusion list (same class, MAD <= {a.near_mad}): "
          f"{fin['test_idx'].nunique()} test images = {fin['test_idx'].nunique() / len(test_img):.2%} of the test release")
    print(f"\nExact copies: {ex['test_idx'].nunique()} test images "
          f"= {ex['test_idx'].nunique() / len(test_img):.2%} of the test release.")
    print("Wrote near_dup_pairs_scored.csv, near_dup_check.png and two exclusion lists "
          f"in {cache}. Look at near_dup_check.png.")


if __name__ == "__main__":
    main()
