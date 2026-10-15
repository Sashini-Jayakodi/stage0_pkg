"""Stage 3: tied-Gaussian head, confusability (epsilon) table and atlas (plan section 6, Stage 3).

Reads the Stage 2 embeddings (nothing else about the model is touched) and:
  1. fits 43 class means + ONE shared covariance (Ledoit-Wolf shrinkage) on the fine-tune-TRAIN embeddings only;
  2. turns every embedding into a posterior (label = top-1, margin = top-1 minus top-2 log-posterior)
     and an OOD score s(x) = min_k Mahalanobis distance to a class mean;
  3. builds the 43x43 confusability table  eps_ij = Phi(-Delta_ij / 2)  (stored as LOG eps, see below);
  4. checks head-vs-softmax agreement and the eps-vs-empirical-confusion Spearman correlation
     (DESCRIPTIVE ONLY: nothing is tuned on them);
  5. draws the atlas heatmap and the top-10 confusable pairs.

Usage:  py -3.12 -m src.head --run runs/20261003_stage1
        py -3.12 -m src.head --run runs/20261003_stage1 --features raw   # plan mitigation 2, only if the gap is >2 pp
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp
from scipy.stats import norm, spearmanr

from src.train import sha256_file

K = 43
# Standard GTSRB sign names (written from memory: check against your signnames file before quoting them).
NAMES = ["Speed 20", "Speed 30", "Speed 50", "Speed 60", "Speed 70", "Speed 80", "End speed 80", "Speed 100",
         "Speed 120", "No passing", "No passing >3.5t", "Right-of-way next junction", "Priority road", "Yield",
         "Stop", "No vehicles", "No vehicles >3.5t", "No entry", "General caution", "Curve left",
         "Curve right", "Double curve", "Bumpy road", "Slippery road", "Road narrows right", "Road work",
         "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Ice/snow", "Wild animals",
         "End all limits", "Turn right ahead", "Turn left ahead", "Ahead only", "Straight or right",
         "Straight or left", "Keep right", "Keep left", "Roundabout", "End no passing", "End no passing >3.5t"]
# Stallkamp et al. subsets, used only to ORDER the atlas axes.
SUBSETS = {"speed limit": [0, 1, 2, 3, 4, 5, 7, 8], "other prohibitory": [9, 10, 15, 16],
           "derestriction": [6, 32, 41, 42], "mandatory": [33, 34, 35, 36, 37, 38, 39, 40],
           "danger": [11] + list(range(18, 32)), "unique": [12, 13, 14, 17]}


def stallkamp_order():
    order = [c for v in SUBSETS.values() for c in v]
    assert sorted(order) == list(range(K)), "subset lists must cover each class exactly once"
    return order


# ------------------------------------------------------------------ head
def ledoit_wolf(Xc):
    """Ledoit-Wolf shrinkage of the covariance of CENTRED data Xc (n,d) towards mu*I.
    Written out (same formula as scikit-learn's, tested against it) so you need no extra package.
    Shrinkage makes the matrix invertible even though l2-normalised features lie on a sphere."""
    X = Xc.astype(np.float64)
    n, p = X.shape
    emp = X.T @ X / n
    x2 = X ** 2
    mu = np.trace(emp) / p
    beta_ = (x2.T @ x2).sum()
    delta_ = ((X.T @ X) ** 2).sum() / n ** 2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * np.trace(emp) + p * mu ** 2) / p
    beta = min(beta, delta)
    s = 0.0 if beta == 0 else beta / delta
    return (1 - s) * emp + s * mu * np.eye(p), float(s)


def fit_head(emb, y):
    """Class means + pooled WITHIN-class covariance (each embedding centred on its OWN class mean)."""
    emb = emb.astype(np.float64)
    assert set(np.unique(y)) == set(range(K)), "every class needs fine-tune frames to fit its mean"
    mu = np.stack([emb[y == k].mean(0) for k in range(K)])
    cov, shrink = ledoit_wolf(emb - mu[y])
    ev, V = np.linalg.eigh(cov)
    assert ev.min() > 0, "covariance is not positive definite"
    W = (V * ev ** -0.5) @ V.T              # whitening: Mahalanobis distance becomes Euclidean distance
    prec = (V / ev) @ V.T
    priors = np.bincount(y, minlength=K) / len(y)
    return {"mu": mu, "W": W, "prec": prec, "priors": priors, "shrinkage": shrink,
            "cond": float(ev.max() / ev.min()), "eig_min": float(ev.min()), "eig_max": float(ev.max())}


def mahal_sq(head, X):
    """Squared Mahalanobis distance from each row of X to each class mean, via the whitening transform."""
    Xw, Mw = X.astype(np.float64) @ head["W"], head["mu"] @ head["W"]
    D2 = (Xw ** 2).sum(1)[:, None] + (Mw ** 2).sum(1)[None, :] - 2 * Xw @ Mw.T
    return np.maximum(D2, 0.0)


def score(head, X):
    """Per frame: top-3 labels, their normalised log-posteriors, margin (top1-top2), OOD score s(x)."""
    D2 = mahal_sq(head, X)
    lp = np.log(head["priors"])[None, :] - 0.5 * D2
    lp = lp - logsumexp(lp, axis=1, keepdims=True)             # normalise with log-sum-exp
    top3 = np.argsort(-lp, axis=1)[:, :3]
    lp3 = np.take_along_axis(lp, top3, axis=1)
    return {"top3": top3.astype(np.int16), "lp3": lp3.astype(np.float32),
            "margin": (lp3[:, 0] - lp3[:, 1]).astype(np.float32),
            "s": np.sqrt(D2.min(1)).astype(np.float32)}


def eps_table(head):
    """Delta_ij = Mahalanobis distance between class means; eps_ij = Phi(-Delta_ij/2) (equal priors).
    We store LOG eps with a log-CDF: for well-separated pairs plain eps underflows to 0."""
    Mw = head["mu"] @ head["W"]
    delta = np.sqrt(np.maximum(((Mw[:, None, :] - Mw[None, :, :]) ** 2).sum(-1), 0.0))
    log_eps = norm.logcdf(-delta / 2)
    np.fill_diagonal(delta, np.nan); np.fill_diagonal(log_eps, np.nan)
    return delta, log_eps


# ------------------------------------------------------------------ checks
def pair_table(log_eps, y, pred_soft):
    """For all 903 unordered pairs: log eps and the symmetrised empirical confusion rate
    (n(i->j) + n(j->i)) / (n_i + n_j), computed with the SOFTMAX labels."""
    C = np.zeros((K, K))
    np.add.at(C, (y, pred_soft), 1)
    n = np.bincount(y, minlength=K)
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            rows.append((i, j, log_eps[i, j], C[i, j] + C[j, i], (C[i, j] + C[j, i]) / (n[i] + n[j])))
    return pd.DataFrame(rows, columns=["i", "j", "log_eps", "confusions", "rate"])


def correlation_report(pairs, top_emp=10, top_eps=20):
    rho = spearmanr(pairs["log_eps"], pairs["rate"])[0]
    nz = pairs[pairs["rate"] > 0]
    rho_nz = spearmanr(nz["log_eps"], nz["rate"])[0] if len(nz) > 2 else float("nan")
    emp = pairs.sort_values(["rate", "confusions"], ascending=False).head(top_emp)
    eps = pairs.sort_values("log_eps", ascending=False).head(top_eps)
    key = lambda d: set(zip(d["i"], d["j"]))
    return {"spearman_all_903": float(rho), "spearman_nonzero_pairs": float(rho_nz),
            "n_pairs_with_zero_confusion": int((pairs["rate"] == 0).sum()),
            "top10_empirical_in_top20_eps": int(len(key(emp) & key(eps)))}


# ------------------------------------------------------------------ figures
def atlas_figure(log_eps, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = stallkamp_order()
    M = log_eps[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(M, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(order, fontsize=6, rotation=90); ax.set_yticklabels(order, fontsize=6)
    pos = 0
    for v in SUBSETS.values():                                # lines between Stallkamp subsets
        pos += len(v)
        ax.axhline(pos - .5, color="w", lw=.8); ax.axvline(pos - .5, color="w", lw=.8)
    ax.set_title("Confusability atlas: log eps_ij (brighter = more confusable)\nclasses grouped by Stallkamp subset")
    fig.colorbar(im, ax=ax, shrink=.8, label="log eps")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def top_pairs_figure(pairs_top, images, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(pairs_top)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 1.45 * n), gridspec_kw={"width_ratios": [1, 1, 3.2]})
    for r, row in enumerate(pairs_top.itertuples()):
        for c, k in enumerate((row.i, row.j)):
            axes[r, c].imshow(images[k]); axes[r, c].axis("off")
        axes[r, 2].axis("off")
        axes[r, 2].text(0, .5, f"{row.i} {NAMES[row.i]}\n{row.j} {NAMES[row.j]}\nDelta {row.delta:.2f} | log eps {row.log_eps:.2f}"
                        f"\nempirical confusion rate {row.rate:.3%}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def sample_images(cache, size, meta):
    """One representative frame per class: frame 25 (close, sharp) of the class's first track."""
    X = np.load(cache / f"train_{size}.npy", mmap_mode="r")
    imgs = {}
    for k in range(K):
        sub = meta[(meta["class_id"] == k)]
        first = sub[sub["track_id"] == sub["track_id"].iloc[0]]
        pick = first.iloc[(first["frame_idx"] - 25).abs().argmin()]
        imgs[k] = np.array(X[pick.name])
    return imgs


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--features", default="emb", choices=["emb", "raw"],
                    help="emb = l2-normalised (default, plan); raw = un-normalised (plan mitigation 2; log which you used)")
    ap.add_argument("--out", help="default <run>/head_<features>")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    cache, size = Path(cfg["data"]["cache_dir"]), cfg["data"]["image_size"]
    run = Path(a.run)
    emb_dir = run / "embeddings"
    out = Path(a.out) if a.out else run / f"head_{a.features}"
    out.mkdir(parents=True, exist_ok=True)

    # --- inputs must be the Stage 2 files, built from the frozen checkpoint ----------------------------
    summ = json.loads((emb_dir / "embeddings_summary.json").read_text())
    assert summ["checkpoint_sha256"] == cfg["embed"]["expected_checkpoint_sha256"], "embeddings are not from the frozen checkpoint"
    for f, h in summ["files_sha256"].items():
        assert sha256_file(emb_dir / f) == h, f"{f} changed since Stage 2 wrote it"
    ft, tr = np.load(emb_dir / "ft_train.npz"), np.load(emb_dir / "train_release.npz")
    Xft, yft, Xtr, ytr = ft[a.features], ft["y"], tr[a.features], tr["y"]
    soft_ft, soft_tr = ft["logits"].argmax(1), tr["logits"].argmax(1)
    print(f"features: {a.features} | fit on {len(yft)} fine-tune-train frames | apply to {len(ytr)} training-release frames")

    # --- fit and score ---------------------------------------------------------------------------------
    head = fit_head(Xft, yft)
    delta, log_eps = eps_table(head)
    s_ft, s_tr = score(head, Xft), score(head, Xtr)
    assert np.isfinite(log_eps[~np.eye(K, dtype=bool)]).all(), "log eps has non-finite values"
    plain = norm.cdf(-delta[~np.eye(K, dtype=bool)] / 2)
    res = {"features": a.features, "shrinkage": head["shrinkage"], "cov_condition_number": head["cond"],
           "cov_eig_min": head["eig_min"], "cov_eig_max": head["eig_max"],
           "n_pairs_where_plain_eps_underflows_to_0": int((plain == 0).sum() // 2),
           "log_eps_min": float(np.nanmin(log_eps)), "log_eps_max": float(np.nanmax(log_eps))}
    assert np.isfinite(head["cond"]), "covariance condition number is not finite"

    # --- head vs softmax -------------------------------------------------------------------------------
    for name, s, soft, y in (("ft_train", s_ft, soft_ft, yft), ("train_release", s_tr, soft_tr, ytr)):
        head_acc, soft_acc = float((s["top3"][:, 0] == y).mean()), float((soft == y).mean())
        res[name] = {"head_accuracy": head_acc, "softmax_accuracy": soft_acc,
                     "label_agreement": float((s["top3"][:, 0] == soft).mean()),
                     "gap_pp_softmax_minus_head": 100 * (soft_acc - head_acc)}
    gap = res["train_release"]["gap_pp_softmax_minus_head"]
    res["head_gap_over_2pp"] = bool(gap > 2.0)

    # --- epsilon vs empirical confusion (descriptive only) ---------------------------------------------
    pairs = pair_table(log_eps, ytr, soft_tr)
    pairs["delta"] = [delta[i, j] for i, j in zip(pairs["i"], pairs["j"])]
    res.update(correlation_report(pairs))
    pairs["name_i"] = [NAMES[i] for i in pairs["i"]]; pairs["name_j"] = [NAMES[j] for j in pairs["j"]]
    pairs.sort_values("log_eps", ascending=False).to_csv(out / "pairs_903.csv", index=False)
    top_eps = pairs.sort_values("log_eps", ascending=False).head(10)
    top_eps.to_csv(out / "top10_confusable_pairs.csv", index=False)

    # --- save ------------------------------------------------------------------------------------------
    np.savez(out / "head.npz", mu=head["mu"], W=head["W"], prec=head["prec"], priors=head["priors"],
             shrinkage=head["shrinkage"], delta=delta, log_eps=log_eps)
    def pack(s, y):
        # eps for the top-2 pair of each frame, looked up now so later stages need no table logic
        return dict(top3=s["top3"], lp3=s["lp3"], margin=s["margin"], s=s["s"],
                    log_eps12=log_eps[s["top3"][:, 0], s["top3"][:, 1]].astype(np.float32), y=y)
    np.savez(out / "scores_train_release.npz", **pack(s_tr, ytr),
             track=tr["track"], frame=tr["frame"], softmax_pred=soft_tr.astype(np.int16))
    np.savez(out / "scores_ft_train.npz", **pack(s_ft, yft))
    try:
        atlas_figure(log_eps, out / "atlas.png")
        meta = pd.read_csv(cache / "train_meta.csv")
        top_pairs_figure(top_eps, sample_images(cache, size, meta), out / "top10_pairs.png")
    except Exception as e:                                    # figures must never block the numbers
        print("WARNING: figure step failed:", repr(e))
    res["files_sha256"] = {f: sha256_file(out / f) for f in ("head.npz", "scores_train_release.npz", "scores_ft_train.npz")}
    (out / "head_summary.json").write_text(json.dumps(res, indent=2))

    print("\n=== Stage 3 summary ===")
    print(f"Ledoit-Wolf shrinkage {res['shrinkage']:.4f} | covariance condition number {res['cov_condition_number']:.3g}")
    for name in ("ft_train", "train_release"):
        r = res[name]
        print(f"{name}: head acc {r['head_accuracy']:.4f} | softmax acc {r['softmax_accuracy']:.4f} | "
              f"label agreement {r['label_agreement']:.4f} | softmax minus head {r['gap_pp_softmax_minus_head']:+.2f} pp")
    print(f"Spearman(log eps, empirical confusion) over 903 pairs: {res['spearman_all_903']:.3f} "
          f"(non-zero-confusion pairs only: {res['spearman_nonzero_pairs']:.3f}; "
          f"{res['n_pairs_with_zero_confusion']} pairs have zero confusion)")
    print(f"top-10 empirical pairs found in the top-20 eps pairs: {res['top10_empirical_in_top20_eps']} of 10")
    print(f"pairs whose plain eps would underflow to 0: {res['n_pairs_where_plain_eps_underflows_to_0']} (log eps min {res['log_eps_min']:.1f})")
    print("\nTen most confusable pairs by eps:")
    print(top_eps[["i", "j", "name_i", "name_j", "delta", "log_eps", "confusions", "rate"]].to_string(index=False))
    if res["head_gap_over_2pp"]:
        print("\nHEAD GAP > 2 pp. Plan mitigations in order: (1) covariance is pooled WITHIN classes (done here); "
              "(2) try --features raw and log which version you use; (3) check the priors; "
              "(4) if it persists keep the Gaussian head and report the gap. NEVER mix the softmax label with the Gaussian margin.")
    if res["spearman_all_903"] < 0.1:
        print("\nSpearman is near zero or negative: investigate BEFORE Stage 5 (plan failure-mode table).")
    return res


if __name__ == "__main__":
    main()
