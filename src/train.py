"""Stage 1: fine-tune ResNet-18 ONCE on the official test release (minus the duplicate list).

Plan section 5. What this script does, in order:
  1. Load the 112x112 cache of the test release and DROP the 26 images on the exclusion list
     (they are copies/near-copies of evaluation frames, see LOG.md).
  2. Split what is left 90/10 by class: fine-tune-train / early-stopping-val.
  3. Fine-tune an ImageNet ResNet-18 (new 43-way last layer) with LIGHT augmentation.
  4. Keep the best epoch (by val accuracy), save checkpoint + curves + results.json.
  5. Sanity check: accuracy on the whole GTSRB TRAINING release (never called "test accuracy").

Why augmentation runs on whole batches with one affine warp instead of a DataLoader:
no worker processes (they are slow/fragile on Windows), same behaviour on CPU, GPU and Colab,
and every random draw comes from one seeded generator, so a run is reproducible.

Usage:   py -3.11 -m src.train --smoke --no-pretrained     # 1-minute pipeline check
         py -3.11 -m src.train --tag 20261003_stage1       # the real run
         py -3.11 -m src.train --tag 20261003_stage1 --resume   # continue after a crash
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml

N_CLASSES = 43
# ImageNet statistics: the pretrained weights expect inputs normalised this way.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------- helpers
def sha256_file(path):
    """Fingerprint of a file; stored next to results so a checkpoint can be verified later."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def normalize(x):
    """x: float (B,3,H,W) in [0,1] -> ImageNet-normalised."""
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def stratified_split(labels, val_fraction, seed):
    """Per-class random split. Every class keeps >= 1 image in BOTH parts, so the smallest
    classes (60 images in the test release) are still represented in training and validation."""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_val = int(min(max(1, round(len(idx) * val_fraction)), len(idx) - 1))
        va.extend(idx[:n_val])
        tr.extend(idx[n_val:])
    return np.sort(np.array(tr)), np.sort(np.array(va))


def augment(x, gen, aug):
    """Light, label-safe augmentation on a float batch (B,3,H,W) in [0,1].
    One random affine warp per image = small zoom-crop (area, aspect) + rotation, then contrast
    (and brightness if enabled). No flips (they swap left/right meaning) and none of the
    stress-test corruptions (blur/fog/rain/noise/perspective/shear)."""
    B = x.shape[0]

    def U(lo, hi):                      # uniform draw per image from the seeded CPU generator
        return lo + (hi - lo) * torch.rand(B, generator=gen)

    area = U(*aug["crop_area_scale"])
    ratio = torch.exp(U(math.log(aug["crop_aspect"][0]), math.log(aug["crop_aspect"][1])))
    sx = torch.sqrt(area * ratio).clamp(max=1.0)     # fraction of width kept
    sy = torch.sqrt(area / ratio).clamp(max=1.0)     # fraction of height kept
    tx = (1 - sx) * (2 * torch.rand(B, generator=gen) - 1)   # crop window stays inside the image
    ty = (1 - sy) * (2 * torch.rand(B, generator=gen) - 1)
    r = aug["rotation_deg"] * math.pi / 180
    ang = U(-r, r) if r > 0 else torch.zeros(B)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = sx * cos, -sy * sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sx * sin, sy * cos, ty
    grid = F.affine_grid(theta.to(x.device), list(x.shape), align_corners=False)
    # 'border' padding: rotated corners repeat edge pixels instead of showing black wedges
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if aug["contrast"] > 0:
        c = U(1 - aug["contrast"], 1 + aug["contrast"]).view(B, 1, 1, 1).to(x.device)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - m) * c + m
    if aug.get("brightness", 0) > 0:
        b = U(1 - aug["brightness"], 1 + aug["brightness"]).view(B, 1, 1, 1).to(x.device)
        x = x * b
    return x.clamp(0, 1)


def build_model(pretrained):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, N_CLASSES)     # 512 -> 43
    return m


@torch.no_grad()
def evaluate(model, images, labels, device, batch=512, limit=None):
    """Plain float32, no augmentation. Used for val and for the training-release sanity number.
    (Stage 2 must reproduce these predictions exactly, so no mixed precision here.)"""
    model.eval()
    n = len(labels) if limit is None else min(limit, len(labels))
    preds, loss_sum = np.empty(n, dtype=np.int64), 0.0
    for s in range(0, n, batch):
        e = min(s + batch, n)
        x = torch.from_numpy(np.array(images[s:e])).to(device).permute(0, 3, 1, 2).float().div_(255)
        out = model(normalize(x))
        y = torch.from_numpy(np.asarray(labels[s:e])).long().to(device)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        preds[s:e] = out.argmax(1).cpu().numpy()
    return float((preds == np.asarray(labels[:n])).mean()), loss_sum / n, preds


def pick_device(arg):
    if arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA GPU is visible to PyTorch")
    return torch.device("cuda" if (arg == "cuda" or (arg == "auto" and torch.cuda.is_available())) else "cpu")


def save_curves(history, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h["epoch"], h["train_loss"], label="train (augmented)")
    ax[0].plot(h["epoch"], h["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(h["epoch"], h["train_acc"], label="train (augmented)")
    ax[1].plot(h["epoch"], h["val_acc"], label="val (optimistic, for stopping only)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--tag", help="run folder name; default <today>_stage1")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<tag>/last.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline tests only)")
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to check the pipeline")
    ap.add_argument("--smoke-n", type=int, default=1000)
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    tc = copy.deepcopy(cfg["train"])
    if a.epochs:
        tc["epochs"] = a.epochs
    if a.batch_size:
        tc["batch_size"] = a.batch_size
    if a.smoke and not a.epochs:
        tc["epochs"] = 2
    cache, size, seed = Path(cfg["data"]["cache_dir"]), cfg["data"]["image_size"], cfg["seed"]
    device = pick_device(a.device)
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- data: test release minus the duplicate exclusion list -------------------------
    meta = pd.read_csv(cache / "test_meta.csv")
    excl = pd.read_csv(tc["exclude_file"])["test_idx"].to_numpy()
    assert len(set(excl)) == len(excl) and excl.min() >= 0 and excl.max() < len(meta), "bad exclusion list"
    keep = np.setdiff1d(np.arange(len(meta)), excl)
    if a.smoke:
        keep = np.sort(np.random.default_rng(seed).choice(keep, min(a.smoke_n, len(keep)), replace=False))
    X_all = np.load(cache / f"test_{size}.npy", mmap_mode="r")
    X = np.ascontiguousarray(X_all[keep])                  # (N,112,112,3) uint8, ~0.47 GB at full size
    labels = meta["class_id"].to_numpy()[keep]
    if not a.smoke:
        assert set(labels) == set(range(N_CLASSES)), "some class has no images after exclusion"
    tr_idx, va_idx = stratified_split(labels, tc["val_fraction"], seed)
    assert not set(tr_idx) & set(va_idx)
    print(f"fine-tune pool: {len(keep)} images ({len(excl)} excluded) -> train {len(tr_idx)} / val {len(va_idx)}")

    imgs_t, labels_t = torch.from_numpy(X), torch.from_numpy(labels).long()
    X_val, y_val = X[va_idx], labels[va_idx]

    # --- run folder --------------------------------------------------------------------
    tag = (a.tag or f"{datetime.date.today():%Y%m%d}_stage1") + ("_smoke" if a.smoke else "")
    run = Path(tc["run_dir"]) / tag
    if (run / "last.pt").exists() and not a.resume:
        raise SystemExit(f"{run} already has a run. Use --resume to continue it or a new --tag.")
    run.mkdir(parents=True, exist_ok=True)
    used = {"train": tc, "seed": seed, "image_size": size, "pretrained": not a.no_pretrained,
            "smoke": a.smoke, "device": str(device), "amp": use_amp}
    (run / "config_used.yaml").write_text(yaml.safe_dump(used))
    chash = config_hash(used)

    # --- model, optimiser, schedule ----------------------------------------------------
    model = build_model(pretrained=not a.no_pretrained).to(device)
    head = list(model.fc.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": tc["lr_backbone"]},
                             {"params": head, "lr": tc["lr_head"]}], weight_decay=tc["weight_decay"])
    bs = tc["batch_size"]
    steps_per_epoch = math.ceil(len(tr_idx) / bs)
    warm, total = tc["warmup_epochs"] * steps_per_epoch, tc["epochs"] * steps_per_epoch

    def lr_lambda(step):                      # linear warm-up, then cosine to zero
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best = [], {"val_acc": -1.0, "val_loss": 9e9, "epoch": 0}
    start_epoch, bad = 1, 0
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        history, best, bad, start_epoch = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']}")

    # --- training loop -------------------------------------------------------------------
    t0 = time.time()
    for epoch in range(start_epoch, tc["epochs"] + 1):
        model.train()
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)     # per-epoch seed => resumable
        order = torch.from_numpy(tr_idx)[torch.randperm(len(tr_idx), generator=gen)]
        loss_sum = correct = seen = 0
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            if len(b) < 2:                                            # BatchNorm needs >= 2 samples
                continue
            x = imgs_t[b].to(device).permute(0, 3, 1, 2).float().div_(255)
            y = labels_t[b].to(device)
            x = normalize(augment(x, gen, tc["aug"]))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = F.cross_entropy(out, y)          # plain CE: NO label smoothing (plan section 5)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(b); correct += (out.argmax(1) == y).sum().item(); seen += len(b)
        v_acc, v_loss, _ = evaluate(model, X_val, y_val, device)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "val_loss": v_loss, "val_acc": v_acc, "lr_backbone": opt.param_groups[0]["lr"],
               "lr_head": opt.param_groups[1]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        improved = (v_acc > best["val_acc"]) or (v_acc == best["val_acc"] and v_loss < best["val_loss"])
        if improved:
            best = {"val_acc": v_acc, "val_loss": v_loss, "epoch": epoch}
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": v_acc}, run / "best.pt")
            bad = 0
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "history": history, "best": best, "bad": bad,
                    "epoch": epoch}, run / "last.pt")         # every epoch: a crash costs one epoch
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
              f"val loss {v_loss:.4f} acc {v_acc:.4f} {'*' if improved else ''} | {row['seconds']:.0f}s")
        if bad >= tc["patience"]:
            print(f"early stop: no val improvement for {tc['patience']} epochs")
            break

    # --- sanity accuracy on the TRAINING release with the best epoch ---------------------
    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=False)["model"])
    Xtr = np.load(cache / f"train_{size}.npy", mmap_mode="r")
    ytr = pd.read_csv(cache / "train_meta.csv")["class_id"].to_numpy()
    acc, _, pred = evaluate(model, Xtr, ytr, device, limit=600 if a.smoke else None)
    n_eval = len(pred)
    per_class = pd.Series(pred == ytr[:n_eval]).groupby(ytr[:n_eval]).mean().sort_values()
    save_curves(history, run / "curves.png")
    ck_hash = sha256_file(run / "best.pt")
    results = {
        "sanity_accuracy_training_release": acc,
        "sanity_accuracy_label": "accuracy on the GTSRB training release (backbone fine-tuned on the "
                                 "official test release); NOT test accuracy",
        "frames_evaluated": n_eval, "worst_5_classes": {int(k): float(v) for k, v in per_class.head(5).items()},
        "best_epoch": best["epoch"], "best_val_acc_optimistic": best["val_acc"],
        "epochs_run": len(history), "n_pool": int(len(keep)), "n_excluded": int(len(excl)),
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "checkpoint": str(run / "best.pt"), "checkpoint_sha256": ck_hash, "config_sha256": chash,
        "seed": seed, "device": str(device), "amp": use_amp, "smoke": a.smoke,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__},
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (run / "results.json").write_text(json.dumps(results, indent=2))
    print("\n=== Stage 1 summary ===")
    print(f"best epoch {best['epoch']} (val acc {best['val_acc']:.4f}; optimistic: same signs can sit in train and val)")
    print(f"accuracy on the GTSRB TRAINING release: {acc:.4f}  ({n_eval} frames; not test accuracy)")
    print(f"worst 5 classes: {results['worst_5_classes']}")
    print(f"checkpoint sha256: {ck_hash}")
    if not a.smoke:
        if acc < 0.90:
            print("CHECK FAILED: below 90%. Suspect a label-mapping, normalisation or resize bug. Do not continue.")
        elif acc > 0.995:
            print("NOTE: above 99.5%. Do NOT weaken the model; few clean errors puts G1(ii) at risk (plan section 9).")
        else:
            print("Accuracy is inside the expected band (about 95-99%).")
    return results


if __name__ == "__main__":
    main()
