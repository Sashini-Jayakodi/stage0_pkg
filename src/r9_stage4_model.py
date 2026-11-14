"""POC-3 (R9) Stage 4, model part: 62-class ResNet-18 fine-tune (POC-1 recipe), embeddings, tied-Gaussian head.

Definitions: LOG_r9_stage4_entry.md Part A (f) to (h). POC-1 code is imported, never edited: src.train (augment,
normalize, evaluate, save_curves), src.embed (extract, l2_normalise, check_block) and src.head (ledoit_wolf, score).
POC-1 fixes the number of classes at 43 inside those modules, so the model builder and head fit are written here for
any K and tested against the POC-1 versions.
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src import train as T
from src.head import ledoit_wolf

IMAGENET_FILE = "resnet18-f37072fd.pth"     # torchvision IMAGENET1K_V1 weights for ResNet-18
RECIPE_KEYS = ("epochs", "patience", "batch_size", "lr_backbone", "lr_head", "weight_decay", "warmup_epochs", "amp",
               "aug")


def check_recipe(r9_cfg, poc_cfg):
    """The training recipe must be the POC-1 recipe (Part A (f)); stop on any difference."""
    for k in RECIPE_KEYS:
        if r9_cfg["train"][k] != poc_cfg["train"][k]:
            raise SystemExit(f"STOP: recipe value {k} differs from configs/poc.yaml")
    if r9_cfg["seed"] != poc_cfg["seed"]:
        raise SystemExit("STOP: seed differs from configs/poc.yaml")
    if r9_cfg["val_fraction"][0] / r9_cfg["val_fraction"][1] != poc_cfg["train"]["val_fraction"]:
        raise SystemExit("STOP: validation fraction differs from configs/poc.yaml")


def imagenet_weights_path():
    """Local copy only: training stops rather than downloading weights (downloads need the author's approval)."""
    p = Path(torch.hub.get_dir()) / "checkpoints" / IMAGENET_FILE
    if not p.is_file():
        raise SystemExit(f"STOP: {p} not found; no download is made")
    return p


def build_model(n_classes, pretrained):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def fine_tune(X, labels, tr_idx, va_idx, tc, seed, device, out_dir, n_classes, pretrained=True, log=print):
    """POC-1 Stage 1 training loop (src/train.py main), unchanged apart from K and the given train/val rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)
    imgs_t, labels_t = torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(labels).long()
    X_val, y_val = np.ascontiguousarray(X[va_idx]), labels[va_idx]
    model = build_model(n_classes, pretrained).to(device)
    head = list(model.fc.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": tc["lr_backbone"]},
                             {"params": head, "lr": tc["lr_head"]}], weight_decay=tc["weight_decay"])
    bs = tc["batch_size"]
    steps_per_epoch = math.ceil(len(tr_idx) / bs)
    warm, total = tc["warmup_epochs"] * steps_per_epoch, tc["epochs"] * steps_per_epoch

    def lr_lambda(step):                      # linear warm-up, then cosine to zero (POC-1)
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best, bad, stopped_early = [], {"val_acc": -1.0, "val_loss": 9e9, "epoch": 0}, 0, False
    t0 = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.from_numpy(tr_idx)[torch.randperm(len(tr_idx), generator=gen)]
        loss_sum = correct = seen = 0
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            if len(b) < 2:                                            # BatchNorm needs >= 2 samples
                continue
            x = imgs_t[b].to(device).permute(0, 3, 1, 2).float().div_(255)
            y = labels_t[b].to(device)
            x = T.normalize(T.augment(x, gen, tc["aug"]))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = F.cross_entropy(out, y)                        # plain CE, no label smoothing
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(b); correct += (out.argmax(1) == y).sum().item(); seen += len(b)
        v_acc, v_loss, _ = T.evaluate(model, X_val, y_val, device)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen, "val_loss": v_loss,
               "val_acc": v_acc, "lr_backbone": opt.param_groups[0]["lr"], "lr_head": opt.param_groups[1]["lr"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        improved = (v_acc > best["val_acc"]) or (v_acc == best["val_acc"] and v_loss < best["val_loss"])
        if improved:
            best = {"val_acc": v_acc, "val_loss": v_loss, "epoch": epoch}
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": v_acc}, out_dir / "best.pt")
            bad = 0
        else:
            bad += 1
        log(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
            f"val loss {v_loss:.4f} acc {v_acc:.4f} {'*' if improved else ''} | {row['seconds']:.0f}s")
        if bad >= tc["patience"]:
            stopped_early = True
            log(f"early stop: no val improvement for {tc['patience']} epochs")
            break
    try:
        T.save_curves(history, out_dir / "curves.png")
    except Exception as e:                                            # a figure must never block the numbers
        log(f"WARNING: curves figure failed: {e!r}")
    return {"best_epoch": best["epoch"], "best_val_acc": best["val_acc"], "best_val_loss": best["val_loss"],
            "epochs_run": len(history), "stopped_early": stopped_early, "use_amp": use_amp,
            "seconds": round(time.time() - t0, 1)}


def load_checkpoint(path, n_classes, device):
    model = build_model(n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    model.eval()
    return model


def fit_head_k(emb, y, K):
    """src.head.fit_head for any K: class means, pooled within-class covariance with Ledoit-Wolf shrinkage,
    whitening, precision and class-frequency priors (POC-1 Stage 3 settings, no PCA)."""
    emb = emb.astype(np.float64)
    assert set(np.unique(y)) == set(range(K)), "every class needs fit embeddings to fit its mean"
    mu = np.stack([emb[y == k].mean(0) for k in range(K)])
    cov, shrink = ledoit_wolf(emb - mu[y])
    ev, V = np.linalg.eigh(cov)
    assert ev.min() > 0, "covariance is not positive definite"
    W = (V * ev ** -0.5) @ V.T
    prec = (V / ev) @ V.T
    priors = np.bincount(y, minlength=K) / len(y)
    return {"mu": mu, "W": W, "prec": prec, "priors": priors, "shrinkage": shrink,
            "cond": float(ev.max() / ev.min()), "eig_min": float(ev.min()), "eig_max": float(ev.max())}
