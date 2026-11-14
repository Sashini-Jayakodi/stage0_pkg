"""POC-3 (R9) Stage 4 runner: split, cache, train, embed, head, vchecks (LOG_r9_stage4_entry.md Part A).

Usage:  py -3.12 -m src.r9_stage4 split | cache | train | embed | head | vchecks

Each step checks the frozen files (src/r9_frozen.py), re-checks its inputs against the summary JSON that wrote them
(hard rule 2), refuses to overwrite a recorded output and appends its printout to runs/btsc/stage4/stage4_printout.txt.
No step computes a label, error or accuracy on pool poles: testing labels come only through src/r9_btsc_guard.py.
"""
import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml

from src import embed as E
from src import r9_stage4_data as D
from src import r9_stage4_model as M
from src import train as T
from src.head import score
from src.r9_btsc_guard import GuardedTestLabels, f3_closed
from src.r9_frozen import check_frozen
from src.r9_stage1_download import sha256_file

CFG = Path("configs/r9_btsc_stage4.yaml")
POC_CFG = Path("configs/poc.yaml")
CODE = ["src/r9_stage4.py", "src/r9_stage4_data.py", "src/r9_stage4_model.py", "src/r9_btsc_guard.py",
        "configs/r9_btsc_stage4.yaml"]
STEPS = ["split", "cache", "train", "embed", "head"]
SETS = ["train_fit", "train_val", "test_design", "test_pool"]


class Tee:
    """Write the printout to the console and to the stage printout file at once."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def provenance():
    return {"run_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_head": git("rev-parse", "HEAD"), "worktree_clean": git("status", "--porcelain") == "",
            "code_sha256": {f: sha256_file(f) for f in CODE},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "torchvision": torchvision.__version__, "numpy": np.__version__}}


def summary_path(out, step):
    return Path(out) / f"{step}_summary.json"


def check_prior(out, step):
    """Every earlier step's outputs must still match the hashes its summary recorded."""
    sums = {}
    for s in STEPS[:STEPS.index(step)] if step in STEPS else STEPS:
        p = summary_path(out, s)
        if not p.is_file():
            D.stop(f"{p} missing; run step {s} first")
        sums[s] = D.check_summary_files(p, out)
    return sums


def load_cfg():
    return yaml.safe_load(CFG.read_text()), yaml.safe_load(POC_CFG.read_text())


def design_guard(out, rows):
    poles = json.loads((Path(out) / "split_poles.json").read_text())
    return GuardedTestLabels([r["pole"] for r in rows], [r["cls"] for r in rows], poles["design_poles"]), poles


# ---------------------------------------------------------------- split
def step_split(paths, cfg):
    s1, rels, in_hashes = D.check_inputs(paths)
    tr, ndup_tr = D.dedupe(rels["training"]["views"])
    te, ndup_te = D.dedupe(rels["testing"]["views"])
    f_tr = D.check_counts(s1, "training", tr, rels["training"]["n_images"])
    f_te = D.check_counts(s1, "testing", te, rels["testing"]["n_images"])
    kept, removed = D.apply_removal(s1, te)
    f_kept = D.facts(kept)
    print(f"V4.1 counts after de-duplication: training {f_tr} | testing {f_te} | testing after removal {f_kept}")
    print(f"duplicated GT rows dropped: training {ndup_tr}, testing {ndup_te}; removed units {len(removed)}")

    test_units = sorted({(v["pole"], v["cls"]) for v in kept})
    design = D.split_by_pole(test_units, tuple(cfg["design_fraction"]), cfg["seed"])
    all_test_poles = sorted({p for p, _ in test_units})
    pool = sorted(set(all_test_poles) - set(design))
    train_units = sorted({(v["pole"], v["cls"]) for v in tr})
    val = D.split_by_pole(train_units, tuple(cfg["val_fraction"]), cfg["seed"])
    fit = sorted({p for p, _ in train_units} - set(val))
    exp = cfg["expected"]
    if (len(design), len(pool), len(val)) != (exp["design_poles"], exp["pool_poles"], exp["train_val_poles"]):
        D.stop(f"split sizes {len(design)}/{len(pool)}/{len(val)} differ from Part A")
    fit_classes = {v["cls"] for v in tr if v["pole"] not in set(val)}
    if fit_classes != set(range(cfg["n_classes"])):
        D.stop(f"classes missing from the training part: {sorted(set(range(cfg['n_classes'])) - fit_classes)}")

    out = Path(paths.out)
    out.mkdir(parents=True, exist_ok=True)
    ds, vs = set(design), set(val)
    D.write_table(out / "train_views.csv", D.view_rows(tr, paths.extract_dir, lambda p: "val" if p in vs else "fit"))
    D.write_table(out / "test_views.csv",
                  D.view_rows(kept, paths.extract_dir, lambda p: "design" if p in ds else "pool"))
    lists = {"design_poles": design, "pool_poles": pool, "train_val_poles": val, "train_fit_poles": fit}
    list_sha = {k: D.sha256_bytes(json.dumps(v).encode()) for k, v in lists.items()}
    (out / "split_poles.json").write_text(json.dumps({**lists, "list_sha256": list_sha}, indent=1) + "\n")
    test_t = D.side_tables(kept, design, ("design", "pool"))
    train_t = D.side_tables(tr, val, ("val", "fit"))
    summ = {"stage": "R9 Stage 4, split", **provenance(), "inputs_sha256": in_hashes,
            "counts": {"training": f_tr, "testing": f_te, "testing_after_removal": f_kept},
            "duplicated_rows_dropped": {"training": ndup_tr, "testing": ndup_te}, "removed_units": len(removed),
            "seed": cfg["seed"], "test_split": test_t, "train_split": train_t, "pole_list_sha256": list_sha,
            "files_sha256": {f: sha256_file(out / f) for f in ("train_views.csv", "test_views.csv", "split_poles.json")}}
    sha = D.write_summary(summary_path(out, "split"), summ)
    print_split(summ)
    print(f"split_summary.json SHA-256 {sha}")


def print_split(s):
    t = s["test_split"]
    for side in ("design", "pool"):
        print(f"testing {side}: {t[side]['poles']} poles, {t[side]['units']} units, {t[side]['views']} views, "
              f"{t[side]['classes']} classes")
    print(f"testing strata: {t['strata']}; strata on one side only: {len(t['strata_one_side_only'])}")
    for r in t["strata_one_side_only"]:
        print(f"  stratum class {r['stratum_class']}: {r['poles']} pole(s), all {r['side']}")
    print(f"testing classes on one side only: {len(t['classes_one_side_only'])}")
    for r in t["classes_one_side_only"]:
        print(f"  class {r['cls']}: {r['units']} unit(s), all {r['side']}")
    print("testing units per class (class: design / pool):")
    cls = sorted({int(c) for side in ("design", "pool") for c in t[side]["units_per_class"]})
    line = [f"{c}: {t['design']['units_per_class'].get(str(c), 0)}/{t['pool']['units_per_class'].get(str(c), 0)}"
            for c in cls]
    for i in range(0, len(line), 10):
        print("  " + " | ".join(line[i:i + 10]))
    r = s["train_split"]
    print(f"training fit part: {r['fit']['poles']} poles, {r['fit']['units']} units, {r['fit']['views']} views, "
          f"{r['fit']['classes']} classes | validation: {r['val']['poles']} poles, {r['val']['units']} units, "
          f"{r['val']['views']} views, {r['val']['classes']} classes")
    print("pole list SHA-256: " + ", ".join(f"{k} {v}" for k, v in s["pole_list_sha256"].items()))


# ---------------------------------------------------------------- cache
def step_cache(paths, cfg):
    D.check_inputs(paths)                     # the extracted images are re-verified before they are read
    out = Path(paths.out)
    check_prior(out, "cache")
    (out / "cache").mkdir(exist_ok=True)
    files, rows = {}, {}
    for name in ("train", "test"):
        tab = D.read_table(out / f"{name}_views.csv")
        npy = out / "cache" / f"{name}_{cfg['image_size']}.npy"
        if npy.exists():
            D.stop(f"{npy} exists; refusing to overwrite")
        D.build_cache([Path(paths.extract_dir) / r["relpath"] for r in tab], npy, cfg["image_size"])
        files[f"cache/{npy.name}"] = sha256_file(npy)
        rows[name] = len(tab)
        print(f"cache {name}: {len(tab)} views -> {npy.name} SHA-256 {files[f'cache/{npy.name}']}")
    sha = D.write_summary(summary_path(out, "cache"), {"stage": "R9 Stage 4, cache", **provenance(),
                                                       "image_size": cfg["image_size"], "input_region": "whole image",
                                                       "rows": rows, "files_sha256": files})
    print(f"cache_summary.json SHA-256 {sha}")


# ---------------------------------------------------------------- train
def step_train(paths, cfg, device):
    out = Path(paths.out)
    check_prior(out, "train")
    r9, poc = load_cfg()
    M.check_recipe(r9, poc)
    wfile = M.imagenet_weights_path()
    wsha = sha256_file(wfile)
    tab = D.read_table(out / "train_views.csv")
    X = np.load(out / "cache" / f"train_{cfg['image_size']}.npy", mmap_mode="r")
    assert len(X) == len(tab), "train cache and table differ in length"
    y = np.array([r["cls"] for r in tab], dtype=np.int64)
    part = np.array([r["part"] for r in tab])
    tr_idx, va_idx = np.flatnonzero(part == "fit"), np.flatnonzero(part == "val")
    print(f"fine-tune: {len(tr_idx)} training-part views, {len(va_idx)} validation views (training release only); "
          f"ImageNet weights {wfile.name} SHA-256 {wsha}; device {device}")
    if (out / "model" / "best.pt").exists():
        D.stop("model/best.pt exists; refusing to overwrite")
    res = M.fine_tune(X, y, tr_idx, va_idx, cfg["train"], cfg["seed"], device, out / "model", cfg["n_classes"])
    ck = sha256_file(out / "model" / "best.pt")
    summ = {"stage": "R9 Stage 4, train", **provenance(), **res, "n_fit_views": int(len(tr_idx)),
            "n_val_views": int(len(va_idx)), "seed": cfg["seed"], "recipe": cfg["train"],
            "imagenet_weights_file": wfile.name, "imagenet_weights_sha256": wsha, "checkpoint_sha256": ck,
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "files_sha256": {"model/best.pt": ck, "model/history.csv": sha256_file(out / "model" / "history.csv")}}
    sha = D.write_summary(summary_path(out, "train"), summ)
    print(f"epochs run {res['epochs_run']} (stopped early: {res['stopped_early']}); best epoch {res['best_epoch']} "
          f"(validation accuracy {res['best_val_acc']:.4f}, loss {res['best_val_loss']:.4f})")
    print(f"checkpoint SHA-256 {ck}")
    print(f"train_summary.json SHA-256 {sha}")


# ---------------------------------------------------------------- embed
def set_rows(out):
    tr, te = D.read_table(out / "train_views.csv"), D.read_table(out / "test_views.csv")
    p_tr, p_te = np.array([r["part"] for r in tr]), np.array([r["part"] for r in te])
    return tr, te, {"train_fit": ("train", np.flatnonzero(p_tr == "fit")),
                    "train_val": ("train", np.flatnonzero(p_tr == "val")),
                    "test_design": ("test", np.flatnonzero(p_te == "design")),
                    "test_pool": ("test", np.flatnonzero(p_te == "pool"))}


def step_embed(paths, cfg, device):
    out = Path(paths.out)
    sums = check_prior(out, "embed")
    ck_path = out / "model" / "best.pt"
    assert sha256_file(ck_path) == sums["train"]["checkpoint_sha256"], "checkpoint differs from train_summary.json"
    model = M.load_checkpoint(ck_path, cfg["n_classes"], device)
    fc_w, fc_b = model.fc.weight.detach().cpu().numpy(), model.fc.bias.detach().cpu().numpy()
    tr, te, rows = set_rows(out)
    guard, _ = design_guard(out, te)
    units = sorted({(r["pole"], r["cls"]) for r in te})
    code = {u: i for i, u in enumerate(units)}                # opaque unit number, no class in the pool file
    X = {k: np.load(out / "cache" / f"{k}_{cfg['image_size']}.npy", mmap_mode="r") for k in ("train", "test")}
    (out / "embeddings").mkdir(exist_ok=True)
    bs, files, info_all = cfg["embed"]["batch_size"], {}, {}
    for name in SETS:
        src, idx = rows[name]
        tab = tr if src == "train" else te
        Xs = np.ascontiguousarray(X[src][idx])
        raw, logits = E.extract(model, Xs, device, bs)
        emb = E.l2_normalise(raw)
        # Label-free checks: y is used by check_block only for row counts, so zeros stand in for every set.
        info = E.check_block(name, raw, emb, logits, np.zeros(len(idx)), fc_w, fc_b, cfg["embed"]["norm_tol"])
        # Prediction path check: evaluate() also returns an accuracy and loss against the zeros; both are discarded.
        _, _, pred_ref = T.evaluate(model, Xs, np.zeros(len(idx), dtype=np.int64), device, batch=bs)
        info["argmax_equals_evaluate"] = bool((logits.argmax(1) == pred_ref).all())
        assert info["argmax_equals_evaluate"], f"{name}: logits argmax differs from the evaluation path"
        arrays = dict(emb=emb, raw=raw, logits=logits, row=idx.astype(np.int64),
                      pole=np.array([tab[i]["pole"] for i in idx], dtype=np.int64),
                      view=np.array([tab[i]["view"] for i in idx], dtype=np.int64),
                      size=np.array([tab[i]["size"] for i in idx], dtype=np.int64))
        if src == "test":
            arrays["unit"] = np.array([code[(tab[i]["pole"], tab[i]["cls"])] for i in idx], dtype=np.int64)
        if name == "test_design":
            arrays["y"] = guard.labels(idx)
        elif src == "train":
            arrays["y"] = np.array([tab[i]["cls"] for i in idx], dtype=np.int64)
        # test_pool: no labels (Part A (g))
        np.savez(out / "embeddings" / f"{name}.npz", **arrays)
        files[f"embeddings/{name}.npz"] = sha256_file(out / "embeddings" / f"{name}.npz")
        info_all[name] = info
        print(f"{name}: {info['rows']} rows | max|norm-1| {info['max_norm_error']:.1e} | "
              f"max fc consistency error {info['max_fc_consistency_error']:.1e} | "
              f"argmax equals evaluate path: {info['argmax_equals_evaluate']}")
    sha = D.write_summary(summary_path(out, "embed"), {"stage": "R9 Stage 4, embed", **provenance(),
                                                       "checkpoint_sha256": sums["train"]["checkpoint_sha256"],
                                                       "checks": info_all, "files_sha256": files})
    print(f"embed_summary.json SHA-256 {sha}")


# ---------------------------------------------------------------- head
def accuracy_block(sc, logits, y):
    head, soft = sc["top3"][:, 0].astype(np.int64), logits.argmax(1)
    return {"rows": int(len(y)), "head_accuracy": float((head == y).mean()), "softmax_accuracy": float((soft == y).mean()),
            "label_agreement": float((head == soft).mean())}


def last_view_units(pole, view, y, head_pred, soft_pred):
    """Per design unit (pole, class), the prediction at its last view in view order (ascending view number)."""
    last = {}
    for i, u in enumerate(zip(pole.tolist(), y.tolist())):
        if u not in last or view[i] > view[last[u]]:
            last[u] = i
    idx = np.array(sorted(last.values()))
    return {"units": int(len(idx)), "head_accuracy": float((head_pred[idx] == y[idx]).mean()),
            "softmax_accuracy": float((soft_pred[idx] == y[idx]).mean())}


def step_head(paths, cfg):
    out = Path(paths.out)
    check_prior(out, "head")
    _, te, rows = set_rows(out)
    guard, _ = design_guard(out, te)
    data = {n: np.load(out / "embeddings" / f"{n}.npz") for n in SETS}
    assert "y" not in data["test_pool"].files, "the pool embeddings file must hold no labels"
    head = M.fit_head_k(data["train_fit"]["emb"], data["train_fit"]["y"], cfg["n_classes"])
    (out / "head").mkdir(exist_ok=True)
    np.savez(out / "head" / "head.npz", mu=head["mu"], W=head["W"], prec=head["prec"], priors=head["priors"],
             shrinkage=head["shrinkage"])
    files, res = {"head/head.npz": sha256_file(out / "head" / "head.npz")}, {}
    for n in SETS:
        sc = score(head, data[n]["emb"])
        arrays = dict(top3=sc["top3"], lp3=sc["lp3"], margin=sc["margin"], s=sc["s"],
                      softmax_pred=data[n]["logits"].argmax(1).astype(np.int16), row=data[n]["row"],
                      pole=data[n]["pole"], view=data[n]["view"], size=data[n]["size"])
        if n != "test_pool":
            arrays["y"] = data[n]["y"]
        if n.startswith("test"):
            arrays["unit"] = data[n]["unit"]
        np.savez(out / "head" / f"scores_{n}.npz", **arrays)
        files[f"head/scores_{n}.npz"] = sha256_file(out / "head" / f"scores_{n}.npz")
        if n == "test_design":
            y = guard.labels(data[n]["row"])               # through the guard again; must equal the stored labels
            assert np.array_equal(y, data[n]["y"]), "design labels differ from the guarded labels"
            res[n] = {"per_view": accuracy_block(sc, data[n]["logits"], y),
                      "per_unit_last_view": last_view_units(data[n]["pole"], data[n]["view"], y,
                                                            sc["top3"][:, 0].astype(np.int64),
                                                            data[n]["logits"].argmax(1))}
        elif n == "test_pool":
            res[n] = {"rows": int(len(data[n]["row"]))}      # row count only (hard rule 3)
        else:
            res[n] = accuracy_block(sc, data[n]["logits"], data[n]["y"])
    summ = {"stage": "R9 Stage 4, head", **provenance(), "features": "l2-normalised 512-d embeddings, no PCA",
            "fit_set": "train_fit", "shrinkage": head["shrinkage"], "cov_condition_number": head["cond"],
            "cov_eig_min": head["eig_min"], "cov_eig_max": head["eig_max"], "results": res, "files_sha256": files}
    sha = D.write_summary(summary_path(out, "head"), summ)
    print(f"head: fit on {res['train_fit']['rows']} training-part views, Ledoit-Wolf shrinkage {head['shrinkage']:.4f}, "
          f"covariance condition number {head['cond']:.3g}")
    for n in ("train_fit", "train_val"):
        r = res[n]
        print(f"{n}: {r['rows']} views | head acc {r['head_accuracy']:.4f} | softmax acc {r['softmax_accuracy']:.4f} | "
              f"agreement {r['label_agreement']:.4f}")
    d = res["test_design"]
    print(f"test_design per view: {d['per_view']['rows']} views | head acc {d['per_view']['head_accuracy']:.4f} | "
          f"softmax acc {d['per_view']['softmax_accuracy']:.4f} | agreement {d['per_view']['label_agreement']:.4f}")
    print(f"test_design per unit at last view: {d['per_unit_last_view']['units']} units | head acc "
          f"{d['per_unit_last_view']['head_accuracy']:.4f} | softmax acc {d['per_unit_last_view']['softmax_accuracy']:.4f}")
    print(f"test_pool: {res['test_pool']['rows']} views (row count only; no label read)")
    print(f"head_summary.json SHA-256 {sha}")


# ---------------------------------------------------------------- vchecks
def step_vchecks(paths, cfg, device):
    out = Path(paths.out)
    if (out / "stage4_summary.json").exists():
        D.stop("stage4_summary.json exists; refusing to overwrite")
    v = {}
    # V4.3 (part 1): all inputs and every step's outputs against the hashes recorded when they were written.
    s1, rels, in_hashes = D.check_inputs(paths)
    sums = check_prior(out, "vchecks")
    v["V4.3"] = "PASS"
    # V4.1: counts recomputed from the inputs.
    tr, _ = D.dedupe(rels["training"]["views"])
    te, _ = D.dedupe(rels["testing"]["views"])
    D.check_counts(s1, "training", tr, rels["training"]["n_images"])
    D.check_counts(s1, "testing", te, rels["testing"]["n_images"])
    kept, removed = D.apply_removal(s1, te)
    v["V4.1"] = "PASS"
    # V4.2: the split recomputed from the rule equals the recorded split, and its properties hold.
    poles = json.loads((out / "split_poles.json").read_text())
    tab_tr, tab_te = D.read_table(out / "train_views.csv"), D.read_table(out / "test_views.csv")
    test_units = sorted({(x["pole"], x["cls"]) for x in kept})
    train_units = sorted({(x["pole"], x["cls"]) for x in tr})
    design, pool, val, fit = (set(poles[k]) for k in ("design_poles", "pool_poles", "train_val_poles", "train_fit_poles"))
    side_of = {r["pole"]: r["part"] for r in tab_te}
    ok = [
        sorted(design) == D.split_by_pole(test_units, tuple(cfg["design_fraction"]), cfg["seed"]),
        sorted(val) == D.split_by_pole(train_units, tuple(cfg["val_fraction"]), cfg["seed"]),
        (len(design), len(pool), len(val)) == (cfg["expected"]["design_poles"], cfg["expected"]["pool_poles"],
                                               cfg["expected"]["train_val_poles"]),
        not design & pool, design | pool == {p for p, _ in test_units},
        all((r["part"] == "design") == (r["pole"] in design) for r in tab_te),
        all(side_of[p] == ("design" if p in design else "pool") for p, _ in test_units),
        not {(r["pole"], r["cls"]) for r in tab_te} & removed,
        {(r["pole"], r["cls"]) for r in tab_te} == set(test_units),
        not val & fit, val | fit == {p for p, _ in train_units},
        all((r["part"] == "val") == (r["pole"] in val) for r in tab_tr),
        {r["cls"] for r in tab_tr if r["part"] == "fit"} == set(range(cfg["n_classes"])),
        all(D.sha256_bytes(json.dumps(poles[k]).encode()) == poles["list_sha256"][k]
            for k in ("design_poles", "pool_poles", "train_val_poles", "train_fit_poles")),
    ]
    v["V4.2"] = "PASS" if all(ok) else f"FAIL (conditions {[i for i, b in enumerate(ok) if not b]})"
    # V4.4: the guard refuses a pool label before F3 and passes design labels.
    guard, _ = design_guard(out, tab_te)
    drows = guard.design_rows()
    prow = int(np.flatnonzero(np.array([r["part"] for r in tab_te]) == "pool")[0])
    try:
        guard.labels([prow])
        refused = False
    except AssertionError:
        refused = True
    v["V4.4"] = "PASS" if (refused and not f3_closed() and len(guard.labels(drows)) == len(drows)) else "FAIL"
    # V4.5: embeddings re-read and re-checked against the checkpoint's last layer.
    model = M.load_checkpoint(out / "model" / "best.pt", cfg["n_classes"], torch.device("cpu"))
    fc_w, fc_b = model.fc.weight.detach().numpy(), model.fc.bias.detach().numpy()
    _, _, rows = set_rows(out)
    ok5 = []
    for n in SETS:
        z = np.load(out / "embeddings" / f"{n}.npz")
        E.check_block(n, z["raw"], z["emb"], z["logits"], np.zeros(len(z["row"])), fc_w, fc_b, cfg["embed"]["norm_tol"])
        ok5 += [np.array_equal(z["row"], rows[n][1]), sums["embed"]["checks"][n]["argmax_equals_evaluate"]]
    ok5.append("y" not in np.load(out / "embeddings" / "test_pool.npz").files)
    v["V4.5"] = "PASS" if all(ok5) else "FAIL"
    # V4.6: cache rows and order equal the view tables (hashes were checked in V4.3).
    ok6 = []
    for name, tab in (("train", tab_tr), ("test", tab_te)):
        X = np.load(out / "cache" / f"{name}_{cfg['image_size']}.npy", mmap_mode="r")
        ok6.append(X.shape == (len(tab), cfg["image_size"], cfg["image_size"], 3))
        for i in (0, len(tab) // 2, len(tab) - 1):        # spot re-decode: same image in the same row
            ok6.append(np.array_equal(X[i], D.load_whole(Path(paths.extract_dir) / tab[i]["relpath"], cfg["image_size"])))
    v["V4.6"] = "PASS" if all(ok6) else "FAIL"
    all_pass = all(x == "PASS" for x in v.values())
    summ = {"stage": "R9 Stage 4", **provenance(), "inputs_sha256": in_hashes,
            "step_summaries_sha256": {f"{s}_summary.json": sha256_file(summary_path(out, s)) for s in STEPS},
            "checkpoint_sha256": sums["train"]["checkpoint_sha256"], "vchecks": v, "all_pass": all_pass,
            "files_sha256": {k: h for s in STEPS for k, h in sums[s]["files_sha256"].items()}}
    sha = D.write_summary(out / "stage4_summary.json", summ)
    print("V-checks: " + " | ".join(f"{k} {x}" for k, x in sorted(v.items())) + f" | ALL PASS: {all_pass}")
    print(f"stage4_summary.json SHA-256 {sha}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=STEPS + ["vchecks"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args(argv)
    paths = D.Paths()
    cfg, _ = load_cfg()
    Path(paths.out).mkdir(parents=True, exist_ok=True)
    with open(Path(paths.out) / "stage4_printout.txt", "a", encoding="utf-8") as log:
        old = sys.stdout
        sys.stdout = Tee(old, log)
        try:
            print(f"\n===== R9 Stage 4 step {a.step} | {provenance()['run_utc']} | HEAD {git('rev-parse', 'HEAD')} "
                  f"| worktree clean {git('status', '--porcelain') == ''} =====")
            blobs = check_frozen()
            print(f"freeze check PASS ({len(blobs)} files by git blob ID)")
            device = T.pick_device(a.device)
            if a.step == "split":
                step_split(paths, cfg)
            elif a.step == "cache":
                step_cache(paths, cfg)
            elif a.step == "train":
                step_train(paths, cfg, device)
            elif a.step == "embed":
                step_embed(paths, cfg, device)
            elif a.step == "head":
                step_head(paths, cfg)
            else:
                step_vchecks(paths, cfg, device)
        finally:
            sys.stdout = old


if __name__ == "__main__":
    main()
