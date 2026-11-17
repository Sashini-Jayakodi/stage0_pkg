"""EXPLORATORY (post hoc). d1: a finer corruption grid on the 261 GTSRB design tracks only.

No pool track is read: the design loader of POC-2 Stage 2 refuses the pool table and keeps design rows only, and the
image cache is indexed at design-track rows only (asserted). Output: runs/20261003_stage1/d1/.

Levels per corruption (fog, motion_blur, gaussian_noise, contrast, brightness; imagecorruptions 1.1.2, fog through
src/d0_corrupt_compat.py):
  L1-L6  x' = round(clip((1 - t) * clean + t * corrupt(clean, severity 1))) as uint8, t in {0.1, 0.2, 0.3, 0.5, 0.75, 1.0}
  L7-L10 corrupt(clean, severity 2..5)
Every frame of a track gets the same corruption and level; the global numpy RNG is seeded once per track with
[20261117, track code] (as d0), so L1-L6 share one severity-1 draw per frame and L6 equals severity 1 exactly.

Per condition (clean + 5 x 10), on the design tracks: frame error (top-1 head error over all frames, before the gate),
gate pass rate, track-consensus error (a0 rule: majority vote over all frames, ties to the larger summed pre-gate
margin, then the smaller class id), and for S1, S2-sum, S3, S4 the commit rate and committed error at the Stage 3
reference threshold per alpha in {2, 3, 5}% (the loosest grid point whose CLEAN design committed error is <= alpha;
fixed in stage3_design.json). The gate is recomputed on corrupted frames with the frozen tau. S2-sum's ICP/ECDF was
fitted on these same design tracks (two halves), so its numbers here are in-sample for its calibration.

Usage:  py -3.12 -m src.d1_grid_check --run runs/20261003_stage1 --config configs/poc.yaml
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src import d0_corrupt_compat as DC
from src import d0_envelope_check as D0
from src import r8_stage2 as S2
from src import r9_frozen as FZ
from src import r9_stage3 as R3
from src import r9_stage3_scores as SC
from src import train as TR
from src.a0_silent_error_check import consensus
from src.head import mahal_sq
from src.r8_stage3_b1design import frame_margins
from src.util import sha256_file

LABEL = "EXPLORATORY (post hoc; design tracks only; GTSRB tracks all seen before)"
ALPHAS = (0.02, 0.03, 0.05)
SCORES = ("S1", "S2-sum", "S3", "S4")
BLEND_T = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
LEVELS = tuple(range(1, 11))                                   # L1-L6 blends of severity 1, L7-L10 severities 2-5
CHUNK = 8


def level_spec(level):
    """(severity, t): t = None means the plain package output at that severity."""
    assert level in LEVELS
    return (1, BLEND_T[level - 1]) if level <= 6 else (level - 5, None)


def blend(clean, corrupted, t):
    """(1 - t) * clean + t * corrupted, rounded to the nearest integer, clipped to [0, 255], uint8."""
    x = (1.0 - t) * clean.astype(np.float64) + t * corrupted.astype(np.float64)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def corrupt_track_level(frames, name, level, seed):
    """All frames of one track at one level; RNG seeded once per track, frames in frame order."""
    sev, t = level_spec(level)
    out = DC.corrupt_track(frames, name, sev, seed)
    return out if t is None else np.stack([blend(f, c, t) for f, c in zip(frames, out)])


def at_threshold(seqs, y_track, grid, j):
    """Commit rate (over all tracks) and committed error at grid index j; (None, None) if j < 0."""
    if j < 0:
        return None, None
    tab = SC.track_table(seqs, y_track, grid[j:j + 1])
    c, w = tab["committed"][:, 0], tab["wrong"][:, 0] & tab["committed"][:, 0]
    return c.mean(), (w.sum() / c.sum()) if c.sum() else None


# ------------------------------------------------------------------ workers
_X = None


def _worker_init(path):
    global _X
    _X = np.load(path, mmap_mode="r")
    DC.install()


def _worker(task):
    name, level, items = task
    return np.concatenate([corrupt_track_level(np.asarray(_X[g]), name, level, [D0.D0_SEED, int(code)])
                           for code, g in items])


def pc(x, nd=2):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    t0 = time.time()
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    blobs = FZ.check_frozen()
    run = Path(a.run)
    dpath = run / "poc3" / "stage3" / "stage3_design.json"
    D0.verify_sha(dpath, R3.DESIGN_SHA256)
    js, des, models_all = R3.load_design(dpath)
    models = {"S2-sum": models_all["S2-sum"]}
    poc_cfg = yaml.safe_load(Path(a.config).read_text())
    cache = Path(poc_cfg["data"]["cache_dir"])
    wheels = {f: D0.verify_sha(Path("data/d0_wheels") / f, h) for f, h in D0.WHEELS_SHA256.items()}

    # design rows only (hard rule 3 style: the loader refuses the pool table)
    S2.OPENED.clear()
    d = S2.load_inputs(run)
    assert not any(S2.POOL_FILE in p for p in S2.OPENED), "the pool table was opened"
    tr_path = run / "embeddings" / "train_release.npz"
    D0.verify_sha(tr_path, d["inputs_sha256"]["train_release.npz"])
    sp_path = run / "splits" / "seed0.npz"
    D0.verify_sha(sp_path, d["inputs_sha256"]["seed0.npz"])
    E = np.load(tr_path)
    track_all = E["track"].astype(int)
    design_mask = np.asarray(np.load(sp_path)["track_split"]) == 0
    grow = np.flatnonzero(design_mask[track_all])                       # design image rows; nothing else is read
    checks = {"image_rows_design_only": bool(design_mask[track_all[grow]].all() and len(grow) == len(d["emb"])),
              "image_rows_line_up": bool(np.array_equal(E["emb"][grow], d["emb"]) and (E["y"][grow] == d["y"]).all())}
    del E
    rows = SC.R1.track_rows(d["track"], d["frame"])
    codes = np.array([d["track"][r[0]] for r in rows])
    y = np.array([d["y"][r[0]] for r in rows])
    assert (codes == d["design_codes"]).all() and len(rows) == S2.N_DESIGN == 261

    # frozen backbone and cache
    ck = run / "best.pt"
    D0.verify_sha(ck, poc_cfg["embed"]["expected_checkpoint_sha256"].lower())
    img_path = cache / "train_112.npy"
    D0.verify_sha(img_path, D0.TRAIN112_SHA256)
    X = np.load(img_path, mmap_mode="r")
    device = TR.pick_device("auto")
    model = TR.build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
    model.eval()
    gpu = [0.0]
    emb_clean = D0.embed_timed(model, np.asarray(X[grow]), device, gpu)
    clean_diff = float(np.abs(emb_clean.astype(np.float64) - d["emb"]).max())
    checks["clean_reembedding_matches_stored"] = clean_diff < D0.CLEAN_TOL
    K = len(d["head"]["priors"])

    def evaluate(emb, passed):
        F = SC.frame_inputs(d["head"], emb, passed, d["track"], d["frame"])
        seqs = SC.sequences(F, models)
        m_all, top_all = frame_margins(d["head"], emb, np.ones(len(emb), bool))      # before the gate
        yhat = consensus(rows, top_all, m_all, K)
        r = {"frame_error": float((F["top"] != d["y"]).mean()), "gate_pass": float(passed.mean()),
             "consensus_error": float((yhat != y).mean()), "scores": {}}
        for s in SCORES:
            r["scores"][s] = {}
            for al in ALPHAS:
                j = int(des[s]["reference"][al])
                c, e = at_threshold(seqs[s], y, des[s]["grid"], j)
                r["scores"][s][al] = {"j": j, "commit_rate": c, "committed_error": e}
        return r, seqs

    res = {}
    res[("clean", 0)], seqs_clean = evaluate(d["emb"], d["passed"])
    # clean checks: design commit fractions reproduce; reference error <= alpha by construction
    ok_fd, ok_ref = True, True
    for s in SCORES:
        tab = SC.track_table(seqs_clean[s], y, des[s]["grid"])
        ok_fd &= bool(np.allclose(tab["committed"].mean(0), des[s]["f_d"], atol=1e-12))
        for al in ALPHAS:
            e = res[("clean", 0)]["scores"][s][al]["committed_error"]
            ok_ref &= e is None or e <= al + 1e-12
    checks["design_f_d_reproduces"], checks["clean_reference_error_le_alpha"] = ok_fd, ok_ref
    fog_ref = DC.reference_check([np.asarray(X[grow[i]]) for i in (0, 3000, 6000)])
    checks["fog_patch_equals_package_3_images"] = all(fog_ref)
    assert all(checks.values()), f"checks failed: {[k for k, v in checks.items() if not v]}"

    items = [(int(codes[i]), grow[r]) for i, r in enumerate(rows)]
    local = np.concatenate(rows)
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    l6_equals_sev1 = True
    with mp.get_context("spawn").Pool(a.workers, initializer=D0._worker_init, initargs=(str(img_path),)) as pool_s1, \
            mp.get_context("spawn").Pool(a.workers, initializer=_worker_init, initargs=(str(img_path),)) as pool:
        for name in DC.CORRUPTIONS:
            for lv in LEVELS:
                tc = time.time()
                imgs = list(pool.imap(_worker, [(name, lv, ch) for ch in chunks]))
                if lv == 6:                                              # L6 (t = 1) must equal severity 1 (d0 path)
                    ref = list(pool_s1.imap(D0._worker, [(name, 1, ch) for ch in chunks]))
                    l6_equals_sev1 &= all(np.array_equal(p, q) for p, q in zip(imgs, ref))
                parts = [D0.embed_timed(model, im, device, gpu) for im in imgs]
                emb_c = np.empty_like(d["emb"])
                emb_c[local] = np.concatenate(parts)
                passed_c = np.sqrt(mahal_sq(d["head"], emb_c).min(1)) <= d["tau"]
                res[(name, lv)], _ = evaluate(emb_c, passed_c)
                print(f"  {name} L{lv}: frame error {pc(res[(name, lv)]['frame_error'])}% ({time.time() - tc:.0f} s)",
                      flush=True)
    checks["L6_equals_severity_1"] = bool(l6_equals_sev1)

    # ---- printout: one table
    out(f"\n=== d1 corruption grid on the design tracks: {LABEL} ===")
    out(f"Freeze check passed: {', '.join(f'{k} {v[:8]}' for k, v in blobs.items())}")
    out(f"Design {len(rows)} tracks, {len(d['emb'])} frames; no pool track read | thresholds = Stage 3 reference j per "
        f"alpha (loosest grid point with clean design committed error <= alpha)")
    out("Reference j: " + "; ".join(f"{s} " + "/".join(str(int(des[s]['reference'][al])) for al in ALPHAS) for s in SCORES)
        + " (alpha 2/3/5%)")
    out(f"Checks: {sum(checks.values())}/{len(checks)} " + ("PASS" if all(checks.values()) else
        "FAIL: " + ", ".join(k for k, v in checks.items() if not v)) + f" (clean re-embedding max |diff| {clean_diff:.2e})")
    out("Levels: L1-L6 = blend t 0.1/0.2/0.3/0.5/0.75/1.0 with severity 1; L7-L10 = severity 2-5. ferr = frame top-1 "
        "error, pass = gate pass, cons = track-consensus error; per score: commit rate / committed error at the "
        "reference j. All in %. S2-sum is in-sample for its calibration on these tracks.")
    hdr = ["cond", "a%", "ferr", "pass", "cons"] + [f"{s} c/e" for s in SCORES]
    body = []
    for (name, lv), r in res.items():
        for al in ALPHAS:
            body.append([name if lv == 0 else f"{name}-L{lv}", f"{100 * al:g}", pc(r["frame_error"]),
                         pc(r["gate_pass"], 1), pc(r["consensus_error"])]
                        + [f"{pc(r['scores'][s][al]['commit_rate'], 1)} / {pc(r['scores'][s][al]['committed_error'])}"
                           for s in SCORES])
    w = [max(len(h), *(len(x[i]) for x in body)) for i, h in enumerate(hdr)]
    out("  ".join(h.rjust(w[i]) for i, h in enumerate(hdr)))
    for x in body:
        out("  ".join(c.rjust(w[i]) for i, c in enumerate(x)))
    secs = round(time.time() - t0, 1)
    out(f"GPU time (embedding forward passes incl. transfer): {gpu[0]:.1f} s on "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | total {secs} s ({a.workers} CPU workers)")

    od = run / "d1"
    od.mkdir(exist_ok=True)
    summary = {"label": LABEL, "frozen_blobs": blobs,
               "code_sha256": {f"{m}.py": sha256_file(Path(__file__).parent / f"{m}.py")
                               for m in ("d1_grid_check", "d0_corrupt_compat", "d0_envelope_check")},
               "inputs_sha256": {**d["inputs_sha256"], "stage3_design.json": R3.DESIGN_SHA256,
                                 "train_112.npy": D0.TRAIN112_SHA256,
                                 "best.pt": poc_cfg["embed"]["expected_checkpoint_sha256"].lower()},
               "wheels_sha256": wheels,
               "config": {"alphas": ALPHAS, "scores": SCORES, "corruptions": DC.CORRUPTIONS, "blend_t": BLEND_T,
                          "levels": {f"L{lv}": list(level_spec(lv)) for lv in LEVELS}, "seed": D0.D0_SEED,
                          "seed_rule": "np.random.seed([20261117, track code]) once per track, every condition",
                          "blend": "round(clip((1 - t) * clean + t * severity-1 output)) as uint8",
                          "thresholds": "Stage 3 reference j per alpha", "gate": "recomputed with frozen tau",
                          "consensus": "a0 rule", "workers": a.workers},
               "reference_j": {s: {str(al): int(des[s]["reference"][al]) for al in ALPHAS} for s in SCORES},
               "checks": checks, "clean_reembedding_max_abs_diff": clean_diff,
               "results": {(n if lv == 0 else f"{n}-L{lv}"): {**{k: v for k, v in r.items() if k != "scores"},
                                                              "scores": {s: {str(al): c for al, c in cs.items()}
                                                                         for s, cs in r["scores"].items()}}
                           for (n, lv), r in res.items()},
               "gpu_seconds": round(gpu[0], 1), "seconds": secs}
    (od / "d1_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=1))
    out(f"Output: {od / 'd1_summary.json'}. {LABEL}")
    (od / "d1_printout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
