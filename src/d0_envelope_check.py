"""EXPLORATORY (post hoc; no pre-registration, no criteria). d0: corruption envelope of the sign-level and the
frame-level certificate on the 1,046 GTSRB pool tracks.

Every GTSRB track has been seen before, so nothing here is confirmatory. No frozen file and no earlier result is
changed. Output: runs/20261003_stage1/d0/.

Conditions: clean (stored embeddings) plus 5 corruptions (imagecorruptions 1.1.2: fog, motion_blur, gaussian_noise,
contrast, brightness; fog through src/d0_corrupt_compat.py) x severity 1..5, applied to every frame of a pool track
(112 px cache images, same corruption and severity for the whole track; the global numpy RNG seeded once per track with
[20261117, track code], the same seed for every condition). The frozen POC-1 ResNet-18 embeds the corrupted frames;
the frozen head, the frozen gate tau (gate recomputed on the corrupted frames) and the Stage 3 design values (grids,
design commit fractions, S2-sum ICP/ECDF model, all fitted on clean design tracks) are used unchanged.

Per condition x score (S1, S2-sum, S3, S4) x alpha (2, 3, 5%) x re-split (the 5 POC-2 Stage 4 re-splits, regenerated
and checked as in R9 Stage 3; Cert = 784 tracks):
  sign level: frozen exact-binomial certificate (src/r9_certify.py, rule "binom") on the Cert tracks, n = 784.
  frame level: the same engine, unit = frame. A frame of a Cert track is committed at lambda when the score's running
    value at that frame exists and is >= lambda; it is wrong when the running label at that frame is not the track's
    class. n = frames of the Cert tracks; start from the design FRAME commit fraction (clean design tracks).
  pool risk at each certified threshold, on all 1,046 corrupted pool tracks:
    frame risk = wrong committed frames / committed frames (the frame-level test's own claim; PRIMARY flag);
    sign risk  = wrong committed tracks / committed tracks (track commits at its first committed frame; SECONDARY).
  Sign-level certifications are flagged when their sign risk > alpha.
Envelope boundary (per corruption, score, alpha, level, re-split): the last severity s such that severities 1..s are
all certified (0 if severity 1 is not); also the minimum over re-splits.

Usage:  py -3.12 -m src.d0_envelope_check --run runs/20261003_stage1 --config configs/poc.yaml
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
from src import embed as EM
from src import r8_stage2 as S2
from src import r8_stage3_resample as RS
from src import r8_stage4 as S4
from src import r9_certify as C
from src import r9_frozen as FZ
from src import r9_stage3 as R3
from src import r9_stage3_scores as SC
from src import train as TR
from src.head import mahal_sq
from src.r8_frozen import load_frozen
from src.util import sha256_file

LABEL = "EXPLORATORY (post hoc; no pre-registration, no criteria; GTSRB tracks all seen before)"
ALPHAS = (0.02, 0.03, 0.05)
SCORES = ("S1", "S2-sum", "S3", "S4")
D0_SEED = 20261117
BATCH = 512                                                   # Stage 1 evaluation batch size
CHUNK = 16                                                    # tracks per corruption task
CLEAN_TOL = 1e-3                                              # max |emb - stored emb| for the clean re-embedding check
TRAIN112_SHA256 = "67a541a74081e5ff51dbcddf68547b5409e354f424ff0e826f01c409ecd56a53"       # LOG.md, POC-1 Stage 0
STAGE3_SUMMARY_SHA256 = "aad2c03ff94487bbc4743dbedf6b76a3e51a4167f50a8d3b831ba7360fa052e2"  # LOG_r9_stage3_entry.md
WHEELS_SHA256 = {                                             # data/d0_wheels/, downloaded 17 Nov 2026 from PyPI
    "imagecorruptions-1.1.2-py3-none-any.whl": "d0e4bf529ba6fbb27d8aba1bde7c5db43bf2b969b2c767c11419e939e3e365b8",
    "opencv_python-5.0.0.93-cp37-abi3-win_amd64.whl": "f90ba04b8f73bc5c3814037699739f0156f597338a98f05956c684e7c3ca10d2",
    "scikit_image-0.26.0-cp312-cp312-win_amd64.whl": "abed017474593cd3056ae0fe948d07d0747b27a085e92df5474f4955dd65aec0",
    "imageio-2.38.2-py3-none-any.whl": "d71174e7af8369a4f73525c5b1c0d2621eb2827ed3cfd7327560be02ac1b4084",
    "lazy_loader-0.6-py3-none-any.whl": "77253be3391b06124a0e16105bd663b6c54470af1a9ca8e1cf026f38d58ed056",
    "tifffile-2026.11.12-py3-none-any.whl": "f7d46b81f8e55e607d0ad3d18c82acac6f791828fc0329d9263a664f12b6e32d",
}


def verify_sha(path, expected):
    """Hard rule 2: refuse a file whose bytes differ from the recorded hash."""
    got = sha256_file(Path(path))
    if got != expected:
        raise AssertionError(f"{path} differs from its recorded SHA-256 ({got[:12]} != {expected[:12]})")
    return got


# ------------------------------------------------------------------ tables (pure numpy; tested against loops)
def frame_table(seqs, y_track, grid):
    """Per track and lambda: committed frames A, wrong committed frames W; NF = frames per track (hard rule 4).
    A frame is committed at lambda when its running value exists and is >= lambda (lambda = -inf: any valued frame)."""
    n, G = len(seqs), len(grid)
    A, W, NF = np.zeros((n, G), int), np.zeros((n, G), int), np.zeros(n, int)
    for i, (conf, lab) in enumerate(seqs):
        com = np.isfinite(conf)[:, None] & (conf[:, None] >= grid[None, :])
        A[i], W[i], NF[i] = com.sum(0), (com & (lab != y_track[i])[:, None]).sum(0), len(conf)
    return A, W, NF


def certify_both(tab, A, W, NF, cert, alpha, f_sign, f_frame, n_design, n_design_frames):
    """(sign-level j, frame-level j) on the Cert tracks; -1 = nothing certified."""
    com = tab["committed"][cert]
    j_s = C.certify((com & tab["wrong"][cert]).sum(0), com.sum(0), int(cert.sum()), alpha, f_sign,
                    test="binom", n_design=n_design)
    j_f = C.certify(W[cert].sum(0), A[cert].sum(0), int(NF[cert].sum()), alpha, f_frame,
                    test="binom", n_design=n_design_frames)
    return int(j_s), int(j_f)


def pool_at(tab, A, W, NF, j):
    """Pool quantities at grid index j (all pool tracks); None where nothing is certified or nothing commits."""
    if j < 0:
        return None
    nc, k = int(tab["committed"][:, j].sum()), int((tab["committed"][:, j] & tab["wrong"][:, j]).sum())
    af, wf = int(A[:, j].sum()), int(W[:, j].sum())
    return {"j": int(j), "commit_rate": nc / len(NF), "n_c": nc, "k": k, "sign_risk": (k / nc) if nc else None,
            "frame_commit_rate": af / int(NF.sum()), "frame_risk": (wf / af) if af else None}


def boundary(certified):
    """certified[s - 1] for severities 1..5 in mild-to-severe order -> last s with 1..s all certified (0 if none)."""
    b = 0
    for ok in certified:
        if not ok:
            break
        b += 1
    return b


# ------------------------------------------------------------------ corruption workers (CPU)
_X = None


def _worker_init(path):
    global _X
    _X = np.load(path, mmap_mode="r")
    DC.install()


def _worker(task):
    name, sev, items = task
    return np.concatenate([DC.corrupt_track(np.asarray(_X[g]), name, sev, [D0_SEED, int(code)]) for code, g in items])


# ------------------------------------------------------------------ embedding (GPU)
def embed_timed(model, images, device, clock):
    """Frozen backbone forward (embed.extract, Stage 2 path); l2-normalised embeddings. Adds GPU wall time to clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    raw, _ = EM.extract(model, images, device, BATCH)
    if device.type == "cuda":
        torch.cuda.synchronize()
    clock[0] += time.perf_counter() - t
    return EM.l2_normalise(raw)


# ------------------------------------------------------------------ one condition
def evaluate(emb, passed, d, y, models, des, f_frame, n_design, n_design_frames, tests):
    F = SC.frame_inputs(d["head"], emb, passed, d["track"], d["frame"])
    seqs = SC.sequences(F, models)
    out = {"frame_error": float((F["top"] != d["y"]).mean()), "gate_pass": float(passed.mean()), "scores": {}}
    for s in SCORES:
        grid = des[s]["grid"]
        tab = SC.track_table(seqs[s], y, grid)
        A, W, NF = frame_table(seqs[s], y, grid)
        cell = {}
        for al in ALPHAS:
            per = []
            for seed in sorted(tests):
                j_s, j_f = certify_both(tab, A, W, NF, ~tests[seed], al, des[s]["f_d"], f_frame[s], n_design,
                                        n_design_frames)
                per.append({"seed": seed, "sign": pool_at(tab, A, W, NF, j_s), "frame": pool_at(tab, A, W, NF, j_f)})
            cell[al] = per
        out["scores"][s] = cell
    return out


# ------------------------------------------------------------------ printing
def pc(x, nd=2):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def med_max(v, nd=2):
    v = [x for x in v if x is not None]
    return "-" if not v else f"{pc(float(np.median(v)), nd)} [{pc(max(v), nd)}]"


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

    blobs = FZ.check_frozen()                                          # frozen engine first, as every r9 runner
    cfg = load_frozen()
    run = Path(a.run)
    assert C.DELTA == 0.1
    dpath = run / "poc3" / "stage3" / "stage3_design.json"
    verify_sha(dpath, R3.DESIGN_SHA256)
    js, des, models_all = R3.load_design(dpath)
    n_cert, n_design = int(js["n_cert"]), int(js["n_design"])
    models = {"S2-sum": models_all["S2-sum"]}
    s4dir = run / "poc2" / "stage4"
    verify_sha(s4dir / "stage4_summary.json", R3.STAGE4_SUMMARY_SHA256)
    s4 = json.loads((s4dir / "stage4_summary.json").read_text())
    assert sha256_file(Path(S4.__file__)) == s4["code_sha256"]["r8_stage4.py"] == R3.STAGE4_CODE_SHA256
    verify_sha(s4dir / "baselines_design.json", S4.BASELINES_SHA256)
    poc_cfg = yaml.safe_load(Path(a.config).read_text())
    cache = Path(poc_cfg["data"]["cache_dir"])
    verify_sha(cache / "train_meta.csv", R3.TRAIN_META_SHA256)
    s3sum_path = run / "poc3" / "stage3" / "stage3_summary.json"
    verify_sha(s3sum_path, STAGE3_SUMMARY_SHA256)
    s3sum = json.loads(s3sum_path.read_text())
    wheels = {f: verify_sha(Path("data/d0_wheels") / f, h) for f, h in WHEELS_SHA256.items()}

    # pool, POC-2 reproduction checks, regenerated Stage 4 splits (as R9 Stage 3)
    d = RS.load_pool(run, cfg)
    _, checks = RS.pool_tables(d, cfg)
    bdes = json.loads((s4dir / "baselines_design.json").read_text())
    bdes["n_all_tracks"] = len(np.load(run / "splits" / "seed0.npz")["track_names"])
    roi = S4.roi_sizes(run, poc_cfg, d)
    T4, y, _, c2 = S4.method_tables(d, cfg, bdes, roi)
    checks.update(c2)
    assert all(checks.values()), f"POC-2 pool reproduction checks failed: {checks}"
    tests, ok, n_cells = R3.reproduce_stage4(s4, T4, y, int(cfg["splits"]["test_tracks"]))
    checks["stage4_coverages_reproduce"] = ok
    assert ok, "regenerated splits do not reproduce Stage 4"
    assert all(int((~t).sum()) == n_cert for t in tests.values())
    rows = SC.R1.track_rows(d["track"], d["frame"])
    codes = np.array([d["track"][r[0]] for r in rows])
    assert (codes == d["pool_codes"]).all(), "pool track order differs"

    # global image rows of the pool frames (train_release.npz rows = train_112.npy rows, src/embed.py)
    tr_path = run / "embeddings" / "train_release.npz"
    verify_sha(tr_path, d["inputs_sha256"]["train_release.npz"])
    E = np.load(tr_path)
    pool_mask = np.asarray(np.load(run / "splits" / "seed0.npz")["track_split"]) != 0
    grow = np.flatnonzero(pool_mask[E["track"].astype(int)])
    checks["image_rows_line_up"] = bool((E["y"][grow] == d["y"]).all() and np.array_equal(E["emb"][grow], d["emb"]))

    # design frame commit fractions (clean design tracks only; the design loader refuses the pool table)
    S2.OPENED.clear()
    dd = S2.load_inputs(run)
    Fd = SC.frame_inputs(dd["head"], dd["emb"], dd["passed"], dd["track"], dd["frame"])
    yd = np.array([dd["y"][r[0]] for r in Fd["rows"]])
    seqs_d = SC.sequences(Fd, models)
    f_frame, n_design_frames = {}, None
    for s in SCORES:
        tab_d = SC.track_table(seqs_d[s], yd, des[s]["grid"])
        checks[f"design_f_d_reproduces_{s}"] = bool(np.allclose(tab_d["committed"].mean(0), des[s]["f_d"], atol=1e-12))
        A_d, _, NF_d = frame_table(seqs_d[s], yd, des[s]["grid"])
        f_frame[s] = A_d.sum(0) / NF_d.sum()
        n_design_frames = int(NF_d.sum())

    # frozen backbone (POC-1 checkpoint, hash-checked) and image cache
    ck = run / "best.pt"
    es = json.loads((run / "embeddings" / "embeddings_summary.json").read_text())
    verify_sha(ck, poc_cfg["embed"]["expected_checkpoint_sha256"].lower())
    assert es["checkpoint_sha256"] == poc_cfg["embed"]["expected_checkpoint_sha256"].lower()
    img_path = cache / "train_112.npy"
    verify_sha(img_path, TRAIN112_SHA256)
    X = np.load(img_path, mmap_mode="r")
    device = TR.pick_device("auto")
    model = TR.build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
    model.eval()
    fog_ref = DC.reference_check([np.asarray(X[grow[i]]) for i in (0, 10_000, 20_000)])
    checks["fog_patch_equals_package_3_images"] = all(fog_ref)

    gpu = [0.0]
    emb_clean = np.concatenate([embed_timed(model, X[grow[i:i + 4096]], device, gpu)      # chunks: 16 GB RAM
                                for i in range(0, len(grow), 4096)])
    clean_diff = float(np.abs(emb_clean.astype(np.float64) - d["emb"]).max())
    checks["clean_reembedding_matches_stored"] = clean_diff < CLEAN_TOL
    assert all(checks.values()), f"checks failed: {[k for k, v in checks.items() if not v]}"

    # clean condition from the stored embeddings and gate flags; sign level must equal R9 Stage 3
    res = {("clean", 0): evaluate(d["emb"], d["passed"], d, y, models, des, f_frame, n_design, n_design_frames, tests)}
    same = True
    for s in SCORES:
        for al in ALPHAS:
            for p in res[("clean", 0)]["scores"][s][al]:
                want = s3sum["results"][s][str(p["seed"])]["cert"][str(al)]["binom"]
                same &= (-1 if p["sign"] is None else p["sign"]["j"]) == want
    checks["clean_sign_level_equals_stage3"] = bool(same)
    assert same, "clean sign-level certificates differ from R9 Stage 3"

    # corrupted conditions: CPU workers corrupt chunks of tracks, the GPU embeds them as they arrive
    items = [(int(codes[i]), grow[r]) for i, r in enumerate(rows)]
    local = np.concatenate(rows)
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    cpu_wall = 0.0
    with mp.get_context("spawn").Pool(a.workers, initializer=_worker_init, initargs=(str(img_path),)) as pool:
        for name in DC.CORRUPTIONS:
            for sev in DC.SEVERITIES:
                tc = time.time()
                parts = []
                for imgs in pool.imap(_worker, [(name, sev, ch) for ch in chunks]):
                    parts.append(embed_timed(model, imgs, device, gpu))
                emb_c = np.empty_like(d["emb"])
                emb_c[local] = np.concatenate(parts)
                cpu_wall += time.time() - tc
                passed_c = np.sqrt(mahal_sq(d["head"], emb_c).min(1)) <= d["tau"]
                res[(name, sev)] = evaluate(emb_c, passed_c, d, y, models, des, f_frame, n_design, n_design_frames, tests)
                r = res[(name, sev)]
                print(f"  {name} {sev}: frame error {pc(r['frame_error'])}%, gate pass {pc(r['gate_pass'])}% "
                      f"({time.time() - tc:.0f} s)", flush=True)

    # envelope boundaries
    env = {}
    for s in SCORES:
        for al in ALPHAS:
            for name in DC.CORRUPTIONS:
                for lvl in ("sign", "frame"):
                    b = [boundary([res[(name, sev)]["scores"][s][al][i][lvl] is not None for sev in DC.SEVERITIES])
                         for i in range(len(tests))]
                    env[(s, al, name, lvl)] = b

    # ---- printout: one table per score, then its envelope
    seeds = sorted(tests)
    out(f"\n=== d0 corruption envelope: {LABEL} ===")
    out(f"Freeze check passed: {', '.join(f'{k} {v[:8]}' for k, v in blobs.items())}")
    out(f"Pool {len(rows)} tracks, {len(d['emb'])} frames | 5 re-splits, Cert {n_cert} tracks | delta {C.DELTA} | "
        f"exact binomial (frozen engine) | design n {n_design} tracks, {n_design_frames} frames")
    out(f"Checks: {sum(checks.values())}/{len(checks)} PASS (clean re-embedding max |diff| {clean_diff:.2e}; "
        f"{n_cells} Stage 4 cells reproduce; clean sign level equals R9 Stage 3)")
    out("Columns: ferr = frame top-1 error, pass = gate pass; sign: cert = re-splits certified, j = grid index per "
        "seed 0-4, commit/err = pool commit rate and pool committed sign error at j (median [max]), S>a = re-splits "
        "with pool sign risk > alpha; frame: frisk = pool frame risk at the frame-level j, srisk = pool sign risk at "
        "that j (median [max]), F>a = re-splits with frame risk > alpha (primary flag), S>a = with sign risk > alpha "
        "(secondary). All in %.")
    hdr = ["cond", "a%", "ferr", "pass", "| cert", "j sign", "commit", "err", "S>a", "| cert", "j frame", "frisk",
           "srisk", "F>a", "S>a"]
    for s in SCORES:
        out(f"\n--- {s}: {len(des[s]['grid'])} grid points ---")
        body = []
        for (name, sev), r in res.items():
            for al in ALPHAS:
                ps = r["scores"][s][al]
                sg, fr = [p["sign"] for p in ps], [p["frame"] for p in ps]
                sgc, frc = [x for x in sg if x], [x for x in fr if x]
                js_ = lambda xs: " ".join("-" if x is None else str(x["j"]) for x in xs)
                body.append([name if sev == 0 else f"{name}-{sev}", f"{100 * al:g}", pc(r["frame_error"]),
                             pc(r["gate_pass"], 1), f"| {len(sgc)}/5", js_(sg),
                             med_max([x["commit_rate"] for x in sgc], 1), med_max([x["sign_risk"] for x in sgc]),
                             str(sum(x["sign_risk"] is not None and x["sign_risk"] > al for x in sgc)),
                             f"| {len(frc)}/5", js_(fr), med_max([x["frame_risk"] for x in frc]),
                             med_max([x["sign_risk"] for x in frc]),
                             str(sum(x["frame_risk"] is not None and x["frame_risk"] > al for x in frc)),
                             str(sum(x["sign_risk"] is not None and x["sign_risk"] > al for x in frc))])
        w = [max(len(h), *(len(x[i]) for x in body)) for i, h in enumerate(hdr)]
        out("  ".join(h.rjust(w[i]) for i, h in enumerate(hdr)))
        for x in body:
            out("  ".join(c.rjust(w[i]) for i, c in enumerate(x)))
        out(f"Envelope ({s}): last severity certified in order 1..5, per seed 0-4 (min); sign level | frame level")
        for name in DC.CORRUPTIONS:
            cells = []
            for al in ALPHAS:
                bs, bf = env[(s, al, name, "sign")], env[(s, al, name, "frame")]
                cells.append(f"{100 * al:g}%: {' '.join(map(str, bs))} ({min(bs)}) | {' '.join(map(str, bf))} ({min(bf)})")
            out(f"  {name:>14}  " + "   ".join(cells))

    n_frame_cert = sum(1 for r in res.values() for s in SCORES for al in ALPHAS for p in r["scores"][s][al] if p["frame"])
    n_flag_f = sum(1 for r in res.values() for s in SCORES for al in ALPHAS for p in r["scores"][s][al]
                   if p["frame"] and p["frame"]["frame_risk"] is not None and p["frame"]["frame_risk"] > al)
    n_flag_fs = sum(1 for (cn, _), r in res.items() for s in SCORES for al in ALPHAS for p in r["scores"][s][al]
                    if p["frame"] and p["frame"]["sign_risk"] is not None and p["frame"]["sign_risk"] > al)
    n_sign_cert = sum(1 for r in res.values() for s in SCORES for al in ALPHAS for p in r["scores"][s][al] if p["sign"])
    n_flag_s = sum(1 for r in res.values() for s in SCORES for al in ALPHAS for p in r["scores"][s][al]
                   if p["sign"] and p["sign"]["sign_risk"] is not None and p["sign"]["sign_risk"] > al)
    secs = round(time.time() - t0, 1)
    out(f"\nTotals over {len(res)} conditions x {len(SCORES)} scores x {len(ALPHAS)} alphas x 5 re-splits: "
        f"sign level certified {n_sign_cert}, pool sign risk > alpha {n_flag_s}; frame level certified {n_frame_cert}, "
        f"pool frame risk > alpha {n_flag_f} (primary), pool sign risk > alpha {n_flag_fs} (secondary).")
    out(f"GPU time (embedding forward passes incl. transfer, {len(res) - 1} corrupted conditions + clean check): "
        f"{gpu[0]:.1f} s on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | corruption + "
        f"embedding wall time {cpu_wall:.1f} s ({a.workers} CPU workers) | total {secs} s")

    od = run / "d0"
    od.mkdir(exist_ok=True)
    import imagecorruptions, skimage, cv2
    summary = {"label": LABEL, "frozen_blobs": blobs,
               "code_sha256": {f"{m}.py": sha256_file(Path(__file__).parent / f"{m}.py")
                               for m in ("d0_envelope_check", "d0_corrupt_compat")},
               "inputs_sha256": {**d["inputs_sha256"], "stage3_design.json": R3.DESIGN_SHA256,
                                 "stage4_summary.json": R3.STAGE4_SUMMARY_SHA256, "train_meta.csv": R3.TRAIN_META_SHA256,
                                 "stage3_summary.json": STAGE3_SUMMARY_SHA256, "train_112.npy": TRAIN112_SHA256,
                                 "best.pt": poc_cfg["embed"]["expected_checkpoint_sha256"].lower()},
               "packages": {"imagecorruptions": "1.1.2", "scikit-image": skimage.__version__, "opencv-python": cv2.__version__,
                            "numpy": np.__version__, "torch": torch.__version__, "wheels_sha256": wheels,
                            "wheel_source": "https://files.pythonhosted.org (PyPI), pinned with --require-hashes",
                            "patch": DC.PATCH_NOTE, "imagecorruptions_path": imagecorruptions.__file__},
               "config": {"alphas": ALPHAS, "scores": SCORES, "corruptions": DC.CORRUPTIONS, "severities": DC.SEVERITIES,
                          "seed": D0_SEED, "seed_rule": "np.random.seed([20261117, track code]) once per track, every condition",
                          "delta": C.DELTA, "rule": "binom", "n_cert": n_cert, "n_design": n_design,
                          "n_design_frames": n_design_frames, "cert_sets": "POC-2 Stage 4 re-splits 0-4, Cert = 784",
                          "pool_risk": "all 1,046 corrupted pool tracks", "flag_primary": "frame risk > alpha",
                          "flag_secondary": "sign risk > alpha", "gate": "recomputed with frozen tau on corrupted frames",
                          "batch": BATCH, "workers": a.workers},
               "f_frame_design": f_frame, "checks": checks, "clean_reembedding_max_abs_diff": clean_diff,
               "results": {f"{n}-{sv}" if sv else n: {"frame_error": r["frame_error"], "gate_pass": r["gate_pass"],
                                                      "scores": {s: {str(al): v for al, v in c.items()}
                                                                 for s, c in r["scores"].items()}}
                           for (n, sv), r in res.items()},
               "envelope": {f"{s}|{al}|{n}|{l}": b for (s, al, n, l), b in env.items()},
               "totals": {"sign_certified": n_sign_cert, "sign_flag": n_flag_s, "frame_certified": n_frame_cert,
                          "frame_flag_primary": n_flag_f, "frame_flag_secondary": n_flag_fs},
               "gpu_seconds": round(gpu[0], 1), "corrupt_embed_wall_seconds": round(cpu_wall, 1), "seconds": secs}
    (od / "d0_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=1))
    out(f"Output: {od / 'd0_summary.json'}. {LABEL}")
    (od / "d0_printout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
