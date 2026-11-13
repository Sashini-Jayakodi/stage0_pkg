"""POC-2 (R8) Stage 2: design phase and D-gate. DESIGN SET ONLY (the 261 seed-0 Fit tracks).
Plan: docs/poc2-implementation-plan-r8.md, sections 2, 5, 6, 8 (D-gate) and 10 (Stage 2).
Definitions: LOG_r8_stage2_entry.md Part A, committed (f19cbfb) before this code existed. The letters (a)..(m) in the
comments below point to that entry, so every rule here can be checked against the version on record.

What it does
  1. Loads the hash-checked Stage 1 design table and, for the per-frame curves, the Stage 1 inputs cut to design rows.
  2. Recomputes the design tracks' running scores and checks they reproduce the stored Stage 1 table exactly.
  3. Prints D1, D2 (plug-in table), D3, the error-vs-frame curve, then the design choices that follow from fixed rules:
     primary alpha, variant, ~30-point working grid, threshold order, s, a Certificate B preview (risk R6), P2 level.

What it does NOT do: open track_table_pool.npz, compute anything on a pool track, or write configs/frozen_r8.yaml.
F1 comes only after the author reviews the D-gate.

Usage:  py -3.12 -m src.r8_stage2 --run runs/20261003_stage1
Output: runs/20261003_stage1/poc2/stage2/stage2_summary.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from src import r8_stage1 as R1        # the exact Stage 1 rule code, so the recomputation is the same rule
from src.certify import hb_pvalue      # the POC-1 Hoeffding-Bentkus p-value (plan section 6)
from src.util import sha256_file       # torch-free copy of the hash function

PART_A_COMMIT = "f19cbfb"              # git commit of the definitions (LOG_r8_stage2_entry.md Part A)
AMENDMENT_1_COMMIT = "ccc400a"         # Amendment 1 to (d): no fallback for s
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)
DELTA = 0.1                            # fixed in every phase (plan section 2)
GAMMA = 0.10                           # (d)
N_CERT = 784                           # plug-in sample size (plan section 8)
SAFETY = 10                            # p <= delta / 10 (plan section 8)
C_MIN_D2 = 0.70
N_WORK = 30                            # (h)
D3_MIN_WRONG = 25                      # (k)
N_DESIGN = 261                         # design-set size; tests set it to the fake run's size
VARIANTS = R1.VARIANTS
DESIGN_FILE, POOL_FILE, EVID_FILE = "track_table_design.npz", "track_table_pool.npz", "frame_evidence.npz"
TOL = 1e-12
OPENED = []                            # every .npz this module opens (check S5)


# ------------------------------------------------------------------ guarded file access (m)
def open_npz(path):
    """The only way this module opens an .npz. Refuses the pool table outright (plan section 2) and records every path."""
    path = Path(path)
    assert path.name != POOL_FILE, "Stage 2 must never open the pool table (plan section 2)"
    OPENED.append(str(path))
    return np.load(path)


def load_inputs(run):
    """Hash-check everything against stage1_summary.json (hard rule 2), then cut every full-release array to design
    rows at once. Only design rows leave this function."""
    run = Path(run)
    s1dir = run / "poc2" / "stage1"
    s1 = json.loads((s1dir / "stage1_summary.json").read_text())
    assert s1.get("all_pass") is True, "Stage 1 did not pass; Stage 2 refuses to run"
    for f in (DESIGN_FILE, EVID_FILE):
        assert sha256_file(s1dir / f) == s1["files_sha256"][f], f"{f} changed since Stage 1"
    ins = s1["inputs_sha256"]
    es = json.loads((run / "embeddings" / "embeddings_summary.json").read_text())
    assert es["checkpoint_sha256"].lower() == ins["checkpoint"].lower(), "embeddings are not from the checkpoint Stage 1 used"
    for rel, key in (("embeddings/train_release.npz", "train_release.npz"), ("head_emb/head.npz", "head.npz"),
                     ("splits/seed0.npz", "seed0.npz")):
        assert sha256_file(run / rel) == ins[key], f"{key} changed since Stage 1"
    hs = json.loads((run / "head_emb" / "head_summary.json").read_text())
    sc_path = run / "head_emb" / "scores_train_release.npz"
    assert sha256_file(sc_path) == hs["files_sha256"]["scores_train_release.npz"], "scores_train_release.npz changed since Stage 3"

    Sp = open_npz(run / "splits" / "seed0.npz")
    names = np.asarray(Sp["track_names"]).astype(str)
    design_mask = np.asarray(Sp["track_split"]) == 0                  # 0 = Fit in POC-1 = the R8 design set
    design_codes = np.flatnonzero(design_mask)
    assert len(design_codes) == N_DESIGN, f"design set has {len(design_codes)} tracks, expected {N_DESIGN}"

    E = open_npz(run / "embeddings" / "train_release.npz")
    track_all = E["track"].astype(int)
    assert (np.asarray(E["track_names"]).astype(str) == names).all(), "track order in seed0.npz differs from the embeddings"
    rows = np.flatnonzero(design_mask[track_all])                     # design frame rows; nothing else is kept
    d = {"emb": E["emb"][rows], "y": E["y"][rows].astype(int), "track": track_all[rows], "frame": E["frame"][rows].astype(int)}
    del E, track_all
    d["top1"] = open_npz(sc_path)["top3"][rows, 0].astype(int)       # POC-1 head top-1 per frame (no gate)
    ev = open_npz(s1dir / EVID_FILE)
    d["passed"], d["s_stored"] = ev["passed"][rows], ev["s"][rows]
    H = open_npz(run / "head_emb" / "head.npz")
    d["head"] = {k: H[k] for k in ("mu", "W", "priors")}
    T = open_npz(s1dir / DESIGN_FILE)
    d["table"] = {k: T[k] for k in T.files}

    # hard rule 3 assertion: every row used belongs to a design track, and the table is the design table, in this order
    assert design_mask[d["track"]].all() and set(np.unique(d["track"])) == set(design_codes), "a non-design track reached Stage 2"
    assert (np.asarray(d["table"]["track_names"]).astype(str) == names[design_codes]).all(), "design table order differs from seed0"
    d.update(design_codes=design_codes, names=names[design_codes], tau=float(s1["results"]["tau"]),
             inputs_sha256={**ins, "scores_train_release.npz": hs["files_sha256"]["scores_train_release.npz"],
                            DESIGN_FILE: s1["files_sha256"][DESIGN_FILE], EVID_FILE: s1["files_sha256"][EVID_FILE]})
    return d


# ------------------------------------------------------------------ recomputation of the design tracks (m)
def recompute(d):
    """Running scores of the design tracks with the Stage 1 code. Gives the per-frame top class (curve), the last-frame
    margin (D3) and a second copy of the commit table, which must equal the stored one."""
    ll, s = R1.frame_evidence(d["head"], d["emb"])
    rows = R1.track_rows(d["track"], d["frame"])                      # ordered by track code = design table order
    log_prior = np.log(d["head"]["priors"])
    T, L = len(rows), max(len(r) for r in rows)
    out = {"s": s, "passed_re": s <= d["tau"], "rows": rows, "len": np.array([len(r) for r in rows])}
    npass = np.zeros((T, L), int)
    for v in VARIANTS:
        grid = d["table"][f"{v}_grid"]
        top_t = np.full((T, L), -1, int)
        M_last = np.full(T, -np.inf)
        com = np.zeros((T, len(grid)), bool)
        pred = np.full((T, len(grid)), -1, int)
        pos = np.full((T, len(grid)), -1, int)
        for i, r in enumerate(rows):
            S, np_ = R1.running_scores(ll[r], d["passed"][r], log_prior, v)   # stored gate flags, as Stage 1 used
            M, top = R1.margins(S)
            top_t[i, :len(r)], M_last[i] = top, M[-1]
            npass[i, :len(r)] = np_
            idx = R1.first_commit(M, grid)
            c = idx < len(r)
            com[i], pred[i, c], pos[i, c] = c, top[idx[c]], idx[c] + 1
        out[v] = {"top_t": top_t, "M_last": M_last, "committed": com, "pred": pred, "pos": pos}
    out["npass"] = npass
    return out


def reproduces(d, rc):
    tb, ok = d["table"], {}
    ok["gate"] = bool((rc["passed_re"] == d["passed"]).all() and np.allclose(rc["s"], d["s_stored"], rtol=1e-4, atol=1e-3))
    y_track = np.array([d["y"][r[0]] for r in rc["rows"]])
    same = (y_track == tb["y_track"]).all()
    for v in VARIANTS:
        R = rc[v]
        same &= (R["committed"] == tb[f"{v}_committed"]).all() and (R["pred"] == tb[f"{v}_pred"]).all() \
            and (R["pos"] == tb[f"{v}_pos"]).all() and (R["top_t"][np.arange(len(rc["rows"])), rc["len"] - 1] == tb[f"{v}_top_last"]).all()
    ok["table"] = bool(same)
    return ok


# ------------------------------------------------------------------ per-lambda statistics (b), (e)
def grid_stats(committed, wrong, size, pos):
    """Per lambda: commit count k, error count w, commit fraction c (over ALL design tracks), selective error e (NaN if
    nothing commits), median commit position and size over committed tracks."""
    n = committed.shape[0]
    k, w = committed.sum(0), (wrong & committed).sum(0)
    e = np.where(k > 0, w / np.maximum(k, 1), np.nan)
    med_pos = np.array([np.median(pos[committed[:, j], j]) if k[j] else np.nan for j in range(committed.shape[1])])
    med_size = np.array([np.median(size[committed[:, j], j]) if k[j] else np.nan for j in range(committed.shape[1])])
    return {"n": n, "k": k, "w": w, "c": k / n, "e": e, "med_pos": med_pos, "med_size": med_size}


def mean_L(st, alpha):
    """(c): mean of L_i = wrong_i - alpha * committed_i + alpha = alpha * (1 - c) + w / n."""
    return alpha * (1.0 - st["c"]) + st["w"] / st["n"]


def d2_table(stats):
    """(e): plug-in p at n = 784 for every variant, alpha and lambda. Predicted certifiable: p <= delta/10 and c >= 0.70."""
    out = {}
    for v, st in stats.items():
        out[v] = {}
        for a in ALPHAS:
            p = hb_pvalue(mean_L(st, a), N_CERT, a)
            elig = st["c"] >= C_MIN_D2 - TOL
            ok = elig & (p <= DELTA / SAFETY)
            j = None
            if elig.any():
                pe = np.where(elig, p, np.inf)
                j = int(np.flatnonzero(pe <= pe.min() + TOL).max())  # best eligible lambda, ties to the larger lambda
            out[v][a] = {"p": p, "pass": bool(ok.any()), "best_j": j}
    return out


def choose_primary_alpha(d2):
    """(f): smallest alpha predicted certifiable for at least one variant; None if there is none (D2 fails)."""
    for a in ALPHAS:
        if any(d2[v][a]["pass"] for v in VARIANTS):
            return a
    return None


def choose_variant(stats, d2, alpha):
    """(g): among variants predicted certifiable at alpha, highest commit count among lambdas with e <= alpha/2.
    Strict '>' while scanning V1, V2, V3 sends ties to the earlier variant (V1 first)."""
    best_v, best_k, per = None, None, {}
    for v in VARIANTS:
        if not d2[v][alpha]["pass"]:
            continue
        st = stats[v]
        elig = (st["k"] > 0) & (st["e"] <= alpha / 2 + TOL)
        kk = int(st["k"][elig].max()) if elig.any() else -1
        per[v] = kk
        if best_k is None or kk > best_k:
            best_v, best_k = v, kk
    return best_v, per


def working_grid(st, alpha, n_work=N_WORK):
    """(h): useful = c >= 0.70 and e <= alpha. n_work points evenly spaced by index from the lowest to the highest useful
    index; all indices in that range if there are n_work or fewer. Returns Stage 1 grid indices (ascending lambda)."""
    useful = (st["c"] >= C_MIN_D2 - TOL) & (st["k"] > 0) & (st["e"] <= alpha + TOL)
    if not useful.any():
        return np.array([], int)
    lo, hi = int(np.flatnonzero(useful).min()), int(np.flatnonzero(useful).max())
    if hi - lo + 1 <= n_work:
        return np.arange(lo, hi + 1)
    return np.unique(np.round(np.linspace(lo, hi, n_work)).astype(int))


def start_and_order(mu_work):
    """(c): start = argmin of mean L over the working grid, ties to the larger lambda; then relax one point at a time."""
    m = mu_work.min()
    start = int(np.flatnonzero(mu_work <= m + TOL).max())
    return start, list(range(start, -1, -1))


def walk(p_work, order, level):
    """Fixed-sequence walk: test in order, stop at the first p > level. Returns the last passing position or None."""
    last = None
    for j in order:
        if p_work[j] > level:
            break
        last = j
    return last


def late_frac(committed, size, s):
    """(d): late_i = 1 - committed_i * [size at commit <= s], averaged over ALL design tracks, per lambda column."""
    return 1.0 - (committed & (np.nan_to_num(size, nan=np.inf) <= s)).mean(0)


def ceil5(x):
    return int(5 * math.ceil(x / 5.0 - 1e-9))


def cert_b_p(committed_w, size_w, p_err_w, s):
    """(i): p = max(p_error, p_late) per working-grid lambda."""
    late = late_frac(committed_w, size_w, s)
    p_late = hb_pvalue(late, N_CERT, GAMMA)
    return np.maximum(p_err_w, p_late), late, p_late


def choose_s(committed_w, size_w, p_err_w):
    """(d) with Amendment 1: smallest multiple of 5 at which the Certificate B plug-in passes (min p <= delta/10).
    No fallback: if no s passes, s is None ("not certifiable at n = 784"), with no substitute value.
    Scanning stops at the largest commit size: beyond it late_i no longer depends on s, so no larger s (up to the
    largest ROI size) can pass if none has by then. Returns (s or None, scan) with scan = [(s, min p), ...]."""
    scan = []
    sizes = size_w[committed_w]
    if sizes.size:
        for s in range(5, ceil5(float(np.max(sizes))) + 5, 5):
            mp = float(cert_b_p(committed_w, size_w, p_err_w, s)[0].min())
            scan.append((s, mp))
            if mp <= DELTA / SAFETY:
                return s, scan
    return None, scan


def cp_upper(k, n, conf=0.95):
    """Upper end of the two-sided Clopper-Pearson interval (Beta quantile 1 - (1 - conf)/2), (a)."""
    return 1.0 if k >= n else float(beta_dist.ppf(1 - (1 - conf) / 2, k + 1, n - k))


# ------------------------------------------------------------------ D1, D3, curve
def d1(d, rc):
    tb = d["table"]
    has = tb["n_passing"] > 0
    k, n = int(tb["V1_wrong_last"][has].sum()), int(has.sum())
    wrong_f = d["top1"] != d["y"]
    err_pass = float(wrong_f[d["passed"]].mean())
    up = cp_upper(k, n)
    return {"track_err": k / n, "k": k, "n": n, "no_passing": int((~has).sum()), "cp_upper": up,
            "frame_err_passing": err_pass, "frame_err_all": float(wrong_f.mean()), "pass": bool(up < err_pass)}


def d3(d, rc):
    tb, out = d["table"], []
    wrong_f = d["top1"] != d["y"]
    for i, r in enumerate(rc["rows"]):
        nw = int(wrong_f[r].sum())
        if nw >= D3_MIN_WRONG:
            out.append({"track": str(d["names"][i]), "label": int(tb["y_track"][i]), "frames": len(r), "wrong_frames": nw,
                        "passing": int(tb["n_passing"][i]),
                        **{f"{v}_M_last": float(rc[v]["M_last"][i]) for v in VARIANTS},
                        **{f"{v}_last_right": bool(tb[f"{v}_top_last"][i] == tb["y_track"][i]) for v in VARIANTS}})
    return out


def curve(d, rc):
    """(l): per position t, among tracks with evidence by t: top-class error per variant and single-frame error at t."""
    y_track = d["table"]["y_track"]
    out = []
    for t in range(1, rc["len"].max() + 1):
        alive = rc["len"] >= t
        ev = alive & (rc["npass"][:, t - 1] > 0)
        row = {"t": t, "tracks": int(alive.sum()), "with_evidence": float(ev.sum() / alive.sum())}
        for v in VARIANTS:
            row[v] = float((rc[v]["top_t"][ev, t - 1] != y_track[ev]).mean()) if ev.any() else float("nan")
        fr = np.array([d["top1"][rc["rows"][i][t - 1]] for i in np.flatnonzero(ev)], dtype=int)
        row["frame"] = float((fr != y_track[ev]).mean()) if ev.any() else float("nan")
        out.append(row)
    return out


# ------------------------------------------------------------------ design choices (f)-(j)
def design_choices(stats, tables, d2):
    """Everything after D2, by the fixed rules. Returns None if D2 failed (no primary alpha)."""
    a = choose_primary_alpha(d2)
    if a is None:
        return None
    v, per_variant = choose_variant(stats, d2, a)
    st, tb = stats[v], tables[v]
    work = working_grid(st, a)
    mu_w = mean_L(st, a)[work]
    p_err_w = hb_pvalue(mu_w, N_CERT, a)
    start, order = start_and_order(mu_w)
    last = walk(p_err_w, order, DELTA)
    com_w, size_w = tb["committed"][:, work], tb["size"][:, work]
    s, s_scan = choose_s(com_w, size_w, p_err_w)
    # no certifiable s -> no Certificate B threshold choice at all (Amendment 1 to (d))
    pB, late, p_late = cert_b_p(com_w, size_w, p_err_w, s) if s is not None else (None, None, None)
    b_order = sorted(range(len(work)), key=lambda j: (pB[j], -j)) if pB is not None else []
    b_pass = [j for j in range(len(work)) if pB is not None and pB[j] <= DELTA / SAFETY]
    b_report = min(b_pass, key=lambda j: (late[j], -j)) if b_pass else None
    pred_cov = float(st["c"][work[last]]) if last is not None else None
    return {"alpha": a, "variant": v, "variant_scores": per_variant, "work": work, "grid": tb["grid"][work],
            "mu": mu_w, "p_err": p_err_w, "start": start, "order": order, "walk_last": last,
            "pred_cov": pred_cov, "c_min": None if pred_cov is None else pred_cov - 0.10,
            "s": s, "s_certifiable": s is not None, "s_scan": s_scan, "pB": pB, "late": late, "p_late": p_late,
            "b_order": b_order, "b_report": b_report}


# ------------------------------------------------------------------ output helpers
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def pct(x):
    return "  n/a " if x is None or not np.isfinite(x) else f"{100 * x:6.2f}%"


def print_report(R):
    D1, D2, ch, st = R["D1"], R["D2"], R["choices"], R["stats"]
    print("\n=== R8 Stage 2: D-gate (design set only, 261 tracks) ===")
    print(f"Definitions: LOG_r8_stage2_entry.md Part A, commit {PART_A_COMMIT}; Amendment 1 to (d), commit {AMENDMENT_1_COMMIT}")
    print("\n--- D1: track error at last frame (V1, commit everything) vs single-frame error")
    print(f"track error {pct(D1['track_err'])} ({D1['k']}/{D1['n']} tracks with >= 1 passing frame; {D1['no_passing']} with none, left out)")
    print(f"upper 95% Clopper-Pearson (two-sided) {pct(D1['cp_upper'])}")
    print(f"single-frame error, passing frames {pct(D1['frame_err_passing'])}  (all frames, reference only: {pct(D1['frame_err_all'])})")
    print(f"D1: {'PASS' if D1['pass'] else 'FAIL'}")

    print("\n--- D2: plug-in at n = 784 (pass: p <= 0.01 and c >= 70%); best eligible lambda per cell")
    print(f"{'var':4} {'alpha':>6} {'lambda':>10} {'c':>8} {'e':>8} {'mean L':>8} {'p':>10}  pass")
    for v in VARIANTS:
        for a in ALPHAS:
            cell = D2[v][a]
            j = cell["best_j"]
            if j is None:
                print(f"{v:4} {100 * a:5.1f}% {'(no lambda with c >= 70%)':>40}  FAIL")
                continue
            s_ = st[v]
            print(f"{v:4} {100 * a:5.1f}% {R['tables'][v]['grid'][j]:10.4g} {pct(s_['c'][j])} {pct(s_['e'][j])} "
                  f"{pct(mean_L(s_, a)[j])} {cell['p'][j]:10.3g}  {'PASS' if cell['pass'] else 'FAIL'}")
    print(f"D2: {'PASS' if R['D2_pass'] else 'FAIL'}")

    print(f"\n--- D3: design tracks wrong on >= {D3_MIN_WRONG} frames (head top-1, no gate): {len(R['D3'])}")
    for t in R["D3"]:
        print(f"  {t['track']} label {t['label']:2d}: wrong {t['wrong_frames']}/{t['frames']}, passing {t['passing']}, "
              + ", ".join(f"{v} M_last {t[f'{v}_M_last']:.4g} ({'right' if t[f'{v}_last_right'] else 'WRONG'})" for v in VARIANTS))

    print("\n--- Error-vs-frame curve (tracks with evidence by t)")
    print(f"{'t':>3} {'evid':>7} {'V1':>8} {'V2':>8} {'V3':>8} {'frame':>8}")
    for r in R["curve"]:
        print(f"{r['t']:3d} {pct(r['with_evidence'])} {pct(r['V1'])} {pct(r['V2'])} {pct(r['V3'])} {pct(r['frame'])}")

    if ch is None:
        print("\nD2 FAILED: primary alpha, variant, working grid, s, Certificate B preview and P2 are not defined.")
        print("Plan section 11, fallback 1 applies. Nothing else is chosen here.")
        return
    v, a = ch["variant"], ch["alpha"]
    s_ = st[v]
    print(f"\n--- Primary alpha (f): {100 * a:.1f}%")
    print(f"--- Variant (g): {v}   (commit count at e <= alpha/2, per candidate: {ch['variant_scores']})")
    print(f"--- Working grid (h): {len(ch['work'])} points, lambda {ch['grid'][0]:.4g} .. {ch['grid'][-1]:.4g} "
          f"(Stage 1 indices {int(ch['work'][0])}..{int(ch['work'][-1])})")
    print(f"{'pos':>3} {'lambda':>10} {'c':>8} {'e':>8} {'med fr':>6} {'med px':>6} {'mean L':>8} {'p_err':>10}")
    for i, j in enumerate(ch["work"]):
        mark = " <- start" if i == ch["start"] else (" <- walk stops (last pass at delta)" if i == ch["walk_last"] else "")
        print(f"{i:3d} {ch['grid'][i]:10.4g} {pct(s_['c'][j])} {pct(s_['e'][j])} {s_['med_pos'][j]:6.1f} {s_['med_size'][j]:6.1f} "
              f"{pct(ch['mu'][i])} {ch['p_err'][i]:10.3g}{mark}")
    print(f"Threshold order (c): start at position {ch['start']} (lambda {ch['grid'][ch['start']]:.4g}), then relax to position 0")
    if ch["s_certifiable"]:
        print(f"\n--- s (d): {ch['s']} px, gamma = 10% (smallest certifiable s)")
    else:
        print("\n--- s: not certifiable at n = 784 (Part A (d))")
    print("    s scan, min Certificate B plug-in p over the working grid (descriptive only; pass needs <= 0.01):")
    scan = [f"{s}px {mp:.3g}" for s, mp in ch["s_scan"]]
    for i in range(0, len(scan), 6):
        print("    " + " | ".join(scan[i:i + 6]))
    print("--- Certificate B plug-in preview (i), risk R6")
    if ch["pB"] is None:
        print("    no certifiable s, so no Certificate B threshold choice (Amendment 1 to (d))")
    else:
        print(f"{'pos':>3} {'lambda':>10} {'late':>8} {'p_late':>10} {'p_err':>10} {'p=max':>10}")
        for j in ch["b_order"][:10]:
            print(f"{j:3d} {ch['grid'][j]:10.4g} {pct(ch['late'][j])} {ch['p_late'][j]:10.3g} {ch['p_err'][j]:10.3g} {ch['pB'][j]:10.3g}")
        print(f"min p = {np.min(ch['pB']):.3g} -> predicted certifiable: {'YES' if np.min(ch['pB']) <= DELTA / SAFETY else 'NO'}"
              + (f"; reported threshold (smallest late among passing): position {ch['b_report']}" if ch["b_report"] is not None else ""))
    print(f"\n--- P2 (j): predicted coverage {pct(ch['pred_cov'])} -> c_min {pct(ch['c_min'])}; P1, P3, P4 as in plan section 8")


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    OPENED.clear()
    d = load_inputs(a.run)
    rc = recompute(d)
    rep = reproduces(d, rc)

    tb = d["table"]
    tables = {v: {"grid": tb[f"{v}_grid"], "committed": tb[f"{v}_committed"], "wrong": tb[f"{v}_wrong"],
                  "size": tb[f"{v}_size"], "pos": tb[f"{v}_pos"]} for v in VARIANTS}
    stats = {v: grid_stats(t["committed"], t["wrong"], t["size"], t["pos"]) for v, t in tables.items()}
    D1 = d1(d, rc)
    D2 = d2_table(stats)
    D2_pass = any(D2[v][x]["pass"] for v in VARIANTS for x in ALPHAS)
    R = {"D1": D1, "D2": D2, "D2_pass": D2_pass, "D3": d3(d, rc), "curve": curve(d, rc), "stats": stats, "tables": tables,
         "choices": design_choices(stats, tables, D2)}

    checks = {"S1_inputs_match_stage1_hashes": True,                  # load_inputs stops otherwise
              "S2_design_set_only": bool(len(d["design_codes"]) == N_DESIGN and len(tb["track_names"]) == N_DESIGN),
              "S3_gate_flags_reproduce": rep["gate"],
              "S4_commit_table_reproduces": rep["table"],
              "S5_pool_file_never_opened": not any(POOL_FILE in p for p in OPENED),
              "D1_pass": D1["pass"], "D2_pass": D2_pass}
    print_report(R)

    out = Path(a.run) / "poc2" / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    ch = R["choices"]
    summary = {"plan": "poc2-implementation-plan-r8 Stage 2 (D-gate only; no freeze)", "definitions_commit": PART_A_COMMIT, "amendment_1_commit": AMENDMENT_1_COMMIT,
               "code_sha256": sha256_file(Path(__file__)), "inputs_sha256": d["inputs_sha256"],
               "files_opened": [Path(p).name for p in OPENED],
               "D1": D1, "D2": {v: {str(x): {"pass": D2[v][x]["pass"], "best_j": D2[v][x]["best_j"],
                                             "best_p": None if D2[v][x]["best_j"] is None else D2[v][x]["p"][D2[v][x]["best_j"]]}
                                    for x in ALPHAS} for v in VARIANTS},
               "D3": R["D3"], "curve": R["curve"],
               "per_variant": {v: {"grid": tables[v]["grid"], **{k: stats[v][k] for k in ("k", "w", "c", "e", "med_pos", "med_size")}}
                               for v in VARIANTS},
               "choices": None if ch is None else {k: ch[k] for k in ch},
               "checks": checks, "all_pass": all(checks.values())}
    (out / "stage2_summary.json").write_text(json.dumps(jsonable(summary), indent=2))

    print("\nChecks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nALL PASS: {summary['all_pass']}")
    print(f"Output: {out / 'stage2_summary.json'}")
    print("Design set only. track_table_pool.npz was not opened; no number on the 1,046 pool tracks exists (plan section 2).")
    print("STOP: D-gate printed. configs/frozen_r8.yaml is NOT written; F1 waits for the author's review.")
    return summary


if __name__ == "__main__":
    main()
