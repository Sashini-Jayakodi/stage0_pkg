"""EXPLORATORY (post hoc; no certificate, no criteria). Silent-error measurement on the 1,046 GTSRB pool tracks.

Every GTSRB track has been seen before, so nothing here is confirmatory. Stored inputs only; no frozen file and no
earlier result is changed. Output: runs/20261003_stage1/a0/.

Per pool track t: consensus yhat_t = majority vote of the head's frame predictions over ALL frames of the track
(gate-failing frames included); ties go to the class with the larger summed single-frame Gaussian margin (margin taken
before the gate), then to the smaller class id. For each score (S1, S2-sum, S2-min, S3, S4) and alpha in {1,2,3,5}%,
the threshold is the one the frozen exact-binomial sign-level certificate (src/r9_certify.py, rule "binom") certifies
with the whole pool as the Cert set (n = 1,046; grids and design commit fractions from the hash-checked Stage 3 design
file). The certificate and the measurement use the same tracks: in-sample by construction.
  c_t = committed; L_t = committed label; e_t = c_t and L_t != y_t; v_t = c_t and L_t != yhat_t (visible);
  s_t = c_t and L_t == yhat_t != y_t (silent). On consensus-correct tracks e_t == v_t, so e_t <= v_t + s_t always.
  q = E[s] (per track, not per committed track).
  gap = alpha * CP_lower(c_N, N, delta/3) - CP_upper(v_N, N, delta/3), N = 784, c_N = round(E[c] N), v_N = round(E[v] N),
        one-sided Clopper-Pearson bounds at level delta/3.
  n_A  = smallest n <= 5,000 with CP_upper(ceil(q n), n, delta/3) < gap ("impossible" if gap <= q); n_A0 = same with
         zero silent errors.
  n_full = smallest n (total labelled tracks) at which the exact binomial test at level delta rejects with
         n_c = round(E[c] n) and k = round(E[e]/E[c] n_c); searched up to 1,000,000. round = half up.

Usage:  py -3.12 -m src.a0_silent_error_check --run runs/20261003_stage1
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

from src import r8_stage2 as S2
from src import r8_stage3_resample as RS
from src import r9_certify as C
from src import r9_frozen as FZ
from src import r9_stage3 as R3
from src import r9_stage3_scores as SC
from src.r8_frozen import load_frozen
from src.r8_stage3_b1design import frame_margins
from src.util import sha256_file

LABEL = "EXPLORATORY (post hoc; no certificate, no criteria; GTSRB tracks all seen before)"
ALPHAS = (0.01, 0.02, 0.03, 0.05)
N_AUDIT = 784                      # audit size for the gap (Cert size of the POC-2 Stage 4 re-splits)
N_A_MAX = 5_000
N_FULL_MAX = 1_000_000
DESIGN_SHA256 = R3.DESIGN_SHA256   # stage3_design.json, recorded in LOG_r9_stage3_entry.md Part A


def verify_sha(path, expected):
    """Hard rule 2: refuse a file whose bytes differ from the recorded hash."""
    got = sha256_file(Path(path))
    if got != expected:
        raise AssertionError(f"{path} differs from its recorded SHA-256 ({got[:12]} != {expected[:12]})")
    return got


def round_half_up(x):
    return int(math.floor(x + 0.5))


# ------------------------------------------------------------------ consensus and decomposition
def consensus(rows, top, margin, K):
    """Majority vote of frame predictions per track; tie -> larger summed margin; then smaller class id."""
    out = np.empty(len(rows), int)
    for i, r in enumerate(rows):
        votes = np.bincount(top[r], minlength=K)
        tied = np.flatnonzero(votes == votes.max())
        if len(tied) > 1:
            msum = np.array([margin[r][top[r] == k].sum() for k in tied])
            tied = tied[msum == msum.max()]
        out[i] = tied[0]
    return out


def committed_labels(seqs, grid_value):
    """Per track at one threshold: committed flag and committed label (-1 if not committed); plan 5.1 rule."""
    c = np.zeros(len(seqs), bool)
    lab = np.full(len(seqs), -1)
    for i, (conf, lb) in enumerate(seqs):
        idx = int(SC.first_commit(conf, np.array([grid_value]))[0])
        if idx < len(conf):
            c[i], lab[i] = True, lb[idx]
    return c, lab


def decompose(c, lab, yhat, y):
    e = c & (lab != y)
    v = c & (lab != yhat)
    s = c & (lab == yhat) & (yhat != y)
    return e, v, s


# ------------------------------------------------------------------ bounds and sample sizes
def cp_lower(x, n, a):
    """One-sided lower Clopper-Pearson bound at level 1 - a."""
    return 0.0 if x <= 0 else float(beta.ppf(a, x, n - x + 1))


def cp_upper(x, n, a):
    """One-sided upper Clopper-Pearson bound at level 1 - a."""
    return 1.0 if x >= n else float(beta.ppf(1 - a, x + 1, n - x))


def cp_two_sided(x, n, conf=0.95):
    a = (1 - conf) / 2
    return cp_lower(x, n, a), cp_upper(x, n, a)


def gap_value(alpha, Ec, Ev, N=N_AUDIT, delta=C.DELTA):
    cN, vN = round_half_up(Ec * N), round_half_up(Ev * N)
    return alpha * cp_lower(cN, N, delta / 3) - cp_upper(vN, N, delta / 3), cN, vN


def n_audit(q, gap, delta=C.DELTA, n_max=N_A_MAX):
    """Smallest n with CP_upper(ceil(q n), n, delta/3) < gap. Returns (n or None, note)."""
    if gap <= q:
        return None, "impossible (gap <= q)"
    ns = np.arange(1, n_max + 1)
    x = np.ceil(q * ns - 1e-12).astype(int)
    ub = np.where(x >= ns, 1.0, beta.ppf(1 - delta / 3, x + 1, np.maximum(ns - x, 1)))
    hit = np.flatnonzero(ub < gap)
    return (int(ns[hit[0]]), "") if len(hit) else (None, f"none <= {n_max}")


def n_full(Ec, err, alpha, delta=C.DELTA, n_max=N_FULL_MAX):
    """Smallest total n at which the frozen exact binomial test rejects with the rounded counts; (n or None, note)."""
    for lo in range(1, n_max + 1, 100_000):
        ns = np.arange(lo, min(lo + 100_000, n_max + 1))
        nc = np.floor(Ec * ns + 0.5)
        k = np.floor(err * nc + 0.5)
        hit = np.flatnonzero((nc > 0) & (C.binom_pvalue(k, nc, alpha) <= delta))
        if len(hit):
            return int(ns[hit[0]]), ""
    return None, f"none <= {n_max}"


# ------------------------------------------------------------------ main
def fmt_n(n, note):
    return note if n is None else f"{n}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    blobs = FZ.check_frozen()                                          # frozen engine first, as every r9 runner
    cfg = load_frozen()
    run = Path(a.run)
    dpath = run / "poc3" / "stage3" / "stage3_design.json"
    verify_sha(dpath, DESIGN_SHA256)
    js, des, models = R3.load_design(dpath)
    n_design = int(js["n_design"])
    d = RS.load_pool(run, cfg)                                         # hash-checks Stage 1, head, splits, b1_design
    delta = C.DELTA
    assert delta == 0.1

    F = SC.frame_inputs(d["head"], d["emb"], d["passed"], d["track"], d["frame"])
    rows = F["rows"]
    T = len(rows)
    assert T == int(cfg["splits"]["pool_tracks"]) == 1046
    y = np.array([d["y"][r[0]] for r in rows])
    assert all((d["y"][r] == y[i]).all() for i, r in enumerate(rows)), "label not constant within a track"
    m_all, top_all = frame_margins(d["head"], d["emb"], np.ones(len(d["emb"]), bool))   # before the gate
    K = len(d["head"]["priors"])
    checks = {"frame_top_equals_stored_top1": bool((top_all == d["top1"]).all())}
    yhat = consensus(rows, top_all, m_all, K)
    cw = yhat != y
    n_ties = int(sum((lambda v: (v == v.max()).sum() > 1)(np.bincount(top_all[r], minlength=K)) for r in rows))

    seqs = SC.sequences(F, models)
    res = {}
    for s in SC.SCORES:
        grid, f_d = des[s]["grid"], des[s]["f_d"]
        tab = SC.track_table(seqs[s], y, grid)
        k_all, nc_all = (tab["committed"] & tab["wrong"]).sum(0), tab["committed"].sum(0)
        res[s] = {}
        for al in ALPHAS:
            j = int(C.certify(k_all, nc_all, T, al, f_d, test="binom", n_design=n_design))
            start = int(C.start_for("binom", f_d, T, al))
            r = {"j": j, "start": start, "lambda": None if j < 0 else float(grid[j])}
            if j >= 0:
                c, lab = committed_labels(seqs[s], grid[j])
                checks[f"{s}_{al}_table_agrees"] = bool((c == tab["committed"][:, j]).all()
                                                        and ((c & (lab != y)) == (tab["committed"][:, j] & tab["wrong"][:, j])).all())
                e, v, sl = decompose(c, lab, yhat, y)
                nc, ne, nv, ns_ = int(c.sum()), int(e.sum()), int(v.sum()), int(sl.sum())
                mism = int((e[~cw] != v[~cw]).sum())
                Ec, Ev, q = nc / T, nv / T, ns_ / T
                err = ne / nc
                gap, cN, vN = gap_value(al, Ec, Ev, N_AUDIT, delta)
                nA, nA_note = n_audit(q, gap, delta)
                nA0, nA0_note = n_audit(0.0, gap, delta)
                nF, nF_note = n_full(Ec, err, al, delta)
                r.update({"n_c": nc, "e": ne, "v": nv, "s": ns_, "commit_rate": Ec, "committed_error": err,
                          "visible_rate_committed": nv / nc, "silent_rate_committed": ns_ / nc,
                          "e_ne_v_on_consensus_correct": mism, "e_le_v_plus_s": bool((e <= (v | sl)).all()),
                          "q": q, "q_cp95": cp_two_sided(ns_, T), "gap": gap, "c_N": cN, "v_N": vN,
                          "n_A": nA, "n_A_note": nA_note, "n_A0": nA0, "n_A0_note": nA0_note,
                          "n_full": nF, "n_full_note": nF_note,
                          "ratio_nA_nfull": (nA / nF) if (nA is not None and nF) else None})
                checks[f"{s}_{al}_mismatch_zero"] = mism == 0
            res[s][al] = r

    # ---- printout: one table
    pc = lambda x, nd=2: "-" if x is None else f"{100 * x:.{nd}f}"
    print(f"\n=== a0 silent-error check: {LABEL} ===")
    print(f"Freeze check passed: {', '.join(f'{k} {v[:8]}' for k, v in blobs.items())}")
    print(f"Pool {T} tracks | certificate on the full pool (n = {T}, start at n = {T}) | delta {delta} | audit N {N_AUDIT}")
    print(f"Consensus-wrong tracks: {int(cw.sum())} / {T} = {pc(cw.mean(), 3)}% | tracks with a vote tie: {n_ties}")
    hdr = ["score", "a%", "j", "cw% (n)", "E[c]% (n_c)", "e/c% (e)", "v/c% (v)", "s/c% (s)", "mism",
           "q% [95% CP]", "gap%", "n_A", "n_A0", "n_full", "n_A/n_full"]
    rows_out = []
    for s in SC.SCORES:
        for al in ALPHAS:
            r = res[s][al]
            if r["j"] < 0:
                rows_out.append([s, f"{100 * al:g}", "-", f"{pc(cw.mean())} ({int(cw.sum())})"] + ["not certified"] + ["-"] * 10)
                continue
            rows_out.append([s, f"{100 * al:g}", str(r["j"]), f"{pc(cw.mean())} ({int(cw.sum())})",
                             f"{pc(r['commit_rate'], 1)} ({r['n_c']})", f"{pc(r['committed_error'], 3)} ({r['e']})",
                             f"{pc(r['visible_rate_committed'], 3)} ({r['v']})", f"{pc(r['silent_rate_committed'], 3)} ({r['s']})",
                             str(r["e_ne_v_on_consensus_correct"]),
                             f"{pc(r['q'], 3)} [{pc(r['q_cp95'][0], 3)}, {pc(r['q_cp95'][1], 3)}]", pc(r["gap"], 3),
                             fmt_n(r["n_A"], r["n_A_note"]), fmt_n(r["n_A0"], r["n_A0_note"]),
                             fmt_n(r["n_full"], r["n_full_note"]),
                             "-" if r["ratio_nA_nfull"] is None else f"{r['ratio_nA_nfull']:.3f}"])
    w = [max(len(h), *(len(x[i]) for x in rows_out)) for i, h in enumerate(hdr)]
    print("  ".join(h.rjust(w[i]) for i, h in enumerate(hdr)))
    for x in rows_out:
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(x)))
    all_pass = all(checks.values())
    print(f"Checks: {sum(checks.values())}/{len(checks)} PASS" + ("" if all_pass else
          " | FAIL: " + ", ".join(k for k, v in checks.items() if not v)))

    od = run / "a0"
    od.mkdir(exist_ok=True)
    summary = {"label": LABEL, "frozen_blobs": blobs,
               "code_sha256": {"a0_silent_error_check.py": sha256_file(Path(__file__))},
               "inputs_sha256": {**d["inputs_sha256"], "stage3_design.json": DESIGN_SHA256},
               "config": {"alphas": ALPHAS, "delta": delta, "n_audit": N_AUDIT, "n_a_max": N_A_MAX,
                          "n_full_max": N_FULL_MAX, "cert_n": T, "start_n": T, "rule": "binom",
                          "consensus": "majority over all frames (gate ignored); tie: larger summed pre-gate margin, then smaller id",
                          "cp_bounds_delta_over_3": "one-sided", "rounding": "half up"},
               "n_pool": T, "consensus_wrong": int(cw.sum()), "consensus_wrong_rate": float(cw.mean()), "vote_ties": n_ties,
               "results": {s: {str(al): r for al, r in v.items()} for s, v in res.items()},
               "checks": checks, "all_pass": all_pass, "seconds": round(time.time() - t0, 1)}
    (od / "a0_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=1))
    print(f"Output: {od / 'a0_summary.json'} ({summary['seconds']} s). {LABEL}")
    return summary


if __name__ == "__main__":
    main()
