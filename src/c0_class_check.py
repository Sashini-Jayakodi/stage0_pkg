"""EXPLORATORY (post hoc; no certificate, no criteria). Per-class-group view of the full-pool certified thresholds on
the 1,046 GTSRB pool tracks, with the sample size a direct per-group exact binomial test would need.

Every GTSRB track has been seen before, so nothing here is confirmatory. Stored inputs only; no frozen file and no
earlier result is changed. Output: runs/20261003_stage1/c0/.

Thresholds: for each score (S1, S2-sum, S2-min, S3, S4) and alpha in {1,2,3,5}%, the threshold the frozen
exact-binomial sign-level certificate certifies with the whole pool as the Cert set (n = 1,046), as in
src/a0_silent_error_check.py; j, n_c and errors are cross-checked against the hash-checked a0_summary.json.

Class ids (GTSRB has no class-name file; names from src/head.py, checked on cached crops): stop 14, yield 13,
no entry 17, priority road 12, speed limits {0-5, 7, 8}. Class 6 is "End of speed limit 80" (a derestriction sign),
so it is in "others", not in the speed-limit group.
  Grouping 1: g1 stop, g2 yield, g3 no entry, g4 speed limits, g5 priority road, g6 others (G = 6).
  Grouping 2: r1 right-of-way {14, 13, 17, 12}, r2 speed limits, r3 others (G = 3).

Per group, at the pooled certified threshold: committed tracks n_g and errors k_g on the pool, e_g = k_g / n_g,
ratio e_g / (pooled committed error). Direct test on the group's committed tracks (no fixed-sequence walk):
p = P(Binom(m, alpha) <= k) with the frozen engine's binom_pvalue, at level delta/2 = 0.05 (scheme b) or delta/(2G)
(scheme a, Bonferroni). Minimum committed tracks:
  m0      = smallest m that rejects with k = 0;
  m_first = smallest m that rejects with k = round_half_up(e_g m);
  m_all   = smallest m from which the test rejects for every m' in [m, M_CAP] (rounding makes the first hit unstable);
  "impossible" when e_g >= alpha. Feasible with 784 calibration tracks: E[committed in group] = 784 n_g / 1,046 >= m0
  (zero errors) and >= m_all (observed error).

Usage:  py -3.12 -m src.c0_class_check --run runs/20261003_stage1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import a0_silent_error_check as A0
from src import r8_stage2 as S2
from src import r8_stage3_resample as RS
from src import r9_certify as C
from src import r9_frozen as FZ
from src import r9_stage3 as R3
from src import r9_stage3_scores as SC
from src.head import NAMES
from src.r8_frozen import load_frozen
from src.util import sha256_file

LABEL = "EXPLORATORY (post hoc; no certificate, no criteria; GTSRB tracks all seen before)"
ALPHAS = (0.01, 0.02, 0.03, 0.05)
N_CAL = 784                        # calibration (Cert) size of the POC-2 Stage 4 re-splits
M_CAP = 1_000_000                  # search cap for the minimum committed tracks
A0_SUMMARY_SHA256 = "4c4c907ee19a493ec0df0e188f6f34aa51f3e350af156985e235bfc4d8791928"   # LOG.md, a0 entry
K_CLASSES = 43

# Class ids and the names the check expects (src/head.py NAMES; verified on cached crops before the run).
EXPECTED_NAMES = {0: "Speed 20", 1: "Speed 30", 2: "Speed 50", 3: "Speed 60", 4: "Speed 70", 5: "Speed 80",
                  6: "End speed 80", 7: "Speed 100", 8: "Speed 120", 12: "Priority road", 13: "Yield", 14: "Stop",
                  17: "No entry"}
SPEED = (0, 1, 2, 3, 4, 5, 7, 8)


def _others(*used):
    taken = {c for g in used for c in g}
    return tuple(c for c in range(K_CLASSES) if c not in taken)


GROUPINGS = {
    "Grouping 1 (plan)": [("g1 stop", (14,)), ("g2 yield", (13,)), ("g3 no entry", (17,)), ("g4 speed limits", SPEED),
                          ("g5 priority road", (12,)), ("g6 others", _others((14,), (13,), (17,), SPEED, (12,)))],
    "Grouping 2 (merged)": [("r1 right-of-way", (14, 13, 17, 12)), ("r2 speed limits", SPEED),
                            ("r3 others", _others((14, 13, 17, 12), SPEED))],
}


def group_index(y, groups):
    """Group number per track; asserts the groups partition the class ids."""
    ids = [c for _, cs in groups for c in cs]
    assert sorted(ids) == list(range(K_CLASSES)), "groups do not partition the 43 classes"
    lut = np.empty(K_CLASSES, int)
    for g, (_, cs) in enumerate(groups):
        lut[list(cs)] = g
    return lut[np.asarray(y)]


# ------------------------------------------------------------------ minimum committed tracks (direct test)
def _reject(err, alpha, level, ms):
    k = np.floor(err * ms + 0.5)                                    # round half up, as a0
    return C.binom_pvalue(k, ms, alpha) <= level


def min_committed(err, alpha, level, cap=M_CAP):
    """(m_first, m_all, note). m_all: smallest m with rejection at every m' in [m, cap]; None with a note otherwise."""
    if err >= alpha:
        return None, None, "impossible (e >= alpha)"
    ms = np.arange(1, cap + 1, dtype=float)
    rej = _reject(err, alpha, level, ms)
    if not rej.any():
        return None, None, f"none <= {cap}"
    first = int(ms[np.argmax(rej)])
    if not rej[-1]:
        return first, None, f"not stable <= {cap}"
    bad = np.flatnonzero(~rej)
    return first, (int(ms[bad[-1]]) + 1 if len(bad) else 1), ""


# ------------------------------------------------------------------ per-group rows
def group_rows(groups, gidx_pool, gidx_design, committed, wrong, alpha, n_pool, n_cal=N_CAL, delta=C.DELTA, cap=M_CAP,
               cache=None):
    """Rows for one score x alpha at one threshold. committed, wrong: bool per pool track."""
    G = len(groups)
    levels = {"b": delta / 2, "a": delta / (2 * G)}
    nc_all, k_all = int(committed.sum()), int((committed & wrong).sum())
    pooled = k_all / nc_all if nc_all else None
    cache = {} if cache is None else cache
    out = []
    for g, (name, cs) in enumerate(groups):
        m = gidx_pool == g
        ng, kg = int((committed & m).sum()), int((committed & wrong & m).sum())
        eg = kg / ng if ng else None
        e784 = n_cal * ng / n_pool
        r = {"group": name, "classes": list(cs), "n_pool": int(m.sum()), "n_design": int((gidx_design == g).sum()),
             "exp_tracks_784": n_cal * int(m.sum()) / n_pool, "n_committed": ng, "errors": kg, "group_error": eg,
             "ratio_to_pooled": (eg / pooled) if (eg is not None and pooled) else None, "exp_committed_784": e784}
        for sch, lev in levels.items():
            key0 = (0.0, alpha, lev)
            if key0 not in cache:
                cache[key0] = min_committed(0.0, alpha, lev, cap)
            m0 = cache[key0][0]
            r[f"{sch}_level"] = lev
            r[f"{sch}_m0"] = m0
            r[f"{sch}_feasible_zero"] = bool(m0 is not None and e784 >= m0)
            if eg is None:
                r.update({f"{sch}_m_first": None, f"{sch}_m_all": None, f"{sch}_note": "no committed tracks",
                          f"{sch}_feasible_obs": False})
                continue
            key = (eg, alpha, lev)
            if key not in cache:
                cache[key] = min_committed(eg, alpha, lev, cap)
            mf, ma, note = cache[key]
            r.update({f"{sch}_m_first": mf, f"{sch}_m_all": ma, f"{sch}_note": note,
                      f"{sch}_feasible_obs": bool(ma is not None and e784 >= ma)})
        out.append(r)
    return out, {"n_committed": nc_all, "errors": k_all, "pooled_error": pooled}


# ------------------------------------------------------------------ printout
def _pc(x, nd=2):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def _m(r, sch):
    if r[f"{sch}_m_all"] is not None:
        return f"{r[f'{sch}_m_first']}/{r[f'{sch}_m_all']}"
    return r[f"{sch}_note"] if r[f"{sch}_m_first"] is None else f"{r[f'{sch}_m_first']}/{r[f'{sch}_note']}"


def print_table(title, groups, res):
    G = len(groups)
    print(f"\n--- {title}: G = {G}; scheme b level {C.DELTA / 2:g}, scheme a level {C.DELTA / (2 * G):.6f} ---")
    hdr = ["group", "score", "a%", "N_pool", "N_des", "N/784", "n_c", "k", "e_g%", "e_g/e", "n_c/784",
           "b:m0", "b:m_first/m_all", "b:feas0", "b:feasObs", "a:m0", "a:m_first/m_all", "a:feas0", "a:feasObs"]
    yn = lambda b: "yes" if b else "no"
    lines = []
    for gname, _ in groups:
        for s in SC.SCORES:
            for al in ALPHAS:
                cell = res[s][str(al)]
                if cell["j"] < 0:
                    lines.append([gname, s, f"{100 * al:g}"] + ["not certified"] + ["-"] * 15)
                    continue
                r = next(x for x in cell["groups"] if x["group"] == gname)
                lines.append([gname, s, f"{100 * al:g}", str(r["n_pool"]), str(r["n_design"]),
                              f"{r['exp_tracks_784']:.1f}", str(r["n_committed"]), str(r["errors"]),
                              _pc(r["group_error"], 3), "-" if r["ratio_to_pooled"] is None else f"{r['ratio_to_pooled']:.2f}",
                              f"{r['exp_committed_784']:.1f}",
                              str(r["b_m0"]), _m(r, "b"), yn(r["b_feasible_zero"]), yn(r["b_feasible_obs"]),
                              str(r["a_m0"]), _m(r, "a"), yn(r["a_feasible_zero"]), yn(r["a_feasible_obs"])])
    w = [max(len(h), *(len(x[i]) for x in lines)) for i, h in enumerate(hdr)]
    print("  ".join(h.rjust(w[i]) for i, h in enumerate(hdr)))
    prev = None
    for x in lines:
        if prev is not None and x[0] != prev:
            print()
        prev = x[0]
        print("  ".join(c.rjust(w[i]) for i, c in enumerate(x)))


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    blobs = FZ.check_frozen()                                          # frozen engine first, as every r9 runner
    cfg = load_frozen()
    run = Path(a.run)
    delta = C.DELTA
    assert delta == 0.1

    # class-name check (the dataset ships ids only; names come from src/head.py)
    checks = {f"name_{c}": NAMES[c] == n for c, n in EXPECTED_NAMES.items()}

    # inputs, hash-checked (hard rule 2)
    dpath = run / "poc3" / "stage3" / "stage3_design.json"
    A0.verify_sha(dpath, A0.DESIGN_SHA256)
    js, des, models = R3.load_design(dpath)
    n_design = int(js["n_design"])
    s1dir = run / "poc2" / "stage1"
    s1 = json.loads((s1dir / "stage1_summary.json").read_text())
    A0.verify_sha(s1dir / S2.DESIGN_FILE, s1["files_sha256"][S2.DESIGN_FILE])
    y_design = np.asarray(S2.open_npz(s1dir / S2.DESIGN_FILE)["y_track"]).astype(int)
    a0p = run / "a0" / "a0_summary.json"
    A0.verify_sha(a0p, A0_SUMMARY_SHA256)
    a0 = json.loads(a0p.read_text())
    d = RS.load_pool(run, cfg)                                         # hash-checks Stage 1, head, splits, b1_design

    F = SC.frame_inputs(d["head"], d["emb"], d["passed"], d["track"], d["frame"])
    rows = F["rows"]
    T = len(rows)
    assert T == int(cfg["splits"]["pool_tracks"]) == 1046 and len(y_design) == n_design == 261
    y = np.array([d["y"][r[0]] for r in rows])
    assert all((d["y"][r] == y[i]).all() for i, r in enumerate(rows)), "label not constant within a track"
    checks["all_43_classes_present"] = set(np.concatenate([y, y_design]).tolist()) == set(range(K_CLASSES))

    seqs = SC.sequences(F, models)
    gidx = {t: (group_index(y, g), group_index(y_design, g)) for t, g in GROUPINGS.items()}
    res = {t: {} for t in GROUPINGS}
    pooled = {}
    cache = {}
    for s in SC.SCORES:
        grid, f_d = des[s]["grid"], des[s]["f_d"]
        tab = SC.track_table(seqs[s], y, grid)
        k_all, nc_all = (tab["committed"] & tab["wrong"]).sum(0), tab["committed"].sum(0)
        pooled[s] = {}
        for al in ALPHAS:
            j = int(C.certify(k_all, nc_all, T, al, f_d, test="binom", n_design=n_design))
            ref = a0["results"][s][str(al)]
            checks[f"{s}_{al}_j_equals_a0"] = j == ref["j"]
            cell = {"j": j, "lambda": None if j < 0 else float(grid[j])}
            if j >= 0:
                com, wr = tab["committed"][:, j], tab["wrong"][:, j]
                for t, groups in GROUPINGS.items():
                    grp, tot = group_rows(groups, *gidx[t], com, wr, al, T, N_CAL, delta, M_CAP, cache)
                    res[t].setdefault(s, {})[str(al)] = {**cell, "groups": grp, "pooled": tot}
                    checks[f"{t}_{s}_{al}_groups_sum"] = (sum(r["n_committed"] for r in grp) == tot["n_committed"]
                                                         and sum(r["errors"] for r in grp) == tot["errors"])
                checks[f"{s}_{al}_counts_equal_a0"] = (int(com.sum()) == ref["n_c"] and int((com & wr).sum()) == ref["e"])
                pooled[s][str(al)] = {**cell, "n_committed": int(com.sum()), "errors": int((com & wr).sum())}
            else:
                for t in GROUPINGS:
                    res[t].setdefault(s, {})[str(al)] = cell
                pooled[s][str(al)] = cell

    # ---- printout
    print(f"\n=== c0 class-group check: {LABEL} ===")
    print(f"Freeze check passed: {', '.join(f'{k} {v[:8]}' for k, v in blobs.items())}")
    print(f"Pool {T} tracks, design {n_design} | thresholds certified on the full pool (n = {T}), as a0 | delta {delta} "
          f"| calibration size {N_CAL} | search cap {M_CAP:,}")
    print("Class ids and names (src/head.py; GTSRB ships no name file):")
    for c, n in EXPECTED_NAMES.items():
        print(f"  {c:2d} {NAMES[c]:<14s} {'ok' if NAMES[c] == n else 'MISMATCH'}")
    print("  Speed-limit group = {0-5, 7, 8}; class 6 (End speed 80) is in 'others'.")
    print("Pooled committed tracks / errors per score x alpha (pool, at the certified threshold):")
    for s in SC.SCORES:
        print("  " + s.ljust(7) + "  ".join(
            f"{100 * al:g}%: j {pooled[s][str(al)]['j']}, {pooled[s][str(al)].get('n_committed', '-')}/"
            f"{pooled[s][str(al)].get('errors', '-')}" for al in ALPHAS))
    print("Columns: N_pool, N_des = tracks in the group; N/784 = expected tracks per 784-track draw; n_c, k = committed "
          "tracks and errors in the group; e_g/e = group error / pooled committed error; n_c/784 = expected committed "
          "per 784-track draw; m0 = minimum committed with 0 errors; m_first/m_all = minimum with the observed group "
          "error (first rejection / rejection at every larger m up to the cap); feas0, feasObs = n_c/784 >= m0, >= m_all.")
    for t, groups in GROUPINGS.items():
        print_table(t, groups, res[t])
    all_pass = all(checks.values())
    print(f"\nChecks: {sum(checks.values())}/{len(checks)} PASS" + ("" if all_pass else
          " | FAIL: " + ", ".join(k for k, v in checks.items() if not v)))

    od = run / "c0"
    od.mkdir(exist_ok=True)
    summary = {"label": LABEL, "frozen_blobs": blobs,
               "code_sha256": {"c0_class_check.py": sha256_file(Path(__file__)),
                               "a0_silent_error_check.py": sha256_file(Path(A0.__file__))},
               "inputs_sha256": {**d["inputs_sha256"], "stage3_design.json": A0.DESIGN_SHA256,
                                 S2.DESIGN_FILE: s1["files_sha256"][S2.DESIGN_FILE], "a0_summary.json": A0_SUMMARY_SHA256},
               "config": {"alphas": ALPHAS, "delta": delta, "n_cal": N_CAL, "m_cap": M_CAP, "cert_n": T, "start_n": T,
                          "rule": "binom", "levels": {"b": "delta/2", "a": "delta/(2G)"}, "rounding": "half up",
                          "names": {str(c): NAMES[c] for c in EXPECTED_NAMES},
                          "groupings": {t: {n: list(cs) for n, cs in g} for t, g in GROUPINGS.items()}},
               "n_pool": T, "n_design": n_design, "pooled": pooled, "results": res,
               "checks": checks, "all_pass": all_pass, "seconds": round(time.time() - t0, 1)}
    (od / "c0_summary.json").write_text(json.dumps(S2.jsonable(summary), indent=1))
    print(f"Output: {od / 'c0_summary.json'} ({summary['seconds']} s). {LABEL}")
    return summary


if __name__ == "__main__":
    main()
