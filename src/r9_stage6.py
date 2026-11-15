"""POC-3 (R9) Stage 6, BelgiumTSC confirmation (confirm phase, after F3): RQ1 resampling with the pre-registered
V-check, RQ2 certification on the 5 frozen re-splits, the baselines, the plan section 8 comparison table and the S2
against B6 table (LOG_r9_stage6_entry.md Part A). Writes runs/btsc/stage6/stage6_summary.json and the printout.

Every value comes from configs/frozen_r9_btsc.yaml, which check_frozen() verifies by git blob ID before anything is
read. The design phase is recomputed from the hash-checked Stage 4 inputs and must write the frozen yaml again byte
for byte (after CRLF -> LF); the S2 calibration scores used on the pool come from that recomputation.

Usage:  py -3.12 -m src.r9_stage6
"""
import argparse
import datetime
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src import r8_stage5 as S5
from src import r8_stage6_design as B5D
from src import r8_stage6_methods as M
from src import r9_certify as C
from src import r9_frozen as FZ
from src import r9_stage5_design as DS
from src import r9_stage5_freeze as FR
from src import r9_stage5_scores as SC
from src import r9_stage6_methods as SM
from src.head import score
from src.r8_stage2 import jsonable
from src.r9_btsc_guard import GuardedTestLabels, f3_closed
from src.r9_stage4_data import stop
from src.util import sha256_file

YAML_PATH = Path("configs/frozen_r9_btsc.yaml")
STAGE5_DESIGN = Path("runs/btsc/stage5/stage5_design.json")
OUT = Path("runs/btsc/stage6")
# Part A (a): recorded values (Stage 5 Part A design-pole values and the F3 record)
DESIGN_SHA256 = "115f98025f60b14aedad9c0d75df6d5deafc568d77b427e2bf9804be1d854a11"
DESIGN_BLOB = "b06a09f8c64bbd789bd02addab0b518ecde66d0e"
POOL_INPUTS = ("embeddings/test_pool.npz", "head/scores_test_pool.npz")
EXPECTED_POOL = {"pool_poles": 530, "pool_units": 626, "pool_views": 1860}
N_DESIGN_UNITS = 157
CODE = ("src/r9_stage6.py", "src/r9_stage6_methods.py")
OPENED = []
COLS = ("S1", "S2-sum", "S2-min", "S3", "S4", "B1", "B2", "B5", "B5c", "B6 S2-sum", "B6 S2-min")
NOT_RUN_B1 = "not run (B1 is pre-registered for the resampling only)"
NOT_RQ1 = "not in the pre-registered resampling"


class Tee(DS.Tee):
    pass


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ inputs (a)
def load_yaml(path=YAML_PATH):
    """Read the frozen yaml (check_frozen() has verified its blob ID before this is called)."""
    text = Path(path).read_text()
    OPENED.append(Path(path).as_posix())
    return yaml.safe_load(text), text.replace("\r\n", "\n")


def open_pool(base, summary):
    """Hash-check the pool score and embedding files against stage4_summary.json and load them."""
    out = {}
    for rel in POOL_INPUTS:
        p = Path(base) / rel
        if not p.is_file():
            stop(f"{p} is missing")
        if sha256_file(p) != summary["files_sha256"][rel]:
            stop(f"{p} changed since stage4_summary.json recorded it")
        OPENED.append(rel)
        out[rel] = dict(np.load(p))
        out.setdefault("sha256", {})[rel] = sha256_file(p)
    return out


def reproduce_yaml(d, design, yaml_text, design_sha, design_blob):
    """Part A (k): the recomputed design phase must write the frozen yaml again, byte for byte (LF)."""
    R = DS.run_design(d)
    built = yaml.safe_dump(FR.build(R, design, design_sha, design_blob), sort_keys=False, default_flow_style=None,
                           width=120)
    if built != yaml_text:
        stop("the recomputed design phase does not reproduce configs/frozen_r9_btsc.yaml")
    return R


# ------------------------------------------------------------------ pool tables (c)
def pool_tables(d, P, R, Y):
    rows, head = d["rows"], d["head"]
    E, S = P["embeddings/test_pool.npz"], P["head/scores_test_pool.npz"]
    integ = {}
    part = np.array([r["part"] for r in rows])
    integ["npz_rows_are_pool_rows"] = bool(np.array_equal(np.sort(E["row"]), np.flatnonzero(part == "pool")))
    integ["npz_matches_view_table"] = bool(all(
        (rows[i]["pole"], rows[i]["view"], rows[i]["size"]) == (int(p), int(v), int(s))
        for i, p, v, s in zip(E["row"], E["pole"], E["view"], E["size"])))
    integ["scores_rows_match"] = bool(np.array_equal(S["row"], E["row"]))
    design_poles = sorted(d["poles"]["design_poles"])
    integ["no_design_pole_in_pool"] = not (set(E["pole"].tolist()) & set(design_poles))
    integ["f3_closed_before_pool_labels"] = bool(f3_closed())
    assert integ["f3_closed_before_pool_labels"], "F3 is not closed; pool labels are refused (hard rule 3)"
    guard = GuardedTestLabels([r["pole"] for r in rows], [r["cls"] for r in rows], design_poles)
    y_view = guard.labels(E["row"])
    sc = score(head, E["emb"])
    integ["pool_head_top3_reproduces"] = bool(np.array_equal(sc["top3"], S["top3"]))
    integ["pool_head_margin_reproduces"] = bool(np.array_equal(sc["margin"], S["margin"]))
    V = SC.view_inputs(head, E["emb"])
    integ["pool_top1_equals_head_top1"] = bool(np.array_equal(V["top"], S["top3"][:, 0].astype(int)))
    units, urows = SC.unit_rows(E["pole"], y_view, E["view"])
    facts = {"pool_poles": len({p for p, _ in units}), "pool_units": len(units), "pool_views": int(len(E["row"]))}
    integ["pool_facts"] = facts == EXPECTED_POOL
    if not all(integ.values()):
        stop(f"integrity check failed: {integ} {facts}")
    y_unit = np.array([c for _, c in units])
    size_view = E["size"].astype(float)
    seqs = SC.sequences(V, urows, R["models"])
    tabs = {s: SM.unit_table(seqs[s], y_unit, np.array(Y["scores"][s]["grid"], float), urows, size_view)
            for s in SC.SCORES}
    b1 = SC.track_table(SM.b1_seqs(V), y_view, np.array(Y["b1"]["grid"], float))
    return {"V": V, "y_view": y_view, "units": units, "urows": urows, "y_unit": y_unit, "size_view": size_view,
            "seqs": seqs, "tabs": tabs, "b1": b1, "view_pole": E["pole"].astype(int), "facts": facts,
            "integrity": integ}


# ------------------------------------------------------------------ RQ1 (d) and the V-check (e)
def run_rq1(T, Y, alphas):
    rq = Y["rq1"]
    unit_pole = np.array([p for p, _ in T["units"]])
    poles = np.unique(unit_pole)
    counts = SM.draw_counts(len(poles), int(rq["draws"]), int(rq["poles_per_draw"]), tuple(rq["salt"]))
    methods = {s: (T["tabs"][s], np.searchsorted(poles, unit_pole), Y["scores"][s]["f_d"], N_DESIGN_UNITS)
               for s in SC.SCORES}
    methods["B1"] = (T["b1"], np.searchsorted(poles, T["view_pole"]), Y["b1"]["f_d"], int(Y["b1"]["n_design_views"]))
    out = {}
    for name, (tab, item_pole, f_d, nd) in methods.items():
        Cp, Wp, n_items = SM.per_pole(tab, item_pole, len(poles))
        K, NC, N = SM.draw_sums(counts, Cp, Wp, n_items)
        risk, f_pool = SM.true_risk(tab)
        cells = {a: {r: SM.rq1_cell(K, NC, N, a, np.array(f_d, float), r, nd, risk) for r in SM.RQ1_RULES}
                 for a in alphas}
        out[name] = {"risk": risk, "f_pool": f_pool, "n_mean": float(N.mean()), "n_range": [int(N.min()), int(N.max())],
                     "cells": cells}
    return out


# ------------------------------------------------------------------ RQ2 (f), (g) and baselines (h)
def run_rq2(T, Y, alphas):
    units, urows, y = T["units"], T["urows"], T["y_unit"]
    unit_pole = np.array([p for p, _ in units])
    pool_poles = sorted(set(unit_pole.tolist()))
    s_values = S5.s_range(float(T["size_view"].max()))
    nv = np.array([len(r) for r in urows])
    integ = {"splits_partition_pool": True, "split_unit_counts": True, "starts_equal_yaml": True,
             "b2_starts_equal_yaml": True}
    # B5 inputs: S1 margins as Ringel scores via F_design, S1 labels, full-sequence label
    F = np.array(Y["b5"]["f_design"], float)
    pis = [B5D.score(c, F) for c, _ in T["seqs"]["S1"]]
    preds = [l.astype(int) for _, l in T["seqs"]["S1"]]
    yfull = np.array([p[-1] for p in preds])
    sizes = [T["size_view"][r] for r in urows]
    tmax = int(Y["b5"]["b5c_tmax"])
    out = {}
    for seed in Y["resplits"]["seeds"]:
        sp = Y["resplits"]["split"][str(seed)]
        tp, cp = set(sp["test_poles"]), set(sp["cert_poles"])
        if tp & cp or tp | cp != set(pool_poles):
            integ["splits_partition_pool"] = False
            stop(f"seed {seed}: Cert and Test poles do not partition the pool")
        test = np.isin(unit_pole, sorted(tp))
        cert = ~test
        n = int(cert.sum())
        if n != sp["n_cert_units"] or int(test.sum()) != sp["n_test_units"]:
            integ["split_unit_counts"] = False
            stop(f"seed {seed}: unit counts differ from the yaml")
        res = {"n_cert": n, "n_test": int(test.sum()), "scores": {}, "b2": {}, "b5": {}, "b5c": {}, "b6": {}}
        for s in SC.SCORES:
            Ys = Y["scores"][s]
            f_d = np.array(Ys["f_d"], float)
            cell = {"cert": {}, "test": {}, "reference": {}, "smallest_alpha": {}}
            for a in alphas:
                ak = SM.akey(a)
                cell["cert"][a], cell["test"][a] = {}, {}
                for r in C.RULES:
                    st = int(C.start_for(r, f_d, n, a, n_design=N_DESIGN_UNITS))
                    if st != Ys["starts"][str(seed)][ak][r]:
                        integ["starts_equal_yaml"] = False
                        stop(f"seed {seed} {s} {ak} {r}: start {st} differs from the frozen start")
                    j = SM.certify_split(T["tabs"][s], cert, a, f_d, r, N_DESIGN_UNITS)
                    cell["cert"][a][r] = j
                    cell["test"][a][r] = SM.test_metrics(T["tabs"][s], j, test, s_values)
                cell["reference"][a] = SM.test_metrics(T["tabs"][s], int(Ys["reference"][ak]), test, s_values)
            for r in C.RULES:
                cell["smallest_alpha"][r] = SM.smallest_alpha({a: cell["cert"][a][r] for a in alphas}, alphas)
            res["scores"][s] = cell
        # B2: one random view per unit, one draw over all pool units in unit order per seed
        k_b2 = SC.b2_draw(nv, np.random.default_rng([*Y["b2"]["salt"], seed]))
        b2_tab = SM.unit_table([SC.seq_b2(T["V"], r, k) for r, k in zip(urows, k_b2)], y,
                               np.array(Y["b2"]["grid"], float), urows, T["size_view"])
        f_b2 = np.array(Y["b2"]["f_d"], float)
        b2 = {"cert": {}, "test": {}, "smallest_alpha": {}}
        for a in alphas:
            b2["cert"][a], b2["test"][a] = {}, {}
            for r in C.RULES:
                st = int(C.start_for(r, f_b2, n, a, n_design=N_DESIGN_UNITS))
                if st != Y["b2"]["starts"][str(seed)][SM.akey(a)][r]:
                    integ["b2_starts_equal_yaml"] = False
                    stop(f"seed {seed} B2 {a} {r}: start differs from the frozen start")
                j = SM.certify_split(b2_tab, cert, a, f_b2, r, N_DESIGN_UNITS)
                b2["cert"][a][r] = j
                b2["test"][a][r] = SM.test_metrics(b2_tab, j, test, s_values)
        for r in C.RULES:
            b2["smallest_alpha"][r] = SM.smallest_alpha({a: b2["cert"][a][r] for a in alphas}, alphas)
        res["b2"] = b2
        # B5 and B5c (Ringel), views as the time axis
        ci, ti = np.flatnonzero(cert), np.flatnonzero(test)
        sub = lambda idx: ([pis[i] for i in idx], [preds[i] for i in idx], y[idx], yfull[idx])
        tsz = [sizes[i] for i in ti]
        d1, d2, n1 = SM.b5c_parts(sorted(cp), unit_pole, ci, seed)
        res["b5c_parts"] = {"d1_poles": n1, "d2_poles": len(cp) - n1, "d1_units": len(d1), "d2_units": len(d2)}
        for a in alphas:
            lam = M.marginal_calibrate(*sub(ci), a, C.DELTA)
            res["b5"][a] = {"lambda": lam, "certified": bool(np.isfinite(lam)),
                            **SM.ringel_eval(*sub(ti), np.full(tmax, lam), tsz)}
            lam_vec, cand = SM.conditional_ext(pis, preds, y, yfull, d1, d2, a, C.DELTA, tmax)
            res["b5c"][a] = {"lambda": lam_vec, "candidate": cand, "certified": bool(np.isfinite(lam_vec).any()),
                             **SM.ringel_eval(*sub(ti), lam_vec, tsz)}
        # B6: uncertified S2 with the frozen theta
        for v in ("S2-sum", "S2-min"):
            res["b6"][v] = {}
            for a in alphas:
                th = Y["b6"]["theta"][v][SM.akey(a)]
                tab = SM.b6_table(T["seqs"][v], y, th, urows, T["size_view"])
                res["b6"][v][a] = {"theta": th, **SM.test_metrics(tab, 0, test, s_values)}
        out[seed] = res
    return out, integ, s_values


# ------------------------------------------------------------------ printout helpers
def pct(x, w=6):
    return f"{'n/a':>{w}}" if x is None else f"{100 * x:{w}.2f}"


def ci(c):
    return "[  n/a ]" if c is None else f"[{100 * c[0]:.2f}, {100 * c[1]:.2f}]"


def mr(vals, as_pct=True, nd=2):
    m = SM.med_range(vals)
    if m is None:
        return "n/a"
    f = (lambda x: f"{100 * x:.{nd}f}%") if as_pct else (lambda x: f"{x:g}")
    return f"{f(m[0])} ({f(m[1])}-{f(m[2])}; {m[3]})"


def num(x):
    return "n/a" if x is None else f"{x:g}"


def alpha_txt(a):
    return "none" if a is None else f"{100 * a:g}%"


# ------------------------------------------------------------------ tables (i), (j)
def layer_rows(cells, seeds, a, rule="binom"):
    """Per seed metrics of a certified-or-not method; returns metrics over the certified seeds only."""
    cert = [s for s in seeds if cells[s]["cert"][a][rule] >= 0]
    return cert, [cells[s]["test"][a][rule] for s in cert]


def comparison_table(rq1, rq2, Y, a):
    seeds = list(rq2)
    T = {}
    sc_cells = {s: {seed: rq2[seed]["scores"][s] for seed in seeds} for s in SC.SCORES}
    b2_cells = {seed: rq2[seed]["b2"] for seed in seeds}
    for c in COLS:
        row = {}
        if c in SC.SCORES or c == "B2":
            cells = sc_cells[c] if c != "B2" else b2_cells
            cs, ms = layer_rows(cells, seeds, a)
            row["Type of promise"] = ("certified real error among committed signs" if c != "B2"
                                      else "certified real error among committed signs (one random view per sign)")
            if c in SC.SCORES:
                r = rq1[c]["cells"][a]["binom"]
                row["Violation rate vs delta (resampling)"] = (f"{100 * r['rate']:.2f}% {ci(r['cp95'])} "
                                                               f"({r['certified']}/2000 certified)")
            else:
                row["Violation rate vs delta (resampling)"] = NOT_RQ1
            row["Certified in how many re-splits"] = f"{len(cs)}/{len(seeds)}"
            row["Test coverage"] = mr([m["coverage"] for m in ms])
            row["Test error among committed"] = mr([m["error"] for m in ms])
            errs = [m["error"] for m in ms if m["error"] is not None]
            row["Worst re-split error vs target"] = (f"{100 * max(errs):.2f}% vs {100 * a:g}% "
                                                     f"({sum(e > a + 1e-12 for e in errs)} above)" if errs else "n/a")
            sa = [cells[s]["smallest_alpha"]["binom"] for s in seeds]
            row["Smallest certifiable alpha"] = (mr(sa) + " | per seed " + ", ".join(alpha_txt(x) for x in sa))
            row["Median commit frame (or views)"] = mr([m["median_view"] for m in ms], as_pct=False)
            row["Median sign size at commit"] = mr([m["median_size"] for m in ms], as_pct=False)
            row["Can escalate"] = "yes"
        elif c == "B1":
            r = rq1["B1"]["cells"][a]["binom"]
            row["Type of promise"] = "certified real error per view (views treated as independent)"
            row["Violation rate vs delta (resampling)"] = (f"{100 * r['rate']:.2f}% {ci(r['cp95'])} "
                                                           f"({r['certified']}/2000 certified)")
            for k in ("Certified in how many re-splits", "Test coverage", "Test error among committed",
                      "Worst re-split error vs target", "Smallest certifiable alpha", "Median commit frame (or views)",
                      "Median sign size at commit"):
                row[k] = NOT_RUN_B1
            row["Can escalate"] = "yes"
        elif c in ("B5", "B5c"):
            key = c.lower()
            res = [rq2[s][key][a] for s in seeds]
            cs = [r for r in res if r["certified"]]
            row["Type of promise"] = "certified accuracy gap, " + ("marginal" if c == "B5" else "conditional on halt view")
            row["Violation rate vs delta (resampling)"] = NOT_RQ1
            row["Certified in how many re-splits"] = f"{len(cs)}/{len(seeds)} (finite lambda)"
            row["Test coverage"] = mr([r["coverage"] for r in cs])
            row["Test error among committed"] = mr([r["error_committed"] for r in cs]) + " (real error)"
            gaps = [r["gap_marginal"] for r in cs]
            errs = [r["error_committed"] for r in cs if r["error_committed"] is not None]
            row["Worst re-split error vs target"] = (
                f"gap {100 * max(gaps):.2f}% vs tolerance {100 * a:g}%; real error {100 * max(errs):.2f}% "
                f"({sum(e > a + 1e-12 for e in errs)} above {100 * a:g}%)" if cs else "n/a")
            alphas = sorted(rq2[seeds[0]][key])
            sa = [next((x for x in alphas if rq2[s][key][x]["certified"]), None) for s in seeds]
            row["Smallest certifiable alpha"] = ("gap tolerance (different promise): " + mr(sa) + " | per seed "
                                                 + ", ".join(alpha_txt(x) for x in sa))
            row["Median commit frame (or views)"] = mr([r["median_commit_frame"] for r in cs], as_pct=False)
            row["Median sign size at commit"] = mr([r["median_size"] for r in cs], as_pct=False)
            row["Can escalate"] = "no"
        else:
            v = c.split()[1]
            ms = [rq2[s]["b6"][v][a] for s in seeds]
            errs = [m["error"] for m in ms if m["error"] is not None]
            row["Type of promise"] = "none (uncertified; target r = alpha)"
            row["Violation rate vs delta (resampling)"] = NOT_RQ1
            row["Certified in how many re-splits"] = "no certificate"
            row["Test coverage"] = mr([m["coverage"] for m in ms])
            row["Test error among committed"] = mr([m["error"] for m in ms])
            row["Worst re-split error vs target"] = (f"{100 * max(errs):.2f}% vs r = {100 * a:g}% "
                                                     f"({sum(e > a + 1e-12 for e in errs)} above)" if errs else "n/a")
            row["Smallest certifiable alpha"] = "none (no certificate)"
            row["Median commit frame (or views)"] = mr([m["median_view"] for m in ms], as_pct=False)
            row["Median sign size at commit"] = mr([m["median_size"] for m in ms], as_pct=False)
            row["Can escalate"] = "yes"
        T[c] = row
    return T


def cost_table(rq2, alphas):
    seeds = list(rq2)
    out = {}
    for v in ("S2-sum", "S2-min"):
        for a in alphas:
            lay = [rq2[s]["scores"][v]["test"][a]["binom"] for s in seeds]
            cert = [rq2[s]["scores"][v]["cert"][a]["binom"] >= 0 for s in seeds]
            groups = {"layer (exact binomial)": [m for m, c in zip(lay, cert) if c],
                      "B6 (uncertified, r = alpha)": [rq2[s]["b6"][v][a] for s in seeds],
                      "uncertified reference": [rq2[s]["scores"][v]["reference"][a] for s in seeds]}
            rows = {}
            for g, ms in groups.items():
                errs = [m["error"] for m in ms if m["error"] is not None]
                rows[g] = {"seeds": f"{sum(cert)}/5 certified" if g.startswith("layer") else f"{len(ms)}/5 run",
                           "coverage": mr([m["coverage"] for m in ms]),
                           "median_view": mr([m["median_view"] for m in ms], as_pct=False),
                           "median_size": mr([m["median_size"] for m in ms], as_pct=False),
                           "error": mr([m["error"] for m in ms]),
                           "worst": f"{100 * max(errs):.2f}%" if errs else "n/a",
                           "above_alpha": sum(e > a + 1e-12 for e in errs)}
            out[(v, a)] = rows
    return out


def above_alpha(rq2, alphas):
    """Every Test cell whose error exceeds alpha: certified layer and B2 cells (all rules), B6 cells against r,
    B5/B5c cells whose gap exceeds the tolerance or whose real error exceeds alpha."""
    out = []
    for seed, res in rq2.items():
        for s in SC.SCORES:
            for a in alphas:
                for r in C.RULES:
                    m = res["scores"][s]["test"][a][r]
                    if m["j"] is not None and m["error"] is not None and m["error"] > a + 1e-12:
                        out.append(("layer", s, seed, a, r, m))
        for a in alphas:
            for r in C.RULES:
                m = res["b2"]["test"][a][r]
                if m["j"] is not None and m["error"] is not None and m["error"] > a + 1e-12:
                    out.append(("B2", "S4 one view", seed, a, r, m))
            for v in ("S2-sum", "S2-min"):
                m = res["b6"][v][a]
                if m["error"] is not None and m["error"] > a + 1e-12:
                    out.append(("B6", v, seed, a, "-", m))
            for key in ("b5", "b5c"):
                m = res[key][a]
                if m["gap_marginal"] > a + 1e-12 or (m["error_committed"] or 0) > a + 1e-12:
                    out.append((key.upper().replace("C", "c"), "S1", seed, a, "-", m))
    return out


# ------------------------------------------------------------------ report
def report(res, Y, alphas):
    prim = float(Y["primary_alpha"])
    print("\n--- (k) integrity checks ---")
    for k, v in res["integrity"].items():
        print(f"  {k}: {v}")
    print(f"  pool facts: {res['facts']}")

    rq1 = res["rq1"]
    cor = Y["correlation_design"]
    print("\n--- (d) RQ1 resampling: 2,000 draws of 397 pool poles with replacement ---")
    print(f"  design-phase within-unit rho {cor['within_unit']['rho']:.3f} "
          f"[{cor['within_unit']['ci95'][0]:.3f}, {cor['within_unit']['ci95'][1]:.3f}], design effect "
          f"{cor['design_effect_m_mean']['value']:.2f} [{cor['design_effect_m_mean']['ci95'][0]:.2f}, "
          f"{cor['design_effect_m_mean']['ci95'][1]:.2f}] at {cor['m_mean_views_per_unit']:.2f} views per unit")
    print("  GTSRB (POC-2 Stage 3b, alpha 1%, HB): frame-level B1 20.0% [18.3, 21.8]%, track-level 1.4% [0.9, 2.0]%")
    for name, r in rq1.items():
        print(f"  {name}: items per draw mean {r['n_mean']:.1f} (range {r['n_range'][0]}-{r['n_range'][1]})")
    print("\n  pool true risk R(j) and pool commit fraction per grid point (whole pool, each item once)")
    for name, r in rq1.items():
        print(f"  {name:>7}: " + " ".join(f"{j}:{pct(x, 0)}/{pct(f, 0)}" for j, (x, f) in
                                           enumerate(zip(r["risk"], r["f_pool"]))))
    print(f"\n  {'alpha':>6} {'method':>7} {'rule':>5} {'no start':>8} {'certified':>9} {'viol':>5} "
          f"{'rate %':>7} {'95% CP':>16} {'mean R %':>9}  certified index: count")
    for a in alphas:
        for name, r in rq1.items():
            for rule in SM.RQ1_RULES:
                c = r["cells"][a][rule]
                if c["no_start"] == c["draws"]:
                    print(f"  {100 * a:5.1f}% {name:>7} {rule:>5}   no start possible in any draw")
                    continue
                idx = ", ".join(f"{j}:{n}" for j, n in c["index_counts"].items())
                print(f"  {100 * a:5.1f}% {name:>7} {rule:>5} {c['no_start']:>8} {c['certified']:>9} "
                      f"{c['violations']:>5} {pct(c['rate'], 7)} {ci(c['cp95']):>16} {pct(c['mean_true_risk'], 9)}  {idx}")

    print(f"\n--- (e) Stage 6 V-check: every score variant at alpha {100 * prim:g}%, exact binomial; "
          f"upper 95% CP <= {SM.V_BOUND} ---")
    for s, v in res["v_check"]["cells"].items():
        print(f"  {s:>7}: certified {v['certified']:>4}/2000, violations {v['violations']:>3}, rate "
              f"{100 * v['rate']:.2f}% {ci(v['cp95'])}, mean true risk {pct(v['mean_true_risk'], 0)}%  "
              f"{'PASS' if v['pass'] else 'FAIL'}")
    print(f"  V-check: {'PASS' if res['v_check']['pass'] else 'FAIL'} (scored as pre-registered, whatever the "
          f"certify counts)")

    rq2 = res["rq2"]
    seeds = list(rq2)
    print("\n--- (f), (g) RQ2 on the 5 frozen re-splits, Test units ---")
    for seed in seeds:
        print(f"  seed {seed}: Cert {rq2[seed]['n_cert']} units, Test {rq2[seed]['n_test']} units; "
              f"B5c D_cal-1/2 {rq2[seed]['b5c_parts']}")
    print("  per cell: certified index j; coverage %; error % [95% CP]; median view; median size px; D90 px")
    for s in SC.SCORES:
        for a in alphas:
            for r in C.RULES:
                print(f"  {s:>7} {100 * a:4.1f}% {r:>5}: " + " | ".join(
                    (f"s{seed} -" if rq2[seed]['scores'][s]['test'][a][r]['j'] is None else
                     f"s{seed} j{m['j']} {pct(m['coverage'], 0)} {pct(m['error'], 0)} {ci(m['cp95'])} "
                     f"v{num(m['median_view'])} {num(m['median_size'])}px D90 {m['d90']}")
                    for seed in seeds for m in [rq2[seed]['scores'][s]['test'][a][r]]))
            m_ = [rq2[seed]["scores"][s]["reference"][a] for seed in seeds]
            print(f"  {s:>7} {100 * a:4.1f}%   ref: " + " | ".join(
                (f"s{seed} -" if m["j"] is None else f"s{seed} j{m['j']} {pct(m['coverage'], 0)} {pct(m['error'], 0)} "
                 f"{ci(m['cp95'])} v{num(m['median_view'])}") for seed, m in zip(seeds, m_)))
    print("\n  B2 (one random view per unit, S4 margin)")
    for a in alphas:
        for r in C.RULES:
            print(f"  {'B2':>7} {100 * a:4.1f}% {r:>5}: " + " | ".join(
                (f"s{seed} -" if m['j'] is None else
                 f"s{seed} j{m['j']} {pct(m['coverage'], 0)} {pct(m['error'], 0)} {ci(m['cp95'])} v{num(m['median_view'])}")
                for seed in seeds for m in [rq2[seed]["b2"]["test"][a][r]]))
    print("\n  B5 / B5c (Ringel; alpha = gap tolerance): lambda; gap %; real error % [95% CP]; median view")
    for key in ("b5", "b5c"):
        for a in alphas:
            print(f"  {key:>7} {100 * a:4.1f}%: " + " | ".join(
                f"s{seed} lam {np.round(np.atleast_1d(m['lambda']), 2).tolist()} gap {pct(m['gap_marginal'], 0)} "
                f"err {pct(m['error_committed'], 0)} {ci(m['cp95'])} v{num(m['median_commit_frame'])}"
                for seed in seeds for m in [rq2[seed][key][a]]))
    print("\n  B6 (uncertified S2, frozen theta): coverage %; error % [95% CP]; median view")
    for v in ("S2-sum", "S2-min"):
        for a in alphas:
            print(f"  {v:>7} {100 * a:4.1f}%: " + " | ".join(
                f"s{seed} {pct(m['coverage'], 0)} {pct(m['error'], 0)} {ci(m['cp95'])} v{num(m['median_view'])}"
                for seed in seeds for m in [rq2[seed]["b6"][v][a]]))

    print("\n--- smallest certifiable alpha per score and rule (seeds 0-4) ---")
    for s in list(SC.SCORES) + ["B2"]:
        for r in C.RULES:
            vals = [(rq2[seed]["scores"][s] if s != "B2" else rq2[seed]["b2"])["smallest_alpha"][r] for seed in seeds]
            print(f"  {s:>7} {r:>5}: " + ", ".join(alpha_txt(x) for x in vals))

    print(f"\n--- (i) comparison table, BelgiumTSC, alpha {100 * prim:g}%: median (min-max; seeds counted) ---")
    for c, row in res["comparison"].items():
        print(f"  [{c}]")
        for k, v in row.items():
            print(f"    {k}: {v}")
    print("  B1 certify counts in the resampling (exact binomial), per alpha: " + ", ".join(
        f"{100 * a:g}%: {rq1['B1']['cells'][a]['binom']['certified']}" for a in alphas))

    print("\n--- (j) S2 against B6: cost of the certificate (headline alpha first) ---")
    order = [prim] + [a for a in alphas if a != prim]
    for v in ("S2-sum", "S2-min"):
        for a in order:
            print(f"  {v} at {100 * a:g}%:")
            for g, r in res["cost"][(v, a)].items():
                print(f"    {g:>28}: {r['seeds']}; coverage {r['coverage']}; median view {r['median_view']}; "
                      f"median size {r['median_size']}; error {r['error']}; worst {r['worst']}; "
                      f"seeds above alpha {r['above_alpha']}")

    print("\n--- Test cells above alpha ---")
    if not res["above"]:
        print("  none")
    for kind, s, seed, a, r, m in res["above"]:
        if kind in ("B5", "B5c"):
            print(f"  {kind} {s} seed {seed} {100 * a:g}%: gap {pct(m['gap_marginal'], 0)}%, real error "
                  f"{pct(m['error_committed'], 0)}% {ci(m['cp95'])}")
        else:
            print(f"  {kind} {s} seed {seed} {100 * a:g}% {r}: error {pct(m['error'], 0)}% ({m['wrong']}/{m['committed']}) "
                  f"{ci(m['cp95'])}, coverage {pct(m['coverage'], 0)}%")


# ------------------------------------------------------------------ main
def run(d, P, Y, yaml_text, design, design_sha, design_blob):
    R = reproduce_yaml(d, design, yaml_text, design_sha, design_blob)
    integ = {"yaml_reproduced_by_design_recomputation": True}
    T = pool_tables(d, P, R, Y)
    integ.update(T["integrity"])
    alphas = [float(a) for a in Y["alphas"]]
    prim = float(Y["primary_alpha"])
    rq1 = run_rq1(T, Y, alphas)
    rq2, i2, s_values = run_rq2(T, Y, alphas)
    integ.update(i2)
    vc = {s: {**rq1[s]["cells"][prim]["binom"], "pass": SM.v_check(rq1[s]["cells"][prim]["binom"])}
          for s in SC.SCORES}
    return {"integrity": integ, "facts": T["facts"], "rq1": rq1, "rq2": rq2, "s_values": s_values,
            "pool_sha256": P["sha256"],
            "v_check": {"cells": vc, "pass": all(v["pass"] for v in vc.values())},
            "comparison": comparison_table(rq1, rq2, Y, prim), "cost": cost_table(rq2, alphas),
            "above": above_alpha(rq2, alphas)}


def to_json(res, Y, blobs, code, d, t0):
    rq2 = {str(s): {**r, "scores": {k: {kk: {str(a): vv for a, vv in v.items()} for kk, v in c.items()}
                                    for k, c in r["scores"].items()},
                    "b2": {k: {str(a): vv for a, vv in v.items()} for k, v in r["b2"].items()},
                    "b5": {str(a): v for a, v in r["b5"].items()}, "b5c": {str(a): v for a, v in r["b5c"].items()},
                    "b6": {k: {str(a): vv for a, vv in v.items()} for k, v in r["b6"].items()}}
           for s, r in res["rq2"].items()}
    return jsonable({
        "plan": "poc3-implementation-plan-r9 Stage 6, BelgiumTSC confirmation; LOG_r9_stage6_entry.md Part A",
        "run_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git("rev-parse", "HEAD"), "worktree_clean": git("status", "--porcelain") == "",
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "frozen_blobs": blobs, "code_sha256": code, "inputs_sha256": d["inputs_sha256"],
        "pool_inputs_sha256": res["pool_sha256"],
        "files_opened": list(DS.OPENED) + list(OPENED), "integrity": res["integrity"], "facts": res["facts"],
        "primary_alpha": Y["primary_alpha"], "delta": C.DELTA, "v_bound": SM.V_BOUND,
        "v_check": res["v_check"],
        "rq1": {k: {"risk": v["risk"], "f_pool": v["f_pool"], "n_mean": v["n_mean"], "n_range": v["n_range"],
                    "cells": {str(a): c for a, c in v["cells"].items()}} for k, v in res["rq1"].items()},
        "rq2": rq2, "d90_s_values": res["s_values"], "comparison": res["comparison"],
        "cost": {f"{v} {a:g}": r for (v, a), r in res["cost"].items()},
        "above_alpha": [{"kind": k, "method": s, "seed": seed, "alpha": a, "rule": r, "metrics": m}
                        for k, s, seed, a, r, m in res["above"]],
        "seconds": round(time.time() - t0, 1)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4", default=str(DS.STAGE4))
    ap.add_argument("--design", default=str(STAGE5_DESIGN))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    t0 = time.time()
    blobs = FZ.check_frozen()                                   # Part A (b): first, before anything is read
    assert YAML_PATH.as_posix() in blobs, "the frozen BelgiumTSC yaml is not among the frozen files"
    out = Path(a.out)
    target = out / "stage6_summary.json"
    if target.exists():
        stop(f"{target} exists; refusing to overwrite a recorded run")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "stage6_printout.txt", "w", encoding="utf-8")
    old, sys.stdout = sys.stdout, Tee(sys.stdout, log)
    try:
        OPENED.clear()
        DS.OPENED.clear()
        print(f"=== R9 Stage 6 BelgiumTSC confirmation | HEAD {git('rev-parse', 'HEAD')} | worktree clean: "
              f"{git('status', '--porcelain') == ''} ===")
        print(f"freeze check PASS: {len(blobs)} files by git blob ID: {sorted(blobs)}")
        Y, yaml_text = load_yaml()
        design, how_design = FR.load_design(a.design, DESIGN_SHA256, DESIGN_BLOB)
        OPENED.append(Path(a.design).as_posix())
        d = DS.load_stage4(a.stage4)
        print(f"input checks PASS: {d['checks']} | stage5_design.json by {how_design}")
        summary = json.loads((Path(a.stage4) / "stage4_summary.json").read_text())
        P = open_pool(a.stage4, summary)
        print(f"pool inputs PASS (SHA-256 against stage4_summary.json): {list(POOL_INPUTS)}")
        res = run(d, P, Y, yaml_text, design, sha256_file(a.design), FZ.blob_id(a.design))
        print(f"files opened: {list(DS.OPENED) + list(OPENED)}")
        report(res, Y, [float(x) for x in Y["alphas"]])
        code = {f: sha256_file(f) for f in CODE}
        target.write_text(json.dumps(to_json(res, Y, blobs, code, d, t0), indent=1) + "\n")
        print(f"\nOutput: {target}  SHA-256 {sha256_file(target)}  git blob {FZ.blob_id(target)}")
        print(f"code SHA-256: {code}")
        print(f"seconds: {time.time() - t0:.1f}")
    finally:
        sys.stdout = old
        log.close()
    return res


if __name__ == "__main__":
    main()
