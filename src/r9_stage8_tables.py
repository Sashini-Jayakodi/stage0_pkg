"""POC-3 (R9) Stage 8: the final tables (plan R9 section 9 Stage 8; sections 8 and 12), written to
docs/poc3-results-tables.md.

No new analysis and no new numbers. Every value is read from a committed summary whose SHA-256 (or git blob ID, so a
line-ending-only difference does not stop the run) is checked against its record first. Values are only selected,
formatted, or summarised as median (min-max; count) over the five re-splits with the Stage 6 runner's own helpers
(src/r9_stage6.py). The BelgiumTSC comparison and cost tables are built by the Stage 6 runner's own functions; the
GTSRB cost table is built by the same function from the Stage 3 and POC-2 records. check_frozen() runs first.

Usage:  py -3.12 -m src.r9_stage8_tables [--out docs/poc3-results-tables.md]
"""
import argparse
import json
from pathlib import Path

import yaml

from src import r9_frozen as FZ
from src import r9_stage6 as R6
from src import r9_stage6_tables as TB
from src.r9_certify import DELTA, RULES
from src.r9_stage3_scores import SCORES
from src.r9_stage4_data import stop
from src.util import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("docs/poc3-results-tables.md")
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)
GTSRB_ALPHA = 0.01                    # the alpha at which every score certifies on GTSRB (Stage 3 Part B)
B6_KEYS = {"S2-sum": "ecdf_sum_K5", "S2-min": "ecdf_min_K5"}     # POC-2 B6 variants matching S2-sum / S2-min

# name: (path, recorded SHA-256 or None, git blob ID as committed, where the hash is recorded)
SOURCES = {
    "r9_stage2": ("runs/20261003_stage1/poc3/stage2/stage2_summary.json",
                  "8c152579609bb972b394d8c145746eae2975f7e7c79fae640ba829f46ee3729f",
                  "55b862555b878b9ff707137964f563afabfd0fac", "LOG_r9_stage2_entry.md Part B"),
    "r9_stage3": ("runs/20261003_stage1/poc3/stage3/stage3_summary.json",
                  "aad2c03ff94487bbc4743dbedf6b76a3e51a4167f50a8d3b831ba7360fa052e2",
                  "883a04fcfce8217694e91a0d7cfa49fef96d072c", "LOG_r9_stage3_entry.md Part B"),
    "r9_stage6": (str(TB.SUMMARY).replace("\\", "/"), TB.SUMMARY_SHA256, TB.SUMMARY_BLOB,
                  "LOG_r9_stage6_entry.md Part B"),
    "r9_stage1": ("runs/btsc/stage1/stage1_summary.json",
                  "77ba6a7042a471fa3320059d85a5ec967a8d30a613355c05c74815d03277a0dd",
                  "e9f55095163f3847d454999cc29bc63da27d2f8d", "LOG_r9_stage1_entry.md Part B"),
    "r8_stage3b": ("runs/20261003_stage1/poc2/stage3/stage3b_summary.json",
                   "7591a438b569e7231f3a40d8da6420a81b6f65791e707df4c0101338a0bf7367",
                   "cb43656d30711a4f5826fa751df4a3a59cae34a5", "LOG_r8_stage3_entry.md Part B"),
    "r8_stage4": ("runs/20261003_stage1/poc2/stage4/stage4_summary.json",
                  "e66e45f210ebae5e6d21a4d8247584e1affc2d9691a760427814c821b164d2cc",
                  "28a5785aaebd24a3a7df8f984df32a79b1b38b9d", "LOG_r8_stage4_entry.md Part B"),
    "r8_stage6": ("runs/20261003_stage1/poc2/stage6/stage6_summary.json",
                  "59ef05a43e43963bc3cf5a7911b7d262234005df1d4cc8909f5f5e18e9d1c25c",
                  "87f12b6b869635133a8200184992b261446345a1", "LOG_r8_stage6_entry.md Part B"),
    # POC-1 Stage 5: the LOG leaves its SHA-256 blank, so it is checked by git blob ID as committed in a3e0b48
    "r6_stage5": ("runs/20261003_stage1/splits/stage5_results.json", None,
                  "4060ae3f0400ede5911d5d7f28cb5658d04252aa", "git blob as committed in a3e0b48 (no SHA-256 in LOG.md)"),
    # text copied into the document: the Stage 6 Part B statement (append-only entry, as committed in 4e41ee5)
    "r9_stage6_entry": ("LOG_r9_stage6_entry.md", None, "34242a040227aecb9895e2daf88e89604a8c655c",
                        "git blob as committed in 4e41ee5"),
}
PLAN = "docs/poc3-implementation-plan-r9.md"          # frozen; checked by check_frozen()


# ------------------------------------------------------------------ reading (hard rule 2)
def load_text(name, root=ROOT, sources=None):
    """Text of a recorded source after its SHA-256 or git blob ID is checked; stop otherwise."""
    path, sha, blob, _ = (sources or SOURCES)[name]
    p = Path(root) / path
    if not p.is_file():
        stop(f"{path} is missing")
    if not ((sha is not None and sha256_file(p) == sha) or FZ.blob_id(p) == blob):
        stop(f"{path} differs from its recorded SHA-256 and git blob ID")
    return p.read_text(encoding="utf-8")


def load_all(root=ROOT, sources=None):
    T = {n: load_text(n, root, sources) for n in (sources or SOURCES)}
    J = {n: json.loads(t) for n, t in T.items() if (sources or SOURCES)[n][0].endswith(".json")}
    return T, J


# ------------------------------------------------------------------ line references
def json_line(text, *keys):
    """1-based line of a nested dict key in an indented JSON file (the file's own indent step)."""
    lines = text.replace("\r\n", "\n").split("\n")
    step = len(lines[1]) - len(lines[1].lstrip(" "))
    i, ind = 0, 0
    for k in keys:
        ind += step
        tag = " " * ind + json.dumps(k) + ":"
        while True:
            i += 1
            if i >= len(lines):
                stop(f"key path {keys} not found")
            line = lines[i]
            if line.startswith(tag):
                break
            if line.strip() and len(line) - len(line.lstrip(" ")) < ind:
                stop(f"key path {keys} not found")
    return i + 1


def text_line(text, needle, start=1):
    """1-based line of the first line at or after start that contains needle."""
    lines = text.replace("\r\n", "\n").split("\n")
    for i in range(start - 1, len(lines)):
        if needle in lines[i]:
            return i + 1
    stop(f"'{needle}' not found")


def read_rel(rel, root=ROOT):
    return (Path(root) / rel).read_text(encoding="utf-8")


def ref(path, line):
    return f"`{path}` line {line}"


# ------------------------------------------------------------------ shared formatting (the Stage 6 runner's)
mr, ci, alpha_txt = R6.mr, R6.ci, R6.alpha_txt


def akey(a):
    return f"{a:g}"


def worst_txt(errs, a, prefix=""):
    """Largest recorded error and how many recorded errors exceed alpha (R6.comparison_table wording)."""
    if not errs:
        return "n/a"
    return f"{100 * max(errs):.2f}% vs {prefix}{100 * a:g}% ({sum(e > a + 1e-12 for e in errs)} above)"


def md_cell(x):
    return str(x).replace("|", "\\|")


def md_table(header, rows):
    out = ["| " + " | ".join(md_cell(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(md_cell(c) for c in r) + " |" for r in rows]
    return out


ROWS = ("Type of promise", "Violation rate vs delta (resampling)", "Certified in how many re-splits",
        "Test coverage", "Test error among committed", "Worst re-split error vs target", "Smallest certifiable alpha",
        "Median commit frame (or views)", "Median sign size at commit", "Can escalate")


def comparison_md(T, cols, rows=ROWS, labels=None):
    labels = labels or {}
    return md_table(["Row"] + [labels.get(c, c) for c in cols], [[r] + [T[c][r] for c in cols] for r in rows])


# table 1 mixes the R9 layer with POC-2 records: the headers and a "Source / test" row say which is which
GTSRB_LABELS = {"B1": "B1 (POC-2 record, HB)", "B2": "B2 (POC-2 record, HB)", "B5": "B5 (POC-2 record)",
                "B5c": "B5c (POC-2 record)", "B6 S2-sum": "B6 S2-sum (POC-2 record, Cert-thirds calibration)",
                "B6 S2-min": "B6 S2-min (POC-2 record, Cert-thirds calibration)"}
GTSRB_ROWS = ROWS + ("Source / test",)
GTSRB_SOURCE = {
    "S1": "R9 Stage 3; exact binomial, R9 grid, power-aware start",
    "S2-sum": "R9 Stage 3; exact binomial, R9 grid, power-aware start; S2 calibrated on two design halves",
    "S2-min": "R9 Stage 3; exact binomial, R9 grid, power-aware start; S2 calibrated on two design halves",
    "S3": "R9 Stage 3; exact binomial, R9 grid, power-aware start",
    "S4": "R9 Stage 3; exact binomial, R9 grid, power-aware start",
    "B1": "POC-2 Stage 3b resampling; Hoeffding-Bentkus, frame level",
    "B2": "POC-2 Stage 4; Hoeffding-Bentkus, POC-2 grid",
    "B5": "POC-2 Stage 6; Ringel gap calibration (no real-error test)",
    "B5c": "POC-2 Stage 6; Ringel gap calibration (no real-error test)",
    "B6 S2-sum": "POC-2 Stage 6; uncertified, calibrated on Cert thirds (POC-2 Stage 6 Amendment 1)",
    "B6 S2-min": "POC-2 Stage 6; uncertified, calibrated on Cert thirds (POC-2 Stage 6 Amendment 1)"}


# ------------------------------------------------------------------ Stage 3 cells (float alpha keys, as TB.rebuild)
def stage3_cells(J3):
    """{score: [per-seed cell]} with cert/test/reference keyed by float alpha; seeds in order 0-4."""
    out = {}
    for s in SCORES:
        out[s] = []
        for seed in SEEDS:
            c = J3["results"][s][str(seed)]
            out[s].append({"cert": {float(a): v for a, v in c["cert"].items()},
                           "test": {float(a): v for a, v in c["test"].items()},
                           "reference": {float(a): v for a, v in c["reference"].items()},
                           "smallest_alpha": c["smallest_alpha"]})
    return out


# ------------------------------------------------------------------ table 1: GTSRB comparison (exploratory)
def gtsrb_layer_column(cells, a):
    cs = [c for c in cells if c["cert"][a]["binom"] >= 0]
    ms = [c["test"][a]["binom"] for c in cs]
    sa = [c["smallest_alpha"]["binom"] for c in cells]
    return {"Type of promise": "certified real error among committed signs (exact binomial)",
            "Violation rate vs delta (resampling)": "not measured in R9 Stage 3 (see table 3)",
            "Certified in how many re-splits": f"{len(cs)}/{len(cells)}",
            "Test coverage": mr([m["coverage"] for m in ms]),
            "Test error among committed": mr([m["error"] for m in ms]),
            "Worst re-split error vs target": worst_txt([m["error"] for m in ms if m["error"] is not None], a),
            "Smallest certifiable alpha": mr(sa) + " | per seed " + ", ".join(alpha_txt(x) for x in sa),
            "Median commit frame (or views)": mr([m["median_frame"] for m in ms], as_pct=False),
            "Median sign size at commit": mr([m["median_size"] for m in ms], as_pct=False),
            "Can escalate": "yes"}


def p6_row(P6, a, row, col):
    """A cell of the recorded POC-2 Stage 6 comparison table (main_<alpha>)."""
    rows = P6["tables"][f"main_{akey(a)}"]["rows"]
    return rows[next(k for k in rows if k.split(" ", 1)[1] == row)][col]


def gtsrb_table(J, a=GTSRB_ALPHA):
    J3, B3, P4, P6 = J["r9_stage3"], J["r8_stage3b"], J["r8_stage4"], J["r8_stage6"]
    cells = stage3_cells(J3)
    T = {s: gtsrb_layer_column(cells[s], a) for s in SCORES}
    ak = akey(a)
    na = "frame-level; resampling only (POC-2 Stage 3b)"

    r = B3["results"][ak]["B1"]
    T["B1"] = {k: na for k in ROWS}
    T["B1"].update({"Type of promise": p6_row(P6, a, "Type of promise", "B1"),
                    "Violation rate vs delta (resampling)": f"{100 * r['violation_rate']:.2f}% {ci(r['cp95'])} "
                                                            f"(certified in {100 * r['certified_share']:.1f}% of "
                                                            f"{r['runs']} draws; HB)",
                    "Smallest certifiable alpha": p6_row(P6, a, "Smallest certifiable alpha, n = 784", "B1")
                    + " (POC-2 Stage 6 table; not validity evidence)",
                    "Can escalate": p6_row(P6, a, "Can escalate", "B1")})

    b2 = [row["alphas"][ak]["B2"] for row in P4["per_seed"]]
    cs = [m for m in b2 if m["certified"]]
    sa = [row["smallest_certified"]["B2"] for row in P4["per_seed"]]
    T["B2"] = {"Type of promise": "certified real error among committed signs (one random frame per track; HB, "
                                  "POC-2 grid)",
               "Violation rate vs delta (resampling)": p6_row(P6, a, "Valid on linked frames", "B2"),
               "Certified in how many re-splits": f"{len(cs)}/{len(b2)}",
               "Test coverage": mr([m["coverage"] for m in cs]),
               "Test error among committed": mr([m["error_committed"] for m in cs]),
               "Worst re-split error vs target": worst_txt([m["error_committed"] for m in cs
                                                            if m["error_committed"] is not None], a),
               "Smallest certifiable alpha": mr(sa) + " | per seed " + ", ".join(alpha_txt(x) for x in sa),
               "Median commit frame (or views)": mr([m["median_commit_frame"] for m in cs], as_pct=False),
               "Median sign size at commit": mr([m["median_size"] for m in cs], as_pct=False),
               "Can escalate": p6_row(P6, a, "Can escalate", "B2")}

    for c in ("B5", "B5c"):
        ms = [row[c.lower()][ak] for row in P6["per_seed"]]
        own = P6["stage6_checks"][f"{c}_own_target"]
        errs = [m["error_committed"] for m in ms if m["error_committed"] is not None]
        T[c] = {"Type of promise": p6_row(P6, a, "Type of promise", c),
                "Violation rate vs delta (resampling)": p6_row(P6, a, "Valid on linked frames", c),
                "Certified in how many re-splits": f"no real-error certificate; own gap target met in "
                                                   f"{own['count']}/{len(own['per_seed'])} (POC-2 Stage 6 check)",
                "Test coverage": mr([m["coverage"] for m in ms]),
                "Test error among committed": mr(errs) + " (real error)",
                "Worst re-split error vs target": f"gap {100 * max(m['gap_marginal'] for m in ms):.2f}% vs tolerance "
                                                  f"{100 * a:g}%; real error {100 * max(errs):.2f}% "
                                                  f"({sum(e > a + 1e-12 for e in errs)} above {100 * a:g}%)",
                "Smallest certifiable alpha": p6_row(P6, a, "Smallest certifiable alpha, n = 784", c),
                "Median commit frame (or views)": mr([m["median_commit_frame"] for m in ms], as_pct=False),
                "Median sign size at commit": mr([m["median_size"] for m in ms], as_pct=False),
                "Can escalate": p6_row(P6, a, "Can escalate", c)}

    for v, key in B6_KEYS.items():
        ms = [row["b6"][key]["alphas"][ak] for row in P6["per_seed"]]
        errs = [m["error_committed"] for m in ms if m["error_committed"] is not None]
        T[f"B6 {v}"] = {"Type of promise": "none (uncertified; target r = alpha; POC-2 calibration on Cert thirds)",
                        "Violation rate vs delta (resampling)": p6_row(P6, a, "Valid on linked frames", f"B6 {key}"),
                        "Certified in how many re-splits": "no certificate",
                        "Test coverage": mr([m["coverage"] for m in ms]),
                        "Test error among committed": mr([m["error_committed"] for m in ms]),
                        "Worst re-split error vs target": worst_txt(errs, a, "r = "),
                        "Smallest certifiable alpha": "none (no certificate)",
                        "Median commit frame (or views)": mr([m["median_commit_frame"] for m in ms], as_pct=False),
                        "Median sign size at commit": mr([m["median_size"] for m in ms], as_pct=False),
                        "Can escalate": p6_row(P6, a, "Can escalate", f"B6 {key}")}
    for c in T:
        T[c]["Source / test"] = GTSRB_SOURCE[c]
    return T


# ------------------------------------------------------------------ table 2: BelgiumTSC comparison (Stage 6 runner)
def btsc_table(J, Y, a):
    rq1, rq2 = TB.rebuild(J["r9_stage6"])
    return R6.comparison_table(rq1, rq2, Y, a)


# ------------------------------------------------------------------ table 3: RQ1 validity
def rate_txt(r, share=None):
    tail = f"certified share {100 * share:.1f}%" if share is not None else f"{r['certified']} certified"
    rate = r["violation_rate"] if "violation_rate" in r else r["rate"]
    return f"{100 * rate:.2f}% {ci(r['cp95'])} ({tail})"


def rq1_gtsrb_rows(B3):
    out = []
    for ak in sorted(B3["results"], key=float):
        r = B3["results"][ak]
        out.append([f"{100 * float(ak):g}%", rate_txt(r["B1"], r["B1"]["certified_share"]),
                    rate_txt(r["track"], r["track"]["certified_share"])])
    return out


CLUSTERS = ("fixed-3", "view-mix", "fixed-30")
SIM_ROWS = (("sign_binom", "Sign-level, exact binomial"), ("B1_binom", "Frame/view-level B1, exact binomial"),
            ("sign_hb", "Sign-level, HB (primary start)"), ("B1_hb", "Frame/view-level B1, HB (primary start)"))


def sim_settings(J2):
    return [s for p in J2["pairs"] for s in p["settings"]]


def sim_worst(settings, cluster, row):
    """Setting with the largest recorded violation rate for one cluster setting and row; ties counted."""
    ss = [s for s in settings if s["cluster"] == cluster]
    top = max(s["rows"][row]["violation_rate"] for s in ss)
    hit = [s for s in ss if s["rows"][row]["violation_rate"] == top]
    s = hit[0]
    r = s["rows"][row]
    tie = f"; {len(hit)} settings at this rate" if len(hit) > 1 else ""
    return (f"{100 * r['violation_rate']:.2f}% {ci(r['cp95'])} at P_FRAME {100 * s['p_frame']:g}%, rho {s['rho']:g}, "
            f"alpha {100 * s['alpha']:g}%{tie}")


def rq1_sim_rows(J2):
    settings = sim_settings(J2)
    rows = []
    for cl in CLUSTERS:
        n = sum(s["cluster"] == cl for s in settings)
        rows.append([cl, str(n)] + [sim_worst(settings, cl, k) for k, _ in SIM_ROWS])
    return rows


def rq1_btsc_rows(J6, alphas=ALPHAS):
    methods = list(J6["rq1"])
    rows = []
    for a in alphas:
        row = [f"{100 * a:g}%"]
        for m in methods:
            parts = []
            for rule in ("binom", "hb"):
                c = J6["rq1"][m]["cells"][str(a)][rule]
                parts.append("no start" if c["no_start"] == c["draws"] else rate_txt(c))
            row.append(" / ".join(parts))
        rows.append(row)
    return methods, rows


# ------------------------------------------------------------------ table 4: exact binomial against HB
def binom_hb_rows(cells_by_score, alphas=ALPHAS):
    """Per score and alpha: certified re-splits and Test coverage (certified seeds) under each rule."""
    rows = []
    for s, cells in cells_by_score.items():
        for a in alphas:
            row = [s, f"{100 * a:g}%"]
            for r in RULES:
                cs = [c for c in cells if c["cert"][a][r] >= 0]
                row.append(f"{len(cs)}/{len(cells)}; {mr([c['test'][a][r]['coverage'] for c in cs])}")
            rows.append(row)
    return rows


def smallest_rows(cells_by_score):
    return [[s] + [", ".join(alpha_txt(None if c["smallest_alpha"][r] is None else float(c["smallest_alpha"][r]))
                             for c in cells) for r in RULES] for s, cells in cells_by_score.items()]


def btsc_cells(J6):
    _, rq2 = TB.rebuild(J6)
    return {s: [rq2[seed]["scores"][s] for seed in SEEDS] for s in SCORES}


# ------------------------------------------------------------------ table 5: S2 against B6
def gtsrb_cost(J, alphas=ALPHAS):
    """R6.cost_table on an rq2-shaped view of the Stage 3 S2 cells and the POC-2 B6 records (keys renamed only)."""
    cells = stage3_cells(J["r9_stage3"])
    P6 = J["r8_stage6"]

    def frame(m):
        return {**m, "median_view": m["median_frame"]}

    def b6(m):
        return {"coverage": m["coverage"], "error": m["error_committed"], "median_view": m["median_commit_frame"],
                "median_size": m["median_size"]}
    rq2 = {}
    for i, seed in enumerate(SEEDS):
        assert P6["per_seed"][i]["seed"] == seed
        rq2[seed] = {"scores": {v: {"cert": cells[v][i]["cert"],
                                    "test": {a: {r: frame(m) for r, m in t.items()} for a, t in cells[v][i]["test"].items()},
                                    "reference": {a: frame(m) for a, m in cells[v][i]["reference"].items()}}
                                for v in B6_KEYS},
                     "b6": {v: {a: b6(P6["per_seed"][i]["b6"][k]["alphas"][akey(a)]) for a in alphas}
                            for v, k in B6_KEYS.items()}}
    return R6.cost_table(rq2, list(alphas))


def btsc_cost(J6, alphas=ALPHAS):
    """The recorded Stage 6 cost table (cost key 'S2-sum 0.03' -> ('S2-sum', 0.03))."""
    return {(v, a): J6["cost"][f"{v} {a:g}"] for v in B6_KEYS for a in alphas}


COST_GROUPS = ("layer (exact binomial)", "B6 (uncertified, r = alpha)", "uncertified reference")


def cost_rows(C, alphas=ALPHAS):
    rows = []
    for v in B6_KEYS:
        for a in alphas:
            for g in COST_GROUPS:
                r = C[(v, a)][g]
                rows.append([v, f"{100 * a:g}%", g, r["seeds"], r["coverage"], r["median_view"], r["median_size"],
                             r["error"], r["worst"], str(r["above_alpha"])])
    return rows


# ------------------------------------------------------------------ table 6: Ringel gap against real error
def ringel_rows(per_seed_b5, alphas=ALPHAS):
    """per_seed_b5: {method: [ {alpha_key: record} per seed ]} with gap_marginal, error_committed, coverage."""
    rows = []
    for c, seeds in per_seed_b5.items():
        for a in alphas:
            ms = [s[akey(a)] for s in seeds]
            errs = [m["error_committed"] for m in ms]
            rows.append([c, f"{100 * a:g}%", ", ".join(f"{100 * m['gap_marginal']:.2f}" for m in ms),
                         ", ".join("n/a" if e is None else f"{100 * e:.2f}" for e in errs),
                         str(sum(e is not None and e > a + 1e-12 for e in errs)),
                         mr([m["coverage"] for m in ms])])
    return rows


def ringel_sources(J):
    P6, J6 = J["r8_stage6"], J["r9_stage6"]
    g = {c: [row[c.lower()] for row in P6["per_seed"]] for c in ("B5", "B5c")}
    b = {c: [J6["rq2"][str(s)][c.lower()] for s in SEEDS] for c in ("B5", "B5c")}
    return g, b


# ------------------------------------------------------------------ table 7: limitations
def plan_limitations(plan_text):
    lines = plan_text.replace("\r\n", "\n").split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("Limitations (written into"))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("**The one result"))
    return [l[2:] for l in lines[i + 1:j] if l.startswith("- ")], i + 1


def paragraph_after(text, heading):
    lines = text.replace("\r\n", "\n").split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith(heading))
    out, j = [], i + 1
    while lines[j].strip():
        out.append(lines[j].strip())
        j += 1
    return " ".join(out), i + 2


def additions(J, Y):
    """POC-3 additions recorded in the Part B entries, with their numbers read from the summaries."""
    J3, J6, J2 = J["r9_stage3"], J["r9_stage6"], J["r9_stage2"]
    g_cells = stage3_cells(J3)
    g_cert = sum(c["cert"][0.005]["binom"] >= 0 for s in SCORES for c in g_cells[s])
    b_cells = btsc_cells(J6)
    b_cert = sum(c["cert"][0.005]["binom"] >= 0 for s in SCORES for c in b_cells[s])
    b_nostart = all(J6["rq1"][s]["cells"]["0.005"]["binom"]["no_start"] == J6["rq1"][s]["cells"]["0.005"]["binom"]["draws"]
                    for s in SCORES)
    ncert = [J6["rq2"][str(s)]["n_cert"] for s in SEEDS]
    prim = float(Y["primary_alpha"])
    flat = {s: sum(c["cert"][prim]["binom"] == 0 for c in b_cells[s]) for s in ("S1", "S3", "S4")}
    s2 = Y["s2"]["S2-sum"]
    fa3 = J["r9_stage1"]["dits"]["F-A3"]
    return [
        f"alpha 0.5% has no exact-binomial start at these Cert sizes: the exact binomial certified 0.5% in {g_cert} of "
        f"{len(SCORES) * len(SEEDS)} (score, re-split) cells on GTSRB (Cert n = {J3['n_cert']}) and {b_cert} of "
        f"{len(SCORES) * len(SEEDS)} on BelgiumTSC (Cert {min(ncert)}-{max(ncert)} units); in the BelgiumTSC resampling "
        f"{'no draw had a start for any score' if b_nostart else 'some draws had a start'}. The cause is the "
        f"power-aware start rule, not the test: the zero-error minimum for 0.5% is n_c = "
        f"{J2['V1']['min_n_engine'][0]} (Stage 2 V1).",
        f"S2 has low resolution on BelgiumTSC: with (a, b) = ({s2['a'] + 0.0:g}, {s2['b']:g}) and {s2['icp_cal_count']} "
        f"ICP calibration scores, its design grid has {len(Y['scores']['S2-sum']['grid'])} points (S2-sum) and "
        f"{len(Y['scores']['S2-min']['grid'])} (S2-min), against {len(Y['scores']['S1']['grid'])} for S1. Its pool "
        f"error at view 1 is {100 * J6['rq1']['S2-sum']['risk'][0]:.2f}% against "
        f"{100 * J6['rq1']['S1']['risk'][0]:.2f}% for S1. Reported as a design-phase finding; S2 was not amended.",
        f"The primary-alpha BelgiumTSC table is flat: at {100 * prim:g}% S1, S3 and S4 certify grid point 0 (commit "
        f"every unit at view 1) in {flat['S1']}, {flat['S3']} and {flat['S4']} of {len(SEEDS)} re-splits, because the "
        f"pool error at view 1 ({100 * J6['rq1']['S1']['risk'][0]:.2f}%) is below the budget. The primary alpha is not "
        f"changed; the 1% table is reported beside it as pre-registered, not scored.",
        f"DITS was not available (F-A3 {fa3}: the download page could not be reached), so Stage 7 was dropped and "
        f"earliness evidence is GTSRB only, labelled exploratory.",
    ]


# ------------------------------------------------------------------ document
def build(root=ROOT, sources=None):
    T, J = load_all(root, sources)
    Y = yaml.safe_load(read_rel(R6.YAML_PATH, root))
    src = sources or SOURCES
    P = {n: v[0] for n, v in src.items()}
    L = {n: read_rel(f, root) for n, f in (
        ("s2", "LOG_r9_stage2_entry.md"), ("s3", "LOG_r9_stage3_entry.md"), ("s6", "LOG_r9_stage6_entry.md"),
        ("s1", "LOG_r9_stage1_entry.md"), ("s5", "LOG_r9_stage5_entry.md"), ("r83", "LOG_r8_stage3_entry.md"),
        ("r84", "LOG_r8_stage4_entry.md"), ("r86", "LOG_r8_stage6_entry.md"), ("log", "LOG.md"))}
    plan_text = read_rel(PLAN, root)
    d = []
    w = d.append

    w("# POC-3 results tables (plan R9, Stage 8)")
    w("")
    w("Generated by `src/r9_stage8_tables.py` from committed, hash-checked summaries; not edited by hand. No value "
      "here is new: each is read from a source below and only selected, formatted, or summarised as median (min-max; "
      "number of re-splits) over the five re-splits with the Stage 6 helpers. delta = "
      f"{DELTA:g} throughout. Percentages are of committed signs unless stated.")
    w("")
    w("Every GTSRB result produced with the POC-3 layer is **exploratory (post-hoc)**: all GTSRB tracks had been "
      "used before. BelgiumTSC results are confirmatory (frozen at F3 before any pool number existed).")
    w("")
    w("## Sources")
    w("")
    w("Each file was checked against the recorded SHA-256 (or, where only a git blob ID is recorded or the bytes "
      "differ by line endings, its git blob ID) before it was read. The frozen files (plan R9, engine, engine tests, "
      "`configs/frozen_r9_btsc.yaml`) were checked by `check_frozen()` first.")
    w("")
    d += md_table(["Source", "File", "SHA-256 (recorded)", "git blob", "Hash recorded in"],
                  [[n, f"`{v[0]}`", v[1] or "none recorded", v[2], v[3]] for n, v in src.items()])
    w("")

    # ---- 1
    a = GTSRB_ALPHA
    t1 = gtsrb_table(J, a)
    cols = list(SCORES) + ["B1", "B2", "B5", "B5c", "B6 S2-sum", "B6 S2-min"]
    w(f"## 1. GTSRB comparison table at alpha {100 * a:g}% (exploratory, post-hoc)")
    w("")
    w(f"alpha {100 * a:g}% is the smallest alpha at which every score certified in every re-split (exact binomial). "
      "Five POC-2 Stage 4 re-splits, Cert 784 / Test 262 tracks.")
    w("")
    d += comparison_md(t1, cols, GTSRB_ROWS, GTSRB_LABELS)
    w("")
    w("Notes. S1 to S4: R9 layer (exact binomial, R9 grids and power-aware starts, S2 calibrated on two design halves). "
      "B1, B2, B5, B5c and B6 are the POC-2 records on the same re-splits: B1 from the Stage 3b resampling (HB); B2 "
      "with the HB test and the POC-2 grid; B6 calibrated on Cert thirds (POC-2 Stage 6 Amendment 1), not on the design "
      "set as S2 is. The 1% result for S1 to S4 is confounded with the changed test, grid and start against POC-2 "
      "(Stage 3 Part B). B1's smallest certifiable alpha is not validity evidence (B1 over-violated in the resampling).")
    w("")
    w("Source: " + "; ".join([
        ref(P["r9_stage3"], json_line(T["r9_stage3"], "results")),
        ref(P["r8_stage3b"], json_line(T["r8_stage3b"], "results", akey(a), "B1")),
        ref(P["r8_stage4"], json_line(T["r8_stage4"], "per_seed")),
        ref(P["r8_stage6"], json_line(T["r8_stage6"], "per_seed")),
        ref(P["r8_stage6"], json_line(T["r8_stage6"], "tables", f"main_{akey(a)}")),
        ref(P["r8_stage6"], json_line(T["r8_stage6"], "stage6_checks"))])
      + ". Recorded in " + "; ".join([
          ref("LOG_r9_stage3_entry.md", text_line(L["s3"], "#### Test metrics, exact binomial")),
          ref("LOG_r8_stage6_entry.md", text_line(L["r86"], "### Part B"))]) + ".")
    w("")

    # ---- 2
    prim = float(Y["primary_alpha"])
    stmt, stmt_line = paragraph_after(T["r9_stage6_entry"], "#### Comparison table (i)")
    bcols = list(R6.COLS)
    w(f"## 2. BelgiumTSC comparison table at the pre-registered primary alpha {100 * prim:g}% (confirmatory)")
    w("")
    w(f"Statement recorded in Stage 6 Part B ({ref('LOG_r9_stage6_entry.md', stmt_line)}):")
    w("")
    w("> " + stmt)
    w("")
    d += comparison_md(btsc_table(J, Y, prim), bcols)
    w("")
    w("B1 certify counts in the resampling (exact binomial): " + ", ".join(
        f"{100 * x:g}%: {J['r9_stage6']['rq1']['B1']['cells'][str(x)]['binom']['certified']}" for x in ALPHAS) + ".")
    w("")
    w("### 2b. BelgiumTSC comparison table at alpha 1% (pre-registered, reported, not the primary alpha)")
    w("")
    d += comparison_md(btsc_table(J, Y, 0.01), bcols)
    w("")
    w("Source: " + "; ".join([ref(P["r9_stage6"], json_line(T["r9_stage6"], "rq1")),
                              ref(P["r9_stage6"], json_line(T["r9_stage6"], "rq2")),
                              ref(P["r9_stage6"], json_line(T["r9_stage6"], "comparison")) + " (the 3% table as printed)",
                              f"`{R6.YAML_PATH.as_posix()}` (frozen)"])
      + ". Built with `src.r9_stage6.comparison_table`. Recorded in " + "; ".join([
          ref("LOG_r9_stage6_entry.md", text_line(L["s6"], "#### Comparison table (i)")),
          ref("LOG_r9_stage6_entry.md", text_line(L["s6"], "#### Comparison table at 1%"))]) + ".")
    w("")

    # ---- 3
    B3, R5, J2, J6 = J["r8_stage3b"], J["r6_stage5"]["dataset_rho_all_frames"], J["r9_stage2"], J["r9_stage6"]
    cor = Y["correlation_design"]
    w("## 3. RQ1: frame/view-level against sign-level violation rates (budget delta = "
      f"{100 * DELTA:g}%)")
    w("")
    w("Violation rate = share of resampled calibration draws in which the certified threshold's true risk is above "
      "alpha; [two-sided 95% Clopper-Pearson].")
    w("")
    w(f"### 3a. GTSRB, real data (POC-2 Stage 3b; {B3['config']['draws']} draws of {B3['config']['n_cert']} pool "
      "tracks; Hoeffding-Bentkus)")
    w("")
    w(f"Within-track error correlation over all {R5['frames']} frames in {R5['tracks']} tracks: rho "
      f"{R5['rho']:.3f} [{R5['rho_ci95'][0]:.3f}, {R5['rho_ci95'][1]:.3f}], design effect (m = 30) "
      f"{R5['design_effect_m30']:.1f} (POC-1 Stage 5).")
    w("")
    d += md_table(["alpha", "Frame-level B1", "Track-level (sign)"], rq1_gtsrb_rows(B3))
    w("")
    w("Source: " + "; ".join([ref(P["r8_stage3b"], json_line(T["r8_stage3b"], "results")),
                              ref(P["r6_stage5"], json_line(T["r6_stage5"], "dataset_rho_all_frames"))])
      + ". Recorded in " + "; ".join([ref("LOG_r8_stage3_entry.md", text_line(L["r83"], "### Part B")),
                                     ref("LOG.md", text_line(L["log"], "### rho and design effect"))]) + ".")
    w("")
    w(f"### 3b. Simulation (R9 Stage 2; {J2['config']['runs']} runs per setting; worst case per cluster setting)")
    w("")
    w(f"Settings: error levels P_FRAME {', '.join(f'{100 * x:g}%' for x in J2['config']['levels'].values())}; "
      "rho in {0, 0.3, 0.45}; alpha in {" + ", ".join(f"{100 * x:g}%" for x in J2["config"]["alphas"]) + "}. rho is "
      "the simulation setting; no design effect is recorded for the simulation. V2 (sign-level, exact binomial, upper "
      f"95% CP <= delta + 0.02): {'PASS' if J2['V2']['pass'] else 'FAIL'}.")
    w("")
    d += md_table(["Cluster setting", "Settings"] + [lab for _, lab in SIM_ROWS], rq1_sim_rows(J2))
    w("")
    w("Source: " + ref(P["r9_stage2"], json_line(T["r9_stage2"], "pairs")) + " (each pair's `settings`). Recorded in "
      + ref("LOG_r9_stage2_entry.md", text_line(L["s2"], "### Part B")) + ".")
    w("")
    m, rows = rq1_btsc_rows(J6)
    w(f"### 3c. BelgiumTSC (R9 Stage 6; {Y['rq1']['draws']} draws of {Y['rq1']['poles_per_draw']} pool poles)")
    w("")
    w(f"Design-phase within-unit rho {cor['within_unit']['rho']:.3f} [{cor['within_unit']['ci95'][0]:.3f}, "
      f"{cor['within_unit']['ci95'][1]:.3f}], design effect {cor['design_effect_m_mean']['value']:.2f} "
      f"[{cor['design_effect_m_mean']['ci95'][0]:.2f}, {cor['design_effect_m_mean']['ci95'][1]:.2f}] at "
      f"{cor['m_mean_views_per_unit']:.2f} views per unit. Each cell: exact binomial / HB (primary start); count = "
      "certified draws.")
    w("")
    d += md_table(["alpha"] + [x if x != "B1" else "B1 (views)" for x in m], rows)
    w("")
    w("Source: " + "; ".join([ref(P["r9_stage6"], json_line(T["r9_stage6"], "rq1")),
                              f"`{R6.YAML_PATH.as_posix()}` `correlation_design`"])
      + ". Recorded in " + ref("LOG_r9_stage6_entry.md", text_line(L["s6"], "#### RQ1 (d)")) + ".")
    w("")

    # ---- 4
    g_cells, b_cells = stage3_cells(J["r9_stage3"]), btsc_cells(J6)
    head = ["Score", "alpha"] + [f"{r}: certified; Test coverage" for r in RULES]
    w("## 4. Exact binomial against Hoeffding-Bentkus (all three HB starts)")
    w("")
    w("binom = exact binomial (primary); hb = HB primary start (Stage 2 Amendment 2); hb_a1 = Amendment 1 start; "
      "hb_bs = the original start. Coverage is over the certified re-splits.")
    w("")
    w("### 4a. GTSRB (exploratory, post-hoc)")
    w("")
    d += md_table(head, binom_hb_rows(g_cells))
    w("")
    w("Smallest certifiable alpha per re-split 0-4:")
    w("")
    d += md_table(["Score"] + list(RULES), smallest_rows(g_cells))
    w("")
    w("At 0.5% the exact binomial has no start at n = 784 (Stage 3 Amendment 1, "
      + ref("LOG_r9_stage3_entry.md", text_line(L["s3"], "### Amendment 1")) + "); HB at 0.5% is not compared with it.")
    w("")
    w("Source: " + ref(P["r9_stage3"], json_line(T["r9_stage3"], "results")) + ". Recorded in "
      + ref("LOG_r9_stage3_entry.md", text_line(L["s3"], "#### Certification")) + ".")
    w("")
    w("### 4b. BelgiumTSC (confirmatory)")
    w("")
    d += md_table(head, binom_hb_rows(b_cells))
    w("")
    w("Smallest certifiable alpha per re-split 0-4:")
    w("")
    d += md_table(["Score"] + list(RULES), smallest_rows(b_cells))
    w("")
    w("Source: " + ref(P["r9_stage6"], json_line(T["r9_stage6"], "rq2")) + ". Recorded in "
      + ref("LOG_r9_stage6_entry.md", text_line(L["s6"], "#### RQ2 (f), (g)")) + ".")
    w("")

    # ---- 5
    chead = ["Variant", "alpha", "Method", "Re-splits", "Test coverage", "Median commit frame/view", "Median size px",
             "Test error", "Worst", "Re-splits above alpha"]
    w("## 5. S2 against B6: the cost of the certificate, every alpha")
    w("")
    w("Layer = S2 under the exact-binomial certificate (certified re-splits only). Uncertified reference = the same "
      "S2 fit at the loosest grid point whose design error is at most alpha. B6 = Boursinos and Koutsoukos without a "
      "certificate, target r = alpha.")
    w("")
    w("### 5a. GTSRB (exploratory, post-hoc; commit frame out of 30)")
    w("")
    w("Layer and reference from R9 Stage 3; B6 from POC-2 Stage 6 (K = 5, calibrated on Cert thirds and with its own "
      "(a, b), so its fit differs from the Stage 3 S2 fit). Built with `src.r9_stage6.cost_table`.")
    w("")
    d += md_table(chead, cost_rows(gtsrb_cost(J)))
    w("")
    w("Source: " + "; ".join([ref(P["r9_stage3"], json_line(T["r9_stage3"], "results")),
                              ref(P["r8_stage6"], json_line(T["r8_stage6"], "per_seed")) + " (`b6`)"])
      + ". Recorded in " + "; ".join([
          ref("LOG_r9_stage3_entry.md", text_line(L["s3"], "**Cost of the certificate")),
          ref("LOG_r8_stage6_entry.md", text_line(L["r86"], "### Part B"))]) + ".")
    w("")
    w("### 5b. BelgiumTSC (confirmatory; commit view out of at most 3)")
    w("")
    d += md_table(chead, cost_rows(btsc_cost(J6)))
    w("")
    w("Source: " + ref(P["r9_stage6"], json_line(T["r9_stage6"], "cost")) + " (as recorded). Recorded in "
      + ref("LOG_r9_stage6_entry.md", text_line(L["s6"], "#### S2 against B6")) + ".")
    w("")

    # ---- 6
    g, b = ringel_sources(J)
    rhead = ["Method", "alpha (gap tolerance)", "Gap % per re-split 0-4", "Real error % per re-split 0-4",
             "Re-splits with real error above alpha", "Test coverage"]
    w("## 6. Ringel et al. (B5, B5c): accuracy gap against real error")
    w("")
    w("The promise of B5 and B5c is a bound on the accuracy gap to the full-length classifier, not on the real error. "
      "Gap = marginal Test gap; real error = error among committed signs on Test.")
    w("")
    w("### 6a. GTSRB (POC-2 Stage 6)")
    w("")
    d += md_table(rhead, ringel_rows(g))
    w("")
    w("Source: " + ref(P["r8_stage6"], json_line(T["r8_stage6"], "per_seed")) + " (`b5`, `b5c`). Recorded in "
      + ref("LOG_r8_stage6_entry.md", text_line(L["r86"], "### Part B")) + ".")
    w("")
    w("### 6b. BelgiumTSC (R9 Stage 6)")
    w("")
    d += md_table(rhead, ringel_rows(b))
    w("")
    w("Source: " + ref(P["r9_stage6"], json_line(T["r9_stage6"], "rq2")) + " (`b5`, `b5c`). Recorded in "
      + ref("LOG_r9_stage6_entry.md", text_line(L["s6"], "#### B5 and B5c")) + ".")
    w("")

    # ---- 7
    lims, lim_line = plan_limitations(plan_text)
    w("## 7. Limitations")
    w("")
    w(f"From plan R9 section 12 (copied; {ref(PLAN, lim_line)}):")
    w("")
    d += [f"- {x}" for x in lims]
    w("")
    w("Added in POC-3 (recorded in the Part B entries; numbers read from the summaries):")
    w("")
    add_src = [
        ref("LOG_r9_stage3_entry.md", text_line(L["s3"], "### Amendment 1")) + "; "
        + ref("LOG.md", text_line(L["log"], "alpha 0.5% has no",
                                  start=text_line(L["log"], "## POC-3 (R9) Pre-registration block"))),
        ref("LOG.md", text_line(L["log"], "**S2 resolution on BelgiumTSC")) + "; "
        + ref("LOG_r9_stage5_entry.md", text_line(L["s5"], "### Part B")),
        ref("LOG_r9_stage6_entry.md", stmt_line - 1),
        ref("LOG_r9_stage1_entry.md", text_line(L["s1"], "### Part B")) + "; "
        + ref(P["r9_stage1"], json_line(T["r9_stage1"], "dits", "F-A3")),
    ]
    d += [f"- {x} (Source: {s}.)" for x, s in zip(additions(J, Y), add_src)]
    w("")
    return "\n".join(d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    blobs = FZ.check_frozen()                                    # first, before any source is read
    text = build()
    out = ROOT / a.out
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"Freeze check passed: {', '.join(f'{k} {v[:8]}' for k, v in blobs.items())}")
    print(f"Sources hash-checked: {', '.join(SOURCES)}")
    print(f"Wrote {a.out} ({len(text.splitlines())} lines, SHA-256 {sha256_file(out)})")


if __name__ == "__main__":
    main()
