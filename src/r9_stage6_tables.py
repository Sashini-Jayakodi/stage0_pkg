"""POC-3 (R9) Stage 6: rebuild the plan section 8 comparison table at any alpha from the recorded
runs/btsc/stage6/stage6_summary.json (LOG_r9_stage6_entry.md Part B, deviation 2).

The runner printed the table at the primary alpha only. This script reads the hash-checked summary, turns its JSON
keys back into the runner's structures and calls the runner's own comparison_table, so no number is recomputed from
data. tests/test_r9_stage6_tables.py checks that the 3% table it prints equals the printed one line for line.

Usage:  py -3.12 -m src.r9_stage6_tables --alpha 0.01
"""
import argparse
import json
from pathlib import Path

import yaml

from src import r9_frozen as FZ
from src import r9_stage6 as R6
from src.r9_stage4_data import stop
from src.util import sha256_file

SUMMARY = Path("runs/btsc/stage6/stage6_summary.json")
SUMMARY_SHA256 = "71be795779554a62546b1e03089222b53c35bc5a8ffac4bb2406c470b39e185f"   # Part B
SUMMARY_BLOB = "28ceaa6ae6cdd8964b0018510612fe2840dab3e2"


def load_summary(path=SUMMARY, sha=SUMMARY_SHA256, blob=SUMMARY_BLOB):
    """Hash-check the recorded summary (SHA-256 of these bytes, or its git blob ID after CRLF -> LF)."""
    path = Path(path)
    if not path.is_file():
        stop(f"{path} is missing")
    if sha256_file(path) != sha and FZ.blob_id(path) != blob:
        stop(f"{path} differs from its recorded SHA-256 and git blob ID")
    return json.loads(path.read_text())


def _cert(d):
    """Certified indices per alpha and rule; the JSON stores -1 or null for 'none'."""
    return {float(a): {r: (-1 if j is None else j) for r, j in v.items()} for a, v in d.items()}


def _by_alpha(d):
    return {float(k): v for k, v in d.items()}


def rebuild(J):
    """rq1 and rq2 in the shapes R6.comparison_table expects (alpha keys as floats, seeds as ints)."""
    rq1 = {m: {"cells": _by_alpha(v["cells"])} for m, v in J["rq1"].items()}
    rq2 = {}
    for s, r in J["rq2"].items():
        scores = {name: {"cert": _cert(c["cert"]), "test": _by_alpha(c["test"]),
                         "reference": _by_alpha(c["reference"]),
                         "smallest_alpha": {ru: (None if x is None else float(x))
                                            for ru, x in c["smallest_alpha"].items()}}
                  for name, c in r["scores"].items()}
        b2 = {"cert": _cert(r["b2"]["cert"]), "test": _by_alpha(r["b2"]["test"]),
              "smallest_alpha": {ru: (None if x is None else float(x)) for ru, x in r["b2"]["smallest_alpha"].items()}}
        rq2[int(s)] = {"scores": scores, "b2": b2, "b5": _by_alpha(r["b5"]), "b5c": _by_alpha(r["b5c"]),
                       "b6": {v: _by_alpha(x) for v, x in r["b6"].items()}}
    return rq1, rq2


def table_lines(J, Y, alpha):
    """The comparison table at alpha in the runner's printout format (one '  [col]' line, then '    row: value')."""
    rq1, rq2 = rebuild(J)
    lines = []
    for c, row in R6.comparison_table(rq1, rq2, Y, alpha).items():
        lines.append(f"  [{c}]")
        lines += [f"    {k}: {v}" for k, v in row.items()]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--summary", default=str(SUMMARY))
    a = ap.parse_args(argv)
    FZ.check_frozen()                                           # the yaml read below is the frozen one
    J = load_summary(a.summary)
    Y = yaml.safe_load(R6.YAML_PATH.read_text())
    if f"{a.alpha:g}" not in Y["alphas"]:
        stop(f"alpha {a.alpha:g} is not in the frozen alpha grid {Y['alphas']}")
    print(f"--- comparison table, BelgiumTSC, alpha {100 * a.alpha:g}% (rebuilt from {a.summary}) ---")
    print("\n".join(table_lines(J, Y, a.alpha)))


if __name__ == "__main__":
    main()
