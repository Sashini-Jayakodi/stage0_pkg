"""Stage 3 follow-up (read-only, DESCRIPTIVE ONLY): where do the real confusions sit in the epsilon ranking?

The stratification in Stages 4-5 only uses the RANK of eps (hard = high-eps pairs), never its absolute value.
So the question that matters is: do the pairs that really get confused rank high in eps, and how much of all
confusion lies in the high-eps pairs? Nothing here is tuned on or fed back into any later stage.

Usage:  py -3.12 scripts/stage3_pair_ranks.py --dir runs/20261003_stage1/head_emb
Reads pairs_903.csv written by src.head; writes pair_ranks.csv and prints two short tables.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.head import SUBSETS  # noqa: E402

SUBSET_OF = {c: name for name, v in SUBSETS.items() for c in v}


def analyse(pairs, top_n=15, cuts=(20, 50, 100, 200, 451)):
    p = pairs.copy()
    p["eps_rank"] = p["log_eps"].rank(ascending=False, method="min").astype(int)      # 1 = most confusable by eps
    p["same_subset"] = [SUBSET_OF[i] == SUBSET_OF[j] for i, j in zip(p["i"], p["j"])]
    total = p["confusions"].sum()
    emp = p.sort_values(["rate", "confusions"], ascending=False).head(top_n)
    mass = {int(k): float(p.loc[p["eps_rank"] <= k, "confusions"].sum() / total) for k in cuts}
    out = {"total_confusion_events": float(total),
           "share_in_same_subset_pairs": float(p.loc[p["same_subset"], "confusions"].sum() / total),
           "share_of_pairs_that_are_same_subset": float(p["same_subset"].mean()),
           "share_of_confusions_in_top_k_eps_pairs": mass}
    return p, emp, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with pairs_903.csv, e.g. runs/20261003_stage1/head_emb")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    p, emp, out = analyse(pd.read_csv(d / "pairs_903.csv"))
    p.to_csv(d / "pair_ranks.csv", index=False)
    pd.set_option("display.width", 140)
    print("Most-confused pairs in reality, and where eps ranks them (rank 1 = most confusable by eps, of 903):\n")
    print(emp[["i", "j", "name_i", "name_j", "confusions", "rate", "eps_rank", "log_eps", "same_subset"]].round(4).to_string(index=False))
    print(f"\nTotal confusion events: {out['total_confusion_events']:.0f}")
    print(f"Share inside the same Stallkamp subset: {out['share_in_same_subset_pairs']:.1%} "
          f"(same-subset pairs are {out['share_of_pairs_that_are_same_subset']:.1%} of all pairs)")
    print("Share of ALL confusion events that fall in the top-k eps pairs:")
    for k, v in out["share_of_confusions_in_top_k_eps_pairs"].items():
        print(f"   top {k:>3} of 903 pairs by eps: {v:.1%}")
    print("\nDescriptive only. Reading: if the top ~100-450 eps pairs hold most confusion events, stratifying by eps rank can work.")


if __name__ == "__main__":
    main()
