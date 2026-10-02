"""INDEPENDENT PROOF: are the 'exact copy' pairs byte-identical in the ORIGINAL zips?

Why this exists: everything so far (pHash, resizing, the cache, my pairing code) could hide a
mistake. This script bypasses all of it. For each flagged pair it reads the two raw .ppm
files straight out of the official zips and compares their SHA-256. Identical bytes = the
same image file in both releases, no interpretation needed.

Usage (after inspect_dups.py has written near_dup_pairs_scored.csv):
    py -3.11 scripts/verify_exact_copies.py              # pairs with MAD <= 0.5
    py -3.11 scripts/verify_exact_copies.py --max-mad 5  # also test the gray zone
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import yaml


def _index_train(zf):
    """(class folder number, filename) -> full path inside the training zip."""
    idx = {}
    for n in zf.namelist():
        parts = n.split("/")
        if n.lower().endswith(".ppm") and len(parts) >= 2 and parts[-2].isdigit():
            idx[(int(parts[-2]), parts[-1])] = n
    return idx


def _index_test(zf):
    """filename -> full path inside the test zip."""
    return {n.rsplit("/", 1)[-1]: n for n in zf.namelist() if n.lower().endswith(".ppm")}


def compare_pairs(train_zip, test_zip, train_meta, test_meta, pairs):
    """pairs: DataFrame with train_idx/test_idx (row positions in the meta tables).
    Returns one row per pair with both file paths and whether the raw bytes are identical."""
    rows = []
    with zipfile.ZipFile(train_zip) as ztr, zipfile.ZipFile(test_zip) as zte:
        itr, ite = _index_train(ztr), _index_test(zte)
        for r in pairs.itertuples():
            tr = train_meta.iloc[int(r.train_idx)]
            te = test_meta.iloc[int(r.test_idx)]
            p_tr = itr[(int(tr["class_folder"]), tr["filename"])]
            p_te = ite[te["filename"]]
            b_tr, b_te = ztr.read(p_tr), zte.read(p_te)
            rows.append(dict(train_file=p_tr, test_file=p_te,
                             bytes_train=len(b_tr), bytes_test=len(b_te),
                             identical=hashlib.sha256(b_tr).digest() == hashlib.sha256(b_te).digest(),
                             mad=float(r.mad)))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--max-mad", type=float, default=0.5)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    d = cfg["data"]
    raw, cache = Path(d["raw_dir"]), Path(d["cache_dir"])
    scored = pd.read_csv(cache / "near_dup_pairs_scored.csv")
    pairs = scored[scored["mad"] <= a.max_mad].sort_values("mad")
    res = compare_pairs(raw / d["train_zip"], raw / d["test_zip"],
                        pd.read_csv(cache / "train_meta.csv"), pd.read_csv(cache / "test_meta.csv"), pairs)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(res.to_string(index=False))
    print(f"\n{int(res['identical'].sum())} of {len(res)} pairs (MAD <= {a.max_mad}) are BYTE-IDENTICAL files.")


if __name__ == "__main__":
    main()
