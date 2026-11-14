"""R9 Stage 1: BelgiumTSC structure audit (Part A (c) to (g) of LOG_r9_stage1_entry.md, with Amendment 1).

Reads only file names, the GT csv files, the readmes and (for check 2) image bytes. No model, no embedding, no
prediction, no design/pool split: Stage 1 is a design-phase stage and must not see any Cert or Test outcome.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom

from src.r9_stage1_download import sha256_file, utc_now
from src.r9_stage1_integrity import check_hash

MANIFEST = Path("runs/btsc/stage1/downloads_manifest.json")
ZIP_DIR = Path("data/btsc_raw/zips")
EXTRACT_DIR = Path("data/btsc_raw/extracted")
OUT = Path("runs/btsc/stage1/stage1_summary.json")
RELEASES = {"training": "BelgiumTSC_Training", "testing": "BelgiumTSC_Testing"}
NAME_RE = re.compile(r"^(\d{5})_(\d{5})\.ppm$")           # Part A (d)
GT_COLS = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]
F_A1_MIN_POLES = 600                                       # plan R9 section 9, F-A1
DELTA = 0.1                                                # plan R9 section 2
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)                   # plan R9 section 4
CERT_SHARE = 0.6                                           # Part A (g): Cert = 75% of the 80% pool
COMMIT_FRACTION = 0.7                                      # Part A (g)
DITS_URLS = ("https://users.diag.uniroma1.it/bloisi/ds/dits.html",
             "http://users.diag.uniroma1.it/bloisi/ds/dits.html")  # Part A (f): HTTPS first, then HTTP


# ---------- input checks (hard rule 2) ----------
def verify_extracted(zip_path, extracted_dir):
    """Refuse an extracted tree that differs from its hash-checked zip: missing, extra or changed files."""
    with zipfile.ZipFile(zip_path) as z:
        members = {m.filename: m.CRC for m in z.infolist() if not m.is_dir()}
    on_disk = {p.relative_to(extracted_dir).as_posix(): p
               for p in Path(extracted_dir).rglob("*") if p.is_file()}
    if set(on_disk) != set(members):
        missing = sorted(set(members) - set(on_disk))[:5]
        extra = sorted(set(on_disk) - set(members))[:5]
        raise SystemExit(f"{extracted_dir}: file set differs from zip (missing {missing}, extra {extra}); stopping")
    for name, path in on_disk.items():
        if zlib.crc32(path.read_bytes()) & 0xFFFFFFFF != members[name]:
            raise SystemExit(f"{path}: content differs from zip member; stopping")


# ---------- reading one release ----------
def read_release(release_dir):
    """Return views (one dict per GT row with a parsed name), consistency failures, GT and readme hashes."""
    release_dir = Path(release_dir)
    class_dirs = sorted(p for p in release_dir.iterdir() if p.is_dir())
    views, fails, gt_sha, n_images = [], defaultdict(list), {}, 0
    for cdir in class_dirs:
        gts = sorted(cdir.glob("GT-*.csv"))
        images = {p.name for p in cdir.iterdir() if p.is_file() and p.suffix.lower() == ".ppm"}
        n_images += len(images)
        if len(gts) != 1:
            fails["gt_file_count_not_1"].append(cdir.name)
            continue
        gt_sha[gts[0].relative_to(release_dir).as_posix()] = sha256_file(gts[0])
        with open(gts[0], newline="") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        if rows and list(rows[0].keys()) != GT_COLS:
            fails["gt_columns"].append(cdir.name)
            continue
        seen = Counter(r["Filename"] for r in rows)
        for name, k in seen.items():
            if k > 1:
                fails["image_with_more_than_one_row"].append(f"{cdir.name}/{name}")
            if name not in images:
                fails["row_without_image"].append(f"{cdir.name}/{name}")
        for name in sorted(images - set(seen)):
            fails["image_without_row"].append(f"{cdir.name}/{name}")
        for r in rows:
            m = NAME_RE.match(r["Filename"])
            if not m:
                fails["name_does_not_parse"].append(f"{cdir.name}/{r['Filename']}")
                continue
            cls = int(r["ClassId"])
            if cls != int(cdir.name):
                fails["class_folder_ne_classid"].append(f"{cdir.name}/{r['Filename']}")
            if r["Filename"] not in images:
                continue
            x1, y1, x2, y2 = (int(r[c]) for c in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"))
            views.append({"folder": cdir.name, "file": r["Filename"], "pole": int(m.group(1)),
                          "view": int(m.group(2)), "cls": cls, "w": int(r["Width"]), "h": int(r["Height"]),
                          "roi": (x1, y1, x2, y2), "size": min(x2 - x1, y2 - y1),
                          "path": cdir / r["Filename"]})
    # Part A (d): no (pole, view) pair twice inside one class folder.
    pv = Counter((v["folder"], v["pole"], v["view"]) for v in views)
    fails["pole_view_pair_repeated_in_folder"] += [f"{k[0]}/{k[1]:05d}_{k[2]:05d}" for k, n in pv.items() if n > 1]
    readme = release_dir / "Readme.txt"
    readme_sha = sha256_file(readme) if readme.exists() else None
    return {"class_dirs": [p.name for p in class_dirs], "n_images": n_images, "views": views, "fails": dict(fails),
            "gt_sha256": gt_sha, "readme_sha256": readme_sha}


# ---------- counts, Part A (c) ----------
def release_counts(rel):
    views = rel["views"]
    poles = {v["pole"] for v in views}
    units = Counter((v["pole"], v["cls"]) for v in views)
    classes_per_pole = defaultdict(set)
    for p, c in units:
        classes_per_pole[p].add(c)
    multi = [p for p, cs in classes_per_pole.items() if len(cs) > 1]
    vpu = Counter(min(n, 4) for n in units.values())
    vpp = list(Counter(v["pole"] for v in views).values())
    sizes = np.array([v["size"] for v in views]) if views else np.array([0])
    return {
        "images": rel["n_images"],
        "class_folders": len(rel["class_dirs"]),
        "distinct_class_ids": len({v["cls"] for v in views}),
        "pole_numbers": len(poles),
        "sign_units": len(units),
        "views_per_sign_unit": {"1": vpu[1], "2": vpu[2], "3": vpu[3], "4+": vpu[4]},
        "sign_units_per_class": {str(c): n for c, n in sorted(Counter(c for _, c in units).items())},
        "poles_with_more_than_one_class": len(multi),
        "extra_sign_units_from_multi_class_poles": len(units) - len(poles),
        # Amendment 2: views per pole as median and range, no pass/fail.
        "views_per_pole_median": float(np.median(vpp)) if vpp else 0,
        "views_per_pole_min": min(vpp) if vpp else 0,
        "views_per_pole_max": max(vpp) if vpp else 0,
        "roi_shorter_side_quantiles": {str(q): float(np.quantile(sizes, q))
                                        for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)},
        "consistency_failures": {k: len(v) for k, v in rel["fails"].items()},
        "consistency_failure_examples": {k: v[:10] for k, v in rel["fails"].items() if v},
    }


# ---------- shared units, Part A (e) with Amendment rules ----------
def image_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_units(train_views, test_views, check1_applies):
    train_units = {(v["pole"], v["cls"]) for v in train_views}
    train_bytes = {image_sha(v["path"]) for v in train_views}
    train_meta = {(v["cls"], v["w"], v["h"], v["roi"]) for v in train_views}
    test_units = {(v["pole"], v["cls"]) for v in test_views}
    c1 = {u for u in test_units if u in train_units}
    c2 = {(v["pole"], v["cls"]) for v in test_views
          if image_sha(v["path"]) in train_bytes or (v["cls"], v["w"], v["h"], v["roi"]) in train_meta}
    removed = (c1 | c2) if check1_applies else c2
    return {"check1_applies": check1_applies,
            "check1_only": len(c1 - c2), "check2_only": len(c2 - c1), "both": len(c1 & c2),
            "check1_total": len(c1), "check2_total": len(c2), "removed": len(removed),
            "removed_units": sorted(removed)}


# ---------- gates, Part A (f) ----------
def gates(test_views, removed_units, poles_recovered):
    kept = {(v["pole"], v["cls"]) for v in test_views} - set(removed_units)
    poles_after = len({p for p, _ in kept})          # F-A1 counts poles, not sign units
    fa1 = "PASS" if poles_recovered and poles_after >= F_A1_MIN_POLES else "FAIL"
    return {"test_poles_after_removal": poles_after, "test_sign_units_after_removal": len(kept),
            "F-A1": fa1, "F-A2": "PASS" if not removed_units else f"shared units removed ({len(removed_units)})"}


# ---------- n_c projection, Part A (g) ----------
def kmax(n, alpha, delta=DELTA):
    """Largest k with P(Bin(n, alpha) <= k) <= delta; -1 means not certifiable even with 0 errors."""
    k = -1
    while k + 1 <= n and binom.cdf(k + 1, n, alpha) <= delta:
        k += 1
    return k


def projection(test_poles):
    cert = int(round(CERT_SHARE * test_poles))
    n_c = int(round(COMMIT_FRACTION * cert))
    return {"test_poles": test_poles, "expected_cert_size": cert, "expected_n_c_at_70pct": n_c,
            "tolerated_errors": {f"{a:g}": kmax(n_c, a) for a in ALPHAS}}


# ---------- F-A3 page retry, Part A (f) ----------
def dits_retry(runner=subprocess.run):
    attempts = []
    for url in DITS_URLS:
        t = utc_now()
        p = runner(["curl", "-sS", "--max-time", "30", "-o", "-", "-w", "\n%{http_code}", url],
                   capture_output=True, text=True)
        code = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        attempts.append({"url": url, "time_utc": t, "curl_exit_code": p.returncode,
                         "http_status": code, "error": p.stderr.strip()})
    ok = any(a["curl_exit_code"] == 0 and a["http_status"].startswith("2") for a in attempts)
    # A reachable page is not yet a pass: any DITS data file needs a Part A amendment before download.
    return {"attempts": attempts, "F-A3": "page reachable; data not downloaded (needs amendment)" if ok else "FAIL"}


# ---------- run ----------
def audit(zip_dir, extract_dir, manifest, check1_applies, dits=None):
    inputs = {"manifest_sha256": sha256_file(manifest)}
    rels = {}
    for key, stem in RELEASES.items():
        zp = Path(zip_dir) / f"{stem}.zip"
        inputs[f"{stem}.zip"] = check_hash(zp, manifest)
        verify_extracted(zp, Path(extract_dir) / stem)
        inner = [p for p in (Path(extract_dir) / stem).iterdir() if p.is_dir()]
        if len(inner) != 1:
            raise SystemExit(f"{stem}: expected one top folder, found {len(inner)}")
        rels[key] = read_release(inner[0])
    counts = {k: release_counts(r) for k, r in rels.items()}
    # Poles are recovered when the readme gives the meaning (recorded) and every file name parses.
    recovered = all(not r["fails"].get("name_does_not_parse") for r in rels.values())
    shared = shared_units(rels["training"]["views"], rels["testing"]["views"], check1_applies)
    g = gates(rels["testing"]["views"], shared["removed_units"], recovered)
    out = {
        "stage": "R9 Stage 1", "run_utc": utc_now(),
        "code_sha256": {f: sha256_file(Path("src") / f) for f in
                        ("r9_stage1_audit.py", "r9_stage1_integrity.py", "r9_stage1_download.py")}
        if Path("src/r9_stage1_audit.py").exists() else {},
        "inputs_sha256": inputs,
        "gt_sha256": {k: r["gt_sha256"] for k, r in rels.items()},
        "readme_sha256": {k: r["readme_sha256"] for k, r in rels.items()},
        "counts": counts, "poles_recovered": recovered,
        "shared": {k: v for k, v in shared.items() if k != "removed_units"},
        "removed_units": [list(u) for u in shared["removed_units"]],
        "gates": g, "projection": projection(g["test_poles_after_removal"]),
        "dits": dits,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check1-applies", required=True, choices=["yes", "no"],
                    help="author's reading of the readme: are pole numbers global across releases?")
    ap.add_argument("--dits-retry", action="store_true", help="run the one F-A3 page retry (Part A (f))")
    a = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite a recorded run")
    dits = dits_retry() if a.dits_retry else None
    out = audit(ZIP_DIR, EXTRACT_DIR, MANIFEST, a.check1_applies == "yes", dits)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "removed_units"}, indent=2))
    print("summary SHA-256:", sha256_file(OUT))


if __name__ == "__main__":
    main()
