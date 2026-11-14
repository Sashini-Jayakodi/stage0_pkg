"""POC-3 (R9) Stage 4, data part: input checks, view tables, de-duplication, removal, pole split, image cache.

Definitions: LOG_r9_stage4_entry.md Part A (a) to (e). Nothing here reads a prediction; the split uses class counts
only (dataset structure). Every stop raises SystemExit with the reason, as the Stage 1 code does.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.r9_stage1_audit import RELEASES, read_release, verify_extracted
from src.r9_stage1_download import sha256_file
from src.r9_stage1_integrity import check_hash

# Stage 1 Part B values (LOG.md, POC-3 Stage 1 entry); everything else from Stage 1 is read from this summary.
STAGE1_SUMMARY_SHA256 = "77ba6a7042a471fa3320059d85a5ec967a8d30a613355c05c74815d03277a0dd"
MANIFEST_SHA256 = "06c90353f75a8c4133991eb32dad2281d922622cecc0591a1b9b5bc819ae4ca3"
# The only consistency failures Stage 1 found (duplicated GT rows); any other kind means the data changed.
KNOWN_FAILS = {"image_with_more_than_one_row", "pole_view_pair_repeated_in_folder"}
TABLE_COLS = ["relpath", "folder", "file", "pole", "view", "cls", "width", "height",
              "roi_x1", "roi_y1", "roi_x2", "roi_y2", "size", "part"]


def stop(msg):
    raise SystemExit(f"STOP: {msg}")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


@dataclass
class Paths:
    stage1_summary: Path = Path("runs/btsc/stage1/stage1_summary.json")
    manifest: Path = Path("runs/btsc/stage1/downloads_manifest.json")
    zip_dir: Path = Path("data/btsc_raw/zips")
    extract_dir: Path = Path("data/btsc_raw/extracted")
    out: Path = Path("runs/btsc/stage4")


# ---------- (a) inputs, hard rule 2 ----------
def check_inputs(paths, s1_sha=STAGE1_SUMMARY_SHA256, manifest_sha=MANIFEST_SHA256):
    """Hash-check the Stage 1 summary, manifest, zips, extracted trees, GT files and readmes; return what was read."""
    got = sha256_file(paths.stage1_summary)
    if got != s1_sha:
        stop(f"{paths.stage1_summary} SHA-256 {got} differs from the recorded {s1_sha}")
    s1 = json.loads(Path(paths.stage1_summary).read_text())
    got = sha256_file(paths.manifest)
    if got != manifest_sha or got != s1["inputs_sha256"]["manifest_sha256"]:
        stop(f"{paths.manifest} SHA-256 {got} differs from the recorded value")
    rels, hashes = {}, {"stage1_summary.json": s1_sha, "downloads_manifest.json": got}
    for key, stem in RELEASES.items():
        zp = Path(paths.zip_dir) / f"{stem}.zip"
        zsha = check_hash(zp, paths.manifest)            # stops on a manifest mismatch
        if zsha != s1["inputs_sha256"][f"{stem}.zip"]:
            stop(f"{zp.name} SHA-256 differs from the Stage 1 summary")
        verify_extracted(zp, Path(paths.extract_dir) / stem)   # same file set and CRC as the checked zip
        inner = [p for p in (Path(paths.extract_dir) / stem).iterdir() if p.is_dir()]
        if len(inner) != 1:
            stop(f"{stem}: expected one top folder, found {len(inner)}")
        rel = read_release(inner[0])
        if rel["gt_sha256"] != s1["gt_sha256"][key]:
            stop(f"{key}: GT csv hashes differ from the Stage 1 summary")
        if rel["readme_sha256"] != s1["readme_sha256"][key]:
            stop(f"{key}: readme hash differs from the Stage 1 summary")
        if set(rel["fails"]) - KNOWN_FAILS:
            stop(f"{key}: consistency failures not seen in Stage 1: {sorted(set(rel['fails']) - KNOWN_FAILS)}")
        rel["inner"] = inner[0]
        rels[key] = rel
        hashes[f"{stem}.zip"] = zsha
    return s1, rels, hashes


# ---------- (b) views ----------
def dedupe(views):
    """One view per image, keyed by (class folder, file name). Identical duplicate rows collapse to one; duplicates
    that differ in any column stop the run (no rule is chosen after seeing the difference)."""
    first, n_dup = {}, 0
    for v in views:
        key = (v["folder"], v["file"])
        rec = (v["pole"], v["view"], v["cls"], v["w"], v["h"], tuple(v["roi"]))
        if key in first:
            if first[key][1] != rec:
                stop(f"duplicated GT rows of {key[0]}/{key[1]} differ; write an Amendment")
            n_dup += 1
            continue
        first[key] = (v, rec)
    out = sorted((v for v, _ in first.values()), key=lambda v: (v["pole"], v["cls"], v["view"], v["folder"]))
    return out, n_dup


def facts(views):
    return {"views": len(views), "classes": len({v["cls"] for v in views}),
            "poles": len({v["pole"] for v in views}), "units": len({(v["pole"], v["cls"]) for v in views})}


def check_counts(s1, key, views, n_files):
    """V4.1: counts after de-duplication equal the Stage 1 counts."""
    c, f = s1["counts"][key], facts(views)
    want = {"views": c["images"], "classes": c["distinct_class_ids"], "poles": c["pole_numbers"],
            "units": c["sign_units"]}
    if f != want or n_files != c["images"]:
        stop(f"{key}: counts {f} (files {n_files}) differ from Stage 1 {want}")
    return f


def apply_removal(s1, test_views):
    """Drop the Stage 1 removed units (read as stored, never recomputed); check the Stage 1 totals after removal."""
    removed = {(int(p), int(c)) for p, c in s1["removed_units"]}
    if len(removed) != s1["shared"]["removed"] or len(removed) != len(s1["removed_units"]):
        stop("removed-unit list length differs from the Stage 1 count")
    units = {(v["pole"], v["cls"]) for v in test_views}
    if removed - units:
        stop(f"{len(removed - units)} removed units are not testing sign units")
    kept = [v for v in test_views if (v["pole"], v["cls"]) not in removed]
    f = facts(kept)
    if (f["poles"], f["units"]) != (s1["gates"]["test_poles_after_removal"], s1["gates"]["test_sign_units_after_removal"]):
        stop(f"after removal {f} differs from the Stage 1 gate counts")
    return kept, removed


# ---------- (c), (d) split by pole ----------
def round_half_up(num, den):
    return (2 * num + den) // (2 * den)


def rarest_class_strata(units):
    """Stratum of each pole = its class with the fewest units over all given units; ties to the smaller class ID."""
    cnt = Counter(c for _, c in units)
    by_pole = defaultdict(set)
    for p, c in units:
        by_pole[p].add(c)
    return {p: min(cs, key=lambda c: (cnt[c], c)) for p, cs in by_pole.items()}


def split_by_pole(units, frac, seed):
    """Largest-remainder allocation of round(frac x poles) selected poles over rarest-class strata (Part A (c)).
    frac = (num, den) so remainders are exact integers and ties are real ties. Returns the sorted selected poles."""
    num, den = frac
    strata = rarest_class_strata(units)
    by_s = defaultdict(list)
    for p, s in strata.items():
        by_s[s].append(p)
    S = sorted(by_s)
    n_sel = round_half_up(len(strata) * num, den)
    base = {s: len(by_s[s]) * num // den for s in S}
    rem = {s: len(by_s[s]) * num % den for s in S}
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(S))                        # first draw: random rank of each stratum for ties
    extra = n_sel - sum(base.values())
    order = sorted(range(len(S)), key=lambda i: (-rem[S[i]], tie[i]))
    for i in order[:extra]:
        base[S[i]] += 1
    selected = []
    for s in S:                                          # then one permutation per stratum, ascending class ID
        perm = rng.permutation(np.array(sorted(by_s[s]), dtype=np.int64))
        selected.extend(int(p) for p in perm[:base[s]])
    assert len(selected) == n_sel
    return sorted(selected)


def side_tables(views, selected, names=("design", "pool")):
    """Poles, units, views and classes per side, strata and classes found on one side only (counts only)."""
    sel = set(selected)
    units = sorted({(v["pole"], v["cls"]) for v in views})
    strata = rarest_class_strata(units)
    side = lambda p: names[0] if p in sel else names[1]
    out = {}
    for nm in names:
        u = [x for x in units if side(x[0]) == nm]
        out[nm] = {"poles": len({p for p, _ in u}), "units": len(u),
                   "views": sum(1 for v in views if side(v["pole"]) == nm),
                   "classes": len({c for _, c in u}),
                   "units_per_class": {str(c): n for c, n in sorted(Counter(c for _, c in u).items())}}
    st_sides = defaultdict(set)
    st_n = Counter(strata.values())
    for p, s in strata.items():
        st_sides[s].add(side(p))
    cl_sides = defaultdict(set)
    for p, c in units:
        cl_sides[c].add(side(p))
    out["strata"] = len(st_sides)
    out["strata_one_side_only"] = [{"stratum_class": int(s), "poles": st_n[s], "side": next(iter(sd))}
                                   for s, sd in sorted(st_sides.items()) if len(sd) == 1]
    out["classes_one_side_only"] = [{"cls": int(c), "units": sum(1 for _, k in units if k == c), "side": next(iter(sd))}
                                    for c, sd in sorted(cl_sides.items()) if len(sd) == 1]
    return out


# ---------- tables ----------
def view_rows(views, extract_dir, part_of):
    rows = []
    for v in views:
        x1, y1, x2, y2 = v["roi"]
        rows.append({"relpath": Path(v["path"]).relative_to(extract_dir).as_posix(), "folder": v["folder"],
                     "file": v["file"], "pole": v["pole"], "view": v["view"], "cls": v["cls"], "width": v["w"],
                     "height": v["h"], "roi_x1": x1, "roi_y1": y1, "roi_x2": x2, "roi_y2": y2, "size": v["size"],
                     "part": part_of(v["pole"])})
    return rows


def write_table(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLS)
        w.writeheader()
        w.writerows(rows)


def read_table(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in TABLE_COLS:
            if k not in ("relpath", "folder", "file", "part"):
                r[k] = int(r[k])
    return rows


# ---------- (e) cache ----------
def load_whole(path, size):
    """Whole image, no ROI crop, Pillow BILINEAR (antialiased), as POC-1 src.data.load_resized."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))


def build_cache(paths, out_npy, size):
    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8, shape=(len(paths), size, size, 3))
    for i, p in enumerate(paths):
        arr[i] = load_whole(p, size)
    arr.flush()
    del arr


# ---------- summaries ----------
def write_summary(path, obj):
    path = Path(path)
    if path.exists():
        stop(f"{path} exists; refusing to overwrite a recorded run")
    path.write_text(json.dumps(obj, indent=2) + "\n")
    return sha256_file(path)


def check_summary_files(summary_path, base):
    """Hard rule 2: every file a summary lists must still have the SHA-256 the summary recorded."""
    s = json.loads(Path(summary_path).read_text())
    for rel, want in s["files_sha256"].items():
        p = Path(base) / rel
        if not p.is_file():
            stop(f"{p} is missing")
        if sha256_file(p) != want:
            stop(f"{p} changed since {Path(summary_path).name} recorded it")
    return s
