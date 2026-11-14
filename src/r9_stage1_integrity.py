"""R9 Stage 1, Amendment 1 conditions (a) and (b).

(a) zip integrity: SHA-256 against the downloads manifest, then every member's CRC (zipfile.testzip, read-only).
(b) after extraction, count .ppm images only and compare with plan section 7 (4,575 training, 2,520 testing).
No pole, camera or class count is made here (Amendment 1).
"""
import argparse
import json
import zipfile
from pathlib import Path

from src.r9_stage1_download import sha256_file

MANIFEST = Path("runs/btsc/stage1/downloads_manifest.json")
ZIP_DIR = Path("data/btsc_raw/zips")
EXTRACT_DIR = Path("data/btsc_raw/extracted")
EXPECTED_IMAGES = {"BelgiumTSC_Training": 4575, "BelgiumTSC_Testing": 2520}  # plan R9 section 7


def recorded_sha(manifest_path, zip_name):
    # Use the last record for this file name, so a re-download (if ever needed) is the one checked.
    recs = [r for r in json.loads(Path(manifest_path).read_text())["files"] if r["name"] == zip_name]
    if not recs:
        raise SystemExit(f"{zip_name}: no record in {manifest_path}")
    return recs[-1]["sha256"]


def check_hash(zip_path, manifest_path):
    want = recorded_sha(manifest_path, zip_path.name)
    got = sha256_file(zip_path)
    if got != want:
        raise SystemExit(f"{zip_path.name}: SHA-256 {got} differs from manifest {want}; stopping")
    return got


def integrity(zip_path, manifest_path):
    sha = check_hash(zip_path, manifest_path)
    with zipfile.ZipFile(zip_path) as z:
        n_members = len(z.infolist())
        bad = z.testzip()  # reads every member and checks its CRC; writes nothing
    return {"zip": zip_path.name, "sha256": sha, "members": n_members,
            "first_bad_member": bad, "result": "PASS" if bad is None else "FAIL"}


def extract(zip_path, manifest_path, extract_root):
    check_hash(zip_path, manifest_path)
    dest = Path(extract_root) / zip_path.stem
    if dest.exists():
        raise SystemExit(f"{dest} exists; extraction needs a new empty folder")
    dest.mkdir(parents=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            # Untrusted archive: refuse any member that would land outside the target folder.
            target = (dest / m.filename).resolve()
            if root != target and root not in target.parents:
                raise SystemExit(f"unsafe member path {m.filename!r}; stopping")
        z.extractall(dest)
    return dest


def count_images(folder):
    return sum(1 for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() == ".ppm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["a", "b"])
    a = ap.parse_args()
    zips = [ZIP_DIR / f"{stem}.zip" for stem in EXPECTED_IMAGES]
    if a.step == "a":
        out = [integrity(z, MANIFEST) for z in zips]
        print(json.dumps(out, indent=2))
        print("CONDITION (a):", "PASS" if all(r["result"] == "PASS" for r in out) else "FAIL")
    else:
        out = []
        for z in zips:
            dest = extract(z, MANIFEST, EXTRACT_DIR)
            n = count_images(dest)
            out.append({"zip": z.name, "extracted_to": dest.as_posix(), "ppm_images": n,
                        "expected": EXPECTED_IMAGES[z.stem],
                        "result": "PASS" if n == EXPECTED_IMAGES[z.stem] else "FAIL"})
        print(json.dumps(out, indent=2))
        print("CONDITION (b):", "PASS" if all(r["result"] == "PASS" for r in out) else "FAIL")


if __name__ == "__main__":
    main()
