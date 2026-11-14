"""R9 Stage 1, descriptive breakdown after the gate result (changes nothing).

Splits the sign units flagged by shared-unit check 2 into those with at least one byte-identical view and those
matched only by class, image size and ROI box. The removed set in stage1_summary.json is not changed.
"""
import json
from pathlib import Path

from src.r9_stage1_audit import (EXTRACT_DIR, MANIFEST, RELEASES, ZIP_DIR, image_sha, read_release,
                                 verify_extracted)
from src.r9_stage1_download import sha256_file
from src.r9_stage1_integrity import check_hash

SUMMARY = Path("runs/btsc/stage1/stage1_summary.json")
SUMMARY_SHA = "77ba6a7042a471fa3320059d85a5ec967a8d30a613355c05c74815d03277a0dd"  # LOG_r9_stage1_entry.md Part B
OUT = Path("runs/btsc/stage1/check2_breakdown.json")


def split_check2(train_views, test_views):
    train_bytes = {image_sha(v["path"]) for v in train_views}
    train_meta = {(v["cls"], v["w"], v["h"], v["roi"]) for v in train_views}
    by_bytes, by_meta = set(), set()
    for v in test_views:
        u = (v["pole"], v["cls"])
        if image_sha(v["path"]) in train_bytes:
            by_bytes.add(u)
        if (v["cls"], v["w"], v["h"], v["roi"]) in train_meta:
            by_meta.add(u)
    return {"check2_units": len(by_bytes | by_meta),
            "with_byte_identical_view": len(by_bytes),
            "class_size_roi_match_only": len(by_meta - by_bytes),
            "byte_identical_and_metadata": len(by_bytes & by_meta),
            "byte_identical_without_metadata_match": len(by_bytes - by_meta)}


def main(summary=SUMMARY, summary_sha=SUMMARY_SHA, zip_dir=ZIP_DIR, extract_dir=EXTRACT_DIR, manifest=MANIFEST,
         out_path=OUT):
    if sha256_file(summary) != summary_sha:
        raise SystemExit(f"{summary}: SHA-256 differs from the recorded value; stopping")
    s = json.loads(Path(summary).read_text())
    rels = {}
    for key, stem in RELEASES.items():
        zp = Path(zip_dir) / f"{stem}.zip"
        if check_hash(zp, manifest) != s["inputs_sha256"][f"{stem}.zip"]:
            raise SystemExit(f"{stem}: zip differs from the one the audit read; stopping")
        verify_extracted(zp, Path(extract_dir) / stem)
        inner = [p for p in (Path(extract_dir) / stem).iterdir() if p.is_dir()][0]
        rels[key] = read_release(inner)
    out = {"label": "descriptive, after the gate result, changes nothing",
           "summary_sha256": summary_sha,
           "check2_total_in_summary": s["shared"]["check2_total"],
           **split_check2(rels["training"]["views"], rels["testing"]["views"])}
    if out["check2_units"] != out["check2_total_in_summary"]:
        raise SystemExit("check 2 unit count differs from the audit summary; stopping")
    if Path(out_path).exists():
        raise SystemExit(f"{out_path} exists; refusing to overwrite")
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("breakdown SHA-256:", sha256_file(out_path))
    return out


if __name__ == "__main__":
    main()
