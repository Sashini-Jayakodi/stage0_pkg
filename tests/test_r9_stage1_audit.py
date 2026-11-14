"""Tests for the R9 Stage 1 audit on synthetic BelgiumTSC-like trees. No real data is read."""
import json
import random
import subprocess
import sys
import zipfile
from collections import defaultdict
from types import SimpleNamespace

import pytest

import src.r9_stage1_audit as A
from src.r9_stage1_download import sha256_file
from src.r9_stage1_integrity import extract

HEADER = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"


def build_release(stem, inner, views, extra_members=None):
    """views: list of (cls, pole, view, w, h, roi, image_bytes). Returns zip member dict."""
    members, rows = {}, defaultdict(list)
    for cls, pole, view, w, h, roi, data in views:
        name = f"{pole:05d}_{view:05d}.ppm"
        members[f"{inner}/{cls:05d}/{name}"] = data
        rows[cls].append(f"{name};{w};{h};{roi[0]};{roi[1]};{roi[2]};{roi[3]};{cls}")
    for cls, rs in rows.items():
        members[f"{inner}/{cls:05d}/GT-{cls:05d}.csv"] = (HEADER + "\n" + "\n".join(rs) + "\n").encode()
    members[f"{inner}/Readme.txt"] = b"synthetic readme"
    members.update(extra_members or {})
    return members


def setup_tree(tmp_path, train_views, test_views):
    zdir, edir = tmp_path / "zips", tmp_path / "extracted"
    zdir.mkdir()
    recs = []
    for stem, inner, views in (("BelgiumTSC_Training", "Training", train_views),
                               ("BelgiumTSC_Testing", "Testing", test_views)):
        zp = zdir / f"{stem}.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for name, data in build_release(stem, inner, views).items():
                z.writestr(name, data)
        recs.append({"name": zp.name, "sha256": sha256_file(zp)})
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"files": recs}))
    for stem in A.RELEASES.values():
        extract(zdir / f"{stem}.zip", man, edir)
    return zdir, edir, man


def v(cls, pole, view, seed, w=100, h=100, roi=(10, 10, 90, 90)):
    # Unique image bytes and box per seed unless a test sets them on purpose.
    return (cls, pole, view, w + seed % 7, h + seed % 5, (roi[0], roi[1], roi[2] + seed % 3, roi[3]),
            f"img-{seed}".encode())


def brute(views):
    poles, units = set(), set()
    for cls, pole, *_ in views:
        poles.add(pole)
        units.add((pole, cls))
    return len(poles), len(units)


def test_counts_match_brute_force(tmp_path):
    rng = random.Random(0)
    train = [v(rng.randrange(5), rng.randrange(1000, 1100), k, 1000 + k) for k in range(80)]
    seen, test = set(), []
    for k in range(120):
        cls, pole, view = rng.randrange(5), rng.randrange(1, 60), rng.randrange(3)
        if (cls, pole, view) not in seen:
            seen.add((cls, pole, view))
            test.append(v(cls, pole, view, k))
    seen_t, train_u = set(), []
    for t in train:
        if t[:3] not in seen_t:
            seen_t.add(t[:3])
            train_u.append(t)
    zdir, edir, man = setup_tree(tmp_path, train_u, test)
    out = A.audit(zdir, edir, man, check1_applies=False)
    c = out["counts"]["testing"]
    assert c["images"] == len(test)
    assert (c["pole_numbers"], c["sign_units"]) == brute(test)
    assert c["extra_sign_units_from_multi_class_poles"] == c["sign_units"] - c["pole_numbers"]
    assert sum(c["views_per_sign_unit"].values()) == c["sign_units"]
    assert c["consistency_failures"].get("name_does_not_parse", 0) == 0
    assert out["poles_recovered"] is True


def test_parsing_reads_pole_and_view():
    m = A.NAME_RE.match("01153_00002.ppm")
    assert (int(m.group(1)), int(m.group(2))) == (1153, 2)
    for bad in ("1153_00002.ppm", "01153-00002.ppm", "01153_00002.PPM", "01153_00002.png"):
        assert A.NAME_RE.match(bad) is None


def test_multi_class_pole(tmp_path):
    train = [v(0, 900, 0, 1)]
    test = [v(0, 5, 0, 10), v(0, 5, 1, 11), v(3, 5, 0, 12), v(1, 6, 0, 13)]  # pole 5 carries classes 0 and 3
    zdir, edir, man = setup_tree(tmp_path, train, test)
    c = A.audit(zdir, edir, man, check1_applies=False)["counts"]["testing"]
    assert c["pole_numbers"] == 2 and c["sign_units"] == 3
    assert c["poles_with_more_than_one_class"] == 1
    assert c["extra_sign_units_from_multi_class_poles"] == 1
    assert c["views_per_sign_unit"] == {"1": 2, "2": 1, "3": 0, "4+": 0}
    # Views per pole: pole 5 has 3 views, pole 6 has 1.
    assert (c["views_per_pole_min"], c["views_per_pole_median"], c["views_per_pole_max"]) == (1, 2.0, 3)


def shared_case(tmp_path):
    train = [v(0, 7, 0, 100),                                          # pole 7 class 0 (number match target)
             (1, 800, 0, 50, 60, (5, 5, 40, 50), b"same-bytes"),       # byte-identical target
             (2, 801, 0, 77, 88, (3, 4, 70, 80), b"train-meta")]       # metadata target
    test = [v(0, 7, 0, 1),                                             # check 1 only
            (1, 20, 0, 51, 61, (6, 6, 41, 51), b"same-bytes"),         # check 2 only (bytes)
            (2, 21, 0, 77, 88, (3, 4, 70, 80), b"test-meta"),          # check 2 only (metadata)
            (1, 800, 0, 50, 60, (5, 5, 40, 50), b"same-bytes"),        # both: same (pole, class) and same bytes
            v(4, 30, 0, 2)]                                            # not shared
    return train, test


def test_shared_detection_by_each_check(tmp_path):
    train, test = shared_case(tmp_path)
    zdir, edir, man = setup_tree(tmp_path, train, test)
    no = A.audit(zdir, edir, man, check1_applies=False)
    s = no["shared"]
    # (1, 800) matches by number and by bytes and metadata: both.
    assert (s["check1_only"], s["check2_only"], s["both"]) == (1, 2, 1)
    assert s["removed"] == 3                       # check 1 does not apply: only check 2's units are removed
    assert no["gates"]["test_sign_units_after_removal"] == 2
    yes = A.audit(zdir, edir, man, check1_applies=True)
    assert yes["shared"]["removed"] == 4
    assert yes["gates"]["test_sign_units_after_removal"] == 1


def test_fa1_counts_poles_not_units(monkeypatch):
    monkeypatch.setattr(A, "F_A1_MIN_POLES", 3)
    # Two poles, four sign units: units >= 3 but poles < 3, so F-A1 must fail.
    views = [{"pole": p, "cls": c} for p, c in ((1, 0), (1, 1), (2, 0), (2, 5))]
    g = A.gates(views, [], poles_recovered=True)
    assert g["test_sign_units_after_removal"] == 4 and g["test_poles_after_removal"] == 2
    assert g["F-A1"] == "FAIL"
    views.append({"pole": 3, "cls": 0})
    assert A.gates(views, [], True)["F-A1"] == "PASS"
    # Removing every unit of pole 3 drops it from the pole count; removing one unit of pole 1 does not.
    g = A.gates(views, [(3, 0), (1, 1)], True)
    assert g["test_poles_after_removal"] == 2 and g["F-A1"] == "FAIL"
    assert A.gates(views, [], poles_recovered=False)["F-A1"] == "FAIL"


def test_consistency_failures_are_counted(tmp_path):
    train = [v(0, 900, 0, 1)]
    test = [v(0, 5, 0, 10), v(1, 6, 0, 11)]
    zdir, edir, man = setup_tree(tmp_path, train, test)
    root = edir / "BelgiumTSC_Testing" / "Testing"
    # Build a second tree where class folder 00001's csv says ClassId 2 and an extra image has no row.
    gt = root / "00001" / "GT-00001.csv"
    rows = gt.read_text().replace(";1\n", ";2\n")
    gt.write_text(rows)
    (root / "00001" / "00099_00000.ppm").write_bytes(b"x")
    rel = A.read_release(root)
    assert len(rel["fails"]["class_folder_ne_classid"]) == 1
    assert len(rel["fails"]["image_without_row"]) == 1


def test_tamper_changed_zip_refused(tmp_path):
    zdir, edir, man = setup_tree(tmp_path, [v(0, 900, 0, 1)], [v(0, 5, 0, 2)])
    zp = zdir / "BelgiumTSC_Testing.zip"
    zp.write_bytes(zp.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="differs from manifest"):
        A.audit(zdir, edir, man, check1_applies=False)


def test_tamper_changed_extracted_csv_refused(tmp_path):
    zdir, edir, man = setup_tree(tmp_path, [v(0, 900, 0, 1)], [v(0, 5, 0, 2)])
    gt = edir / "BelgiumTSC_Testing" / "Testing" / "00000" / "GT-00000.csv"
    gt.write_text(gt.read_text().replace("00005_", "00006_"))
    with pytest.raises(SystemExit, match="content differs from zip member"):
        A.audit(zdir, edir, man, check1_applies=False)


def test_tamper_extra_file_refused(tmp_path):
    zdir, edir, man = setup_tree(tmp_path, [v(0, 900, 0, 1)], [v(0, 5, 0, 2)])
    (edir / "BelgiumTSC_Testing" / "Testing" / "00000" / "00077_00000.ppm").write_bytes(b"new")
    with pytest.raises(SystemExit, match="file set differs from zip"):
        A.audit(zdir, edir, man, check1_applies=False)


def test_kmax_reproduces_plan_rows_and_part_a_values():
    # Plan R9 section 6 rows; Part A (g) values.
    assert [A.kmax(784, a) for a in A.ALPHAS] == [1, 3, 10, 17, 30]
    assert [A.kmax(300, a) for a in A.ALPHAS] == [-1, 0, 2, 4, 9]
    assert [A.kmax(252, a) for a in A.ALPHAS] == [-1, 0, 1, 3, 7]
    assert A.kmax(387, 0.01) == 0 and A.kmax(388, 0.01) == 1
    assert A.kmax(0, 0.05) == -1
    p = A.projection(600)
    assert (p["expected_cert_size"], p["expected_n_c_at_70pct"]) == (360, 252)


def test_dits_retry_records_both_attempts():
    calls = []

    def fail(cmd, **kw):
        calls.append(cmd[-1])
        return SimpleNamespace(returncode=7, stdout="", stderr="Failed to connect")
    r = A.dits_retry(runner=fail)
    assert calls == list(A.DITS_URLS) and calls[0].startswith("https://")
    assert r["F-A3"] == "FAIL" and all(a["error"] for a in r["attempts"])

    def ok(cmd, **kw):
        return SimpleNamespace(returncode=0, stdout="<html></html>\n200", stderr="")
    assert A.dits_retry(runner=ok)["F-A3"].startswith("page reachable")


def test_module_loads_no_model_code():
    # Design-phase guard (hard rule 3): the audit must not import torch or any model/embedding loader.
    code = ("import sys, src.r9_stage1_audit; "
            "bad=[m for m in sys.modules if m=='torch' or m.startswith('torch.') or m.startswith('src.r8_') "
            "or m in ('src.head','src.train','src.embed')]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
