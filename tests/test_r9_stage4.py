"""Tests for R9 Stage 4 on synthetic data: split rule against a brute-force reference, de-duplication, removal,
input hash checks with tamper tests, summary hash checks, whole-image cache, recipe check, 62-class head against the
POC-1 head, a short CPU fine-tune and the split step end to end. No real BelgiumTSC file is read."""
import json
import random
import sys
import zipfile
from collections import defaultdict
from fractions import Fraction
from math import floor
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.r9_stage1_audit as A  # noqa: E402
from src import head as H  # noqa: E402
from src import r9_stage4 as R  # noqa: E402
from src import r9_stage4_data as D  # noqa: E402
from src import r9_stage4_model as M  # noqa: E402
from src.r9_stage1_download import sha256_file  # noqa: E402
from src.r9_stage1_integrity import extract  # noqa: E402

HEADER = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"


# ---------------------------------------------------------------- split rule
def ref_split(units, num, den, seed):
    """Brute-force reference of Part A (c): strata by loops, quotas as exact fractions, extras one at a time."""
    poles = sorted({p for p, _ in units})
    classes = {c for _, c in units}
    count = {c: sum(1 for _, k in units if k == c) for c in classes}
    stratum = {}
    for p in poles:
        best = None
        for q, c in units:
            if q == p and (best is None or count[c] < count[best] or (count[c] == count[best] and c < best)):
                best = c
        stratum[p] = best
    S = sorted(set(stratum.values()))
    members = {s: sorted(p for p in poles if stratum[p] == s) for s in S}
    n_sel = floor(Fraction(len(poles) * num, den) + Fraction(1, 2))
    quota = {s: Fraction(len(members[s]) * num, den) for s in S}
    base = {s: floor(quota[s]) for s in S}
    frac = {s: quota[s] - base[s] for s in S}
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(S))
    given = set()
    for _ in range(n_sel - sum(base.values())):
        i = max((i for i in range(len(S)) if i not in given), key=lambda i: (frac[S[i]], -tie[i]))
        given.add(i)
        base[S[i]] += 1
    sel = []
    for s in S:
        perm = rng.permutation(np.array(members[s], dtype=np.int64))
        sel += [int(x) for x in perm[:base[s]]]
    return sorted(sel)


def random_units(rng, n_poles, n_classes, multi=0.2):
    units = set()
    for p in range(1, n_poles + 1):
        cls = int(rng.zipf(1.6)) % n_classes
        units.add((p * 7, cls))
        if rng.random() < multi:
            units.add((p * 7, int(rng.integers(n_classes))))
    return sorted(units)


@pytest.mark.parametrize("seed", range(25))
def test_split_matches_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    units = random_units(rng, int(rng.integers(5, 400)), int(rng.integers(2, 60)))
    for frac in ((1, 5), (1, 10), (1, 3)):
        assert D.split_by_pole(units, frac, seed) == ref_split(units, *frac, seed)


def test_split_properties():
    rng = np.random.default_rng(7)
    units = random_units(rng, 663, 53)
    sel = D.split_by_pole(units, (1, 5), 0)
    poles = {p for p, _ in units}
    assert len(sel) == 133 and len(set(sel)) == 133 and set(sel) <= poles
    strata = D.rarest_class_strata(units)
    by_s = defaultdict(list)
    for p, s in strata.items():
        by_s[s].append(p)
    for s, ps in by_s.items():                       # every stratum gets floor or ceil of its quota
        k = sum(1 for p in ps if p in set(sel))
        assert len(ps) // 5 <= k <= -(-len(ps) // 5)
    assert D.split_by_pole(units, (1, 5), 0) == sel  # deterministic
    assert D.split_by_pole(units, (1, 5), 1) != sel  # seed matters


def test_rarest_class_strata_ties_and_multi_class():
    units = [(1, 4), (1, 2), (2, 2), (3, 4), (4, 9), (4, 2)]
    # counts: class 2 -> 3 units, class 4 -> 2, class 9 -> 1
    assert D.rarest_class_strata(units) == {1: 4, 2: 2, 3: 4, 4: 9}
    assert D.rarest_class_strata([(1, 5), (1, 3)]) == {1: 3}   # tie goes to the smaller class ID


def test_round_half_up():
    assert D.round_half_up(663, 5) == 133
    assert D.round_half_up(1364, 10) == 136
    assert D.round_half_up(5, 2) == 3


def test_side_tables_one_side_only():
    views = [{"pole": 1, "cls": 0}, {"pole": 2, "cls": 0}, {"pole": 3, "cls": 1}, {"pole": 3, "cls": 0}]
    t = D.side_tables(views, [1])
    assert (t["design"]["poles"], t["pool"]["poles"]) == (1, 2)
    assert t["classes_one_side_only"] == [{"cls": 1, "units": 1, "side": "pool"}]
    assert t["strata_one_side_only"] == [{"stratum_class": 1, "poles": 1, "side": "pool"}]


# ---------------------------------------------------------------- de-duplication and removal
def view(folder, file, pole, vw, cls, roi=(1, 1, 9, 9)):
    return {"folder": folder, "file": file, "pole": pole, "view": vw, "cls": cls, "w": 10, "h": 10, "roi": roi,
            "size": min(roi[2] - roi[0], roi[3] - roi[1]), "path": Path(folder) / file}


def test_dedupe_identical_rows_collapse():
    a = view("00001", "00005_00000.ppm", 5, 0, 1)
    out, n = D.dedupe([a, dict(a), view("00001", "00005_00001.ppm", 5, 1, 1)])
    assert len(out) == 2 and n == 1


def test_dedupe_differing_rows_stop():
    a = view("00001", "00005_00000.ppm", 5, 0, 1)
    b = view("00001", "00005_00000.ppm", 5, 0, 1, roi=(2, 2, 9, 9))
    with pytest.raises(SystemExit, match="differ"):
        D.dedupe([a, b])


def test_dedupe_key_is_folder_and_file():
    a = view("00001", "00005_00000.ppm", 5, 0, 1)
    b = view("00002", "00005_00000.ppm", 5, 0, 2)
    out, n = D.dedupe([a, b])
    assert len(out) == 2 and n == 0


def test_dedupe_view_order_ascending():
    vs = [view("00001", f"00005_0000{k}.ppm", 5, k, 1) for k in (2, 0, 1)]
    out, _ = D.dedupe(vs)
    assert [v["view"] for v in out] == [0, 1, 2]


def s1_stub(test_views, removed):
    kept = [v for v in test_views if (v["pole"], v["cls"]) not in set(removed)]
    return {"removed_units": [list(u) for u in removed], "shared": {"removed": len(removed)},
            "gates": {"test_poles_after_removal": len({v["pole"] for v in kept}),
                      "test_sign_units_after_removal": len({(v["pole"], v["cls"]) for v in kept})}}


def test_removal_reads_the_stored_list():
    vs = [view("00001", "00005_00000.ppm", 5, 0, 1), view("00002", "00005_00000.ppm", 5, 0, 2),
          view("00001", "00006_00000.ppm", 6, 0, 1)]
    kept, removed = D.apply_removal(s1_stub(vs, [(5, 2)]), vs)
    assert {(v["pole"], v["cls"]) for v in kept} == {(5, 1), (6, 1)} and removed == {(5, 2)}


def test_removal_unknown_unit_stops():
    vs = [view("00001", "00005_00000.ppm", 5, 0, 1)]
    s1 = s1_stub(vs, [])
    s1["removed_units"], s1["shared"]["removed"] = [[9, 9]], 1
    with pytest.raises(SystemExit):
        D.apply_removal(s1, vs)


def test_removal_count_mismatch_stops():
    vs = [view("00001", "00005_00000.ppm", 5, 0, 1), view("00001", "00006_00000.ppm", 6, 0, 1)]
    s1 = s1_stub(vs, [(5, 1)])
    s1["gates"]["test_poles_after_removal"] = 2
    with pytest.raises(SystemExit):
        D.apply_removal(s1, vs)


# ---------------------------------------------------------------- synthetic release tree
def ppm(seed, w, h):
    rng = np.random.default_rng(seed)
    im = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    import io
    b = io.BytesIO()
    im.save(b, format="PPM")
    return b.getvalue()


def build_tree(tmp_path, train, test, dup_row=None):
    """train/test: lists of (cls, pole, view, seed). Writes zips, manifest, extracted trees; returns Paths."""
    zdir, edir = tmp_path / "zips", tmp_path / "extracted"
    zdir.mkdir()
    recs = []
    for stem, inner, views in (("BelgiumTSC_Training", "Training", train), ("BelgiumTSC_Testing", "Testing", test)):
        members, rows = {}, defaultdict(list)
        for cls, pole, vw, seed in views:
            w, h = 30 + seed % 97, 28 + (seed // 97) % 89     # distinct sizes, so check 2 removes nothing by chance
            name = f"{pole:05d}_{vw:05d}.ppm"
            members[f"{inner}/{cls:05d}/{name}"] = ppm(seed, w, h)
            row = f"{name};{w};{h};2;3;{w - 2};{h - 1};{cls}"
            rows[cls].append(row)
            if dup_row == (stem, cls, pole, vw):
                rows[cls].append(row)
        for cls, rs in rows.items():
            members[f"{inner}/{cls:05d}/GT-{cls:05d}.csv"] = (HEADER + "\n" + "\n".join(rs) + "\n").encode()
        members[f"{inner}/Readme.txt"] = b"synthetic readme"
        zp = zdir / f"{stem}.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for n, d in members.items():
                z.writestr(n, d)
        recs.append({"name": zp.name, "sha256": sha256_file(zp)})
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"files": recs}))
    for stem in A.RELEASES.values():
        extract(zdir / f"{stem}.zip", man, edir)
    s1 = A.audit(zdir, edir, man, check1_applies=True)
    s1p = tmp_path / "stage1_summary.json"
    s1p.write_text(json.dumps(s1))
    return D.Paths(stage1_summary=s1p, manifest=man, zip_dir=zdir, extract_dir=edir, out=tmp_path / "out")


def synth_views(seed=0, n_test_poles=40, n_classes=6):
    rng = random.Random(seed)
    train, test, s = [], [], 0
    for p in range(1000, 1000 + 60):
        cls = p % n_classes                       # every class in training
        for vw in range(3):
            train.append((cls, p, vw, s)); s += 1
    for p in range(1, n_test_poles + 1):
        cls = rng.randrange(n_classes)
        for vw in range(3):
            test.append((cls, p, vw, s)); s += 1
        if p % 7 == 0:                            # a multi-class pole
            test.append(((cls + 1) % n_classes, p, 0, s)); s += 1
    test.append((train[0][0], train[0][1], 0, 10_000))   # check-1 shared unit (same pole and class as training)
    return train, test


def inputs(tmp_path, **kw):
    train, test = synth_views()
    paths = build_tree(tmp_path, train, test, **kw)
    return paths, sha256_file(paths.stage1_summary), sha256_file(paths.manifest)


def test_check_inputs_passes_on_clean_tree(tmp_path):
    paths, s1sha, msha = inputs(tmp_path)
    s1, rels, hashes = D.check_inputs(paths, s1sha, msha)
    assert set(rels) == {"training", "testing"} and hashes["stage1_summary.json"] == s1sha


def test_tamper_stage1_summary(tmp_path):
    paths, s1sha, msha = inputs(tmp_path)
    paths.stage1_summary.write_text(paths.stage1_summary.read_text() + " ")
    with pytest.raises(SystemExit):
        D.check_inputs(paths, s1sha, msha)


def test_tamper_manifest(tmp_path):
    paths, s1sha, msha = inputs(tmp_path)
    paths.manifest.write_text(paths.manifest.read_text() + " ")
    with pytest.raises(SystemExit):
        D.check_inputs(paths, s1sha, msha)


def test_tamper_zip(tmp_path):
    paths, s1sha, msha = inputs(tmp_path)
    zp = paths.zip_dir / "BelgiumTSC_Testing.zip"
    b = bytearray(zp.read_bytes()); b[-1] ^= 1
    zp.write_bytes(bytes(b))
    with pytest.raises(SystemExit):
        D.check_inputs(paths, s1sha, msha)


@pytest.mark.parametrize("target", ["image", "gt", "readme"])
def test_tamper_extracted_file(tmp_path, target):
    paths, s1sha, msha = inputs(tmp_path)
    inner = paths.extract_dir / "BelgiumTSC_Testing" / "Testing"
    f = {"image": next(inner.glob("*/*.ppm")), "gt": next(inner.glob("*/GT-*.csv")), "readme": inner / "Readme.txt"}[target]
    b = bytearray(f.read_bytes()); b[-1] ^= 1
    f.write_bytes(bytes(b))
    with pytest.raises(SystemExit):
        D.check_inputs(paths, s1sha, msha)


def test_extra_extracted_file_refused(tmp_path):
    paths, s1sha, msha = inputs(tmp_path)
    (paths.extract_dir / "BelgiumTSC_Training" / "Training" / "extra.ppm").write_bytes(b"x")
    with pytest.raises(SystemExit):
        D.check_inputs(paths, s1sha, msha)


def test_counts_and_duplicate_rows(tmp_path):
    train, test = synth_views()
    paths = build_tree(tmp_path, train, test, dup_row=("BelgiumTSC_Testing",) + test[3][:3])
    s1, rels, _ = D.check_inputs(paths, sha256_file(paths.stage1_summary), sha256_file(paths.manifest))
    te, n = D.dedupe(rels["testing"]["views"])
    assert n == 1
    f = D.check_counts(s1, "testing", te, rels["testing"]["n_images"])
    assert f["views"] == len(test)
    with pytest.raises(SystemExit):                 # a view count that keeps the duplicated row is refused
        D.check_counts(s1, "testing", rels["testing"]["views"], rels["testing"]["n_images"])


def split_cfg(paths):
    s1 = json.loads(paths.stage1_summary.read_text())
    n_te = s1["gates"]["test_poles_after_removal"]
    n_tr = s1["counts"]["training"]["pole_numbers"]
    cfg = yaml.safe_load((ROOT / "configs" / "r9_btsc_stage4.yaml").read_text())
    cfg["n_classes"] = 6
    cfg["expected"] = {"design_poles": D.round_half_up(n_te, 5), "pool_poles": n_te - D.round_half_up(n_te, 5),
                       "train_val_poles": D.round_half_up(n_tr, 10)}
    return cfg


def test_split_step_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)                          # provenance hashes the code files by repo-relative path
    paths, s1sha, msha = inputs(tmp_path)
    monkeypatch.setattr(D, "STAGE1_SUMMARY_SHA256", s1sha)
    monkeypatch.setattr(D, "MANIFEST_SHA256", msha)
    monkeypatch.setattr(D.check_inputs, "__defaults__", (s1sha, msha))
    cfg = split_cfg(paths)
    R.step_split(paths, cfg)
    out = paths.out
    poles = json.loads((out / "split_poles.json").read_text())
    te = D.read_table(out / "test_views.csv")
    s1 = json.loads(paths.stage1_summary.read_text())
    removed = {tuple(u) for u in s1["removed_units"]}
    assert removed and not {(r["pole"], r["cls"]) for r in te} & removed
    assert not set(poles["design_poles"]) & set(poles["pool_poles"])
    for r in te:
        assert (r["part"] == "design") == (r["pole"] in set(poles["design_poles"]))
    D.check_summary_files(out / "split_summary.json", out)
    with pytest.raises(SystemExit, match="exists"):  # a recorded run is never overwritten
        R.step_split(paths, cfg)


# ---------------------------------------------------------------- summaries
def test_check_summary_files_tamper_and_missing(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "best.pt").write_bytes(b"weights")
    D.write_summary(tmp_path / "s_summary.json", {"files_sha256": {
        "a.bin": sha256_file(tmp_path / "a.bin"), "model/best.pt": sha256_file(tmp_path / "model" / "best.pt")}})
    D.check_summary_files(tmp_path / "s_summary.json", tmp_path)
    (tmp_path / "model" / "best.pt").write_bytes(b"weightz")
    with pytest.raises(SystemExit, match="changed"):
        D.check_summary_files(tmp_path / "s_summary.json", tmp_path)
    (tmp_path / "model" / "best.pt").unlink()
    with pytest.raises(SystemExit, match="missing"):
        D.check_summary_files(tmp_path / "s_summary.json", tmp_path)


def test_check_prior_refuses_changed_step_output(tmp_path):
    for s in R.STEPS:
        f = tmp_path / f"{s}.out"
        f.write_bytes(s.encode())
        D.write_summary(R.summary_path(tmp_path, s), {"files_sha256": {f.name: sha256_file(f)}})
    R.check_prior(tmp_path, "vchecks")
    (tmp_path / "cache.out").write_bytes(b"other")
    with pytest.raises(SystemExit):
        R.check_prior(tmp_path, "train")
    R.check_prior(tmp_path, "cache")                 # earlier steps only: the cache output is not read yet


def test_write_summary_refuses_overwrite(tmp_path):
    D.write_summary(tmp_path / "x.json", {})
    with pytest.raises(SystemExit):
        D.write_summary(tmp_path / "x.json", {})


# ---------------------------------------------------------------- cache
def test_cache_is_whole_image_resize_in_table_order(tmp_path):
    files = []
    for k, (w, h) in enumerate([(40, 30), (25, 60), (112, 112)]):
        p = tmp_path / f"{k}.ppm"
        p.write_bytes(ppm(k, w, h))
        files.append(p)
    D.build_cache(files, tmp_path / "c.npy", 112)
    X = np.load(tmp_path / "c.npy", mmap_mode="r")
    assert X.shape == (3, 112, 112, 3) and X.dtype == np.uint8
    for i, p in enumerate(files):
        want = np.asarray(Image.open(p).convert("RGB").resize((112, 112), Image.BILINEAR))
        assert np.array_equal(X[i], want)


def test_load_whole_matches_poc1_loader(tmp_path):
    from src.data import load_resized
    zp = tmp_path / "z.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("a.ppm", ppm(3, 37, 41))
    (tmp_path / "a.ppm").write_bytes(ppm(3, 37, 41))
    with zipfile.ZipFile(zp) as z:
        assert np.array_equal(load_resized(z, "a.ppm", 112), D.load_whole(tmp_path / "a.ppm", 112))


# ---------------------------------------------------------------- model, recipe, head
def test_recipe_equals_poc1():
    r9 = yaml.safe_load((ROOT / "configs/r9_btsc_stage4.yaml").read_text())
    poc = yaml.safe_load((ROOT / "configs/poc.yaml").read_text())
    M.check_recipe(r9, poc)
    bad = json.loads(json.dumps(r9))
    bad["train"]["lr_head"] = 0.002
    with pytest.raises(SystemExit):
        M.check_recipe(bad, poc)
    bad = json.loads(json.dumps(r9))
    bad["train"]["aug"]["rotation_deg"] = 15
    with pytest.raises(SystemExit):
        M.check_recipe(bad, poc)


def test_build_model_62_outputs():
    m = M.build_model(62, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.zeros(2, 3, 112, 112)).shape == (2, 62)


def test_fit_head_k_equals_poc1_head():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.arange(43), rng.integers(0, 43, 400)])
    X = rng.normal(size=(len(y), 16)) + y[:, None] * 0.05
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    a, b = M.fit_head_k(X, y, 43), H.fit_head(X, y)
    for k in ("mu", "W", "prec", "priors"):
        np.testing.assert_allclose(a[k], b[k], rtol=0, atol=1e-12)
    assert a["shrinkage"] == b["shrinkage"]


def test_fit_head_k_needs_every_class():
    X = np.random.default_rng(0).normal(size=(10, 4))
    with pytest.raises(AssertionError):
        M.fit_head_k(X, np.array([0] * 5 + [1] * 5), 3)


def test_fine_tune_smoke_cpu(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (12, 112, 112, 3), dtype=np.uint8)
    y = np.array([0, 1, 2] * 4)
    tc = yaml.safe_load((ROOT / "configs/r9_btsc_stage4.yaml").read_text())["train"]
    tc = {**tc, "epochs": 2, "batch_size": 4}
    res = M.fine_tune(X, y, np.arange(9), np.arange(9, 12), tc, 0, torch.device("cpu"), tmp_path, 3,
                      pretrained=False, log=lambda s: None)
    assert res["epochs_run"] == 2 and 1 <= res["best_epoch"] <= 2
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "history.csv").is_file()
    m = M.load_checkpoint(tmp_path / "best.pt", 3, torch.device("cpu"))
    assert m.fc.out_features == 3


# ---------------------------------------------------------------- head-step helpers
def test_last_view_units_brute_force():
    rng = np.random.default_rng(1)
    pole = rng.integers(0, 15, 80)
    y = pole % 4
    view = rng.permutation(80)
    hp, sp = rng.integers(0, 4, 80), rng.integers(0, 4, 80)
    got = R.last_view_units(pole, view, y, hp, sp)
    units = sorted(set(zip(pole.tolist(), y.tolist())))
    hc = sc = 0
    for u in units:
        rows = [i for i in range(80) if (pole[i], y[i]) == u]
        i = max(rows, key=lambda r: view[r])
        hc += hp[i] == y[i]; sc += sp[i] == y[i]
    assert got == {"units": len(units), "head_accuracy": hc / len(units), "softmax_accuracy": sc / len(units)}


def test_no_pool_poles_in_real_design_guard_path():
    # The runner builds its guard from split_poles.json design poles only; a pool row is refused.
    from src.r9_btsc_guard import GuardedTestLabels
    g = GuardedTestLabels([1, 2], [0, 1], [1], blobs={})   # explicit pre-F3 state (the repo is past F3)
    with pytest.raises(AssertionError):
        g.labels([1])
