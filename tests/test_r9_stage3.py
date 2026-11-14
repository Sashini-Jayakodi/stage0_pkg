"""Tests for the POC-3 Stage 3 runner (src/r9_stage3.py), synthetic only. Brute-force references are plain loops
written from LOG_r9_stage3_entry.md Part A; the end-to-end tests use the POC-2 Stage 4 fake run."""
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import r8_stage2 as S2  # noqa: E402
from src import r8_stage4 as S4  # noqa: E402
from src import r8_stage5 as S5  # noqa: E402
from src import r9_frozen as FZ  # noqa: E402
from src import r9_stage3 as R  # noqa: E402
from src import r9_stage3_design as D  # noqa: E402
from src import r9_stage3_scores as SC  # noqa: E402
from src.util import sha256_file  # noqa: E402
import test_r8_stage5 as T5  # noqa: E402

rng = np.random.default_rng(23)


def synth_tab(n=300, G=12):
    com = rng.random((n, G)) < np.linspace(0.98, 0.4, G)
    com = np.cumsum(com[:, ::-1], 1)[:, ::-1] > 0                    # stricter lambda never commits more
    wrong = com & (rng.random((n, 1)) < 0.03)
    pos = np.where(com, rng.integers(1, 31, (n, G)), 0)
    return {"committed": com, "wrong": wrong, "pos": pos}


# ------------------------------------------------------------------ brute-force references
def brute_binom_walk(tab, cert, alpha, f_d, n, delta=0.1):
    start = -1
    for j in range(len(f_d)):
        if f_d[j] * n * alpha >= 4:
            start = j
    best = -1
    for j in range(start, -1, -1):
        nc = sum(1 for i in range(len(cert)) if cert[i] and tab["committed"][i, j])
        k = sum(1 for i in range(len(cert)) if cert[i] and tab["committed"][i, j] and tab["wrong"][i, j])
        p = 1.0 if nc == 0 else binom.cdf(k, nc, alpha)
        if p > delta:
            break
        best = j
    return best


def test_certify_cell_matches_brute_walk():
    for _ in range(20):
        tab = synth_tab()
        cert = rng.random(300) < 0.75
        f_d = tab["committed"].mean(0)
        for al in (0.01, 0.02, 0.05):
            assert R.certify_cell(tab, cert, al, f_d, "binom", 100) == brute_binom_walk(tab, cert, al, f_d, int(cert.sum()))


def test_cert_counts_use_cert_tracks_only():
    tab = synth_tab()
    cert = rng.random(300) < 0.5
    k, n_c = R.cert_counts(tab, cert)
    for j in range(tab["committed"].shape[1]):
        assert n_c[j] == sum(tab["committed"][i, j] for i in range(300) if cert[i])
        assert k[j] == sum(tab["committed"][i, j] and tab["wrong"][i, j] for i in range(300) if cert[i])


def test_test_metrics_match_loops():
    tab = synth_tab()
    test = rng.random(300) < 0.25
    size = np.where(tab["committed"], rng.integers(15, 150, tab["pos"].shape).astype(float), np.nan)
    S = S5.s_range(150)
    for j in range(tab["committed"].shape[1]):
        m = R.test_metrics(tab, size, j, test, S)
        idx = [i for i in range(300) if test[i]]
        c = [i for i in idx if tab["committed"][i, j]]
        w = [i for i in c if tab["wrong"][i, j]]
        assert m["coverage"] == pytest.approx(len(c) / len(idx)) and m["escalation"] == pytest.approx(1 - len(c) / len(idx))
        assert m["error"] == pytest.approx(len(w) / len(c))
        assert m["median_frame"] == pytest.approx(np.median([tab["pos"][i, j] for i in c]))
        assert m["median_size"] == pytest.approx(np.median([size[i, j] for i in c]))
        want = S5.NOT_REACHED
        for s in S:
            if sum(1 for i in c if size[i, j] <= s) / len(idx) >= 0.9 - 1e-12:
                want = s
                break
        assert m["d90"] == want


def test_uncertified_cell_commits_nothing():
    tab = synth_tab()
    m = R.test_metrics(tab, np.zeros(tab["pos"].shape), -1, np.ones(300, bool), [5])
    assert m["coverage"] == 0.0 and m["escalation"] == 1.0 and m["error"] is None and m["j"] is None


def test_smallest_alpha():
    assert R.smallest_alpha({0.005: -1, 0.01: -1, 0.02: 4, 0.05: 9}, [0.005, 0.01, 0.02, 0.05]) == 0.02
    assert R.smallest_alpha({0.01: -1, 0.02: -1}, [0.01, 0.02]) is None


def test_size_table_reads_the_commit_frame():
    rows = [np.array([4, 2, 0]), np.array([1, 3, 5])]
    roi = np.array([10.0, 11, 12, 13, 14, 15])
    tab = {"committed": np.array([[True, False], [True, True]]), "pos": np.array([[2, 0], [1, 3]])}
    sz = R.size_table(tab, rows, roi)
    assert sz[0, 0] == 12 and np.isnan(sz[0, 1]) and sz[1, 0] == 11 and sz[1, 1] == 15


# ------------------------------------------------------------------ end to end on the POC-2 Stage 4 fake run
def make_3(tmp, monkeypatch):
    run, cfgp = T5.make_5(tmp, monkeypatch)                           # Stage 4 run with alphas that can certify at n = 15
    monkeypatch.setattr(S2, "N_DESIGN", 20)
    monkeypatch.setattr(SC, "starts", functools.partial(SC.starts, n=15))   # fake Cert size
    D.main(["--run", str(run)])
    dp = run / "poc3" / "stage3" / "stage3_design.json"
    js = json.loads(dp.read_text())
    js["n_cert"] = 15
    dp.write_text(json.dumps(js))
    monkeypatch.setattr(SC, "starts", functools.partial(SC.starts.func, n=15))
    monkeypatch.setattr(R, "FROZEN_ARGS", S4.FROZEN_ARGS)
    monkeypatch.setattr(R, "DESIGN_SHA256", sha256_file(dp))
    monkeypatch.setattr(R, "STAGE4_SUMMARY_SHA256", sha256_file(run / "poc2" / "stage4" / "stage4_summary.json"))
    monkeypatch.setattr(R, "STAGE4_CODE_SHA256", sha256_file(Path(S4.__file__)))
    import yaml
    meta = Path(yaml.safe_load(Path(cfgp).read_text())["data"]["cache_dir"]) / "train_meta.csv"
    monkeypatch.setattr(R, "TRAIN_META_SHA256", sha256_file(meta))
    return run, cfgp


def test_end_to_end_on_fake_run(tmp_path, monkeypatch):
    run, cfgp = make_3(tmp_path, monkeypatch)
    s = R.main(["--run", str(run), "--config", str(cfgp)])
    assert all(s["checks"].values()) and s["n_cert"] == 15 and s["n_test"] == 5
    js = json.loads((run / "poc3" / "stage3" / "stage3_summary.json").read_text())
    assert set(js["results"]) == set(SC.SCORES) and set(js["results"]["S1"]) == {"0", "1", "2", "3", "4"}
    for s_ in SC.SCORES:
        for sd in js["results"][s_].values():
            assert set(sd["cert"]) == {str(a) for a in SC.ALPHAS}


def test_refuses_changed_design_file(tmp_path, monkeypatch):
    run, cfgp = make_3(tmp_path, monkeypatch)
    p = run / "poc3" / "stage3" / "stage3_design.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(AssertionError, match="stage3_design.json differs"):
        R.main(["--run", str(run), "--config", str(cfgp)])


def test_refuses_changed_stage4_summary(tmp_path, monkeypatch):
    run, cfgp = make_3(tmp_path, monkeypatch)
    p = run / "poc2" / "stage4" / "stage4_summary.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(AssertionError, match="stage4_summary.json differs"):
        R.main(["--run", str(run), "--config", str(cfgp)])


def test_refuses_changed_stage4_code(tmp_path, monkeypatch):
    run, cfgp = make_3(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "STAGE4_CODE_SHA256", "0" * 64)
    with pytest.raises(AssertionError, match="r8_stage4.py differs"):
        R.main(["--run", str(run), "--config", str(cfgp)])


def test_refuses_changed_train_meta(tmp_path, monkeypatch):
    run, cfgp = make_3(tmp_path, monkeypatch)
    monkeypatch.setattr(R, "TRAIN_META_SHA256", "0" * 64)
    with pytest.raises(AssertionError, match="train_meta.csv differs"):
        R.main(["--run", str(run), "--config", str(cfgp)])


def test_stops_if_coverages_do_not_reproduce(tmp_path, monkeypatch):
    run, cfgp = make_3(tmp_path, monkeypatch)
    p = run / "poc2" / "stage4" / "stage4_summary.json"
    s4 = json.loads(p.read_text())
    hit = False
    for r in s4["per_seed"]:
        for by_m in r["alphas"].values():
            for m in S4.METHODS:
                if by_m[m]["certified"] and not hit:
                    by_m[m]["coverage"] += 0.01
                    hit = True
    assert hit, "the fake Stage 4 must certify something for this test to mean anything"
    p.write_text(json.dumps(s4))
    monkeypatch.setattr(R, "STAGE4_SUMMARY_SHA256", sha256_file(p))
    with pytest.raises(AssertionError, match="do not reproduce"):
        R.main(["--run", str(run), "--config", str(cfgp)])


def test_freeze_check_runs_first(tmp_path, monkeypatch):
    run, cfgp = make_3(tmp_path, monkeypatch)
    monkeypatch.setattr(FZ, "FROZEN_BLOBS", dict(FZ.FROZEN_BLOBS, **{FZ.ENGINE: "0" * 40}))
    p = run / "poc3" / "stage3" / "stage3_design.json"
    p.write_text(p.read_text() + " ")                                   # would fail later; the guard must fail first
    with pytest.raises(AssertionError, match="frozen blob ID"):
        R.main(["--run", str(run), "--config", str(cfgp)])


def test_runner_names_no_btsc_folder():
    src = (ROOT / "src" / "r9_stage3.py").read_text(encoding="utf-8")
    for bad in ("btsc_raw", "runs/btsc", "import torch"):
        assert bad not in src
