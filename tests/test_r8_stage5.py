"""Tests for POC-2 Stage 5 (src/r8_stage5.py) on synthetic data. No real GTSRB is read.
Brute-force references are plain loops written from LOG_r8_stage5_entry.md Part A and the frozen late definition."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import r8_stage4 as S4  # noqa: E402
from src import r8_stage5 as S5  # noqa: E402
from src.util import sha256_file  # noqa: E402
import test_r8_stage4 as T4  # noqa: E402

rng = np.random.default_rng(31)


def brute_share(committed, size, s):
    """1 - mean late_i(s), late_i(s) = 1 - committed_i * [size at commit <= s] (frozen_r8.yaml late_definition)."""
    late = [1 - (1 if (c and z <= s) else 0) for c, z in zip(committed, size)]
    return 1 - sum(late) / len(late)


def test_curve_matches_frozen_late_definition():
    committed = rng.random(262) < 0.9
    size = np.where(committed, rng.integers(15, 120, 262).astype(float), np.nan)
    S = S5.s_range(np.nanmax(size))
    curve = S5.committed_by_size(committed, size, S)
    assert all(curve[i] == pytest.approx(brute_share(committed, size, s)) for i, s in enumerate(S))
    assert (np.diff(curve) >= 0).all() and curve[-1] == pytest.approx(committed.mean())


def test_s_range_and_d90():
    assert S5.s_range(29.0) == [5, 10, 15, 20, 25, 30] and S5.s_range(30.0) == [5, 10, 15, 20, 25, 30]
    S = [5, 10, 15, 20]
    assert S5.d90(np.array([0.1, 0.89, 0.9, 0.95]), S) == 15
    assert S5.d90(np.array([0.1, 0.5, 0.8, 0.85]), S) == S5.NOT_REACHED
    n = S5.summary_numbers(np.array([0.2, 0.4, 0.6, 0.91, 0.92]), [5, 25, 50, 75, 80])
    assert n == {"share_at_25px": 0.4, "share_at_50px": 0.6, "share_at_75px": 0.91, "d90_size": 75}


def test_baseline_reported_only_where_both_are_certified():
    sa = lambda c, b: {"commit": {"certified": c}, "B2": {"certified": b}}
    assert S5.report_baseline(sa(True, True), "B2")
    assert not S5.report_baseline(sa(True, False), "B2")
    assert not S5.report_baseline(sa(False, True), "B2")


def test_the_word_readable_is_never_used():
    """Stage 5 Part A (d), author's rule: D90 is never called 'readable' in code, printouts or the log."""
    for p in (ROOT / "src" / "r8_stage5.py", ROOT / "LOG_r8_stage5_entry.md"):
        assert not re.search("readable", p.read_text(encoding="utf-8"), re.I), p


# ------------------------------------------------------------------ end to end on a fake run folder
def make_5(tmp, monkeypatch):
    run, cfgp = T4.make_4(tmp, monkeypatch)
    # 15 fake Cert tracks cannot certify 1% or 5% (zero-error floor ~14% at n = 15), so give the fake frozen config
    # larger alphas; otherwise the end-to-end path would never draw a curve
    args = S4.FROZEN_ARGS
    fz = args["root"] / "configs" / "frozen_r8.yaml"
    cfg = yaml.safe_load(fz.read_text())
    cfg["certificate_A"]["alpha_grid"] = [0.01, 0.3, 0.5]
    cfg["certificate_B"]["s"] = "not certifiable at n = 784"          # as in the real frozen file
    fz.write_text(yaml.safe_dump(cfg))
    args["frozen_sha"] = sha256_file(fz)
    S4.main(["--run", str(run), "--config", str(cfgp)])
    monkeypatch.setattr(S5, "FROZEN_ARGS", S4.FROZEN_ARGS)
    monkeypatch.setattr(S5, "STAGE4_SUMMARY_SHA256", sha256_file(run / "poc2" / "stage4" / "stage4_summary.json"))
    monkeypatch.setattr(S5, "STAGE4_CODE_SHA256", sha256_file(Path(S4.__file__)))
    return run, cfgp


def test_stage5_end_to_end_reproduces_stage4(tmp_path, monkeypatch):
    run, cfgp = make_5(tmp_path, monkeypatch)
    s4 = json.loads((run / "poc2" / "stage4" / "stage4_summary.json").read_text())
    if not any(r["alphas"][k]["commit"]["certified"] for r in s4["per_seed"] for k in r["alphas"]):
        pytest.skip("the fake run certified nothing for the commit rule; nothing to draw")
    s = S5.main(["--run", str(run), "--config", str(cfgp)])
    assert s["all_pass"], s["checks"]
    for c in s["cells"]:
        if c["certified"]:
            assert c["curve"][-1] == pytest.approx(c["coverage"]) and c["never_committed"] == pytest.approx(1 - c["coverage"])
            assert ("d90_size" in c) == c["report"]


def test_stage5_refuses_changed_stage4_summary(tmp_path, monkeypatch):
    run, cfgp = make_5(tmp_path, monkeypatch)
    p = run / "poc2" / "stage4" / "stage4_summary.json"
    p.write_text(p.read_text() + " ")
    with pytest.raises(AssertionError, match="stage4_summary.json differs"):
        S5.main(["--run", str(run), "--config", str(cfgp)])


def test_stage5_refuses_changed_stage4_code(tmp_path, monkeypatch):
    run, cfgp = make_5(tmp_path, monkeypatch)
    monkeypatch.setattr(S5, "STAGE4_CODE_SHA256", "0" * 64)
    with pytest.raises(AssertionError, match="r8_stage4.py differs"):
        S5.main(["--run", str(run), "--config", str(cfgp)])


def test_stage5_stops_if_coverages_do_not_reproduce(tmp_path, monkeypatch):
    """A Stage 4 coverage that the regenerated split cannot reproduce must stop the stage (Part A (c))."""
    run, cfgp = make_5(tmp_path, monkeypatch)
    p = run / "poc2" / "stage4" / "stage4_summary.json"
    s4 = json.loads(p.read_text())
    hit = False
    for r in s4["per_seed"]:
        for k, by_m in r["alphas"].items():
            if by_m["commit"]["certified"] and not hit:
                by_m["commit"]["coverage"] += 0.01
                hit = True
    if not hit:
        pytest.skip("the fake run certified nothing for the commit rule")
    p.write_text(json.dumps(s4))
    monkeypatch.setattr(S5, "STAGE4_SUMMARY_SHA256", sha256_file(p))
    with pytest.raises(AssertionError, match="do not reproduce"):
        S5.main(["--run", str(run), "--config", str(cfgp)])
