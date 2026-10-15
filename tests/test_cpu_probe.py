import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cpu_probe as P  # noqa: E402


def _rows(idle, load, ms_fast=20.0, ms_slow=65.0):
    """idle / load: strings of F and S, one window each."""
    rows, t = [], 0.0
    for ph, s in (("idle", idle), ("load", load)):
        for c in s:
            rows.append({"start_s": t, "phase": ph, "n": 5, "median_ms": ms_fast if c == "F" else ms_slow, "min_ms": 1.0, "max_ms": 1.0})
            t += 5
    return rows


def test_classify_constant_idle_slow_throttle_oscillation_intermittent():
    assert P.classify(_rows("FFFF", "FFFFFFFF"))[1] == "constant"
    assert P.classify(_rows("SSSS", "SSSSSSSS"))[1] == "constant"                       # uniformly slow, no outside reference: indistinguishable from uniformly fast (the loophole)
    assert P.classify(_rows("SSSS", "SSSSSSSS"), reference_ms=20.0)[1] == "slow_when_idle"      # ... which the reference closes
    assert P.classify(_rows("FFFF", "FFFFFFFF"), reference_ms=20.0)[1] == "constant"
    assert P.classify(_rows("FFFF", "FFSSSSSS"))[1] == "slows_under_load"
    assert P.classify(_rows("FFFF", "FFSSFFSS"))[1] == "oscillates"
    assert P.classify(_rows("FSFF", "FFFFFFFF"))[1] == "intermittent"
    pat, key, text = P.classify(_rows("FFFF", "FFSSFFSS"))
    assert pat == "FFFFFFSSFFSS" and "comes back" in text


def test_small_jitter_is_not_called_slow():
    rows = _rows("FFFF", "FFFF", ms_fast=20.0)
    for i, r in enumerate(rows):
        r["median_ms"] = 20.0 + (i % 3) * 2.0               # up to +20%
    assert P.classify(rows)[1] == "constant"


def test_window_table_never_mixes_phases_and_uses_medians():
    s = [(0.0, "idle", 10.0), (1.0, "idle", 12.0), (2.0, "idle", 14.0), (5.2, "idle", 30.0), (5.9, "load", 50.0), (6.1, "load", 70.0)]
    rows = P.window_table(s, 5.0)
    assert [r["phase"] for r in rows] == ["idle", "idle", "load"]
    assert rows[0]["median_ms"] == 12.0 and rows[1]["median_ms"] == 30.0 and rows[2]["median_ms"] == 60.0 and rows[2]["n"] == 2


def test_short_real_probe_runs_and_writes_a_complete_json(tmp_path):
    out = tmp_path / "p" / "cpu_probe.json"
    r = P.run_probe(seconds=4, idle_seconds=1.5, window_s=1.0, out=out, quiet=True, reference_ms=None)
    d = json.loads(out.read_text())
    assert d["verdict_key"] in {"constant", "slow_when_idle", "slows_under_load", "oscillates", "intermittent"}
    assert d["n_samples"] > 20 and d["windows"] and set(d["windows"][0]) >= {"phase", "median_ms", "state", "vs_fastest"}
    assert d["fastest_window_median_ms"] > 0 and len(d["pattern"]) == len(d["windows"]) and r["pattern"] == d["pattern"]
    assert {"idle", "load"} == {w["phase"] for w in d["windows"]}


def test_probe_canary_workload_matches_the_benchmarks_canary():
    from src import latency
    a = min(P.canary_once() for _ in range(5))
    b = latency.canary_ms(5)
    assert 0.5 < a / b < 2.0
