"""Stage 8 helper: does the CPU change speed while a benchmark runs?  (pure Python, no numpy, no torch, no model)

  py -3.12 -m src.cpu_probe                       about 90 s; run it from the same plain PowerShell window, laptop plugged in, nothing else open

Why: in three backbone runs every CPU-bound number jumped by roughly 4x at some point (1 thread 23 -> 101 ms, 14 threads 9.7 -> 44.8 ms, GPU path 3.9 -> 15 ms) and a
fixed CPU workload (the "canary") slowed by 2.7-3.4x at the same time. That is the machine, not the model. This probe finds out WHEN and under WHAT CONDITIONS:

  Phase A (idle, first 20 s)        one canary reading per second (best of 3)       -> is the CPU already slow when nothing is running?
  Phase B (sustained load, the rest) one thread running the canary back to back      -> does sustained load push it into a slow state, and does it come back?

The canary is the same loop as `latency.canary_ms` (300,000 integer multiply-adds in pure Python, ~20 ms when the CPU is fast). Readings are grouped into 5 s windows; a window
is SLOW if its median is more than 1.5x the fastest window's median. Also recorded (Windows only, each in try/except): AC power status, the active power plan, and the CPU's
reported performance (% of nominal frequency) once when idle and once under load. Nothing here decides anything about the project; it only explains the instrument.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

SLOW_RATIO = 1.5
REFERENCE_MS = 20.0          # fast-state canary on the author's laptop (20.0 and 20.4 ms at the start of the CPU rows of backbone run 3); use --reference-ms 0 to turn off


def canary_once():
    t0 = time.perf_counter_ns()
    s = 0
    for i in range(300000):
        s += i * i
    return (time.perf_counter_ns() - t0) / 1e6


def window_table(samples, window_s):
    """samples: list of (t_seconds, phase, ms). -> list of {start, phase, n, median_ms, min_ms, max_ms} per window (a window never mixes phases)."""
    out, cur, key = [], [], None
    for t, ph, ms in samples:
        k = (int(t // window_s), ph)
        if key is not None and k != key:
            out.append(cur)
            cur = []
        key = k
        cur.append((t, ph, ms))
    if cur:
        out.append(cur)
    rows = []
    for w in out:
        v = [x[2] for x in w]
        rows.append({"start_s": round(w[0][0], 1), "phase": w[0][1], "n": len(v), "median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)})
    return rows


def classify(rows, ratio=SLOW_RATIO, reference_ms=None):
    """Pure function on the window table. Returns (pattern string F/S per window, verdict key, plain-language verdict).
    A window is judged against the faster of (the fastest window in this probe, reference_ms). Without an outside reference a machine that is slow in EVERY window
    would look 'constant' (that is how the GPU row of backbone run 3 was called stable while 3x slow)."""
    if not rows:
        return "", "no_data", "no readings"
    ref = min(r["median_ms"] for r in rows)
    if reference_ms:
        ref = min(ref, float(reference_ms))
    flags = ["S" if r["median_ms"] > ratio * ref else "F" for r in rows]
    for r, f in zip(rows, flags):
        r["state"] = f
        r["vs_fastest"] = r["median_ms"] / ref
    pat = "".join(flags)
    idle = [f for r, f in zip(rows, flags) if r["phase"] == "idle"]
    load = [f for r, f in zip(rows, flags) if r["phase"] == "load"]
    if "S" not in flags:
        return pat, "constant", "CPU speed stayed constant (all windows within %.1fx of the fastest) in this probe: the slowdown is not triggered by plain single-thread load." % ratio
    if idle and idle.count("S") * 2 >= len(idle):
        return pat, "slow_when_idle", "CPU is already slow when NOTHING is running: a power limit, not load (battery / power plan / OEM 'quiet' mode / Windows efficiency mode for this window)."
    first = load.index("S") if "S" in load else None
    if first is not None and "F" in load[first:]:
        return pat, "oscillates", "CPU is fast at first, slows under sustained load AND comes back: throttling that cycles (thermal or power-limit control, or periodic background work)."
    if first is not None:
        return pat, "slows_under_load", "CPU is fast when idle and stays slow once sustained load starts: thermal or power-limit throttling (check AC adapter wattage, OEM performance mode, fans)."
    return pat, "intermittent", "CPU is fast under load but slow in some idle windows: background activity or power-state changes between bursts."


def ac_status():
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class PS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        p = PS()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p))
        return {"ac_line": {0: "on battery", 1: "plugged in", 255: "unknown"}.get(p.ACLineStatus, p.ACLineStatus), "battery_percent": int(p.BatteryLifePercent),
                "battery_saver_on": bool(p.SystemStatusFlag)}
    except Exception as e:
        return {"error": repr(e)}


def run_text(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def cpu_perf_percent():
    """CPU performance as % of nominal frequency (Windows performance counter, can exceed 100 with turbo)."""
    if not sys.platform.startswith("win"):
        return None
    txt = run_text(["typeperf", r"\Processor Information(_Total)\% Processor Performance", "-sc", "1"], timeout=20)
    for line in txt.splitlines():
        parts = [p.strip().strip('"') for p in line.split('","')]
        if len(parts) >= 2:
            try:
                return float(parts[-1].strip('"'))
            except ValueError:
                continue
    return txt[:200]


def run_probe(seconds=90, idle_seconds=20, window_s=5.0, out=None, quiet=False, reference_ms=REFERENCE_MS):
    say = (lambda *a, **k: None) if quiet else (lambda *a, **k: print(*a, flush=True, **k))
    info = {"platform": platform.platform(), "python": platform.python_version(), "power": ac_status(),
            "power_plan": run_text(["powercfg", "/getactivescheme"]) if sys.platform.startswith("win") else None}
    say(f"power: {info['power']} | plan: {info['power_plan']}")
    samples, t_start = [], time.perf_counter()
    say(f"phase A: idle for {idle_seconds} s (one reading per second) ...")
    while time.perf_counter() - t_start < idle_seconds:
        t0 = time.perf_counter()
        ms = min(canary_once() for _ in range(3))
        samples.append((t0 - t_start, "idle", ms))
        time.sleep(max(0.0, 1.0 - (time.perf_counter() - t0)))
    info["cpu_performance_percent_idle"] = cpu_perf_percent()
    say(f"cpu performance reported while idle: {info['cpu_performance_percent_idle']} %")
    say(f"phase B: sustained single-thread load for {max(0, seconds - idle_seconds)} s ...")
    load_start, sampled_perf = time.perf_counter(), False
    while time.perf_counter() - t_start < seconds:
        t0 = time.perf_counter()
        samples.append((t0 - t_start, "load", canary_once()))
        if not sampled_perf and t0 - load_start > 0.6 * max(1.0, seconds - idle_seconds):
            sampled_perf = True                                                                  # one reading under load (the call itself pauses the loop for ~1-2 s)
            info["cpu_performance_percent_under_load"] = cpu_perf_percent()
            say(f"cpu performance reported under load: {info['cpu_performance_percent_under_load']} %")
    info["power_after"] = ac_status()
    rows = window_table(samples, window_s)
    pat, key, text = classify(rows, reference_ms=reference_ms)
    say("\nwindow  phase  median canary (ms)   min / max      state   vs fastest")
    for r in rows:
        say(f"{r['start_s']:>5.0f}s  {r['phase']:<5}  {r['median_ms']:>10.1f}        {r['min_ms']:>6.1f} / {r['max_ms']:<7.1f}  {r['state']}       {r['vs_fastest']:.2f}x")
    ref = min(r["median_ms"] for r in rows) if rows else float("nan")
    say(f"\nreference fast-state canary: {reference_ms} ms (windows are called SLOW above {SLOW_RATIO}x the faster of that and the fastest window)")
    say(f"pattern (F = fast, S = slow, one letter per {window_s:.0f} s window): {pat}")
    say(f"fastest window median {ref:.1f} ms (a fast CPU gives about 20 ms; the three backbone runs showed 20 ms at the start of a row and 55-68 ms after the first round)")
    say(f"verdict: {text}")
    result = {"info": info, "window_s": window_s, "slow_ratio": SLOW_RATIO, "windows": rows, "pattern": pat, "verdict_key": key, "verdict": text,
              "fastest_window_median_ms": ref, "reference_ms": reference_ms, "n_samples": len(samples), "samples_ms": [[round(t, 3), ph, round(ms, 3)] for t, ph, ms in samples]}
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))
        say(f"saved {out}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU-speed probe (idle, then sustained single-thread load)")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--idle-seconds", type=float, default=20.0)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--out", default="runs/20261003_stage1/stage8/cpu_probe.json")
    ap.add_argument("--reference-ms", type=float, default=REFERENCE_MS, help="canary time of this machine in its fast state; 0 = judge only against the fastest window of this probe")
    a = ap.parse_args(argv)
    return run_probe(a.seconds, a.idle_seconds, a.window, a.out, reference_ms=a.reference_ms or None)


if __name__ == "__main__":
    main()
