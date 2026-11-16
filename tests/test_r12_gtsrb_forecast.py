"""r12: synthetic tests (LOG_r12_entry.md Part A (h)). No GTSRB image, embedding or track table is read; the tamper and
loader tests read only the committed POC-2 stage3b_summary.json (its hash and config)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from src import r11_forecast_ucb as F
from src import r12_gtsrb_forecast as G
from src import r9_certify as C


def synth_pilot(seed, nd=261, m=30, G_=20, base=0.008):
    """Per-track accepted and accepted-and-wrong counts with track-level difficulty; acceptance falls along the grid."""
    rng = np.random.default_rng(seed)
    acc_frac = np.linspace(0.83, 0.71, G_)
    acc = rng.binomial(m, acc_frac[None, :].repeat(nd, 0))
    p_track = rng.beta(0.4, 0.4 / base, nd)
    shrink = np.linspace(1.0, 0.3, G_)
    wr = rng.binomial(acc, np.clip(p_track[:, None] * shrink[None, :] * 1.0, 0, 1))
    return acc, wr


# ------------------------------------------------------------------ the forecast is r11's, unchanged
@pytest.mark.parametrize("seed,alpha,test", [(0, 0.005, "hb"), (1, 0.01, "hb"), (2, 0.01, "binom"), (3, 0.02, "binom")])
def test_pilot_forecast_equals_r11_run_forecast(seed, alpha, test):
    acc, wr = synth_pilot(seed)
    rec = G.pilot_forecast(acc, wr, alpha, test)
    start = C.start_for(test, acc.sum(0) / (261 * 30), 784 * 30, alpha, 261 * 30)
    ref = F.run_forecast(acc, wr, 30, alpha, test, start, 261, 784)
    assert rec["start"] == start
    for k, v in ref.items():
        assert rec[k] == v, k


def test_start_is_19_for_gtsrb_like_fractions():
    """Part A (b): with accept fractions near GTSRB's (71-83%) and n = 23,520, both rules start at the top position."""
    f = np.linspace(0.832, 0.707, 20)
    for a in (0.005, 0.01, 0.02):
        for t in ("hb", "binom"):
            assert C.start_for(t, f, 784 * 30, a, 261 * 30) == 19


# ------------------------------------------------------------------ observed exact-binomial walk
def test_observed_binom_against_loop():
    rng = np.random.default_rng(7)
    T, Gg, D = 60, 20, 40
    A = rng.integers(15, 31, (T, Gg))
    W = rng.binomial(A, 0.012)
    Wd = np.stack([np.bincount(rng.integers(0, T, 50), minlength=T) for _ in range(D)]).astype(float)
    for a in (0.01, 0.02, 0.05):
        got = G.observed_binom(A, W, Wd, a, start=19)
        for i in range(D):
            k, n = Wd[i] @ W, Wd[i] @ A
            best = -1
            for j in range(19, -1, -1):
                if binom.cdf(k[j], n[j], a) > C.DELTA:
                    break
                best = j
            assert got[i] == best


def test_per_track_counts_against_loop():
    rng = np.random.default_rng(3)
    n_tracks, nfr, Gg = 12, 300, 5
    inv = rng.integers(0, n_tracks, nfr)
    acc = rng.random((nfr, Gg)) < 0.7
    wr = acc & (rng.random((nfr, Gg)) < 0.1)
    A, W, NF = G.per_track(inv, acc, wr, n_tracks)
    for t in range(n_tracks):
        assert NF[t] == (inv == t).sum()
        assert (A[t] == acc[inv == t].sum(0)).all() and (W[t] == wr[inv == t].sum(0)).all()


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_recomputes_every_step():
    acc, wr = synth_pilot(5)
    idx = np.random.default_rng([1, 2]).integers(0, 261, (4, 261))
    bs = G.bootstrap(acc, wr, idx, 0.01, "hb")
    for r, b in zip(idx, bs):
        a2, w2 = acc[r], wr[r]
        start = C.start_for("hb", a2.sum(0) / (261 * 30), 784 * 30, 0.01, 261 * 30)
        U = F.cp_upper(w2.sum(0), a2.sum(0))
        js = next((j for j in range(start, -1, -1) if U[j] > 0.01), -1)
        assert b["j_star"] == js and b["start"] == start
        assert b["forecast"] == F.run_forecast(a2, w2, 30, 0.01, "hb", start, 261, 784)["forecast"]


def test_bootstrap_key_is_new():
    assert G.BOOT_KEY[0] not in (20261114, 20261115, 20261116, 20261117, 20261118)


# ------------------------------------------------------------------ criteria (Part A (f))
@pytest.mark.parametrize("f,o,want", [(0.20, 0.17, (True, True)), (0.2001, 0.17, (False, True)),
                                      (0.15, 0.17, (True, True)), (0.1499, 0.17, (True, False)),
                                      (0.13, 0.17, (False, False)), (0.47, 0.20, (False, True))])
def test_criteria_hand_cases(f, o, want):
    c = G.criteria(f, o)
    assert (c["a_abs_le_3"], c["b_not_below_2"]) == want


def test_heuristic_value():
    from src import r9_stage10 as S10
    assert abs(100 * S10.de_to_delta_eff(G.DE_GTSRB) - 36.55) < 0.005


# ------------------------------------------------------------------ tamper test (hard rule 2)
def test_tampered_stage3b_refused(tmp_path):
    src = Path("runs/20261003_stage1/poc2/stage3/stage3b_summary.json")
    bad = tmp_path / "stage3b_summary.json"
    shutil.copy(src, bad)
    with open(bad, "ab") as fh:
        fh.write(b" ")
    with pytest.raises(SystemExit) as e:
        G.main(["--stage3b", str(bad), "--out", str(tmp_path / "out")])
    assert "SHA-256" in str(e.value) and not (tmp_path / "out").exists()


def test_stage3b_loader_accepts_recorded_file():
    d = G.load_stage3b(Path("runs/20261003_stage1/poc2/stage3/stage3b_summary.json"))
    assert d["config"]["draws"] == 2000 and d["config"]["n_cert"] == 784
