"""Tests for src/r9_stage8_tables.py.

Every table cell is checked against a brute-force reading of its source JSON (independent code: statistics.median,
own formatting); the BelgiumTSC 3% table and cost table must equal what Stage 6 recorded; the document on disk must
equal a fresh build; changed sources are refused. Reads the committed summaries; skipped where they are absent."""
import json
import shutil
import statistics
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import r9_frozen as FZ  # noqa: E402
from src import r9_stage6 as R6  # noqa: E402
from src import r9_stage8_tables as S8  # noqa: E402

present = all((ROOT / v[0]).is_file() for v in S8.SOURCES.values())
recorded = pytest.mark.skipif(not present, reason="recorded summaries not present")
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.005, 0.01, 0.02, 0.03, 0.05)


@pytest.fixture(scope="module")
def J():
    return {n: json.loads((ROOT / v[0]).read_text(encoding="utf-8"))
            for n, v in S8.SOURCES.items() if v[0].endswith(".json")}


@pytest.fixture(scope="module")
def Y():
    return yaml.safe_load((ROOT / R6.YAML_PATH).read_text())


# ------------------------------------------------------------------ brute-force formatting (independent of R6.mr)
def bf_mr(vals, pct=True):
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    f = (lambda x: f"{100 * x:.2f}%") if pct else (lambda x: f"{x:g}")
    return f"{f(statistics.median(v))} ({f(min(v))}-{f(max(v))}; {len(v)})"


def bf_ci(c):
    return f"[{100 * c[0]:.2f}, {100 * c[1]:.2f}]"


# ------------------------------------------------------------------ sources and document
@recorded
def test_every_source_passes_its_hash_check():
    T, J_ = S8.load_all()
    assert set(T) == set(S8.SOURCES)


@recorded
def test_document_on_disk_equals_a_fresh_build():
    FZ.check_frozen()
    assert (ROOT / S8.OUT).read_text(encoding="utf-8") == S8.build()


@recorded
def test_changed_json_source_refused(tmp_path):
    for v in S8.SOURCES.values():
        (tmp_path / v[0]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / v[0], tmp_path / v[0])
    S8.load_text("r9_stage3", root=tmp_path)                          # unchanged copy passes
    p = tmp_path / S8.SOURCES["r9_stage3"][0]
    p.write_text(p.read_text(encoding="utf-8").replace('"n_cert": 784', '"n_cert": 785', 1), encoding="utf-8")
    with pytest.raises(SystemExit):
        S8.load_text("r9_stage3", root=tmp_path)


@recorded
def test_changed_copied_log_entry_refused(tmp_path):
    rel = S8.SOURCES["r9_stage6_entry"][0]
    (tmp_path / rel).write_bytes((ROOT / rel).read_bytes().replace(b"1.44%", b"1.45%", 1))
    with pytest.raises(SystemExit):
        S8.load_text("r9_stage6_entry", root=tmp_path)


def test_missing_source_refused(tmp_path):
    with pytest.raises(SystemExit):
        S8.load_text("r9_stage2", root=tmp_path)


def test_line_ending_only_difference_accepted(tmp_path):
    rel = "x.json"
    data = b'{\n "a": {\n  "b": 1\n }\n}\n'
    (tmp_path / rel).write_bytes(data.replace(b"\n", b"\r\n"))      # CRLF copy; SHA-256 deliberately wrong
    (tmp_path / "lf.json").write_bytes(data)
    src = {"x": (rel, "0" * 64, FZ.blob_id(tmp_path / "lf.json"), "")}
    assert json.loads(S8.load_text("x", root=tmp_path, sources=src)) == {"a": {"b": 1}}


def test_json_line_synthetic():
    text = '{\n  "a": {\n    "b": 1,\n    "c": {\n      "b": 2\n    }\n  },\n  "c": {\n    "b": 3\n  }\n}'
    assert S8.json_line(text, "a") == 2
    assert S8.json_line(text, "a", "c", "b") == 5
    assert S8.json_line(text, "c", "b") == 9                           # the top-level c, not a.c
    with pytest.raises(SystemExit):
        S8.json_line(text, "a", "d")


@recorded
def test_cited_json_lines_hold_their_keys():
    T, _ = S8.load_all()
    for name, keys in (("r9_stage3", ("results",)), ("r8_stage3b", ("results", "0.01", "B1")),
                       ("r9_stage6", ("cost",)), ("r9_stage1", ("dits", "F-A3"))):
        line = T[name].replace("\r\n", "\n").split("\n")[S8.json_line(T[name], *keys) - 1]
        assert line.strip().startswith(json.dumps(keys[-1]) + ":")


@recorded
def test_check_frozen_runs_before_any_source_is_read(monkeypatch):
    calls = []
    monkeypatch.setattr(S8.FZ, "check_frozen", lambda: calls.append("frozen") or (_ for _ in ()).throw(SystemExit(1)))
    monkeypatch.setattr(S8, "build", lambda *a, **k: calls.append("build"))
    with pytest.raises(SystemExit):
        S8.main(["--out", "unused.md"])
    assert calls == ["frozen"]


# ------------------------------------------------------------------ table 1: GTSRB at 1%
@recorded
def test_gtsrb_layer_cells(J):
    T = S8.gtsrb_table(J, 0.01)
    for s in S8.SCORES:
        cells = [J["r9_stage3"]["results"][s][str(seed)] for seed in SEEDS]
        cert = [c for c in cells if c["cert"]["0.01"]["binom"] >= 0]
        ms = [c["test"]["0.01"]["binom"] for c in cert]
        errs = [m["error"] for m in ms]
        sa = [c["smallest_alpha"]["binom"] for c in cells]
        col = T[s]
        assert col["Certified in how many re-splits"] == f"{len(cert)}/5"
        assert col["Test coverage"] == bf_mr([m["coverage"] for m in ms])
        assert col["Test error among committed"] == bf_mr(errs)
        assert col["Worst re-split error vs target"] == \
            f"{100 * max(errs):.2f}% vs 1% ({sum(e > 0.01 + 1e-12 for e in errs)} above)"
        assert col["Smallest certifiable alpha"] == \
            bf_mr(sa) + " | per seed " + ", ".join(f"{100 * x:g}%" for x in sa)
        assert col["Median commit frame (or views)"] == bf_mr([m["median_frame"] for m in ms], False)
        assert col["Median sign size at commit"] == bf_mr([m["median_size"] for m in ms], False)
    # the Stage 3 Part B values at 1% (coverage medians and commit frames)
    assert [T[s]["Test coverage"].split("%")[0] for s in S8.SCORES] == ["81.30", "90.84", "89.31", "85.88", "56.49"]
    assert [T[s]["Median commit frame (or views)"].split(" ")[0] for s in S8.SCORES] == ["17", "5", "5", "21", "2"]


@recorded
def test_gtsrb_baseline_cells(J):
    T = S8.gtsrb_table(J, 0.01)
    b1 = J["r8_stage3b"]["results"]["0.01"]["B1"]
    assert T["B1"]["Violation rate vs delta (resampling)"].startswith(
        f"{100 * b1['violation_rate']:.2f}% {bf_ci(b1['cp95'])}")
    b2 = [r["alphas"]["0.01"]["B2"] for r in J["r8_stage4"]["per_seed"]]
    assert T["B2"]["Certified in how many re-splits"] == f"{sum(m['certified'] for m in b2)}/5"
    P6 = J["r8_stage6"]["per_seed"]
    for c in ("B5", "B5c"):
        ms = [r[c.lower()]["0.01"] for r in P6]
        assert T[c]["Test coverage"] == bf_mr([m["coverage"] for m in ms])
        assert T[c]["Test error among committed"] == bf_mr([m["error_committed"] for m in ms]) + " (real error)"
        assert f"gap {100 * max(m['gap_marginal'] for m in ms):.2f}%" in T[c]["Worst re-split error vs target"]
        assert f"real error {100 * max(m['error_committed'] for m in ms):.2f}%" in T[c]["Worst re-split error vs target"]
    for v, k in S8.B6_KEYS.items():
        ms = [r["b6"][k]["alphas"]["0.01"] for r in P6]
        errs = [m["error_committed"] for m in ms]
        col = T[f"B6 {v}"]
        assert col["Test coverage"] == bf_mr([m["coverage"] for m in ms])
        assert col["Test error among committed"] == bf_mr(errs)
        assert col["Worst re-split error vs target"] == \
            f"{100 * max(errs):.2f}% vs r = 1% ({sum(e > 0.01 + 1e-12 for e in errs)} above)"
        assert col["Median commit frame (or views)"] == bf_mr([m["median_commit_frame"] for m in ms], False)
    # POC-2 Stage 6 recorded table cells copied as they are
    rows = J["r8_stage6"]["tables"]["main_0.01"]["rows"]
    assert T["B5"]["Type of promise"] == rows["1 Type of promise"]["B5"]
    assert T["B1"]["Smallest certifiable alpha"].startswith(rows["5 Smallest certifiable alpha, n = 784"]["B1"])
    # every column names its source and test; every baseline header says it is a POC-2 record
    assert all(T[c]["Source / test"] for c in T)
    md = "\n".join(S8.comparison_md(T, list(T), S8.GTSRB_ROWS, S8.GTSRB_LABELS))
    assert "| Source / test |" in md
    assert all("POC-2 record" in S8.GTSRB_LABELS[c] for c in T if c.startswith("B"))


# ------------------------------------------------------------------ table 2: BelgiumTSC
@recorded
def test_btsc_primary_table_equals_recorded_comparison(J, Y):
    assert S8.btsc_table(J, Y, float(Y["primary_alpha"])) == J["r9_stage6"]["comparison"]


@recorded
def test_btsc_1pct_cells(J, Y):
    T = S8.btsc_table(J, Y, 0.01)
    rq2 = J["r9_stage6"]["rq2"]
    for s in S8.SCORES:
        cells = [rq2[str(seed)]["scores"][s] for seed in SEEDS]
        cert = [c for c in cells if c["cert"]["0.01"]["binom"] is not None and c["cert"]["0.01"]["binom"] >= 0]
        ms = [c["test"]["0.01"]["binom"] for c in cert]
        assert T[s]["Certified in how many re-splits"] == f"{len(cert)}/5"
        assert T[s]["Test coverage"] == bf_mr([m["coverage"] for m in ms])
        assert T[s]["Test error among committed"] == bf_mr([m["error"] for m in ms])
        r = J["r9_stage6"]["rq1"][s]["cells"]["0.01"]["binom"]
        assert T[s]["Violation rate vs delta (resampling)"].startswith(f"{100 * r['rate']:.2f}% {bf_ci(r['cp95'])}")
    for v in ("S2-sum", "S2-min"):
        ms = [rq2[str(seed)]["b6"][v]["0.01"] for seed in SEEDS]
        assert T[f"B6 {v}"]["Test coverage"] == bf_mr([m["coverage"] for m in ms])
        assert T[f"B6 {v}"]["Test error among committed"] == bf_mr([m["error"] for m in ms])


# ------------------------------------------------------------------ table 3: RQ1
@recorded
def test_rq1_gtsrb_rows(J):
    rows = S8.rq1_gtsrb_rows(J["r8_stage3b"])
    res = J["r8_stage3b"]["results"]
    assert [r[0] for r in rows] == [f"{100 * float(a):g}%" for a in sorted(res, key=float)]
    for row, a in zip(rows, sorted(res, key=float)):
        for cell, m in zip(row[1:], ("B1", "track")):
            r = res[a][m]
            assert cell == (f"{100 * r['violation_rate']:.2f}% {bf_ci(r['cp95'])} "
                            f"(certified share {100 * r['certified_share']:.1f}%)")


@recorded
def test_rq1_simulation_worst_cases(J):
    settings = [s for p in J["r9_stage2"]["pairs"] for s in p["settings"]]
    rows = S8.rq1_sim_rows(J["r9_stage2"])
    for row, cl in zip(rows, S8.CLUSTERS):
        ss = [s for s in settings if s["cluster"] == cl]
        assert row[1] == str(len(ss))
        for cell, (k, _) in zip(row[2:], S8.SIM_ROWS):
            top = sorted(ss, key=lambda s: -s["rows"][k]["violation_rate"])[0]["rows"][k]
            assert cell.startswith(f"{100 * top['violation_rate']:.2f}% {bf_ci(top['cp95'])}")
    # the largest sign-level rate recorded in Stage 2 Part B: 5.35%, fixed-30, rho 0.45, alpha 2%
    assert rows[2][2].startswith("5.35% [4.40, 6.43] at P_FRAME 7.4%, rho 0.45, alpha 2%")


@recorded
def test_rq1_btsc_rows(J):
    methods, rows = S8.rq1_btsc_rows(J["r9_stage6"])
    assert methods == ["S1", "S2-sum", "S2-min", "S3", "S4", "B1"]
    for row, a in zip(rows, ALPHAS):
        for cell, m in zip(row[1:], methods):
            parts = cell.split(" / ")
            for part, rule in zip(parts, ("binom", "hb")):
                c = J["r9_stage6"]["rq1"][m]["cells"][str(a)][rule]
                want = "no start" if c["no_start"] == c["draws"] else \
                    f"{100 * c['rate']:.2f}% {bf_ci(c['cp95'])} ({c['certified']} certified)"
                assert part == want
    assert rows[0][-1].startswith("14.00% [12.51, 15.60] (1083 certified)")   # Stage 6 Part B


# ------------------------------------------------------------------ table 4: exact binomial against HB
def _bf_binom_hb(cells, a, r):
    cs = [c for c in cells if c["cert"][a][r] is not None and c["cert"][a][r] >= 0]
    return f"{len(cs)}/{len(cells)}; {bf_mr([c['test'][a][r]['coverage'] for c in cs])}"


@recorded
def test_binom_hb_rows_both_datasets(J):
    g = {s: [J["r9_stage3"]["results"][s][str(seed)] for seed in SEEDS] for s in S8.SCORES}
    b = {s: [J["r9_stage6"]["rq2"][str(seed)]["scores"][s] for seed in SEEDS] for s in S8.SCORES}
    for raw, cells in ((g, S8.stage3_cells(J["r9_stage3"])), (b, S8.btsc_cells(J["r9_stage6"]))):
        rows = iter(S8.binom_hb_rows(cells))
        for s in S8.SCORES:
            for a in ALPHAS:
                row = next(rows)
                assert row[:2] == [s, f"{100 * a:g}%"]
                assert row[2:] == [_bf_binom_hb(raw[s], str(a), r) for r in S8.RULES]
        for row, s in zip(S8.smallest_rows(cells), S8.SCORES):
            assert row[1:] == [", ".join("none" if c["smallest_alpha"][r] is None else f"{100 * float(c['smallest_alpha'][r]):g}%"
                                         for c in raw[s]) for r in S8.RULES]
    # Stage 3 Part B: HB primary start certified 1% in 5 / 5 / 2 / 1 / 0 re-splits
    hb1 = [r[3].split(";")[0] for r in S8.binom_hb_rows(S8.stage3_cells(J["r9_stage3"])) if r[1] == "1%"]
    assert hb1 == ["5/5", "5/5", "2/5", "1/5", "0/5"]


# ------------------------------------------------------------------ table 5: S2 against B6
@recorded
def test_cost_function_reproduces_recorded_btsc_cost(J):
    from src import r9_stage6_tables as TB
    _, rq2 = TB.rebuild(J["r9_stage6"])
    assert R6.cost_table(rq2, list(ALPHAS)) == S8.btsc_cost(J["r9_stage6"])


@recorded
def test_gtsrb_cost_cells(J):
    C = S8.gtsrb_cost(J)
    P6 = J["r8_stage6"]["per_seed"]
    for v, k in S8.B6_KEYS.items():
        for a in ALPHAS:
            cells = [J["r9_stage3"]["results"][v][str(seed)] for seed in SEEDS]
            cert = [c["cert"][str(a)]["binom"] >= 0 for c in cells]
            lay = [c["test"][str(a)]["binom"] for c, ok in zip(cells, cert) if ok]
            ref = [c["reference"][str(a)] for c in cells]
            b6 = [r["b6"][k]["alphas"][f"{a:g}"] for r in P6]
            row = C[(v, a)]
            assert row["layer (exact binomial)"]["seeds"] == f"{sum(cert)}/5 certified"
            assert row["layer (exact binomial)"]["coverage"] == bf_mr([m["coverage"] for m in lay])
            assert row["layer (exact binomial)"]["median_view"] == bf_mr([m["median_frame"] for m in lay], False)
            assert row["uncertified reference"]["error"] == bf_mr([m["error"] for m in ref])
            assert row["B6 (uncertified, r = alpha)"]["coverage"] == bf_mr([m["coverage"] for m in b6])
            assert row["B6 (uncertified, r = alpha)"]["error"] == bf_mr([m["error_committed"] for m in b6])
            errs = [m["error_committed"] for m in b6 if m["error_committed"] is not None]
            assert row["B6 (uncertified, r = alpha)"]["worst"] == f"{100 * max(errs):.2f}%"
            assert row["B6 (uncertified, r = alpha)"]["above_alpha"] == sum(e > a + 1e-12 for e in errs)


# ------------------------------------------------------------------ table 6: Ringel
@recorded
def test_ringel_rows(J):
    g, b = S8.ringel_sources(J)
    for src, rows in ((g, S8.ringel_rows(g)), (b, S8.ringel_rows(b))):
        it = iter(rows)
        for c in ("B5", "B5c"):
            for a in ALPHAS:
                row = next(it)
                ms = [seed[f"{a:g}"] for seed in src[c]]
                assert row[2] == ", ".join(f"{100 * m['gap_marginal']:.2f}" for m in ms)
                assert row[3] == ", ".join(f"{100 * m['error_committed']:.2f}" for m in ms)
                assert row[4] == str(sum(m["error_committed"] > a + 1e-12 for m in ms))
                assert row[5] == bf_mr([m["coverage"] for m in ms])
    # Stage 6 Part B: B5 real error 1.29, 0.65, 0.62, 0.65, 1.90 at 0.5%
    assert S8.ringel_rows(b)[0][3] == "1.29, 0.65, 0.62, 0.65, 1.90"


# ------------------------------------------------------------------ table 7: limitations
@recorded
def test_plan_limitations_copied_verbatim():
    plan = (ROOT / S8.PLAN).read_text(encoding="utf-8")
    lims, _ = S8.plan_limitations(plan)
    assert len(lims) == 7
    for x in lims:
        assert f"- {x}\n" in plan.replace("\r\n", "\n")


@recorded
def test_additions_numbers_from_sources(J, Y):
    add = S8.additions(J, Y)
    assert len(add) == 4
    assert f"n_c = {J['r9_stage2']['V1']['min_n_engine'][0]}" in add[0]
    assert f"{len(Y['scores']['S2-sum']['grid'])} points (S2-sum)" in add[1]
    assert f"{100 * J['r9_stage6']['rq1']['S1']['risk'][0]:.2f}%" in add[2]
    assert f"F-A3 {J['r9_stage1']['dits']['F-A3']}" in add[3]
