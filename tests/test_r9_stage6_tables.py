"""Tests for src/r9_stage6_tables.py: the table rebuilt from the recorded stage6_summary.json equals the 3% table of
the recorded printout line for line; a changed summary is refused; an alpha outside the frozen grid is refused.
Reads the recorded Stage 6 outputs; skipped where they are not present."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import r9_stage6 as R6  # noqa: E402
from src import r9_stage6_tables as TB  # noqa: E402
from src.r9_frozen import blob_id  # noqa: E402
from src.util import sha256_file  # noqa: E402

SUMMARY = ROOT / TB.SUMMARY
PRINTOUT = ROOT / "runs/btsc/stage6/stage6_printout.txt"
PRINTOUT_SHA256 = "b2d6eef1a8f11d673882b01d08ff4ecca0d6c4eb57bdad04bfbd844c1898f30c"   # Part B
PRINTOUT_BLOB = "4dd6c061f9c80345ffc941a90aecb235d1ed4c2b"
recorded = pytest.mark.skipif(not (SUMMARY.is_file() and PRINTOUT.is_file()),
                              reason="recorded Stage 6 outputs not present")


def printed_table(text):
    """Lines of the printed comparison table: from the first '  [' after its header up to the B1 note."""
    lines = text.replace("\r\n", "\n").split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("--- (i) comparison table"))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("  B1 certify counts"))
    return lines[start + 1:end]


@recorded
def test_rebuilt_3pct_table_equals_printout_line_for_line():
    assert sha256_file(PRINTOUT) == PRINTOUT_SHA256 or blob_id(PRINTOUT) == PRINTOUT_BLOB
    J = TB.load_summary(SUMMARY)
    Y = yaml.safe_load((ROOT / R6.YAML_PATH).read_text())
    want = printed_table(PRINTOUT.read_text(encoding="utf-8"))
    got = TB.table_lines(J, Y, float(Y["primary_alpha"]))
    assert len(want) == len(got) == len(R6.COLS) * 11
    for w, g in zip(want, got):
        assert w == g


@recorded
def test_changed_summary_refused(tmp_path):
    p = tmp_path / "s.json"
    p.write_bytes(SUMMARY.read_bytes().replace(b'"primary_alpha": "0.03"', b'"primary_alpha": "0.02"'))
    assert p.read_bytes() != SUMMARY.read_bytes()
    with pytest.raises(SystemExit):
        TB.load_summary(p)


@recorded
def test_alpha_outside_frozen_grid_refused():
    with pytest.raises(SystemExit, match="not in the frozen alpha grid"):
        TB.main(["--alpha", "0.04", "--summary", str(SUMMARY)])
