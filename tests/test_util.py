"""The CPU-only stages must import without torch (Windows Application Control can block torch's DLLs)."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.util import sha256_file  # noqa: E402


def test_sha256_file_matches_hashlib_on_a_multi_chunk_file(tmp_path):
    f = tmp_path / "x.bin"
    data = os.urandom((1 << 20) * 2 + 123)                      # more than two read chunks
    f.write_bytes(data)
    assert sha256_file(f) == hashlib.sha256(data).hexdigest()


def test_stage3_to_6_modules_import_even_when_torch_is_blocked(tmp_path):
    # Poison torch the way Windows does (an OSError at import) and import the CPU-only stage modules in a fresh interpreter.
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text("raise OSError('[WinError 4551] Application Control policy has blocked this file')")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)])}
    r = subprocess.run([sys.executable, "-c", "import src.splits, src.fitdiag, src.stage5, src.certify, src.certify_synth, src.stage6; print('ok')"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]
