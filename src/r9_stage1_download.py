"""R9 Stage 1: download one source file and append its record to the downloads manifest.

Why a script: the record must carry the download tool's own exit code, HTTP status, times, size and SHA-256, so that
every later step can check the file against it (hard rule 2). Only URLs listed in Part A (a) are allowed.
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

# Part A (a): the only URLs this step may fetch.
ALLOWED = {
    "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip",
    "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def download(url, out_path, manifest_path, attempt_note=""):
    if url not in ALLOWED:
        raise SystemExit(f"URL not listed in Part A (a): {url}")
    out_path = Path(out_path)
    if out_path.exists():
        # Never overwrite a recorded file; a re-download goes to a new name so both attempts stay on record.
        raise SystemExit(f"Refusing to overwrite {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = utc_now()
    proc = subprocess.run(
        ["curl", "-sS", "--fail", "--max-time", "1200", "-o", str(out_path),
         "-w", "%{http_code} %{url_effective} %{size_download}", url],
        capture_output=True, text=True)
    end = utc_now()
    parts = proc.stdout.split()
    rec = {
        "name": out_path.name, "path": out_path.as_posix(), "url": url,
        "download_start_utc": start, "download_end_utc": end,
        "curl_exit_code": proc.returncode, "curl_stderr": proc.stderr.strip(),
        "http_status": int(parts[0]) if parts else None,
        "url_effective": parts[1] if len(parts) > 1 else None,
        "size_bytes": out_path.stat().st_size if out_path.exists() else None,
        "sha256": sha256_file(out_path) if out_path.exists() else None,
        "note": attempt_note,
    }
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    manifest.setdefault("files", []).append(rec)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="runs/btsc/stage1/downloads_manifest.json")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    download(a.url, a.out, a.manifest, a.note)
