#!/usr/bin/env python3
"""Package generated artefacts and push them to GitHub.

After each trading or training session the raw outputs ``trades_raw.csv``,
``metrics.csv`` and ``model.json`` are compressed into a single
``run_<timestamp>.tar.gz`` archive.  A ``manifest.json`` containing the current
commit hash, schema version and SHA256 checksums of each file is included in the
archive.  The archive is committed to the repository and the raw files are
removed locally. Authentication is performed using the ``GITHUB_TOKEN``
environment variable which must contain a personal access token with permission
to push to the repository.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if os.name != "nt" and (xdg := os.getenv("XDG_DATA_HOME")):
    LOG_DIR = Path(xdg) / "botcopier" / "logs"
else:
    LOG_DIR = REPO_ROOT / "logs"

TRADES_FILE = LOG_DIR / "trades_raw.csv"
METRICS_FILE = LOG_DIR / "metrics.csv"
MODEL_FILE = REPO_ROOT / "model.json"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_archive() -> Path | None:
    """Create a ``run_<timestamp>.tar.gz`` archive with a manifest."""

    files = [TRADES_FILE, METRICS_FILE, MODEL_FILE]
    if not all(p.exists() for p in files):
        return None

    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
        .decode()
        .strip()
    )
    schema_version = os.environ.get("SCHEMA_VERSION", "1.0")
    manifest = {
        "commit": commit,
        "schema_version": schema_version,
        "files": {p.name: {"sha256": _sha256(p)} for p in files},
    }
    manifest_path = LOG_DIR / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive = LOG_DIR / f"run_{timestamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in files + [manifest_path]:
            tar.add(p, arcname=p.name)

    for p in files + [manifest_path]:
        p.unlink()

    return archive


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN environment variable is required", file=sys.stderr)
        return 1

    archive = create_archive()
    if not archive:
        print("Required log files are missing", file=sys.stderr)
        return 0

    run(["git", "add", str(archive)])
    status = (
        subprocess.check_output(
            ["git", "status", "--porcelain", str(archive)], cwd=REPO_ROOT
        )
        .decode()
        .strip()
    )
    if not status:
        print("No changes to commit")
        return 0

    commit_message = f"upload logs {dt.date.today().isoformat()}"
    run(["git", "commit", "-m", commit_message])

    origin_url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=REPO_ROOT
    ).decode().strip()
    if origin_url.startswith("https://"):
        push_url = origin_url.replace("https://", f"https://{token}@")
    elif origin_url.startswith("git@github.com:"):
        repo_path = origin_url.split(":", 1)[1]
        push_url = f"https://{token}@github.com/{repo_path}"
    else:
        push_url = origin_url

    run(["git", "push", push_url, "HEAD"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
