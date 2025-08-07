#!/usr/bin/env python3
"""Commit generated log files and push them to GitHub.

The script adds any ``*.csv.gz`` files under ``logs/`` to the Git repository,
commits them if they have changed and pushes the new commit to the ``origin``
remote. Authentication is performed using the ``GITHUB_TOKEN`` environment
variable which must contain a personal access token with permission to push to
the repository.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN environment variable is required", file=sys.stderr)
        return 1

    logs = sorted(LOG_DIR.glob("*.csv.gz"))
    existing = [str(p) for p in logs if p.exists()]
    if not existing:
        print("No log files found", file=sys.stderr)
        return 0

    run(["git", "add", *existing])
    status = subprocess.check_output(
        ["git", "status", "--porcelain", *existing], cwd=REPO_ROOT
    ).decode().strip()
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
