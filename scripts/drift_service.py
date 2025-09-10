#!/usr/bin/env python3
"""Background service to periodically run ``drift_monitor.py``."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from logging_utils import setup_logging


def _run_monitor(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("drift_monitor.py")),
        "--baseline-file",
        str(args.baseline_file),
        "--recent-file",
        str(args.recent_file),
        "--drift-threshold",
        str(args.drift_threshold),
        "--model-json",
        str(args.model_json),
        "--log-dir",
        str(args.log_dir),
        "--out-dir",
        str(args.out_dir),
        "--files-dir",
        str(args.files_dir),
    ]
    if args.flag_file is not None:
        cmd += ["--flag-file", str(args.flag_file)]
    with subprocess.Popen(cmd) as proc:
        proc.wait()


def main() -> int:
    p = argparse.ArgumentParser(description="Run drift monitor in a loop")
    p.add_argument("--baseline-file", type=Path, required=True)
    p.add_argument("--recent-file", type=Path, required=True)
    p.add_argument("--drift-threshold", type=float, default=0.2)
    p.add_argument("--model-json", type=Path, default=Path("model.json"))
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--files-dir", type=Path, required=True)
    p.add_argument("--flag-file", type=Path)
    p.add_argument("--interval", type=int, default=3600, help="seconds between checks")
    args = p.parse_args()

    logger = setup_logging(__name__)

    try:
        while True:
            try:
                _run_monitor(args)
            except subprocess.SubprocessError:
                logger.exception("drift monitor failed")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("drift service stopping")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
