#!/usr/bin/env python3
"""Write updated model parameters to the MT4 Files directory."""

import argparse
import json
import gzip
import shutil
from pathlib import Path


def publish(model_path: Path, files_dir: Path) -> None:
    """Copy model artifacts to ``files_dir``.

    If ``model_path`` points to an ONNX file (or a directory containing one),
    that file is copied directly.  Otherwise minimal fields from a JSON model
    are extracted and written to the destination, preserving compression.
    """

    if model_path.is_dir():
        for name in ["model.onnx", "model.json", "model.json.gz"]:
            candidate = model_path / name
            if candidate.exists():
                model_path = candidate
                break

    if model_path.suffix == ".onnx":
        files_dir.mkdir(parents=True, exist_ok=True)
        dest = files_dir / "model.onnx"
        shutil.copy(model_path, dest)
        print(f"ONNX model copied to {dest}")
        return

    open_func = gzip.open if model_path.suffix == ".gz" else open
    with open_func(model_path, "rt") as f:
        data = json.load(f)

    out = {
        "coefficients": data.get("coefficients") or data.get("coef_vector", []),
        "intercept": data.get("intercept", 0.0),
        "threshold": data.get("threshold", 0.5),
        "hourly_thresholds": data.get("hourly_thresholds", []),
        "probability_table": data.get("probability_table", []),
    }

    files_dir.mkdir(parents=True, exist_ok=True)
    dest = files_dir / model_path.name
    open_func_out = gzip.open if dest.suffix == ".gz" else open
    with open_func_out(dest, "wt") as f:
        json.dump(out, f, indent=2)
    print(f"Model parameters written to {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description="Publish model parameters")
    p.add_argument("model", help="path to trained model directory or file")
    p.add_argument("files_dir", help="MT4 Files directory")
    args = p.parse_args()
    publish(Path(args.model), Path(args.files_dir))


if __name__ == "__main__":
    main()
