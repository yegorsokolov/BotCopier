#!/usr/bin/env python3
"""Write updated model parameters to the MT4 Files directory."""

import argparse
import json
from pathlib import Path


def publish(model_json: Path, files_dir: Path) -> None:
    """Copy minimal model fields to ``files_dir/model.json``."""
    with open(model_json) as f:
        data = json.load(f)

    out = {
        "coefficients": data.get("coefficients") or data.get("coef_vector", []),
        "intercept": data.get("intercept", 0.0),
        "threshold": data.get("threshold", 0.5),
        "hourly_thresholds": data.get("hourly_thresholds", []),
        "probability_table": data.get("probability_table", []),
    }

    files_dir.mkdir(parents=True, exist_ok=True)
    dest = files_dir / "model.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Model parameters written to {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description="Publish model parameters")
    p.add_argument("model_json", help="path to trained model.json")
    p.add_argument("files_dir", help="MT4 Files directory")
    args = p.parse_args()
    publish(Path(args.model_json), Path(args.files_dir))


if __name__ == "__main__":
    main()
