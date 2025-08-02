#!/usr/bin/env python3
"""Plot feature importances stored in model.json."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean absolute SHAP feature importances"
    )
    parser.add_argument("model", type=Path, help="Path to model.json")
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top features to display"
    )
    args = parser.parse_args()

    data = json.loads(args.model.read_text())
    importance = data.get("feature_importance", {})
    if not importance:
        print("No feature_importance data found in model.json")
        return

    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[: args.top]
    features, values = zip(*items)

    plt.figure(figsize=(8, max(4, len(features) * 0.4)))
    plt.barh(features, values)
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
