import json
import math
import re
import subprocess
import sys
from pathlib import Path

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def train_simple(x, y, epochs=200, lr=0.1):
    w = 0.0
    b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(x, y):
            z = w * xi + b
            p = sigmoid(z)
            w -= lr * (p - yi) * xi
            b -= lr * (p - yi)
    return w, b

def action(prob, thr):
    if prob > thr:
        return "buy"
    if (1.0 - prob) > thr:
        return "sell"
    return "hold"

def test_python_vs_mql4_parity(tmp_path):
    spreads = [1.0, 1.2, 1.1, 1.4, 1.6, 1.3, 1.5, 1.7, 1.8, 1.9]
    labels = [1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
    w, b = train_simple(spreads, labels)
    model = {
        "feature_names": ["spread"],
        "models": {
            "logreg": {
                "coefficients": [w],
                "intercept": b,
                "threshold": 0.5,
                "feature_mean": [0.0],
                "feature_std": [1.0],
                "conformal_lower": 0.0,
                "conformal_upper": 1.0,
            }
        },
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model))
    template_src = Path(__file__).resolve().parents[1] / "StrategyTemplate.mq4"
    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text(template_src.read_text())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4.py",
            "--model",
            model_path,
            "--template",
            template,
            "--out",
            out_dir / "strategy.mq4",
        ],
        check=True,
    )
    content = (out_dir / "strategy.mq4").read_text()
    coeff_match = re.search(r"g_coeffs_logreg\[] = \{([^}]+)\};", content)
    assert coeff_match
    coeffs = [float(x) for x in coeff_match.group(1).split(",")]
    threshold_match = re.search(r"g_threshold_logreg = ([0-9eE+\-.]+);", content)
    assert threshold_match
    thr = float(threshold_match.group(1))
    b_mql, w_mql = coeffs[0], coeffs[1]
    py_actions = []
    mql_actions = []
    for s in spreads:
        p_py = sigmoid(w * s + b)
        p_mql = sigmoid(w_mql * s + b_mql)
        py_actions.append(action(p_py, thr))
        mql_actions.append(action(p_mql, thr))
    parity = sum(p == m for p, m in zip(py_actions, mql_actions)) / len(spreads)
    assert parity >= 0.95
