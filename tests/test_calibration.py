import json
import subprocess
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from botcopier.training.pipeline import train


def _make_dataset(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "symbol": ["EURUSD"] * 20 + ["GBPUSD"] * 20,
            "feature": list(range(40)),
            "label": [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10,
        }
    )
    file = tmp_path / "trades_raw.csv"
    df.to_csv(file, index=False)
    return file


def test_calibrated_probabilities_and_symbol_thresholds(tmp_path):
    data_file = _make_dataset(tmp_path)
    out = tmp_path / "out"
    train(data_file, out, n_splits=2)
    model = json.loads((out / "model.json").read_text())
    # verify calibration alters probabilities
    df = pd.read_csv(data_file)
    X = df[["feature"]].to_numpy()
    y = df["label"].to_numpy()
    scaler = StandardScaler().fit(X)
    clf = SGDClassifier(loss="log_loss").fit(scaler.transform(X), y)
    calib = CalibratedClassifierCV(clf, method="isotonic", cv="prefit").fit(
        scaler.transform(X), y
    )
    probs_raw = clf.predict_proba(scaler.transform(X))[:, 1]
    probs_cal = calib.predict_proba(scaler.transform(X))[:, 1]
    assert not np.allclose(probs_raw, probs_cal)
    raw_brier = brier_score_loss(y, probs_raw)
    cal_brier = brier_score_loss(y, probs_cal)
    assert cal_brier <= raw_brier



def test_threshold_lookup_in_mql4(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": [],
                "models": {
                    "logreg": {
                        "coefficients": [1.0],
                        "intercept": 0.0,
                        "threshold": 0.5,
                        "feature_mean": [0.0],
                        "feature_std": [1.0],
                        "conformal_lower": 0.0,
                        "conformal_upper": 1.0,
                    }
                },
                "symbol_thresholds": {"EURUSD": 0.7},
            }
        )
    )
    template_src = Path(__file__).resolve().parents[1] / "StrategyTemplate.mq4"
    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text(template_src.read_text())
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            model,
            "--template",
            template,
        ],
        check=True,
    )
    content = template.read_text()
    assert 'if(s == "EURUSD") return 0.7;' in content
    assert "prob > SymbolThreshold()" in content
