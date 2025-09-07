import json
import subprocess
import sys
from pathlib import Path

from scripts.train_target_clone import train


def test_session_conformal_bounds_written(tmp_path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,profit,hour,spread\n"
        "1,1.0,1,1.0\n"
        "0,-0.5,2,1.1\n"
        "1,0.2,9,1.2\n"
        "0,-0.3,10,1.3\n"
        "1,0.4,17,1.4\n"
        "0,-0.6,18,1.5\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    for params in model["session_models"].values():
        assert "conformal_lower" in params and "conformal_upper" in params


def test_generated_ea_skips_uncertain_trades(tmp_path):
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
                        "conformal_lower": 0.4,
                        "conformal_upper": 0.6,
                    }
                },
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
    assert "prob >= g_conformal_lower && prob <= g_conformal_upper" in content
    assert 'decision = "skip"' in content
    assert 'reason = "uncertain_prob"' in content
    # ensure skip check occurs before trade decision thresholds
    assert content.index('if(uncertain)') < content.index('else if(prob > SymbolThreshold())')
