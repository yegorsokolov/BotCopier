import json
import subprocess
import sys


def test_generated_features(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"feature_names": ["spread", "hour"]}))

    template = tmp_path / "StrategyTemplate.mq4"
    # minimal template containing placeholder for insertion
    template.write_text("#property strict\n\n// __GET_FEATURE__\n")

    subprocess.run(
        [sys.executable, "scripts/generate_mql4_from_model.py", "--model", model, "--template", template],
        check=True,
    )

    content = template.read_text()
    assert "case 0: return MarketInfo(Symbol(), MODE_SPREAD); // spread" in content
    assert "case 1: return TimeHour(TimeCurrent()); // hour" in content

    data = json.loads(model.read_text())
    assert data["feature_names"] == ["spread", "hour"]
