import json
import subprocess
import sys
from pathlib import Path

from botcopier.training.pipeline import train


def test_calendar_features(tmp_path: Path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,event_time\n" "1,1.0,2024-01-01 00:20:00\n" "0,1.2,2024-01-01 02:00:00\n"
    )
    calendar = Path("tests/sample_calendar.csv")
    out_dir = tmp_path / "out"
    train(data, out_dir, calendar_file=calendar)

    model = json.loads((out_dir / "model.json").read_text())
    feats = model.get("retained_features") or model.get("feature_names")
    assert "event_flag" in feats
    assert "event_impact" in feats

    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text(Path("StrategyTemplate.mq4").read_text())
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            out_dir / "model.json",
            "--template",
            template,
            "--calendar-file",
            calendar,
        ],
        check=True,
    )
    content = template.read_text()
    assert "CalendarFlag()" in content
    assert "CalendarImpact()" in content
    assert str(calendar) in content

