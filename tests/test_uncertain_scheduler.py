import csv
import subprocess
import sys
from pathlib import Path


def test_scheduler_runs_labeler(tmp_path):
    input_file = tmp_path / "uncertain_decisions.csv"
    with input_file.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["action", "probability", "threshold", "features", "label"])
        writer.writerow(["buy", "0.4", "0.5", "0.1:0.2", ""])
    out_file = tmp_path / "labeled.csv"
    subprocess.run(
        [
            sys.executable,
            "scripts/uncertain_scheduler.py",
            "--once",
            "--label",
            "1",
            "--input",
            str(input_file),
            "--output",
            str(out_file),
        ],
        check=True,
    )
    rows = list(csv.reader(out_file.open(), delimiter=";"))
    assert rows[1][-1] == "1"
