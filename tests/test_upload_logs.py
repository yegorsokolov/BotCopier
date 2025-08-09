import json
import tarfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.upload_logs as ul


def test_create_archive(tmp_path, monkeypatch):
    repo = tmp_path
    logs = repo / "logs"
    logs.mkdir()
    trades = logs / "trades_raw.csv"
    trades.write_text("a,b\n1,2\n")
    metrics = logs / "metrics.csv"
    metrics.write_text("c,d\n3,4\n")
    model = repo / "model.json"
    model.write_text("{}")

    subprocess.run(["git", "init"], cwd=repo, check=True)
    (repo / "dummy.txt").write_text("x")
    subprocess.run(["git", "add", "dummy.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)

    monkeypatch.setenv("SCHEMA_VERSION", "9.9")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(ul, "REPO_ROOT", repo)
    monkeypatch.setattr(ul, "LOG_DIR", logs)
    monkeypatch.setattr(ul, "TRADES_FILE", trades)
    monkeypatch.setattr(ul, "METRICS_FILE", metrics)
    monkeypatch.setattr(ul, "MODEL_FILE", model)

    archive = ul.create_archive()
    assert archive and archive.exists()
    assert not trades.exists()
    assert not metrics.exists()
    assert not model.exists()

    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        assert {"trades_raw.csv", "metrics.csv", "model.json", "manifest.json"} <= names
        manifest = json.load(tar.extractfile("manifest.json"))

    assert manifest["schema_version"] == "9.9"
    assert "commit" in manifest
    for fname in ("trades_raw.csv", "metrics.csv", "model.json"):
        assert fname in manifest["files"]

