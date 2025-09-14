import http.client
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.skip(reason="requires full metrics_collector runtime")
def test_metrics_collector_handles_sigterm(tmp_path):
    db = tmp_path / "m.db"
    script = (
        Path(__file__).resolve().parents[1]
        / "botcopier"
        / "scripts"
        / "metrics_collector.py"
    )
    port = 8766
    stub_dir = tmp_path / "stub"
    (stub_dir / "pyarrow").mkdir(parents=True)
    (stub_dir / "pyarrow" / "__init__.py").write_text("__version__='0.0'\n")
    (stub_dir / "pyarrow" / "flight.py").write_text(
        "class Ticket:\n    def __init__(self, data):\n        self.data = data\n"
        "class FlightClient:\n    def __init__(self, uri):\n        pass\n"
        "    def do_get(self, ticket):\n        raise RuntimeError('stub')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{stub_dir}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--db",
            str(db),
            "--http-port",
            str(port),
            "--flight-port",
            "9999",
        ],
        env=env,
    )
    try:
        time.sleep(1.0)
        conn = http.client.HTTPConnection("127.0.0.1", port)
        conn.request(
            "POST",
            "/ingest",
            body=json.dumps({"time": "t", "magic": "0"}),
            headers={"Content-Type": "application/json"},
        )
        conn.getresponse().read()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    finally:
        proc.kill()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    assert rows == 1
