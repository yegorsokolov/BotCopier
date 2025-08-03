import json
import subprocess
import sys
import zlib
from pathlib import Path


def test_stream_listener_binary(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "stream_listener.py"
    msg = {
        "schema_version": "1.0",
        "type": "event",
        "foo": "bar",
    }
    payload = zlib.compress(json.dumps(msg).encode("utf-8"))
    packet = len(payload).to_bytes(4, "little") + payload
    subprocess.run([sys.executable, str(script), "--binary"], input=packet, cwd=tmp_path)
    out_file = tmp_path / "logs" / "trades_raw.csv"
    assert out_file.exists()
    lines = [l.strip() for l in out_file.read_text().splitlines() if l.strip()]
    assert "bar" in lines[-1]
