import subprocess
import sys
from pathlib import Path

from proto import observer_pb2, trade_event_pb2


def test_stream_listener_binary(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "stream_listener.py"
    event = trade_event_pb2.TradeEvent(event_id=1, event_time="t", symbol="X", price=1.0)
    envelope = observer_pb2.ObserverMessage(schema_version="1.0", event=event)
    payload = envelope.SerializeToString()
    packet = len(payload).to_bytes(4, "little") + payload
    subprocess.run([sys.executable, str(script), "--binary"], input=packet, cwd=tmp_path)
    out_file = tmp_path / "logs" / "trades_raw.csv"
    assert out_file.exists()
    lines = [l.strip() for l in out_file.read_text().splitlines() if l.strip()]
    assert "X" in lines[-1]
