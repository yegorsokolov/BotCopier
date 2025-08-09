import subprocess
import sys
import time
from pathlib import Path

import capnp
import zmq


def test_stream_listener_binary(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "stream_listener.py"
    endpoint = "tcp://127.0.0.1:6000"
    proc = subprocess.Popen([sys.executable, str(script), "--endpoint", endpoint], cwd=tmp_path)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(endpoint)
    time.sleep(0.5)

    schema_path = Path(__file__).resolve().parents[1] / "proto" / "trade.capnp"
    trade_capnp = capnp.load(str(schema_path))
    msg = trade_capnp.TradeEvent.new_message(
        eventId=1,
        eventTime="t",
        brokerTime="b",
        localTime="l",
        action="OPEN",
        ticket=1,
        magic=0,
        source="mt4",
        symbol="X",
        orderType=0,
        lots=0.0,
        price=1.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        profitAfterTrade=0.0,
        spread=0.0,
        comment="",
        remainingLots=0.0,
        slippage=0.0,
        volume=0,
        openTime="",
        bookBidVol=0.0,
        bookAskVol=0.0,
        bookImbalance=0.0,
        slHitDist=0.0,
        tpHitDist=0.0,
        decisionId=0,
    )
    pub.send(b"\x00" + msg.to_bytes())
    time.sleep(0.5)
    proc.terminate()
    proc.wait(timeout=5)
    out_file = tmp_path / "logs" / "trades_raw.csv"
    assert out_file.exists()
    lines = [l.strip() for l in out_file.read_text().splitlines() if l.strip()]
    assert "X" in lines[-1]
