import logging
import sys
from pathlib import Path

import pandas as pd
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.anomaly_monitor import detect_change_points


def test_change_point_alert(caplog):
    df = pd.DataFrame(
        {
            "rowid": range(1, 16),
            "win_rate": [0.5] * 10 + [0.9] * 5,
            "drawdown": [0.1] * 10 + [0.5] * 5,
        }
    )
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer("test_cp")
    log = logging.getLogger("anomaly_monitor")
    caplog.set_level(logging.WARNING, logger="anomaly_monitor")
    with tracer.start_as_current_span("cp_test"):
        hits = detect_change_points(df, ["win_rate", "drawdown"], penalty=0.5, log=log)
    assert set(hits) == {"win_rate", "drawdown"}
    messages = [r.getMessage() for r in caplog.records if "Change point detected" in r.getMessage()]
    assert any("win_rate" in m for m in messages)
    assert any("trace_id=" in m for m in messages)
