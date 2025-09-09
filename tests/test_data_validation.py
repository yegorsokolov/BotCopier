import pandas as pd
from datetime import datetime

from scripts.data_validation import validate_logs


def test_null_event_time_fails():
    df = pd.DataFrame({
        "event_time": [datetime(2020, 1, 1), None],
        "hour": [0, 1],
    })
    result = validate_logs(df)
    assert not result["success"]


def test_hour_out_of_range_fails():
    df = pd.DataFrame({
        "event_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "hour": [0, 24],
    })
    result = validate_logs(df)
    assert not result["success"]


def test_non_monotonic_time_fails():
    df = pd.DataFrame({
        "event_time": [datetime(2020, 1, 2), datetime(2020, 1, 1)],
        "hour": [1, 0],
    })
    result = validate_logs(df)
    assert not result["success"]
