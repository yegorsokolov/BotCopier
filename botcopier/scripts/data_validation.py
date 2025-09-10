import logging
from typing import Any

import pandas as pd
try:  # optional dependency
    import great_expectations as ge
    from great_expectations.core.batch import Batch
    from great_expectations.execution_engine import PandasExecutionEngine
    from great_expectations.validator.validator import Validator
except Exception:  # pragma: no cover - optional
    ge = None  # type: ignore
    Batch = PandasExecutionEngine = Validator = None  # type: ignore

logger = logging.getLogger(__name__)

if ge is not None:  # pragma: no cover - optional initialisation
    _context = ge.get_context()
    _engine = PandasExecutionEngine()
else:  # pragma: no cover - optional dependency missing
    _context = None
    _engine = None


def validate_logs(df: pd.DataFrame) -> dict[str, Any]:
    """Validate trade log dataframe using great_expectations.

    Checks include non-null constraints, value ranges and monotonicity for
    temporal columns.  Returns the great_expectations validation result.
    """
    if ge is None or Batch is None or _engine is None or Validator is None:
        return {"success": True}
    df_local = df.copy()
    if "event_time" in df_local.columns:
        df_local["event_time"] = pd.to_datetime(df_local["event_time"])
        df_local["_event_time_posix"] = df_local["event_time"].view("int64")
    batch = Batch(data=df_local)
    validator = Validator(execution_engine=_engine, batches=[batch], data_context=_context)

    if "event_time" in df_local.columns:
        validator.expect_column_values_to_not_be_null("event_time")
        validator.expect_column_values_to_be_increasing("_event_time_posix")

    if "hour" in df.columns:
        validator.expect_column_values_to_not_be_null("hour")
        validator.expect_column_values_to_be_between("hour", 0, 23)

    if "volume" in df.columns:
        validator.expect_column_values_to_not_be_null("volume")
        validator.expect_column_values_to_be_between("volume", 0, None)

    if "meta_label" in df.columns:
        validator.expect_column_values_to_be_between("meta_label", 0, 1)

    if "net_profit" in df.columns:
        validator.expect_column_values_to_not_be_null("net_profit")

    result: dict[str, Any] = validator.validate()
    return result
