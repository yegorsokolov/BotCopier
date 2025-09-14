from __future__ import annotations

from typing import Iterable

import pandas as pd
import pyarrow as pa

from botcopier.exceptions import DataError


def validate_columns(
    df: pd.DataFrame,
    schema: pa.Schema,
    *,
    required: Iterable[str] | None = None,
    name: str = "artifact",
) -> None:
    """Validate that ``df`` contains the ``required`` columns.

    Parameters
    ----------
    df:
        DataFrame to validate.
    schema:
        Arrow schema providing the full set of expected columns.
    required:
        Iterable of field names that must be present.  Defaults to all fields in
        ``schema``.
    name:
        Human readable name of the data source used in error messages.
    """

    required_set = set(required) if required is not None else {f.name for f in schema}
    missing = [col for col in required_set if col not in df.columns]
    if missing:
        raise DataError(
            f"{name} missing required columns: {', '.join(sorted(missing))}",
        )


__all__ = ["validate_columns"]
