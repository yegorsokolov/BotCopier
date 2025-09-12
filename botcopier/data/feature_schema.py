"""Pandera schema for engineered features.

This schema validates the subset of feature columns produced by
``_extract_features`` for which we know sensible value ranges.  The model uses
`strict = False` to allow extra feature columns while ensuring that any
specified fields are numeric and fall within the expected bounds.
"""

from __future__ import annotations

import pandera as pa
from pandera import Field
from pandera.typing import Series


class FeatureSchema(pa.DataFrameModel):
    """Validation schema for known feature columns."""

    atr: Series[float] | None = Field(ge=0)
    sl_dist_atr: Series[float] | None = Field(ge=0)
    tp_dist_atr: Series[float] | None = Field(ge=0)
    book_bid_vol: Series[float] | None = Field(ge=0)
    book_ask_vol: Series[float] | None = Field(ge=0)
    book_imbalance: Series[float] | None = Field(ge=-1, le=1)
    equity: Series[float] | None = Field(ge=0)
    margin_level: Series[float] | None = Field(ge=0)

    class Config:
        strict = False
        coerce = True


__all__ = ["FeatureSchema"]
