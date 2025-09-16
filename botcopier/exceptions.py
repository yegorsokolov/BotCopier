from __future__ import annotations

"""Custom exception hierarchy for BotCopier."""

from datetime import datetime
from typing import Optional, Union


class BotCopierError(Exception):
    """Base class for all BotCopier exceptions.

    Parameters
    ----------
    message:
        Human readable description of the error.
    symbol:
        Optional trading symbol associated with the error.
    timestamp:
        Timestamp of the event that triggered the error.  May be a
        :class:`~datetime.datetime` instance or string.
    """

    def __init__(
        self,
        message: str,
        *,
        symbol: Optional[str] = None,
        timestamp: Optional[Union[datetime, str]] = None,
    ) -> None:
        context: list[str] = []
        if symbol:
            context.append(f"symbol={symbol}")
        if timestamp:
            if isinstance(timestamp, datetime):
                ts = timestamp.isoformat()
            else:
                ts = str(timestamp)
            context.append(f"timestamp={ts}")
        if context:
            message = f"{message} ({', '.join(context)})"
        super().__init__(message)
        self.symbol = symbol
        self.timestamp = timestamp


class DataError(BotCopierError):
    """Errors related to data loading or validation."""


class ModelError(BotCopierError):
    """Errors raised during model training or inference."""


class ServiceError(BotCopierError):
    """Errors originating from external services."""


class TrainingPipelineError(BotCopierError):
    """Errors raised during the training pipeline orchestration."""


class PromotionError(BotCopierError):
    """Errors raised when promoting strategies between environments."""


class EvaluationError(BotCopierError):
    """Errors raised while evaluating model or strategy performance."""


__all__ = [
    "BotCopierError",
    "DataError",
    "ModelError",
    "ServiceError",
    "TrainingPipelineError",
    "PromotionError",
    "EvaluationError",
]
