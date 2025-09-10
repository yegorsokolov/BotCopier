import json
import logging
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple
        log = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log)


def setup_logging(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure basic JSON logging and return a logger.

    Parameters
    ----------
    name:
        Name of the logger to return. ``None`` returns the root logger.
    level:
        Logging level for the returned logger.
    """

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)
    return logging.getLogger(name)
