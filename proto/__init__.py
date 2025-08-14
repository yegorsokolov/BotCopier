# Convenient helpers for loading Cap'n Proto schemas.
#
# The stream listener and other utilities import these modules to decode
# binary messages produced by the observer EA.  ``pycapnp`` performs the
# schema compilation at runtime so no generated Python files need to be
# tracked in the repository.

from pathlib import Path

try:  # pragma: no cover - optional dependency
    import capnp  # type: ignore

    _PROTO_DIR = Path(__file__).resolve().parent

    # Expose loaded schemas as ``trade_capnp`` and ``metrics_capnp`` so other
    # modules can simply import them from ``proto``.
    trade_capnp = capnp.load(str(_PROTO_DIR / "trade.capnp"))
    metrics_capnp = capnp.load(str(_PROTO_DIR / "metrics.capnp"))
except Exception:  # pragma: no cover - allow running without pycapnp
    trade_capnp = metrics_capnp = None  # type: ignore

__all__ = ["trade_capnp", "metrics_capnp"]
