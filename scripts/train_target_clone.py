#!/usr/bin/env python3
"""Train model from exported features.

The observer EA continuously exports trade logs as CSV files. This script
loads those logs, extracts a few simple features from each trade entry and
trains a very small predictive model. The resulting parameters along with
some training metadata are written to ``model.json`` so they can be consumed
by other helper scripts.
"""
import argparse
import json
import gzip
from datetime import datetime
import math
import time
from pathlib import Path
from typing import Iterable, List, Optional
import sqlite3
import logging
import subprocess
import shutil
import sys

import importlib.util

import os

DEFAULT_DATA_HOME = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "botcopier"

import pandas as pd

import numpy as np
import psutil
try:
    import pyarrow as pa
    import pyarrow.flight as flight
    import pyarrow.parquet as pq
    if not hasattr(pa, "Table"):
        pa = None
        flight = None
        pq = None
        import sys as _sys
        _sys.modules.pop("pyarrow", None)
        _sys.modules.pop("pyarrow.flight", None)
        _sys.modules.pop("pyarrow.parquet", None)
except Exception:  # pragma: no cover - optional dependency
    pa = None
    flight = None
    pq = None
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import BayesianRidge, LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import requests

from pydantic import ValidationError  # type: ignore
from schemas.trades import TradeEvent  # type: ignore

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    _HAS_TORCH = False

# Optional graph library for node embeddings
try:  # pragma: no cover - optional dependency
    from torch_geometric.nn import Node2Vec  # type: ignore
    _HAS_PYG = True
except Exception:  # pragma: no cover - optional dependency
    Node2Vec = None  # type: ignore
    _HAS_PYG = False


from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
try:  # Optional Jaeger exporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    JaegerExporter = None
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id


SCHEMA_VERSION = 1
START_EVENT_ID = 0

resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "train_target_clone")})
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
elif os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST") and JaegerExporter:
    provider.add_span_processor(
        BatchSpanProcessor(
            JaegerExporter(
                agent_host_name=os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST"),
                agent_port=int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831")),
            )
        )
    )
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

logger_provider = LoggerProvider(resource=resource)
if endpoint:
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint)))
set_logger_provider(logger_provider)
handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {"level": record.levelname}
        if isinstance(record.msg, dict):
            log.update(record.msg)
        else:
            log["message"] = record.getMessage()
        if hasattr(record, "trace_id"):
            log["trace_id"] = format_trace_id(record.trace_id)
        if hasattr(record, "span_id"):
            log["span_id"] = format_span_id(record.span_id)
        return json.dumps(log)


logger = logging.getLogger(__name__)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

try:  # Optional dependency for RL refinement
    import stable_baselines3  # type: ignore  # noqa: F401
    HAS_SB3 = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SB3 = False


def _has_sufficient_ram(min_gb: float = 4.0) -> bool:
    """Return True if the system has at least ``min_gb`` RAM."""
    try:
        return psutil.virtual_memory().total / (1024 ** 3) >= min_gb
    except Exception:  # pragma: no cover - psutil errors
        return False


def _has_sufficient_gpu(min_gb: float = 1.0) -> bool:
    """Return True if a CUDA GPU with ``min_gb`` memory is available."""
    try:  # pragma: no cover - optional dependency
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory
            return mem / (1024 ** 3) >= min_gb
    except Exception:
        pass
    return False


def _compute_decay_weights(event_times: np.ndarray, half_life_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Return age in days and exponential decay weights for ``event_times``."""
    ref_time = event_times.max()
    age_days = (
        (ref_time - event_times).astype("timedelta64[s]").astype(float)
        / (24 * 3600)
    )
    weights = 0.5 ** (age_days / half_life_days)
    return age_days, weights


if _HAS_TORCH:
    class ContrastiveEncoder(nn.Module):
        """Simple linear encoder used for contrastive pretraining."""

        def __init__(self, window: int, dim: int):
            super().__init__()
            self.layer = nn.Linear(window, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
            return self.layer(x)
else:  # pragma: no cover - torch not available
    class ContrastiveEncoder:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available")


def _load_contrastive_encoder(path: Path) -> ContrastiveEncoder:
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for --use-encoder")
    state = torch.load(path, map_location="cpu")
    window = int(state.get("window", 0))
    dim = int(state.get("dim", 0))
    model = ContrastiveEncoder(window, dim)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _encode_features(model: ContrastiveEncoder | None, X: np.ndarray) -> np.ndarray:
    if model is None or X.size == 0 or not _HAS_TORCH:
        return X
    with torch.no_grad():
        t = torch.from_numpy(X.astype("float32"))
        out = model(t).cpu().numpy()
    return out


def detect_resources():
    """Detect available resources and suggest an operating mode.

    The returned dictionary contains RAM, swap, disk and CPU information along
    with GPU capabilities.  Based on these metrics and available ML libraries a
    ``mode`` string is provided that callers can use to toggle heavy features or
    optional refinement steps.
    """

    try:
        mem_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:  # pragma: no cover - psutil errors
        mem_gb = 0.0
    try:
        swap_gb = psutil.swap_memory().total / (1024**3)
    except Exception:  # pragma: no cover - psutil errors
        swap_gb = 0.0
    try:
        cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    except Exception:  # pragma: no cover - psutil errors
        cores = 0

    try:
        cpu_mhz = psutil.cpu_freq().max
    except Exception:  # pragma: no cover - psutil errors
        cpu_mhz = 0.0
    if not cpu_mhz:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("cpu MHz"):
                        cpu_mhz = float(line.split(":")[1].strip())
                        break
        except Exception:
            pass

    if mem_gb == 0.0:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = float(line.split()[1])
                        mem_gb = mem_kb / (1024**2)
                        break
        except Exception:
            pass
    if cores == 0:
        try:
            with open("/proc/cpuinfo") as f:
                cores = sum(1 for line in f if line.startswith("processor"))
        except Exception:
            pass

    # Estimate free disk space to adjust behavior on low-storage systems
    disk_gb = shutil.disk_usage("/").free / (1024**3)
    lite_mode = mem_gb < 4 or cores < 2 or disk_gb < 5
    heavy_mode = mem_gb >= 8 and cores >= 4

    gpu_mem_gb = 0.0
    has_gpu = False
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                has_gpu = True
        except Exception:
            has_gpu = False
    if not has_gpu:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            gpu_mem_gb = float(out.splitlines()[0])
            has_gpu = True
        except Exception:
            gpu_mem_gb = 0.0
            has_gpu = False

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    CPU_MHZ_THRESHOLD = 2500.0

    if lite_mode:
        model_type = "logreg"
    else:
        model_type = "transformer"
        if not (
            has_gpu
            and has("transformers")
            and gpu_mem_gb >= 8.0
            and cpu_mhz >= CPU_MHZ_THRESHOLD
        ):
            model_type = "logreg"

    use_optuna = heavy_mode and has("optuna")
    bayes_steps = 20 if use_optuna else 0

    enable_rl = heavy_mode and has_gpu and gpu_mem_gb >= 8.0 and has("stable_baselines3")

    if enable_rl:
        mode = "rl"
    elif lite_mode:
        mode = "lite"
    elif model_type != "logreg":
        mode = "deep"
    elif heavy_mode:
        mode = "heavy"
    else:
        mode = "standard"

    return {
        "lite_mode": lite_mode,
        "heavy_mode": heavy_mode,
        "model_type": model_type,
        "bayes_steps": bayes_steps,
        "mem_gb": mem_gb,
        "swap_gb": swap_gb,
        "disk_gb": disk_gb,
        "cores": cores,
        "cpu_mhz": cpu_mhz,
        "has_gpu": has_gpu,
        "gpu_mem_gb": gpu_mem_gb,
        "enable_rl": enable_rl,
        "mode": mode,
    }


def sync_with_server(
    model_path: Path,
    server_url: str,
    poll_interval: float = 1.0,
    timeout: float = 30.0,
) -> None:
    """Send local model weights to a federated server and update with averaged weights."""
    open_func = gzip.open if model_path.suffix == ".gz" else open
    try:
        with open_func(model_path, "rt") as f:
            model = json.load(f)
    except FileNotFoundError:
        return
    weights = model.get("coefficients")
    intercept = model.get("intercept")
    if weights is None:
        return
    payload = {"weights": weights}
    if intercept is not None:
        payload["intercept"] = intercept
    try:
        requests.post(f"{server_url}/update", json=payload, timeout=5)
    except Exception:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{server_url}/weights", timeout=5)
            data = resp.json()
        except Exception:
            time.sleep(poll_interval)
            continue
        if data.get("weights"):
            model["coefficients"] = data["weights"]
            if data.get("intercept") is not None:
                model["intercept"] = data["intercept"]
            with open_func(model_path, "wt") as f:
                json.dump(model, f)
            break
        time.sleep(poll_interval)


def _export_onnx(clf, feature_names: List[str], out_dir: Path) -> None:
    """Export ``clf`` to ONNX format if dependencies are available."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onnx_path = out_dir / "model.onnx"
        initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
        model_onnx = convert_sklearn(clf, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(model_onnx.SerializeToString())
        print(f"ONNX model written to {onnx_path}")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"ONNX conversion skipped: {exc}")


def _sma(values, window):
    """Simple moving average for the last ``window`` values."""
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(sum(values[-w:]) / w)


def _atr(values, period):
    """Average true range over ``period`` price changes."""
    if len(values) < 2:
        return 0.0
    diffs = np.abs(np.diff(values[-(period + 1) :]))
    if len(diffs) == 0:
        return 0.0
    w = min(period, len(diffs))
    return float(diffs[-w:].mean())


def _bollinger(values, window, dev=2.0):
    """Return Bollinger Bands for the last ``window`` values."""
    if not values:
        return 0.0, 0.0, 0.0
    w = min(window, len(values))
    arr = np.array(values[-w:])
    sma = arr.mean()
    std = arr.std(ddof=0)
    upper = sma + dev * std
    lower = sma - dev * std
    return float(upper), float(sma), float(lower)


def _safe_float(val, default=0.0):
    """Convert ``val`` to float, treating ``None``/NaN as ``default``."""
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default


def _rsi(values, period):
    """Very small RSI implementation on ``values``."""
    if len(values) < 2:
        return 50.0
    deltas = np.diff(values[-(period + 1) :])
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def _macd_update(state, price, short=12, long=26, signal=9):
    """Update MACD EMA state with ``price`` and return (macd, signal)."""
    alpha_short = 2 / (short + 1)
    alpha_long = 2 / (long + 1)
    alpha_signal = 2 / (signal + 1)

    ema_short = state.get("ema_short")
    ema_long = state.get("ema_long")
    ema_signal = state.get("ema_signal")

    ema_short = price if ema_short is None else alpha_short * price + (1 - alpha_short) * ema_short
    ema_long = price if ema_long is None else alpha_long * price + (1 - alpha_long) * ema_long
    macd = ema_short - ema_long
    ema_signal = macd if ema_signal is None else alpha_signal * macd + (1 - alpha_signal) * ema_signal

    state["ema_short"] = ema_short
    state["ema_long"] = ema_long
    state["ema_signal"] = ema_signal

    return macd, ema_signal


def _stochastic_update(state, price, k_period=14, d_period=3):
    """Update and return stochastic %K and %D values."""
    prices = state.setdefault("prices", [])
    prices.append(price)
    if len(prices) > k_period:
        del prices[0]
    low = min(prices)
    high = max(prices)
    if high == low:
        k = 0.0
    else:
        k = (price - low) / (high - low) * 100.0
    k_history = state.setdefault("k_values", [])
    k_history.append(k)
    if len(k_history) > d_period:
        del k_history[0]
    d = float(sum(k_history) / len(k_history))
    return float(k), d


def _adx_update(state, price, period=14):
    """Update ADX state with ``price`` and return current ADX."""
    prev = state.get("prev_price")
    state["prev_price"] = price
    if prev is None:
        return 0.0

    up_move = price - prev if price > prev else 0.0
    down_move = prev - price if price < prev else 0.0
    tr = abs(price - prev)

    plus_dm = state.setdefault("plus_dm", [])
    minus_dm = state.setdefault("minus_dm", [])
    tr_list = state.setdefault("tr", [])
    dx_list = state.setdefault("dx", [])

    plus_dm.append(up_move)
    minus_dm.append(down_move)
    tr_list.append(tr)
    if len(plus_dm) > period:
        del plus_dm[0]
    if len(minus_dm) > period:
        del minus_dm[0]
    if len(tr_list) > period:
        del tr_list[0]

    atr = sum(tr_list) / len(tr_list)
    if atr == 0:
        di_plus = di_minus = 0.0
    else:
        di_plus = 100.0 * (sum(plus_dm) / len(plus_dm)) / atr
        di_minus = 100.0 * (sum(minus_dm) / len(minus_dm)) / atr

    denom = di_plus + di_minus
    dx = 0.0 if denom == 0 else 100.0 * abs(di_plus - di_minus) / denom
    dx_list.append(dx)
    if len(dx_list) > period:
        del dx_list[0]
    adx = sum(dx_list) / len(dx_list)
    return float(adx)


def _rolling_corr(a, b, window=5):
    """Return correlation of the last ``window`` points of ``a`` and ``b``."""
    if not a or not b:
        return 0.0
    w = min(len(a), len(b), window)
    if w < 2:
        return 0.0
    arr1 = np.array(a[-w:], dtype=float)
    arr2 = np.array(b[-w:], dtype=float)
    if arr1.std(ddof=0) == 0 or arr2.std(ddof=0) == 0:
        return 0.0
    return float(np.corrcoef(arr1, arr2)[0, 1])


def _load_logs_db(db_file: Path) -> pd.DataFrame:
    """Load log rows from a SQLite database."""

    conn = sqlite3.connect(db_file)
    try:
        query = "SELECT * FROM logs"
        params: tuple = ()
        if START_EVENT_ID > 0:
            query += " WHERE CAST(event_id AS INTEGER) > ?"
            params = (START_EVENT_ID,)
        df_logs = pd.read_sql_query(query, conn, params=params, parse_dates=["event_time", "open_time"])
    finally:
        conn.close()

    df_logs.columns = [c.lower() for c in df_logs.columns]

    if "schema_version" in df_logs.columns:
        df_logs["schema_version"] = (
            pd.to_numeric(df_logs["schema_version"], errors="coerce")
            .fillna(SCHEMA_VERSION)
            .astype(int)
        )
        mismatch_mask = df_logs["schema_version"] != SCHEMA_VERSION
        if mismatch_mask.any():
            logging.warning(
                "Dropping %s rows with schema version != %s",
                mismatch_mask.sum(),
                SCHEMA_VERSION,
            )
            df_logs = df_logs[~mismatch_mask]
    else:
        logging.warning("schema_version column missing; assuming %s", SCHEMA_VERSION)
        df_logs["schema_version"] = SCHEMA_VERSION

    if "open_time" in df_logs.columns:
        df_logs["trade_duration"] = (
            pd.to_datetime(df_logs["event_time"]) - pd.to_datetime(df_logs["open_time"])
        ).dt.total_seconds().fillna(0)
    for col in [
        "book_bid_vol",
        "book_ask_vol",
        "book_imbalance",
        "commission",
        "swap",
        "trend_estimate",
        "trend_variance",
    ]:
        if col not in df_logs.columns:
            df_logs[col] = 0.0
        else:
            df_logs[col] = pd.to_numeric(df_logs[col], errors="coerce").fillna(0.0)
    if "is_anomaly" not in df_logs.columns:
        df_logs["is_anomaly"] = 0
    else:
        df_logs["is_anomaly"] = (
            pd.to_numeric(df_logs["is_anomaly"], errors="coerce").fillna(0).astype(int)
        )

    valid_actions = {"OPEN", "CLOSE", "MODIFY"}
    if "action" in df_logs.columns:
        df_logs["action"] = df_logs["action"].fillna("").str.upper()
        df_logs = df_logs[(df_logs["action"] == "") | df_logs["action"].isin(valid_actions)]

    invalid_rows = pd.DataFrame(columns=df_logs.columns)
    if "event_id" in df_logs.columns:
        dup_mask = df_logs.duplicated(subset="event_id", keep="first")
        if dup_mask.any():
            invalid_rows = pd.concat([invalid_rows, df_logs[dup_mask]])
            logging.warning("Dropping %s duplicate event_id rows", dup_mask.sum())
        df_logs = df_logs[~dup_mask]

    if set(["ticket", "action"]).issubset(df_logs.columns):
        crit_mask = (
            df_logs["ticket"].isna()
            | (df_logs["ticket"].astype(str) == "")
            | df_logs["action"].isna()
            | (df_logs["action"].astype(str) == "")
        )
        if crit_mask.any():
            invalid_rows = pd.concat([invalid_rows, df_logs[crit_mask]])
            logging.warning("Dropping %s rows with missing ticket/action", crit_mask.sum())
        df_logs = df_logs[~crit_mask]

    if "lots" in df_logs.columns:
        df_logs["lots"] = pd.to_numeric(df_logs["lots"], errors="coerce")
    if "price" in df_logs.columns:
        df_logs["price"] = pd.to_numeric(df_logs["price"], errors="coerce")
    unreal_mask = pd.Series(False, index=df_logs.index)
    if "lots" in df_logs.columns:
        unreal_mask |= df_logs["lots"] < 0
    if "price" in df_logs.columns:
        unreal_mask |= df_logs["price"].isna()
    if unreal_mask.any():
        invalid_rows = pd.concat([invalid_rows, df_logs[unreal_mask]])
        logging.warning("Dropping %s rows with negative lots or NaN price", unreal_mask.sum())
    df_logs = df_logs[~unreal_mask]

    if not invalid_rows.empty:
        invalid_file = db_file.with_name("invalid_rows.csv")
        try:
            invalid_rows.to_csv(invalid_file, index=False)
        except Exception:  # pragma: no cover - disk issues
            pass

    return df_logs


def _load_logs(
    data_dir: Path,
    *,
    lite_mode: bool = False,
    chunk_size: int = 50000,
    flight_uri: str | None = None,
) -> tuple[pd.DataFrame | Iterable[pd.DataFrame], list[str], list[str]]:
    """Load log rows from ``data_dir``.

    ``MODIFY`` entries are retained alongside ``OPEN`` and ``CLOSE``.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``trades_*.csv`` files or a SQLite ``.db`` file.

    Returns
    -------
    tuple[pandas.DataFrame, list[str], list[str]]
        Parsed rows as a DataFrame along with commit hashes and checksums
        collected from accompanying manifest files.
    """

    if flight_uri:
        client = flight.FlightClient(flight_uri)
        desc = flight.FlightDescriptor.for_path("trades")
        info = client.get_flight_info(desc)
        reader = client.do_get(info.endpoints[0].ticket)
        if lite_mode:
            def _iter_batches():
                for batch in reader:
                    yield pa.Table.from_batches([batch]).to_pandas()
            return _iter_batches(), [], []
        table = reader.read_all()
        df = table.to_pandas()
        df.columns = [c.lower() for c in df.columns]
        return df, [], []

    if data_dir.suffix == ".db":
        df = _load_logs_db(data_dir)
        return df, [], []

    fields = [
        "schema_version",
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "sl_dist",
        "tp_dist",
        "sl_hit_dist",
        "tp_hit_dist",
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
        "open_time",
        "book_bid_vol",
        "book_ask_vol",
        "book_imbalance",
        "equity",
        "margin_level",
        "commission",
        "swap",
        "is_anomaly",
        "trend_estimate",
        "trend_variance",
    ]

    data_commits: list[str] = []
    data_checksums: list[str] = []

    metrics_file = data_dir / "metrics.csv"
    df_metrics = None
    key_col: str | None = None
    if metrics_file.exists():
        df_metrics = pd.read_csv(metrics_file, sep=";")
        df_metrics.columns = [c.lower() for c in df_metrics.columns]
        if "schema_version" in df_metrics.columns:
            df_metrics["schema_version"] = (
                pd.to_numeric(df_metrics["schema_version"], errors="coerce")
                .fillna(SCHEMA_VERSION)
                .astype(int)
            )
            m_mask = df_metrics["schema_version"] != SCHEMA_VERSION
            if m_mask.any():
                logging.warning(
                    "Dropping %s metric rows with schema version != %s",
                    m_mask.sum(),
                    SCHEMA_VERSION,
                )
                df_metrics = df_metrics[~m_mask]
        else:
            logging.warning("schema_version column missing in metrics; assuming %s", SCHEMA_VERSION)
            df_metrics["schema_version"] = SCHEMA_VERSION
        if "magic" in df_metrics.columns:
            key_col = "magic"
        elif "model_id" in df_metrics.columns:
            key_col = "model_id"

    invalid_file = data_dir / "invalid_rows.csv"

    def iter_chunks():
        seen_ids: set[str] = set()
        invalid_rows: list[pd.DataFrame] = []
        for log_file in sorted(data_dir.glob("trades_*.csv")):
            reader = pd.read_csv(
                log_file,
                sep=";",
                header=0,
                dtype=str,
                chunksize=chunk_size,
                engine="python",
            )
            manifest_file = log_file.with_suffix(".manifest.json")
            if manifest_file.exists():
                try:
                    with open(manifest_file) as mf:
                        meta = json.load(mf)
                    commit = meta.get("commit")
                    checksum = meta.get("checksum")
                    if commit:
                        data_commits.append(str(commit))
                    if checksum:
                        data_checksums.append(str(checksum))
                except Exception:
                    pass
            for chunk in reader:
                chunk = chunk.reindex(columns=fields)
                chunk.columns = [c.lower() for c in chunk.columns]
                invalid = pd.DataFrame(columns=chunk.columns)
                chunk["schema_version"] = (
                    pd.to_numeric(chunk.get("schema_version"), errors="coerce")
                    .fillna(SCHEMA_VERSION)
                    .astype(int)
                )
                ver_mask = chunk["schema_version"] != SCHEMA_VERSION
                if ver_mask.any():
                    invalid = pd.concat([invalid, chunk[ver_mask]])
                    logging.warning(
                        "Dropping %s rows with schema version != %s",
                        ver_mask.sum(),
                        SCHEMA_VERSION,
                    )
                chunk = chunk[~ver_mask]
                chunk["event_time"] = pd.to_datetime(chunk.get("event_time"), errors="coerce")
                if "broker_time" in chunk.columns:
                    chunk["broker_time"] = pd.to_datetime(chunk.get("broker_time"), errors="coerce")
                    chunk["broker_time"] = chunk["broker_time"].where(~chunk["broker_time"].isna(), None)
                if "local_time" in chunk.columns:
                    chunk["local_time"] = pd.to_datetime(chunk.get("local_time"), errors="coerce")
                    chunk["local_time"] = chunk["local_time"].where(~chunk["local_time"].isna(), None)
                if "open_time" in chunk.columns:
                    chunk["open_time"] = pd.to_datetime(chunk.get("open_time"), errors="coerce")
                    chunk["trade_duration"] = (
                        chunk["event_time"] - chunk["open_time"]
                    ).dt.total_seconds().fillna(0)
                else:
                    chunk["trade_duration"] = 0.0
                for col in [
                    "book_bid_vol",
                    "book_ask_vol",
                    "book_imbalance",
                    "equity",
                    "margin_level",
                    "commission",
                    "swap",
                    "trend_estimate",
                    "trend_variance",
                ]:
                    chunk[col] = pd.to_numeric(chunk.get(col, 0.0), errors="coerce").fillna(0.0)
                chunk["is_anomaly"] = pd.to_numeric(chunk.get("is_anomaly", 0), errors="coerce").fillna(0)
                for col in ["source", "comment"]:
                    if col in chunk.columns:
                        chunk[col] = chunk[col].fillna("").astype(str)
                valid_actions = {"OPEN", "CLOSE", "MODIFY"}
                chunk["action"] = chunk["action"].fillna("").str.upper()
                chunk = chunk[(chunk["action"] == "") | chunk["action"].isin(valid_actions)]
                if "event_id" in chunk.columns:
                    dup_mask = (
                        chunk["event_id"].isin(seen_ids)
                        | chunk.duplicated(subset="event_id", keep="first")
                    )
                    if dup_mask.any():
                        invalid = pd.concat([invalid, chunk[dup_mask]])
                        logging.warning(
                            "Dropping %s duplicate event_id rows", dup_mask.sum()
                        )
                    seen_ids.update(chunk.loc[~dup_mask, "event_id"].tolist())
                    chunk = chunk[~dup_mask]
                if {"ticket", "action"}.issubset(chunk.columns):
                    crit_mask = (
                        chunk["ticket"].isna()
                        | (chunk["ticket"].astype(str) == "")
                        | chunk["action"].isna()
                        | (chunk["action"].astype(str) == "")
                    )
                    if crit_mask.any():
                        invalid = pd.concat([invalid, chunk[crit_mask]])
                        logging.warning(
                            "Dropping %s rows with missing ticket/action", crit_mask.sum()
                        )
                    chunk = chunk[~crit_mask]
                if "lots" in chunk.columns:
                    chunk["lots"] = pd.to_numeric(chunk["lots"], errors="coerce")
                if "price" in chunk.columns:
                    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
                unreal_mask = pd.Series(False, index=chunk.index)
                if "lots" in chunk.columns:
                    unreal_mask |= chunk["lots"] < 0
                if "price" in chunk.columns:
                    unreal_mask |= chunk["price"].isna()
                if unreal_mask.any():
                    invalid = pd.concat([invalid, chunk[unreal_mask]])
                    logging.warning(
                        "Dropping %s rows with negative lots or NaN price", unreal_mask.sum()
                    )
                chunk = chunk[~unreal_mask]
                if "magic" in chunk.columns:
                    chunk["magic"] = (
                        pd.to_numeric(chunk["magic"], errors="coerce").fillna(0).astype(int)
                    )
                if df_metrics is not None and key_col is not None and "magic" in chunk.columns:
                    if key_col == "magic":
                        chunk = chunk.merge(df_metrics, how="left", on="magic")
                    else:
                        chunk = chunk.merge(
                            df_metrics, how="left", left_on="magic", right_on="model_id"
                        )
                        chunk = chunk.drop(columns=["model_id"])
                records = []
                for row in chunk.to_dict(orient="records"):
                    try:
                        TradeEvent(**row)
                        records.append(row)
                    except ValidationError:
                        invalid = pd.concat([invalid, pd.DataFrame([row])])
                if records:
                    chunk = pd.DataFrame(records, columns=chunk.columns)
                else:
                    chunk = pd.DataFrame(columns=chunk.columns)
                if not invalid.empty:
                    invalid_rows.append(invalid)
                yield chunk
        if invalid_rows:
            try:
                pd.concat(invalid_rows, ignore_index=True).to_csv(invalid_file, index=False)
            except Exception:  # pragma: no cover - disk issues
                pass

    if lite_mode:
        return iter_chunks(), data_commits, data_checksums
    dfs = list(iter_chunks())
    if dfs:
        df_logs = pd.concat(dfs, ignore_index=True)
    else:
        df_logs = pd.DataFrame(columns=[c.lower() for c in fields])
    return df_logs, data_commits, data_checksums


def _load_calendar(file: Path) -> list[tuple[datetime, float]]:
    """Load calendar events from a CSV file.

    The file is expected to have at least two columns: ``time`` and
    ``impact``. ``time`` should be parseable by ``pandas.to_datetime``.

    Parameters
    ----------
    file : Path
        CSV file containing calendar events.

    Returns
    -------
    list[tuple[datetime, float]]
        Sorted list of ``(event_time, impact)`` tuples.
    """

    if not file.exists():
        return []
    df = pd.read_csv(file)
    events: list[tuple[datetime, float]] = []
    for _, row in df.iterrows():
        t = pd.to_datetime(row.get("time"), utc=False, errors="coerce")
        if pd.isna(t):
            continue
        impact = float(row.get("impact", 0.0) or 0.0)
        events.append((t.to_pydatetime(), impact))
    events.sort(key=lambda x: x[0])
    return events


def _load_news_sentiment(file: Path) -> dict[str, list[tuple[datetime, float]]]:
    """Load news sentiment scores from a SQLite or CSV file."""
    if file is None or not file.exists():
        return {}
    data: dict[str, list[tuple[datetime, float]]] = {}
    try:
        if file.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            conn = sqlite3.connect(file)
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT symbol, timestamp, score FROM sentiment ORDER BY timestamp"
            ).fetchall()
            conn.close()
            for sym, ts, sc in rows:
                try:
                    t = datetime.fromisoformat(str(ts))
                except Exception:
                    continue
                data.setdefault(str(sym), []).append((t, float(sc)))
        else:
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                t = pd.to_datetime(row.get("timestamp"), utc=False, errors="coerce")
                if pd.isna(t):
                    continue
                sym = str(row.get("symbol", ""))
                score = float(row.get("score", 0.0) or 0.0)
                data.setdefault(sym, []).append((t.to_pydatetime(), score))
    except Exception:
        return {}
    for lst in data.values():
        lst.sort(key=lambda x: x[0])
    return data


def _load_feature_cache(out_dir: Path, existing: dict | None):
    """Return cached features if metadata matches ``existing`` model."""
    if pq is None or existing is None:
        return None
    cache_file = out_dir / "features.parquet"
    if not cache_file.exists():
        return None
    try:
        pf = pq.ParquetFile(cache_file)
        meta = pf.schema_arrow.metadata or {}
        names = json.loads(meta.get(b"feature_names", b"[]").decode())
        last_id = int(meta.get(b"last_event_id", b"0"))
        existing_names = existing.get("feature_names", [])
        if (
            set(existing_names).issubset(set(names))
            and last_id == int(existing.get("last_event_id", 0))
        ):
            df_cache = pf.read().to_pandas()
            labels = df_cache["label"].to_numpy()
            feats = df_cache[names].to_dict("records")
            sl = df_cache.get("sl_target", pd.Series([])).to_numpy()
            tp = df_cache.get("tp_target", pd.Series([])).to_numpy()
            lots = df_cache.get("lot_target", pd.Series([])).to_numpy()
            hrs = df_cache.get("hours", pd.Series([], dtype=int)).to_numpy(dtype=int)
            times = df_cache.get("event_time", pd.Series([], dtype="datetime64[s]")).to_numpy(
                dtype="datetime64[s]"
            )
            return feats, labels, sl, tp, lots, hrs, times, last_id
    except Exception:
        return None
    return None


def _save_feature_cache(
    out_dir: Path,
    feature_names: list[str],
    features: list[dict[str, float]],
    labels: np.ndarray,
    sl: np.ndarray,
    tp: np.ndarray,
    lots: np.ndarray,
    hours: np.ndarray,
    times: np.ndarray,
    last_event_id: int,
):
    """Persist extracted features to a compressed Parquet file."""
    if pq is None or pa is None:
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_cache = pd.DataFrame(
            [{fn: feat.get(fn, 0.0) for fn in feature_names} for feat in features]
        )
        df_cache["label"] = labels
        df_cache["sl_target"] = sl
        df_cache["tp_target"] = tp
        df_cache["lot_target"] = lots
        df_cache["hours"] = hours
        df_cache["event_time"] = pd.to_datetime(times)
        table = pa.Table.from_pandas(df_cache)
        meta = {
            b"feature_names": json.dumps(feature_names).encode(),
            b"last_event_id": str(last_event_id).encode(),
        }
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, out_dir / "features.parquet", compression="gzip")
    except Exception:
        return


def _read_last_event_id(out_dir: Path) -> int:
    """Read ``last_event_id`` from an existing model file in ``out_dir``."""
    json_path = out_dir / "model.json"
    gz_path = out_dir / "model.json.gz"
    model_file: Path | None = None
    open_func = open
    if gz_path.exists():
        model_file = gz_path
        open_func = gzip.open
    elif json_path.exists():
        model_file = json_path
    if model_file is None:
        return 0
    try:
        with open_func(model_file, "rt") as f:
            data = json.load(f)
        return int(data.get("last_event_id", 0))
    except Exception:
        return 0


def _risk_parity_weights(cov: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Compute risk-parity weights for covariance matrix ``cov``.

    This solves for weights where each asset contributes equally to overall
    portfolio risk using a simple iterative proportional scaling algorithm.
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    w = np.ones(n) / n
    for _ in range(max_iter):
        # Marginal risk contribution of each asset
        marginal = cov @ w
        risk_contrib = w * marginal
        target = risk_contrib.mean()
        if np.max(np.abs(risk_contrib - target)) < tol:
            break
        # Scale weights inversely proportional to their risk contribution
        w *= target / risk_contrib
        w /= w.sum()
    return w


def _compute_risk_parity(
    price_map: dict[str, list[float]], window: int = 100
) -> tuple[dict[str, float], np.ndarray, list[str]]:
    """Return risk-parity weights and covariance given ``price_map``."""
    returns: dict[str, pd.Series] = {}
    for sym, prices in price_map.items():
        if len(prices) > 1:
            s = pd.Series(prices).pct_change().dropna()
            if window > 0:
                s = s.tail(window)
            if not s.empty:
                returns[sym] = s.reset_index(drop=True)
    if len(returns) < 2:
        if len(returns) == 1:
            sym = next(iter(returns))
            s = returns[sym]
            cov = np.array([[float(s.var())]])
            return {sym: 1.0}, cov, [sym]
        return {}, np.array([]), []
    df = pd.DataFrame(returns)
    cov = df.cov().values
    weights = _risk_parity_weights(cov)
    symbols = list(df.columns)
    weight_map = {sym: float(w) for sym, w in zip(symbols, weights)}
    return weight_map, cov, symbols


def _extract_features(
    rows,
    use_sma=False,
    sma_window=5,
    use_rsi=False,
    rsi_period=14,
    use_macd=False,
    use_atr=False,
    atr_period=14,
    use_bollinger=False,
    boll_window=20,
    use_stochastic=False,
    use_adx=False,
    use_volume=False,
    use_orderbook=False,
    volatility=None,
    higher_timeframes=None,
    *,
    corr_map=None,
    extra_price_series=None,
    symbol_graph: dict | str | Path | None = None,
    corr_window: int = 5,
    encoder: dict | None = None,
    calendar_events: list[tuple[datetime, float]] | None = None,
    event_window: float = 60.0,
    perf_budget: float | None = None,
    news_sentiment: dict[str, list[tuple[datetime, float]]] | None = None,
):
    feature_dicts = []
    labels = []
    sl_targets = []
    tp_targets = []
    lot_targets = []
    prices = []
    hours = []
    times = []
    price_map = {sym: [] for sym in (extra_price_series or {}).keys()}
    macd_state = {}
    higher_timeframes = [str(tf).upper() for tf in (higher_timeframes or [])]
    tf_map = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200,
    }
    tf_secs = {tf: tf_map.get(tf, 60) * 60 for tf in higher_timeframes}
    tf_prices = {tf: [] for tf in higher_timeframes}
    tf_macd_state = {tf: {} for tf in higher_timeframes}
    tf_macd = {tf: 0.0 for tf in higher_timeframes}
    tf_macd_sig = {tf: 0.0 for tf in higher_timeframes}
    tf_last_bin = {tf: None for tf in higher_timeframes}
    tf_prev_price = {tf: None for tf in higher_timeframes}
    stoch_state = {}
    adx_state = {}
    extra_series = extra_price_series or {}
    price_map.update({sym: [] for sym in extra_series.keys()})
    if corr_map:
        for base, peers in corr_map.items():
            price_map.setdefault(base, [])
            for p in peers:
                price_map.setdefault(p, [])

    pair_weights: dict[tuple[str, str], float] = {}
    sym_metrics: dict[str, dict[str, float]] = {}
    if symbol_graph:
        try:
            if not isinstance(symbol_graph, dict):
                with open(symbol_graph) as f_g:
                    graph_params = json.load(f_g)
            else:
                graph_params = symbol_graph
            symbols = graph_params.get("symbols", [])
            edge_index = graph_params.get("edge_index", [])
            weights = graph_params.get("edge_weight", [])
            for (i, j), w in zip(edge_index, weights):
                if i < len(symbols) and j < len(symbols):
                    a = symbols[i]
                    b = symbols[j]
                    pair_weights[(a, b)] = float(w)
            metrics = graph_params.get("metrics", {})
            for m_name, vals in metrics.items():
                for i, sym in enumerate(symbols):
                    sym_metrics.setdefault(sym, {})[m_name] = float(vals[i])
        except Exception:
            pair_weights = {}
            sym_metrics = {}
    enc_window = int(encoder.get("window")) if encoder else 0
    enc_weights = (
        np.array(encoder.get("weights", []), dtype=float) if encoder else np.empty((0, 0))
    )
    enc_centers = (
        np.array(encoder.get("centers", []), dtype=float) if encoder else np.empty((0, 0))
    )
    calendar_events = calendar_events or []
    news_indices = {sym: 0 for sym in (news_sentiment or {}).keys()}
    row_idx = 0

    start_time = time.perf_counter()
    psutil.cpu_percent(interval=None)
    heavy_order = [
        "multi_tf",
        "order_book",
        "use_adx",
        "use_stochastic",
        "use_bollinger",
        "use_atr",
    ]

    for r in rows:
        if r.get("action", "").upper() != "OPEN":
            continue
        if str(r.get("is_anomaly", "0")).lower() in ("1", "true", "yes"):
            continue

        t = r["event_time"]
        if not isinstance(t, datetime):
            parsed = None
            for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
                try:
                    parsed = datetime.strptime(str(t), fmt)
                    break
                except Exception:
                    continue
            if parsed is None:
                continue
            t = parsed

        times.append(t)
        order_type = int(float(r.get("order_type", 0)))
        label = 1 if order_type == 0 else 0  # buy=1, sell=0

        price = _safe_float(r.get("price", 0))
        sl = _safe_float(r.get("sl", 0))
        tp = _safe_float(r.get("tp", 0))
        lots = _safe_float(r.get("lots", 0))
        profit = _safe_float(r.get("profit", 0))

        for tf in higher_timeframes:
            tf_bin = int(t.timestamp() // tf_secs[tf])
            if tf_last_bin[tf] is None:
                tf_last_bin[tf] = tf_bin
            elif tf_bin != tf_last_bin[tf]:
                if tf_prev_price[tf] is not None:
                    tf_prices[tf].append(tf_prev_price[tf])
                    if use_macd:
                        tf_macd[tf], tf_macd_sig[tf] = _macd_update(
                            tf_macd_state[tf], tf_prev_price[tf]
                        )
                tf_last_bin[tf] = tf_bin
            tf_prev_price[tf] = price

        symbol = r.get("symbol", "")
        sym_prices = price_map.setdefault(symbol, [])

        spread = _safe_float(r.get("spread", 0))
        slippage = _safe_float(r.get("slippage", 0))
        account_equity = _safe_float(r.get("equity", 0))
        margin_level = _safe_float(r.get("margin_level", 0))
        commission = _safe_float(r.get("commission", 0))
        swap = _safe_float(r.get("swap", 0))
        trend_est = _safe_float(r.get("trend_estimate", 0))
        trend_var = _safe_float(r.get("trend_variance", 0))

        hour_sin = math.sin(2 * math.pi * t.hour / 24)
        hour_cos = math.cos(2 * math.pi * t.hour / 24)
        dow_sin = math.sin(2 * math.pi * t.weekday() / 7)
        dow_cos = math.cos(2 * math.pi * t.weekday() / 7)

        sl_dist = _safe_float(r.get("sl_dist", sl - price))
        tp_dist = _safe_float(r.get("tp_dist", tp - price))
        sl_hit = _safe_float(r.get("sl_hit_dist", 0.0))
        tp_hit = _safe_float(r.get("tp_hit_dist", 0.0))

        feat = {
            "symbol": symbol,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "lots": lots,
            "profit": profit,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "sl_hit_dist": sl_hit,
            "tp_hit_dist": tp_hit,
            "spread": spread,
            "slippage": slippage,
            "equity": account_equity,
            "margin_level": margin_level,
            "commission": commission,
            "swap": swap,
            "trend_estimate": trend_est,
            "trend_variance": trend_var,
            "event_id": int(float(r.get("event_id", 0) or 0)),
        }

        if use_orderbook:
            feat.update(
                {
                    "book_bid_vol": float(r.get("book_bid_vol", 0) or 0),
                    "book_ask_vol": float(r.get("book_ask_vol", 0) or 0),
                    "book_imbalance": float(r.get("book_imbalance", 0) or 0),
                }
            )

        if calendar_events is not None:
            flag = 0.0
            impact_val = 0.0
            for ev_time, ev_imp in calendar_events:
                if abs((t - ev_time).total_seconds()) <= event_window * 60.0:
                    flag = 1.0
                    if ev_imp > impact_val:
                        impact_val = ev_imp
            feat["event_flag"] = flag
            feat["event_impact"] = impact_val

        if use_volume:
            feat["volume"] = float(r.get("volume", 0) or 0)

        if volatility is not None:
            key = t.strftime("%Y-%m-%d %H")
            vol = volatility.get(key)
            if vol is None:
                key = t.strftime("%Y-%m-%d")
                vol = volatility.get(key, 0.0)
            feat["volatility"] = float(vol)

        if use_sma:
            feat["sma"] = _sma(prices, sma_window)

        if use_rsi:
            feat["rsi"] = _rsi(prices, rsi_period)

        if use_macd:
            macd, signal = _macd_update(macd_state, price)
            feat["macd"] = macd
            feat["macd_signal"] = signal

        if use_atr:
            feat["atr"] = _atr(prices, atr_period)

        if use_bollinger:
            upper, mid, lower = _bollinger(prices, boll_window)
            feat["bollinger_upper"] = upper
            feat["bollinger_middle"] = mid
            feat["bollinger_lower"] = lower

        if use_stochastic:
            k, d_val = _stochastic_update(stoch_state, price)
            feat["stochastic_k"] = k
            feat["stochastic_d"] = d_val

        if use_adx:
            feat["adx"] = _adx_update(adx_state, price)

        for tf in higher_timeframes:
            prices_tf = tf_prices.get(tf, [])
            if use_sma:
                feat[f"sma_{tf}"] = _sma(prices_tf, sma_window)
            if use_rsi:
                feat[f"rsi_{tf}"] = _rsi(prices_tf, rsi_period)
            if use_macd:
                feat[f"macd_{tf}"] = tf_macd.get(tf, 0.0)
                feat[f"macd_signal_{tf}"] = tf_macd_sig.get(tf, 0.0)

        if corr_map:
            base_prices = price_map.get(symbol, [])
            for peer in corr_map.get(symbol, []):
                peer_prices = price_map.get(peer, [])
                corr = _rolling_corr(base_prices, peer_prices, corr_window)
                ratio = 0.0
                if base_prices and peer_prices and peer_prices[-1] != 0:
                    ratio = base_prices[-1] / peer_prices[-1]
                feat[f"corr_{peer}"] = corr
                feat[f"ratio_{peer}"] = ratio

        if pair_weights:
            for (a, b), w in pair_weights.items():
                if a == symbol:
                    feat[f"corr_{symbol}_{b}"] = w
        if sym_metrics:
            mvals = sym_metrics.get(symbol)
            if mvals:
                for m_name, m_val in mvals.items():
                    feat[f"graph_{m_name}"] = m_val

        if enc_window > 0 and enc_weights.size > 0:
            seq = (prices + [price])[-(enc_window + 1) :]
            if len(seq) < enc_window + 1:
                seq = [seq[0]] * (enc_window + 1 - len(seq)) + seq
            deltas = np.diff(seq)
            vals = deltas.dot(enc_weights)
            for i, v in enumerate(vals):
                feat[f"ae{i}"] = float(v)
        if enc_centers.size > 0:
            d = ((enc_centers - vals) ** 2).sum(axis=1)
            feat["regime"] = float(int(np.argmin(d)))

        if news_sentiment is not None:
            sent_list = news_sentiment.get(symbol)
            score = 0.0
            if sent_list:
                idx = news_indices.get(symbol, 0)
                while idx + 1 < len(sent_list) and sent_list[idx + 1][0] <= t:
                    idx += 1
                news_indices[symbol] = idx
                if sent_list[idx][0] <= t:
                    score = float(sent_list[idx][1])
            feat["news_sentiment"] = score

        prices.append(price)
        sym_prices.append(price)
        for sym, series in extra_series.items():
            if sym == symbol:
                continue
            if row_idx < len(series):
                price_map.setdefault(sym, []).append(float(series[row_idx]))

        feature_dicts.append(feat)
        labels.append(label)
        sl_targets.append(sl_dist)
        tp_targets.append(tp_dist)
        lot_targets.append(lots)
        hours.append(t.hour)
        row_idx += 1

        if perf_budget is not None:
            elapsed = time.perf_counter() - start_time
            load = psutil.cpu_percent(interval=None)
            while heavy_order and (
                elapsed > perf_budget * row_idx or load > 90.0
            ):
                feat_name = heavy_order.pop(0)
                if feat_name == "use_atr":
                    use_atr = False
                elif feat_name == "use_bollinger":
                    use_bollinger = False
                elif feat_name == "use_stochastic":
                    use_stochastic = False
                elif feat_name == "use_adx":
                    use_adx = False
                elif feat_name == "order_book":
                    use_orderbook = False
                elif feat_name == "multi_tf":
                    higher_timeframes = []
                    tf_prices.clear()
                    tf_macd_state.clear()
                    tf_macd.clear()
                    tf_macd_sig.clear()
                    tf_last_bin.clear()
                    tf_prev_price.clear()
                logging.info("Disabling %s due to performance budget", feat_name)
                elapsed = time.perf_counter() - start_time
                load = psutil.cpu_percent(interval=None)

    enabled_feats = []
    if use_sma:
        enabled_feats.append("sma")
    if use_rsi:
        enabled_feats.append("rsi")
    if use_macd:
        enabled_feats.append("macd")
    if use_atr:
        enabled_feats.append("atr")
    if use_bollinger:
        enabled_feats.append("bollinger")
    if use_stochastic:
        enabled_feats.append("stochastic")
    if use_adx:
        enabled_feats.append("adx")
    if higher_timeframes:
        enabled_feats.extend(f"tf_{tf}" for tf in higher_timeframes)
    if news_sentiment is not None:
        enabled_feats.append("news_sentiment")
    logging.info("Enabled features: %s", sorted(enabled_feats))

    return (
        feature_dicts,
        np.array(labels),
        np.array(sl_targets),
        np.array(tp_targets),
        np.array(hours, dtype=int),
        np.array(lot_targets),
        np.array(times, dtype="datetime64[s]"),
        price_map,
    )


def _best_threshold(y_true, probas):
    """Return probability threshold with best F1 score."""
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 17):
        preds = (probas >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, best_f1


def _train_lite_mode(
    data_dir: Path,
    out_dir: Path,
    *,
    use_sma: bool = False,
    sma_window: int = 5,
    use_rsi: bool = False,
    rsi_period: int = 14,
    use_macd: bool = False,
    use_atr: bool = False,
    atr_period: int = 14,
    use_bollinger: bool = False,
    boll_window: int = 20,
    use_stochastic: bool = False,
    use_adx: bool = False,
    use_volume: bool = False,
    use_orderbook: bool = False,
    volatility_series=None,
    corr_map=None,
    corr_window: int = 5,
    extra_price_series=None,
    news_sentiment=None,
    calendar_events: list[tuple[datetime, float]] | None = None,
    event_window: float = 60.0,
    encoder_file: Path | None = None,
    chunk_size: int = 50000,
    compress_model: bool = False,
    regime_model: dict | None = None,
    flight_uri: str | None = None,
    mode: str = "lite",
) -> None:
    """Stream features and train an SGD classifier incrementally."""

    ae_info = None
    rows_iter, data_commits, data_checksums = _load_logs(
        data_dir, lite_mode=True, chunk_size=chunk_size, flight_uri=flight_uri
    )
    last_event_id = 0
    encoder = None
    enc_model = None
    if encoder_file is not None and encoder_file.exists():
        with open(encoder_file) as f:
            encoder = json.load(f)

    vec = DictVectorizer(sparse=False)
    scaler = StandardScaler()
    clf = SGDClassifier(loss="log_loss")
    if regime_model is not None:
        vec_reg = DictVectorizer(sparse=False)
        vec_reg.fit([{n: 0.0} for n in regime_model.get("feature_names", [])])
        reg_mean = np.array(regime_model.get("mean", []), dtype=float)
        reg_std = np.array(regime_model.get("std", []), dtype=float)
        reg_std[reg_std == 0] = 1.0
        reg_centers = np.array(regime_model.get("centers", []), dtype=float)
    else:
        vec_reg = None
        reg_mean = reg_std = reg_centers = None
    first = True
    sample_count = 0

    for chunk in rows_iter:
        if "event_id" in chunk.columns:
            max_id = pd.to_numeric(chunk["event_id"], errors="coerce").max()
            if not pd.isna(max_id):
                last_event_id = max(last_event_id, int(max_id))
        (
            f_chunk,
            l_chunk,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = _extract_features(
            chunk.to_dict("records"),
            use_sma=use_sma,
            sma_window=sma_window,
            use_rsi=use_rsi,
            rsi_period=rsi_period,
            use_macd=use_macd,
            use_atr=use_atr,
            atr_period=atr_period,
            use_bollinger=use_bollinger,
            boll_window=boll_window,
            use_stochastic=use_stochastic,
            use_adx=use_adx,
            use_volume=use_volume,
            use_orderbook=use_orderbook,
            volatility=volatility_series,
            higher_timeframes=None,
            corr_map=corr_map,
            corr_window=corr_window,
            extra_price_series=extra_price_series,
            encoder=encoder,
            calendar_events=calendar_events,
            event_window=event_window,
            news_sentiment=news_sentiment,
            symbol_graph=graph_params,
        )
        if not f_chunk:
            continue
        if vec_reg is not None and f_chunk:
            Xr = vec_reg.transform(f_chunk)
            Xr = (Xr - reg_mean) / reg_std
            dists = ((Xr[:, None, :] - reg_centers[None, :, :]) ** 2).sum(axis=2)
            rids = dists.argmin(axis=1)
            for i, r in enumerate(rids):
                f_chunk[i][f"regime_{int(r)}"] = 1.0
        sample_count += len(l_chunk)
        if first:
            X = vec.fit_transform(f_chunk)
            scaler.partial_fit(X)
            X = scaler.transform(X)
            X = _encode_features(enc_model, X)
            clf.partial_fit(X, l_chunk, classes=np.array([0, 1]))
            first = False
        else:
            X = vec.transform(f_chunk)
            scaler.partial_fit(X)
            X = scaler.transform(X)
            X = _encode_features(enc_model, X)
            clf.partial_fit(X, l_chunk)

    if first:
        raise ValueError(f"No training data found in {data_dir}")

    feature_names = vec.get_feature_names_out().tolist()
    model = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "model_type": "logreg",
        "weighted": False,
        "training_mode": "lite",
        "mode": mode,
        "train_accuracy": float("nan"),
        "val_accuracy": float("nan"),
        "threshold": 0.5,
        "accuracy": float("nan"),
        "num_samples": int(sample_count),
        "feature_importance": {},
        "mean": scaler.mean_.astype(np.float32).tolist(),
        "std": scaler.scale_.astype(np.float32).tolist(),
        "coefficients": clf.coef_[0].astype(np.float32).tolist(),
        "intercept": float(clf.intercept_[0]),
        "classes": [int(c) for c in clf.classes_],
        "last_event_id": int(last_event_id),
        "mode": mode,
    }
    model["feature_flags"] = {
        "sma": use_sma,
        "rsi": use_rsi,
        "macd": use_macd,
        "atr": use_atr,
        "bollinger": use_bollinger,
        "stochastic": use_stochastic,
        "adx": use_adx,
        "volume": use_volume,
        "order_book": use_orderbook,
        "higher_timeframes": [],
    }
    if ae_info:
        model["autoencoder"] = ae_info
    if regime_model is not None and reg_centers is not None:
        model["regime_centers"] = reg_centers.astype(float).tolist()
        model["regime_feature_names"] = regime_model.get("feature_names", [])
    if data_commits:
        model["data_commit"] = ",".join(sorted(set(data_commits)))
    if data_checksums:
        model["data_checksum"] = ",".join(sorted(set(data_checksums)))
    if calendar_events:
        model["calendar_events"] = [
            [dt.isoformat(), float(imp)] for dt, imp in calendar_events
        ]
        model["event_window"] = float(event_window)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
    open_func = gzip.open if compress_model else open
    with open_func(model_path, "wt") as f:
        json.dump(model, f)
    print(f"Model written to {model_path}")
    if model_type != "tft":
        _export_onnx(clf, feature_names, out_dir)


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    use_sma: bool | None = None,
    sma_window: int = 5,
    use_rsi: bool | None = None,
    rsi_period: int = 14,
    use_macd: bool | None = None,
    use_atr: bool | None = None,
    atr_period: int = 14,
    use_bollinger: bool | None = None,
    boll_window: int = 20,
    use_stochastic: bool | None = None,
    use_adx: bool | None = None,
    use_volume: bool | None = None,
    use_orderbook: bool | None = None,
    volatility_series=None,
    higher_timeframes: list[str] | None = None,
    grid_search: bool | None = None,
    c_values=None,
    model_type: str | None = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    incremental: bool = False,
    sequence_length: int = 5,
    corr_map=None,
    corr_window: int = 5,
    extra_price_series=None,
    symbol_graph: Path | None = None,
    bayes_steps: int | None = None,
    regress_sl_tp: bool = False,
    regress_lots: bool = False,
    early_stop: bool = False,
    encoder_file: Path | None = None,
    cache_features: bool = True,
    calendar_events: list[tuple[datetime, float]] | None = None,
    event_window: float = 60.0,
    calibration: str | None = None,
    stack_models: list[str] | None = None,
    prune_threshold: float = 0.0,
    prune_warn: float = 0.5,
    lite_mode: bool | None = None,
    mode: str | None = None,
    compress_model: bool = False,
    regime_model_file: Path | None = None,
    moe: bool = False,
    flight_uri: str | None = None,
    use_encoder: bool = False,
    uncertain_file: Path | None = None,
    uncertain_weight: float = 2.0,
    decay_half_life: float | None = None,
    news_sentiment_file: Path | None = None,
    rl_finetune: bool = False,
    replay_file: Path | None = None,
    replay_weight: float = 3.0,
):
    """Train a simple classifier model from the log directory."""
    news_data = _load_news_sentiment(news_sentiment_file) if news_sentiment_file else None
    if replay_file:
        try:
            rep_df = pd.read_csv(replay_file)
            replay_weights = {
                int(row["event_id"]): float(row.get("weight", 1.0))
                for _, row in rep_df.iterrows()
                if not pd.isna(row.get("event_id"))
            }
        except Exception as exc:
            logging.warning("failed to read replay file %s: %s", replay_file, exc)
            replay_weights = {}
    else:
        replay_weights = {}
    # Automatically select features and model based on hardware capabilities
    resources = detect_resources()
    heavy_mode = resources["heavy_mode"]
    if lite_mode is None:
        lite_mode = resources["lite_mode"]
    if mode is None:
        mode = resources["mode"]
    if model_type is None:
        model_type = resources["model_type"]
    if bayes_steps is None:
        bayes_steps = 0 if resources["lite_mode"] else resources["bayes_steps"]
    if grid_search is None:
        grid_search = heavy_mode
    if use_sma is None:
        use_sma = heavy_mode
    if use_rsi is None:
        use_rsi = heavy_mode
    if use_macd is None:
        use_macd = heavy_mode
    if use_atr is None:
        use_atr = heavy_mode
    if use_bollinger is None:
        use_bollinger = heavy_mode
    if use_stochastic is None:
        use_stochastic = heavy_mode
    if use_adx is None:
        use_adx = heavy_mode
    if use_volume is None:
        use_volume = heavy_mode
    if use_orderbook is None:
        use_orderbook = heavy_mode
    if lite_mode:
        regime_model = None
        if regime_model_file and regime_model_file.exists():
            with open(regime_model_file) as f:
                regime_model = json.load(f)
        use_orderbook = False
        _train_lite_mode(
            data_dir,
            out_dir,
            use_sma=use_sma,
            sma_window=sma_window,
            use_rsi=use_rsi,
            rsi_period=rsi_period,
            use_macd=use_macd,
            use_atr=use_atr,
            atr_period=atr_period,
            use_bollinger=use_bollinger,
            boll_window=boll_window,
            use_stochastic=use_stochastic,
            use_adx=use_adx,
            use_volume=use_volume,
            use_orderbook=use_orderbook,
            volatility_series=volatility_series,
            corr_map=corr_map,
            corr_window=corr_window,
            extra_price_series=extra_price_series,
            news_sentiment=news_data,
            calendar_events=calendar_events,
            event_window=event_window,
            encoder_file=encoder_file,
            compress_model=compress_model,
            regime_model=regime_model,
            flight_uri=flight_uri,
            mode=mode,
        )
        return
    feature_flags = {
        "sma": use_sma,
        "rsi": use_rsi,
        "macd": use_macd,
        "atr": use_atr,
        "bollinger": use_bollinger,
        "stochastic": use_stochastic,
        "adx": use_adx,
        "volume": use_volume,
        "order_book": use_orderbook,
        "higher_timeframes": higher_timeframes or [],
    }
    if bayes_steps > 0:
        try:
            import optuna  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            logging.warning(
                "optuna is not installed; skipping hyperparameter search"
            )
            bayes_steps = 0
    enc_model: ContrastiveEncoder | None = None
    if use_encoder:
        try:
            enc_model = _load_contrastive_encoder(Path("encoder.pt"))
        except Exception as exc:
            logging.warning("failed to load encoder: %s", exc)
            enc_model = None

    existing_model = None
    last_event_id = 0
    provided_half_life = decay_half_life
    if decay_half_life is None:
        decay_half_life = 0.0
    json_path = out_dir / "model.json"
    gz_path = out_dir / "model.json.gz"
    model_file: Path | None = None
    open_model = open
    if gz_path.exists():
        model_file = gz_path
        open_model = gzip.open
    elif json_path.exists():
        model_file = json_path
    if model_file is not None:
        with open_model(model_file, "rt") as f:
            existing_model = json.load(f)
            last_event_id = int(existing_model.get("last_event_id", 0))
            if decay_half_life == 0.0 and provided_half_life is None:
                decay_half_life = float(
                    existing_model.get("half_life_days")
                    or existing_model.get("weight_decay", {}).get("half_life_days", 0.0)
                )
    elif incremental:
        raise FileNotFoundError(f"{json_path} not found for incremental training")

    features = labels = sl_targets = tp_targets = hours = lot_targets = event_times = None
    loaded_from_cache = False
    data_commits: list[str] = []
    data_checksums: list[str] = []
    graph_params = None
    if symbol_graph:
        try:
            with open(symbol_graph) as f_g:
                graph_params = json.load(f_g)
        except Exception as exc:  # pragma: no cover - file errors
            logging.warning("failed to load symbol graph: %s", exc)

    if cache_features:
        loaded = _load_feature_cache(out_dir, existing_model)
        if loaded is not None:
            (
                features,
                labels,
                sl_targets,
                tp_targets,
                lot_targets,
                hours,
                event_times,
                last_event_id,
            ) = loaded
            loaded_from_cache = True

    price_map_total: dict[str, list[float]] = {}
    if not loaded_from_cache:
        rows_iter, data_commits, data_checksums = _load_logs(
            data_dir, lite_mode=True, flight_uri=flight_uri
        )
        encoder = None
        if encoder_file is not None and encoder_file.exists():
            with open(encoder_file) as f:
                encoder = json.load(f)
        features = []
        labels_list: list[np.ndarray] = []
        sl_list: list[np.ndarray] = []
        tp_list: list[np.ndarray] = []
        hours_list: list[np.ndarray] = []
        lot_list: list[np.ndarray] = []
        time_list: list[np.ndarray] = []
        for chunk in rows_iter:
            if "event_id" in chunk.columns:
                max_id = pd.to_numeric(chunk["event_id"], errors="coerce").max()
                if not pd.isna(max_id):
                    last_event_id = max(last_event_id, int(max_id))
            (
                f_chunk,
                l_chunk,
                sl_chunk,
                tp_chunk,
                h_chunk,
                lot_chunk,
                t_chunk,
                p_chunk,
            ) = _extract_features(
                chunk.to_dict("records"),
                use_sma=use_sma,
                sma_window=sma_window,
                use_rsi=use_rsi,
                rsi_period=rsi_period,
                use_macd=use_macd,
                use_atr=use_atr,
                atr_period=atr_period,
                use_bollinger=use_bollinger,
                boll_window=boll_window,
                use_stochastic=use_stochastic,
                use_adx=use_adx,
                use_volume=use_volume,
                use_orderbook=use_orderbook,
                volatility=volatility_series,
                higher_timeframes=higher_timeframes,
                corr_map=corr_map,
                corr_window=corr_window,
                extra_price_series=extra_price_series,
                encoder=encoder,
                calendar_events=calendar_events,
                event_window=event_window,
                news_sentiment=news_data,
                symbol_graph=graph_params,
            )
            for sym, prices in p_chunk.items():
                price_map_total.setdefault(sym, []).extend(prices)
            features.extend(f_chunk)
            labels_list.append(l_chunk)
            sl_list.append(sl_chunk)
            tp_list.append(tp_chunk)
            hours_list.append(h_chunk)
            lot_list.append(lot_chunk)
            time_list.append(t_chunk)
        labels = np.concatenate(labels_list) if labels_list else np.array([])
        sl_targets = np.concatenate(sl_list) if sl_list else np.array([])
        tp_targets = np.concatenate(tp_list) if tp_list else np.array([])
        hours = np.concatenate(hours_list) if hours_list else np.array([], dtype=int)
        lot_targets = np.concatenate(lot_list) if lot_list else np.array([])
        event_times = (
            np.concatenate(time_list)
            if time_list
            else np.array([], dtype="datetime64[s]")
        )
        # feature cache will be saved later once final feature names are known
    else:
        # even when loading from cache, capture latest event_id for model metadata
        rows_df, _, _ = _load_logs(data_dir, flight_uri=flight_uri)
        if "event_id" in rows_df.columns:
            max_id = pd.to_numeric(rows_df["event_id"], errors="coerce").max()
            if not pd.isna(max_id):
                last_event_id = max(last_event_id, int(max_id))
    # append labeled uncertain decisions
    base_len = len(features)
    ufile = uncertain_file or (data_dir / "uncertain_decisions_labeled.csv")
    added = 0
    if ufile.exists():
        df_u = pd.read_csv(ufile, sep=";")
        extra_feats: list[dict[str, float]] = []
        extra_labels: list[float] = []
        now_ts = np.datetime64(datetime.utcnow(), "s")
        for _, row in df_u.iterrows():
            feat_str = str(row.get("features", ""))
            vals = [v for v in feat_str.split(",") if v != ""]
            feat = {f"f{i}": float(v) for i, v in enumerate(vals)}
            extra_feats.append(feat)
            extra_labels.append(float(row.get("label", 0)))
        if extra_feats:
            features.extend(extra_feats)
            labels = np.concatenate([labels, np.array(extra_labels)])
            zeros = np.zeros(len(extra_feats))
            sl_targets = np.concatenate([sl_targets, zeros])
            tp_targets = np.concatenate([tp_targets, zeros])
            hours = np.concatenate([hours, zeros.astype(int)])
            lot_targets = np.concatenate([lot_targets, zeros])
            event_times = np.concatenate([event_times, np.full(len(extra_feats), now_ts)])
            added = len(extra_feats)
    if added:
        logger.info("loaded %d labeled uncertain decisions from %s", added, ufile)
    uncertainty_mask = np.concatenate([np.zeros(base_len), np.ones(added)])
    if not features:
        raise ValueError(f"No training data found in {data_dir}")
    risk_parity, cov_matrix, cov_symbols = _compute_risk_parity(price_map_total)
    regime_info = None
    reg_centers = None
    try:
        from sklearn.cluster import KMeans

        feat_reg = [dict(f) for f in features]
        for f in feat_reg:
            f.pop("profit", None)
            for k in list(f.keys()):
                if k.startswith("regime"):
                    f.pop(k, None)
        vec_reg = DictVectorizer(sparse=False)
        Xr = vec_reg.fit_transform(feat_reg)
        n_reg = min(3, len(feat_reg))
        if n_reg > 1:
            kmeans = KMeans(n_clusters=n_reg, random_state=42, n_init=10)
            regimes = kmeans.fit_predict(Xr)
            for i, r in enumerate(regimes):
                features[i]["regime"] = int(r)
            reg_centers = kmeans.cluster_centers_
            reg_mean = Xr.mean(axis=0)
            reg_std = Xr.std(axis=0)
            reg_std[reg_std == 0] = 1.0
            regime_info = {
                "feature_names": vec_reg.get_feature_names_out().tolist(),
                "mean": reg_mean.astype(np.float32).tolist(),
                "std": reg_std.astype(np.float32).tolist(),
                "centers": reg_centers.astype(np.float32).tolist(),
            }
    except Exception as exc:
        logging.warning("regime clustering failed: %s", exc)
    ae_info = None
    ae_feature_order = ["price", "sl", "tp", "lots", "spread", "slippage"]
    ae_matrix = np.array(
        [[float(f.get(k, 0.0)) for k in ae_feature_order] for f in features],
        dtype=float,
    )
    ae_hidden = 4
    if ae_matrix.shape[0] >= 10:
        ae_mean = ae_matrix.mean(axis=0)
        ae_std = ae_matrix.std(axis=0)
        ae_std[ae_std == 0] = 1.0
        ae_scaled = (ae_matrix - ae_mean) / ae_std
        ae_model = MLPRegressor(hidden_layer_sizes=(ae_hidden,), max_iter=200, random_state=42)
        ae_model.fit(ae_scaled, ae_scaled)
        recon = ae_model.predict(ae_scaled)
        ae_errors = np.mean((ae_scaled - recon) ** 2, axis=1)
        ae_threshold = float(np.percentile(ae_errors, 95))
        mask = ae_errors <= ae_threshold
        if mask.sum() > 0:
            features = [features[i] for i, m in enumerate(mask) if m]
            labels = labels[mask]
            sl_targets = sl_targets[mask]
            tp_targets = tp_targets[mask]
            hours = hours[mask]
            lot_targets = lot_targets[mask]
            uncertainty_mask = uncertainty_mask[mask]
        ae_info = {
            "weights": [w.astype(np.float32).tolist() for w in ae_model.coefs_],
            "bias": [b.astype(np.float32).tolist() for b in ae_model.intercepts_],
            "mean": ae_mean.astype(np.float32).tolist(),
            "std": ae_std.astype(np.float32).tolist(),
            "threshold": ae_threshold,
            "feature_order": ae_feature_order,
        }
    hidden_size = 8
    logreg_C = 1.0
    best_trial = None
    study = None
    sl_coef = []
    tp_coef = []
    sl_inter = 0.0
    tp_inter = 0.0
    lot_coef = []
    lot_inter = 0.0

    # ------------------------------------------------------------------
    # Graph-based symbol embeddings
    # ------------------------------------------------------------------
    symbol_embeddings: dict[str, list[float]] = {}
    if graph_params:
        for sym, vec in (graph_params.get("embeddings") or {}).items():
            symbol_embeddings[sym] = [float(v) for v in vec]
    if graph_params and not symbol_embeddings and _HAS_TORCH and _HAS_PYG:
        try:
            symbols = graph_params.get("symbols", [])
            edge_index = torch.tensor(
                graph_params.get("edge_index", []), dtype=torch.long
            ).t().contiguous()
            if edge_index.numel() > 0:
                emb_dim = int(graph_params.get("embedding_dim", 8)) or 8
                node2vec = Node2Vec(
                    edge_index,
                    embedding_dim=emb_dim,
                    walk_length=5,
                    context_size=3,
                    walks_per_node=10,
                )
                optimizer = torch.optim.Adam(node2vec.parameters(), lr=0.01)
                loader = node2vec.loader(batch_size=32, shuffle=True)
                for _ in range(20):
                    for pos_rw, neg_rw in loader:
                        optimizer.zero_grad()
                        loss = node2vec.loss(pos_rw, neg_rw)
                        loss.backward()
                        optimizer.step()
                emb = node2vec.embedding.weight.detach().cpu().numpy()
                for i, sym in enumerate(symbols):
                    symbol_embeddings[sym] = emb[i].astype(float).tolist()
                graph_params["embedding_dim"] = emb.shape[1]
        except Exception as exc:  # pragma: no cover - optional dependency
            logging.warning("symbol embedding computation failed: %s", exc)
    if symbol_embeddings:
        for feat in features:
            vec = symbol_embeddings.get(feat.get("symbol"))
            if vec is not None:
                for j, v in enumerate(vec):
                    feat[f"sym_emb_{j}"] = float(v)
        if graph_params and not graph_params.get("embedding_dim"):
            graph_params["embedding_dim"] = len(next(iter(symbol_embeddings.values())))

    if existing_model is not None:
        vec = DictVectorizer(sparse=False)
        vec.fit([{name: 0.0} for name in existing_model.get("feature_names", [])])
    else:
        vec = DictVectorizer(sparse=False)

    if len(labels) < 5 or len(np.unique(labels)) < 2:
        # Not enough data to create a meaningful split
        feat_train, y_train = features, labels
        feat_val, y_val = [], np.array([])
        sl_train = sl_targets
        sl_val = np.array([])
        tp_train = tp_targets
        tp_val = np.array([])
        lot_train = lot_targets
        lot_val = np.array([])
        hours_train = hours
        hours_val = np.array([])
        times_train = event_times
        times_val = np.array([], dtype="datetime64[s]")
        unc_train_mask = uncertainty_mask
    else:
        tscv = TimeSeriesSplit(n_splits=min(5, len(labels) - 1))
        # select the final chronological split for validation
        train_idx, val_idx = list(tscv.split(features))[-1]
        feat_train = [features[i] for i in train_idx]
        feat_val = [features[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        sl_train = sl_targets[train_idx]
        sl_val = sl_targets[val_idx]
        tp_train = tp_targets[train_idx]
        tp_val = tp_targets[val_idx]
        lot_train = lot_targets[train_idx]
        lot_val = lot_targets[val_idx]
        hours_train = hours[train_idx]
        hours_val = hours[val_idx]
        times_train = event_times[train_idx]
        times_val = event_times[val_idx]
        unc_train_mask = uncertainty_mask[train_idx]

        # if the training split ended up with only one class, fall back to using
        # all data for training so the model can be fit
        if len(np.unique(y_train)) < 2:
            feat_train, y_train = features, labels
            feat_val, y_val = [], np.array([])
            hours_train = hours
            hours_val = np.array([])
            lot_train = lot_targets
            lot_val = np.array([])
            times_train = event_times
            times_val = np.array([], dtype="datetime64[s]")
            unc_train_mask = uncertainty_mask

    coef_variances = None
    noise_variance = None
    weight_decay_info = None
    base_weight = np.ones(len(feat_train), dtype=float)
    if model_type in ("logreg", "bayes_logreg") and not grid_search:
        base_weight *= np.array(
            [
                abs(f.get("profit")) if f.get("profit") not in (None, 0)
                else abs(f.get("lots", 1.0))
                for f in feat_train
            ],
            dtype=float,
        )
    if unc_train_mask.size:
        base_weight *= np.where(unc_train_mask > 0, uncertain_weight, 1.0)
        logger.info(
            "emphasizing %d uncertain samples with weight %.2f",
            int(unc_train_mask.sum()),
            uncertain_weight,
        )
    if replay_weights:
        for i, feat in enumerate(feat_train):
            eid = int(feat.get("event_id", -1))
            if eid in replay_weights:
                base_weight[i] *= replay_weight * replay_weights[eid]
        logger.info(
            "emphasizing %d replay corrections with weight %.2f",
            len(replay_weights),
            replay_weight,
        )
    if decay_half_life > 0 and event_times is not None and event_times.size:
        _, decay = _compute_decay_weights(times_train, decay_half_life)
        base_weight *= decay
        weight_decay_info = {
            "half_life_days": float(decay_half_life),
            "ref_time": np.datetime_as_string(event_times.max(), unit="s"),
        }
    sample_weight = base_weight if not np.allclose(base_weight, 1.0) else None

    feat_train_clf = [dict(f) for f in feat_train]
    feat_val_clf = [dict(f) for f in feat_val]
    for f in feat_train_clf:
        f.pop("profit", None)
        f.pop("event_id", None)
    for f in feat_val_clf:
        f.pop("profit", None)
        f.pop("event_id", None)

    feat_train_reg = [dict(f) for f in feat_train_clf]
    feat_val_reg = [dict(f) for f in feat_val_clf]
    for f in feat_train_reg:
        f["sl_dist"] = 0.0
        f["tp_dist"] = 0.0
    for f in feat_val_reg:
        f["sl_dist"] = 0.0
        f["tp_dist"] = 0.0

    feat_train_lot = [dict(f) for f in feat_train_reg]
    feat_val_lot = [dict(f) for f in feat_val_reg]
    for f in feat_train_lot:
        f["lots"] = 0.0
    for f in feat_val_lot:
        f["lots"] = 0.0

    symbols_val_arr = (
        np.array([f.get("symbol", "") for f in feat_val]) if feat_val else np.array([])
    )
    profits_val = (
        np.array([f.get("profit", 0.0) for f in feat_val]) if feat_val else np.array([])
    )

    if existing_model is not None:
        X_train = vec.transform(feat_train_clf)
    else:
        X_train = vec.fit_transform(feat_train_clf)
    if feat_val_clf:
        X_val = vec.transform(feat_val_clf)
    else:
        X_val = np.empty((0, X_train.shape[1]))
    X_train = _encode_features(enc_model, X_train)
    X_val = _encode_features(enc_model, X_val)

    feature_names = vec.get_feature_names_out().tolist()
    X_train_reg = vec.transform(feat_train_reg)
    X_val_reg = (
        vec.transform(feat_val_reg) if feat_val_reg else np.empty((0, X_train_reg.shape[1]))
    )
    X_train_reg = _encode_features(enc_model, X_train_reg)
    X_val_reg = _encode_features(enc_model, X_val_reg)

    X_train_lot = vec.transform(feat_train_lot)
    X_val_lot = (
        vec.transform(feat_val_lot) if feat_val_lot else np.empty((0, X_train_lot.shape[1]))
    )
    X_train_lot = _encode_features(enc_model, X_train_lot)
    X_val_lot = _encode_features(enc_model, X_val_lot)

    if cache_features and not loaded_from_cache and features:
        _save_feature_cache(
            out_dir,
            feature_names,
            features,
            labels,
            sl_targets,
            tp_targets,
            lot_targets,
            hours,
            event_times,
            last_event_id,
        )


    bayes_threshold = None
    if bayes_steps > 0:
        available_models = ["logreg", "bayes_logreg", "random_forest", "xgboost", "nn"]
        try:
            import importlib.util

            if importlib.util.find_spec("xgboost") is None:
                available_models.remove("xgboost")
        except Exception:
            if "xgboost" in available_models:
                available_models.remove("xgboost")

        def _objective(trial):
            model_choice = trial.suggest_categorical("model_type", available_models)
            max_feats = min(len(feature_names), 10)
            sel_idx = []
            for i, name in enumerate(feature_names[:max_feats]):
                if trial.suggest_categorical(f"f_{name}", [True, False]):
                    sel_idx.append(i)
            if not sel_idx:
                sel_idx = list(range(max_feats))
            sel_idx += list(range(max_feats, len(feature_names)))
            X_tr = X_train[:, sel_idx]
            X_v = X_val[:, sel_idx]
            if model_choice == "logreg":
                c = trial.suggest_float("C", 1e-3, 10.0, log=True)
                clf = LogisticRegression(max_iter=200, C=c)
                clf.fit(X_tr, y_train)
            elif model_choice == "bayes_logreg":
                clf = BayesianRidge()
                clf.fit(X_tr, y_train)
            elif model_choice == "random_forest":
                est = trial.suggest_int("n_estimators", 50, 200)
                depth = trial.suggest_int("max_depth", 2, 8)
                clf = RandomForestClassifier(n_estimators=est, max_depth=depth, random_state=42)
                clf.fit(X_tr, y_train)
            elif model_choice == "xgboost":
                from xgboost import XGBClassifier  # type: ignore

                est = trial.suggest_int("n_estimators", 50, 300)
                lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                depth = trial.suggest_int("max_depth", 2, 8)
                clf = XGBClassifier(
                    n_estimators=est,
                    learning_rate=lr,
                    max_depth=depth,
                    eval_metric="logloss",
                    use_label_encoder=False,
                )
                clf.fit(X_tr, y_train)
            else:
                h = trial.suggest_int("hidden_size", 4, 64)
                try:
                    from tensorflow import keras  # type: ignore

                    clf = keras.Sequential(
                        [
                            keras.layers.Input(shape=(X_tr.shape[1],)),
                            keras.layers.Dense(h, activation="relu"),
                            keras.layers.Dense(1, activation="sigmoid"),
                        ]
                    )
                    clf.compile(optimizer="adam", loss="binary_crossentropy")
                    clf.fit(X_tr, y_train, epochs=50, verbose=0)
                except Exception:
                    clf = MLPClassifier(
                        hidden_layer_sizes=(h,), max_iter=500, random_state=42
                    )
                    clf.fit(X_tr, y_train)

            if hasattr(clf, "predict_proba"):
                val_proba = clf.predict_proba(X_v)[:, 1] if len(y_val) > 0 else np.empty(0)
            else:
                val_proba = clf.predict(X_v).reshape(-1) if len(y_val) > 0 else np.empty(0)

            thr = trial.suggest_float("threshold", 0.0, 1.0)
            trial.set_user_attr("features", [feature_names[i] for i in sel_idx])
            if len(y_val) > 0:
                preds = (val_proba >= thr).astype(int)
                if profits_val.size == len(preds):
                    return float(np.sum(profits_val[preds == 1]))
                elif len(np.unique(y_val)) > 1:
                    return roc_auc_score(y_val, preds)
                else:
                    return accuracy_score(y_val, preds)
            return 0.0

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(_objective, n_trials=bayes_steps)
        best_trial = study.best_trial
        model_type = best_trial.params.get("model_type", model_type)
        bayes_threshold = float(best_trial.params.get("threshold", 0.5))
        max_feats = min(len(feature_names), 10)
        sel_idx = [
            i
            for i, name in enumerate(feature_names[:max_feats])
            if best_trial.params.get(f"f_{name}", True)
        ]
        if not sel_idx:
            sel_idx = list(range(max_feats))
        sel_idx += list(range(max_feats, len(feature_names)))
        selected_indices = sel_idx
        X_train = X_train[:, sel_idx]
        X_val = X_val[:, sel_idx]
        X_train_reg = X_train_reg[:, sel_idx]
        X_val_reg = X_val_reg[:, sel_idx]
        feature_names = [feature_names[i] for i in sel_idx]
        if model_type == "logreg" and "C" in best_trial.params:
            logreg_C = float(best_trial.params["C"])
        elif model_type == "bayes_logreg":
            pass
        elif model_type == "random_forest":
            n_estimators = int(best_trial.params["n_estimators"])
            max_depth = int(best_trial.params["max_depth"])
        elif model_type == "xgboost":
            n_estimators = int(best_trial.params["n_estimators"])
            learning_rate = float(best_trial.params["learning_rate"])
            max_depth = int(best_trial.params["max_depth"])
        elif model_type == "nn":
            hidden_size = int(best_trial.params["hidden_size"])

    # statistics for feature scaling
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    if moe:
        from sklearn.preprocessing import LabelEncoder

        all_feat_clf = [dict(f) for f in features]
        for f in all_feat_clf:
            f.pop("profit", None)
        X_all = vec.transform(all_feat_clf)
        X_all = _encode_features(enc_model, X_all)
        symbols_all = np.array([f.get("symbol", "") for f in features])
        le = LabelEncoder()
        sym_idx = le.fit_transform(symbols_all)
        gating_clf = LogisticRegression(max_iter=200, multi_class="multinomial")
        gating_clf.fit(X_all, sym_idx)
        gating_coef = gating_clf.coef_.ravel().astype(np.float32).tolist()
        gating_inter = gating_clf.intercept_.astype(np.float32).tolist()

        expert_models = []
        for idx, sym in enumerate(le.classes_):
            mask = symbols_all == sym
            if not mask.any():
                continue
            X_sym = X_all[mask]
            y_sym = labels[mask]
            sw = np.array(
                [abs(features[i].get("profit", features[i].get("lots", 1.0))) for i in np.where(mask)[0]],
                dtype=float,
            )
            exp_clf = LogisticRegression(max_iter=200)
            exp_clf.fit(X_sym, y_sym, sample_weight=sw)
            expert_models.append(
                {
                    "coefficients": exp_clf.coef_[0].astype(np.float32).tolist(),
                    "intercept": float(exp_clf.intercept_[0]),
                    "classes": [int(c) for c in exp_clf.classes_],
                    "symbol": sym,
                }
            )

        model = {
            "model_id": (existing_model.get("model_id") if existing_model else "target_clone"),
            "trained_at": datetime.utcnow().isoformat(),
            "feature_names": feature_names,
            "model_type": "moe_logreg",
            "gating_coefficients": gating_coef,
            "gating_intercepts": gating_inter,
            "gating_classes": le.classes_.tolist(),
            "session_models": expert_models,
            "last_event_id": int(last_event_id),
            "mean": feature_mean.astype(np.float32).tolist(),
            "std": feature_std.astype(np.float32).tolist(),
            "mode": mode,
        }
        model["feature_flags"] = feature_flags
        model["half_life_days"] = float(decay_half_life or 0.0)
        if weight_decay_info:
            model["weight_decay"] = weight_decay_info
        if ae_info:
            model["autoencoder"] = ae_info
        if data_commits:
            model["data_commit"] = ",".join(sorted(set(data_commits)))
        if data_checksums:
            model["data_checksum"] = ",".join(sorted(set(data_checksums)))
        if calendar_events:
            model["calendar_events"] = [
                [dt.isoformat(), float(imp)] for dt, imp in calendar_events
            ]
            model["event_window"] = float(event_window)
        return

    if regime_info is not None:
        regimes = np.array([int(f.get("regime", 0)) for f in features])
        all_feat = [dict(f) for f in features]
        for f in all_feat:
            f.pop("profit", None)
            f.pop("regime", None)
            for k in list(f.keys()):
                if k.startswith("regime_"):
                    f.pop(k, None)
        X_all = vec.transform(all_feat)
        X_all = _encode_features(enc_model, X_all)
        gating_clf = LogisticRegression(max_iter=200, multi_class="multinomial")
        gating_clf.fit(X_all, regimes)

        regime_models = []
        for r in sorted(set(regimes)):
            mask = regimes == r
            if not mask.any():
                continue
            X_r = X_all[mask]
            y_r = labels[mask]
            if len(np.unique(y_r)) < 2:
                continue
            sw = np.array(
                [abs(features[i].get("profit", features[i].get("lots", 1.0))) for i in np.where(mask)[0]],
                dtype=float,
            )
            clf_r = LogisticRegression(max_iter=200)
            clf_r.fit(X_r, y_r, sample_weight=sw)
            regime_models.append(
                {
                    "regime": int(r),
                    "coefficients": clf_r.coef_[0].astype(np.float32).tolist(),
                    "intercept": float(clf_r.intercept_[0]),
                    "classes": [int(c) for c in clf_r.classes_],
                    "feature_names": feature_names,
                }
            )
        # compute probabilities on train/validation sets
        def _regime_proba(X_mat: np.ndarray) -> np.ndarray:
            reg_preds = gating_clf.predict(X_mat)
            proba = np.zeros(len(reg_preds))
            for rm in regime_models:
                r = rm["regime"]
                mask = reg_preds == r
                if mask.any():
                    coef = np.array(rm["coefficients"], dtype=float)
                    inter = float(rm["intercept"])
                    z = X_mat[mask].dot(coef) + inter
                    proba[mask] = 1.0 / (1.0 + np.exp(-z))
            return proba

        train_feat = [dict(f) for f in feat_train]
        for f in train_feat:
            f.pop("profit", None)
            f.pop("regime", None)
            for k in list(f.keys()):
                if k.startswith("regime_"):
                    f.pop(k, None)
        X_train_all = vec.transform(train_feat)
        X_train_all = _encode_features(enc_model, X_train_all)
        proba_train = _regime_proba(X_train_all)

        val_feat = [dict(f) for f in feat_val]
        for f in val_feat:
            f.pop("profit", None)
            f.pop("regime", None)
            for k in list(f.keys()):
                if k.startswith("regime_"):
                    f.pop(k, None)
        X_val_all = (
            _encode_features(enc_model, vec.transform(val_feat))
            if val_feat
            else np.empty((0, X_train_all.shape[1]))
        )
        proba_val = _regime_proba(X_val_all) if len(val_feat) else np.array([])
        if len(y_val) > 0:
            threshold, _ = _best_threshold(y_val, proba_val)
            val_preds = (proba_val >= threshold).astype(int)
            val_acc = float(accuracy_score(y_val, val_preds))
        else:
            threshold = 0.5
            val_acc = float("nan")
        train_preds = (proba_train >= threshold).astype(int)
        train_acc = float(accuracy_score(y_train, train_preds))
        hours_val_arr = np.array(hours_val, dtype=int) if len(hours_val) else np.array([], dtype=int)
        hourly_thresholds: List[float] = []
        for h in range(24):
            idx = np.where(hours_val_arr == h)[0]
            if len(idx) > 0:
                t, _ = _best_threshold(y_val[idx], proba_val[idx])
            else:
                t = threshold
            hourly_thresholds.append(float(t))

        model = {
            "model_id": (existing_model.get("model_id") if existing_model else "target_clone"),
            "trained_at": datetime.utcnow().isoformat(),
            "feature_names": feature_names,
            "model_type": "regime_logreg",
            "meta_model": {
                "feature_names": feature_names,
                "coefficients": gating_clf.coef_.astype(np.float32).tolist(),
                "intercepts": gating_clf.intercept_.astype(np.float32).tolist(),
            },
            "regime_models": regime_models,
            "coefficients": gating_clf.coef_[0].astype(np.float32).tolist(),
            "intercept": float(gating_clf.intercept_[0]),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "threshold": threshold,
            "accuracy": val_acc,
            "num_samples": int(labels.shape[0])
            + (int(existing_model.get("num_samples", 0)) if existing_model else 0),
            "feature_importance": {},
            "weighted": sample_weight is not None,
            "last_event_id": int(last_event_id),
            "mean": feature_mean.astype(np.float32).tolist(),
            "std": feature_std.astype(np.float32).tolist(),
            "hourly_thresholds": hourly_thresholds,
        }
        model["regime_centers"] = regime_info.get("centers", [])
        model["regime_feature_names"] = regime_info.get("feature_names", [])
        model["regime_model_idx"] = [m.get("regime", i) for i, m in enumerate(regime_models)]
        model["mode"] = mode
        model["feature_flags"] = feature_flags
        model["half_life_days"] = float(decay_half_life or 0.0)
        if weight_decay_info:
            model["weight_decay"] = weight_decay_info
        if ae_info:
            model["autoencoder"] = ae_info
        if data_commits:
            model["data_commit"] = ",".join(sorted(set(data_commits)))
        if data_checksums:
            model["data_checksum"] = ",".join(sorted(set(data_checksums)))
        if calendar_events:
            model["calendar_events"] = [
                [dt.isoformat(), float(imp)] for dt, imp in calendar_events
            ]
            model["event_window"] = float(event_window)
        model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
        open_func = gzip.open if compress_model else open
        with open_func(model_path, "wt") as f:
            json.dump(model, f)
        print(f"Model written to {model_path}")
        if "coefficients" in model and "intercept" in model:
            w = np.array(model["coefficients"], dtype=float)
            b = float(model["intercept"])
            init = {
                "weights": [
                    (w / 2.0).tolist(),
                    (-w / 2.0).tolist(),
                ],
                "intercepts": [b / 2.0, -b / 2.0],
                "feature_names": model.get("feature_names", []),
            }
            with open(out_dir / "policy_init.json", "w") as f_init:
                json.dump(init, f_init, indent=2)
            print(f"Initial policy written to {out_dir / 'policy_init.json'}")
        return

    if stack_models:
        estimators = []
        for mt in stack_models:
            if mt == "logreg":
                estimators.append(("logreg", LogisticRegression(max_iter=200)))
            elif mt == "random_forest":
                estimators.append(("rf", RandomForestClassifier(n_estimators=100, random_state=42)))
            elif mt == "xgboost":
                try:
                    from xgboost import XGBClassifier  # type: ignore

                    estimators.append(
                        (
                            "xgb",
                            XGBClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                eval_metric="logloss",
                                use_label_encoder=False,
                            ),
                        )
                    )
                except Exception:
                    logging.warning(
                        "xgboost is not installed; using LogisticRegression in stack"
                    )
                    estimators.append(("xgb", LogisticRegression(max_iter=200)))
            elif mt == "lgbm":
                try:
                    from lightgbm import LGBMClassifier  # type: ignore

                    estimators.append(
                        (
                            "lgbm",
                            LGBMClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                            ),
                        )
                    )
                except Exception:
                    logging.warning(
                        "lightgbm is not installed; using LogisticRegression in stack"
                    )
                    estimators.append(("lgbm", LogisticRegression(max_iter=200)))
            elif mt == "catboost":
                try:
                    from catboost import CatBoostClassifier  # type: ignore

                    estimators.append(
                        (
                            "cat",
                            CatBoostClassifier(
                                iterations=n_estimators,
                                learning_rate=learning_rate,
                                depth=max_depth,
                                verbose=False,
                            ),
                        )
                    )
                except Exception:
                    logging.warning(
                        "catboost is not installed; using LogisticRegression in stack"
                    )
                    estimators.append(("cat", LogisticRegression(max_iter=200)))
            elif mt == "nn":
                estimators.append(
                    ("nn", MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42))
                )
        final_est = LogisticRegression(max_iter=200)
        clf = StackingClassifier(estimators=estimators, final_estimator=final_est, stack_method="predict_proba")
        clf.fit(X_train, y_train)
        train_proba_raw = clf.predict_proba(X_train)[:, 1]
        val_proba_raw = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
        model_type = "stack"
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        train_proba_raw = clf.predict_proba(X_train)[:, 1]
        val_proba_raw = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore

            clf = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
        except Exception:
            logging.warning(
                "xgboost is not installed; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
    elif model_type == "lgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore

            clf = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
            )
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
        except Exception:
            logging.warning(
                "lightgbm is not installed; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
    elif model_type == "catboost":
        try:
            from catboost import CatBoostClassifier  # type: ignore

            clf = CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=max_depth,
                verbose=False,
            )
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
        except Exception:
            logging.warning(
                "catboost is not installed; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
    elif model_type == "bayes_logreg":
        clf = BayesianRidge()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val) if len(y_val) > 0 else np.empty(0)
        train_proba_raw = 1.0 / (1.0 + np.exp(-train_pred))
        val_proba_raw = (
            1.0 / (1.0 + np.exp(-val_pred)) if len(y_val) > 0 else np.empty(0)
        )
        coef_variances = np.diag(clf.sigma_)
        noise_variance = 1.0 / clf.alpha_
    elif model_type == "nn":
        try:
            from tensorflow import keras  # type: ignore

            model_nn = keras.Sequential(
                [
                    keras.layers.Input(shape=(X_train.shape[1],)),
                    keras.layers.Dense(hidden_size, activation="relu"),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            model_nn.compile(optimizer="adam", loss="binary_crossentropy")
            callbacks = (
                [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
                if early_stop
                else None
            )
            model_nn.fit(
                X_train,
                y_train,
                epochs=50,
                verbose=0,
                callbacks=callbacks,
            )
            train_proba_raw = model_nn.predict(X_train).reshape(-1)
            val_proba_raw = (
                model_nn.predict(X_val).reshape(-1)
                if len(y_val) > 0
                else np.empty(0)
            )
            clf = model_nn
        except Exception:
            logging.warning(
                "TensorFlow not available; using MLPClassifier instead"
            )
            clf = MLPClassifier(
                hidden_layer_sizes=(hidden_size,), max_iter=500, random_state=42
            )
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
    elif model_type == "lstm":
        try:
            from tensorflow import keras  # type: ignore
        except Exception:
            logging.warning(
                "TensorFlow is required for LSTM model; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
        else:
            seq_len = sequence_length
            X_all = (
                vec.fit_transform(features)
                if existing_model is None
                else vec.transform(features)
            )
            X_all = _encode_features(enc_model, X_all)
            X_all = _encode_features(enc_model, X_all)
            sequences = []
            for i in range(len(X_all)):
                start = max(0, i - seq_len + 1)
                seq = X_all[start : i + 1]
                if seq.shape[0] < seq_len:
                    pad = np.zeros((seq_len - seq.shape[0], X_all.shape[1]))
                    seq = np.vstack([pad, seq])
                sequences.append(seq)
            X_all_seq = np.array(sequences)
            if len(labels) < 6 or len(np.unique(labels)) < 2:
                X_train_seq, y_train = X_all_seq, labels
                X_val_seq, y_val = (
                    np.empty((0, seq_len, X_all.shape[1])),
                    np.array([]),
                )
            else:
                tscv = TimeSeriesSplit(n_splits=5)
                train_idx, val_idx = list(tscv.split(X_all_seq))[-1]
                X_train_seq, X_val_seq = X_all_seq[train_idx], X_all_seq[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
            model_nn = keras.Sequential(
                [
                    keras.layers.Input(shape=(seq_len, X_all.shape[1])),
                    keras.layers.LSTM(8),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            model_nn.compile(optimizer="adam", loss="binary_crossentropy")
            callbacks = (
                [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
                if early_stop
                else None
            )
            model_nn.fit(
                X_train_seq,
                y_train,
                epochs=50,
                verbose=0,
                callbacks=callbacks,
            )
            train_proba_raw = model_nn.predict(X_train_seq).reshape(-1)
            val_proba_raw = (
                model_nn.predict(X_val_seq).reshape(-1)
                if len(y_val) > 0
                else np.empty(0)
            )
            clf = model_nn
    elif model_type == "transformer":
        try:
            from tensorflow import keras  # type: ignore
        except Exception:
            logging.warning(
                "TensorFlow is required for transformer model; using LogisticRegression instead"
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
        else:
            seq_len = sequence_length
            X_all = (
                vec.fit_transform(features)
                if existing_model is None
                else vec.transform(features)
            )
            sequences = []
            for i in range(len(X_all)):
                start = max(0, i - seq_len + 1)
                seq = X_all[start : i + 1]
                if seq.shape[0] < seq_len:
                    pad = np.zeros((seq_len - seq.shape[0], X_all.shape[1]))
                    seq = np.vstack([pad, seq])
                sequences.append(seq)
            X_all_seq = np.array(sequences)
            if len(labels) < 6 or len(np.unique(labels)) < 2:
                X_train_seq, y_train = X_all_seq, labels
                X_val_seq, y_val = (
                    np.empty((0, seq_len, X_all.shape[1])),
                    np.array([]),
                )
            else:
                tscv = TimeSeriesSplit(n_splits=5)
                train_idx, val_idx = list(tscv.split(X_all_seq))[-1]
                X_train_seq, X_val_seq = X_all_seq[train_idx], X_all_seq[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
            inp = keras.layers.Input(shape=(seq_len, X_all.shape[1]))
            att = keras.layers.MultiHeadAttention(num_heads=1, key_dim=X_all.shape[1])(inp, inp)
            pooled = keras.layers.GlobalAveragePooling1D()(att)
            out = keras.layers.Dense(1, activation="sigmoid")(pooled)
            model_nn = keras.Model(inp, out)
            model_nn.compile(optimizer="adam", loss="binary_crossentropy")
            callbacks = (
                [keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
                if early_stop
                else None
            )
            model_nn.fit(
                X_train_seq,
                y_train,
                epochs=50,
                verbose=0,
                callbacks=callbacks,
            )
            train_proba_raw = model_nn.predict(X_train_seq).reshape(-1)
            val_proba_raw = (
                model_nn.predict(X_val_seq).reshape(-1)
                if len(y_val) > 0
                else np.empty(0)
            )
            clf = model_nn
    elif model_type == "tft":
        try:
            import torch
            import pytorch_lightning as pl  # type: ignore
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet  # type: ignore
        except Exception:
            logging.warning(
                "pytorch-forecasting is required for tft model; using LogisticRegression instead",
            )
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            model_type = "logreg"
        else:
            df = pd.DataFrame(X_train, columns=feature_names)
            df["y"] = y_train
            df["time_idx"] = np.arange(len(df))
            df["group"] = 0
            ds = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="y",
                group_ids=["group"],
                max_encoder_length=sequence_length,
                max_prediction_length=1,
                time_varying_unknown_reals=feature_names + ["y"],
            )
            train_loader = ds.to_dataloader(train=True, batch_size=32)
            tft = TemporalFusionTransformer.from_dataset(ds, learning_rate=1e-3, log_interval=-1)
            trainer = pl.Trainer(
                max_epochs=5,
                enable_checkpointing=False,
                logger=False,
                enable_model_summary=False,
            )
            trainer.fit(tft, train_loader)
            train_proba_raw = tft.predict(train_loader).detach().cpu().numpy().reshape(-1)
            if len(y_val) > 0:
                dfv = pd.DataFrame(X_val, columns=feature_names)
                dfv["y"] = y_val
                dfv["time_idx"] = np.arange(len(dfv))
                dfv["group"] = 0
                val_ds = TimeSeriesDataSet.from_dataset(ds, dfv, predict=True)
                val_loader = val_ds.to_dataloader(train=False, batch_size=32)
                val_proba_raw = (
                    tft.predict(val_loader).detach().cpu().numpy().reshape(-1)
                )
            else:
                val_proba_raw = np.empty(0)
            clf = tft
    else:
        if grid_search:
            if c_values is None:
                c_values = [0.01, 0.1, 1.0, 10.0]
            param_grid = {"C": c_values}
            gs = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=3)
            gs.fit(X_train, y_train)
            clf = gs.best_estimator_
        else:
            if incremental:
                clf = SGDClassifier(loss="log_loss", alpha=1.0 / logreg_C)
                if existing_model is not None:
                    clf.classes_ = np.array(existing_model.get("classes", [0, 1]))
                    clf.coef_ = np.array([existing_model.get("coefficients", [])])
                    clf.intercept_ = np.array([existing_model.get("intercept", 0.0)])
                classes = getattr(clf, "classes_", np.array([0, 1]))
                batch_size = 1000
                for start in range(0, X_train.shape[0], batch_size):
                    end = start + batch_size
                    sw = sample_weight[start:end] if sample_weight is not None else None
                    clf.partial_fit(
                        X_train[start:end], y_train[start:end], classes=classes, sample_weight=sw
                    )
                train_proba_raw = clf.predict_proba(X_train)[:, 1]
                val_proba_raw = (
                    clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
                )
            else:
                clf = LogisticRegression(max_iter=200, C=logreg_C, warm_start=existing_model is not None)
                if existing_model is not None:
                    clf.classes_ = np.array(existing_model.get("classes", [0, 1]))
                    clf.coef_ = np.array([existing_model.get("coefficients", [])])
                    clf.intercept_ = np.array([existing_model.get("intercept", 0.0)])
                clf.fit(X_train, y_train, sample_weight=sample_weight)
                train_proba_raw = clf.predict_proba(X_train)[:, 1]
                val_proba_raw = (
                    clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
                )

    train_proba = train_proba_raw
    val_proba = val_proba_raw
    cal_coef = 1.0
    cal_inter = 0.0
    if calibration is not None and len(y_val) > 0:
        calibrator = CalibratedClassifierCV(clf, cv="prefit", method=calibration)
        calibrator.fit(X_val, y_val)
        train_proba = calibrator.predict_proba(X_train)[:, 1]
        val_proba = calibrator.predict_proba(X_val)[:, 1]
        if calibration == "sigmoid":
            cal_lr = calibrator.calibrated_classifiers_[0].calibrator
            cal_coef = float(cal_lr.coef_[0][0])
            cal_inter = float(cal_lr.intercept_[0])
    

    if regress_sl_tp or regress_lots:
        from sklearn.linear_model import LinearRegression

    if regress_sl_tp:
        reg_sl = LinearRegression()
        reg_sl.fit(X_train_reg, sl_train)
        reg_tp = LinearRegression()
        reg_tp.fit(X_train_reg, tp_train)
        sl_coef = reg_sl.coef_
        sl_inter = reg_sl.intercept_
        tp_coef = reg_tp.coef_
        tp_inter = reg_tp.intercept_

    if regress_lots:
        reg_lot = LinearRegression()
        reg_lot.fit(X_train_lot, lot_train)
        lot_coef = reg_lot.coef_
        lot_inter = reg_lot.intercept_

    if len(y_val) > 0:
        if bayes_threshold is not None:
            threshold = bayes_threshold
        else:
            threshold, _ = _best_threshold(y_val, val_proba)
        val_preds = (val_proba >= threshold).astype(int)
        val_acc = float(accuracy_score(y_val, val_preds))
        val_f1 = float(f1_score(y_val, val_preds))
        val_roc = float(roc_auc_score(y_val, val_proba))
    else:
        threshold = bayes_threshold if bayes_threshold is not None else 0.5
        val_acc = float("nan")
        val_f1 = float("nan")
        val_roc = float("nan")
    train_preds = (train_proba >= threshold).astype(int)
    train_acc = float(accuracy_score(y_train, train_preds))
    ctx_metrics = trace.get_current_span().get_span_context()
    logger.info(
        {"train_acc": train_acc, "val_acc": val_acc, "val_f1": val_f1, "val_roc_auc": val_roc},
        extra={"trace_id": ctx_metrics.trace_id, "span_id": ctx_metrics.span_id},
    )
    for i in range(min(5, len(train_preds))):
        with tracer.start_as_current_span("decision") as dspan:
            dctx = dspan.get_span_context()
            logger.info(
                {
                    "decision_id": int(i),
                    "prediction": int(train_preds[i]),
                    "prob": float(train_proba[i]),
                },
                extra={"trace_id": dctx.trace_id, "span_id": dctx.span_id},
            )

    hours_val_arr = np.array(hours_val, dtype=int) if len(hours_val) else np.array([], dtype=int)
    hourly_thresholds: List[float] = []
    for h in range(24):
        idx = np.where(hours_val_arr == h)[0]
        if len(idx) > 0:
            t, _ = _best_threshold(y_val[idx], val_proba[idx])
        else:
            t = threshold
        hourly_thresholds.append(float(t))

    threshold_map: dict[str, List[float]] = {}
    if len(y_val) > 0 and symbols_val_arr.size:
        try:  # pragma: no cover - optional dependency
            import optuna  # type: ignore
        except Exception:
            threshold_map = {}
        else:
            for sym in np.unique(symbols_val_arr):
                sym_mask = symbols_val_arr == sym
                sym_thr: List[float] = []
                for h in range(24):
                    mask = sym_mask & (hours_val_arr == h)
                    if mask.any():
                        def _thr_obj(trial: "optuna.trial.Trial") -> float:
                            thr = trial.suggest_float("thr", 0.0, 1.0)
                            preds = (val_proba[mask] >= thr).astype(int)
                            if profits_val.size == len(preds):
                                return float(np.sum(profits_val[mask][preds == 1]))
                            elif len(np.unique(y_val[mask])) > 1:
                                return roc_auc_score(y_val[mask], preds)
                            else:
                                return accuracy_score(y_val[mask], preds)
                        study = optuna.create_study(direction="maximize")
                        study.optimize(_thr_obj, n_trials=20)
                        sym_thr.append(float(study.best_params.get("thr", threshold)))
                    else:
                        sym_thr.append(float(threshold))
                threshold_map[str(sym)] = sym_thr

    # Compute SHAP feature importance on the training set
    keep_idx = list(range(len(feature_names)))
    try:
        import shap  # type: ignore

        if model_type in ("logreg", "bayes_logreg"):
            explainer = shap.LinearExplainer(clf, X_train)
            shap_values = explainer.shap_values(X_train)
        else:
            explainer = shap.Explainer(clf, X_train)
            shap_values = explainer(X_train).values
        importances = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_names, importances.tolist()))
    except Exception:  # pragma: no cover - shap optional
        importances = np.array([])
        feature_importance = {}

    if prune_threshold > 0.0 and feature_importance:
        keep_idx = [i for i, name in enumerate(feature_names) if feature_importance.get(name, 0.0) >= prune_threshold]
        removed_ratio = 1 - len(keep_idx) / len(feature_names)
        if removed_ratio > prune_warn:
            logging.warning("Pruning removed %.1f%% of features", removed_ratio * 100)
        if len(keep_idx) < len(feature_names):
            X_train = X_train[:, keep_idx]
            if X_val.shape[0] > 0:
                X_val = X_val[:, keep_idx]
            X_train_reg = X_train_reg[:, keep_idx]
            if X_val_reg.shape[0] > 0:
                X_val_reg = X_val_reg[:, keep_idx]
            feature_mean = feature_mean[keep_idx]
            feature_std = feature_std[keep_idx]
            feature_names = [feature_names[i] for i in keep_idx]
            selected_indices = [selected_indices[i] for i in keep_idx]

            clf.fit(X_train, y_train, sample_weight=sample_weight)
            train_proba_raw = clf.predict_proba(X_train)[:, 1]
            val_proba_raw = (
                clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
            )
            train_proba = train_proba_raw
            val_proba = val_proba_raw
            if calibration is not None and len(y_val) > 0:
                calibrator = CalibratedClassifierCV(clf, cv="prefit", method=calibration)
                calibrator.fit(X_val, y_val)
                train_proba = calibrator.predict_proba(X_train)[:, 1]
                val_proba = calibrator.predict_proba(X_val)[:, 1]
                if calibration == "sigmoid":
                    cal_lr = calibrator.calibrated_classifiers_[0].calibrator
                    cal_coef = float(cal_lr.coef_[0][0])
                    cal_inter = float(cal_lr.intercept_[0])
            if regress_sl_tp:
                reg_sl.fit(X_train_reg, sl_train)
                reg_tp.fit(X_train_reg, tp_train)
                sl_coef = reg_sl.coef_
                sl_inter = reg_sl.intercept_
                tp_coef = reg_tp.coef_
                tp_inter = reg_tp.intercept_
            if regress_lots:
                reg_lot.fit(X_train_lot, lot_train)
                lot_coef = reg_lot.coef_
                lot_inter = reg_lot.intercept_
            if len(y_val) > 0:
                if bayes_threshold is not None:
                    threshold = bayes_threshold
                else:
                    threshold, _ = _best_threshold(y_val, val_proba)
                val_preds = (val_proba >= threshold).astype(int)
                val_acc = float(accuracy_score(y_val, val_preds))
            else:
                threshold = bayes_threshold if bayes_threshold is not None else 0.5
                val_acc = float("nan")
            train_preds = (train_proba >= threshold).astype(int)
            train_acc = float(accuracy_score(y_train, train_preds))

            try:
                if model_type in ("logreg", "bayes_logreg"):
                    explainer = shap.LinearExplainer(clf, X_train)
                    shap_values = explainer.shap_values(X_train)
                else:
                    explainer = shap.Explainer(clf, X_train)
                    shap_values = explainer(X_train).values
                importances = np.abs(shap_values).mean(axis=0)
                feature_importance = dict(zip(feature_names, importances.tolist()))
            except Exception:  # pragma: no cover
                feature_importance = {}


    # Train a lightweight student model on the teacher's soft probabilities.
    student_coef: list[float] | None = None
    student_inter: float | None = None
    student_val_acc: float | None = None
    if model_type not in ("logreg", "bayes_logreg") and hasattr(clf, "predict_proba"):
        try:
            teacher_probs = train_proba
            X_rep = np.vstack([X_train, X_train])
            y_rep = np.concatenate([np.ones(len(teacher_probs)), np.zeros(len(teacher_probs))])
            sw = np.concatenate([teacher_probs, 1 - teacher_probs])
            student = LogisticRegression(max_iter=200)
            student.fit(X_rep, y_rep, sample_weight=sw)
            student_coef = student.coef_[0].astype(np.float32).tolist()
            student_inter = float(student.intercept_[0])
            if len(y_val) > 0:
                student_val_acc = float(student.score(X_val, y_val))
        except Exception:  # pragma: no cover - distillation best effort
            pass

    out_dir.mkdir(parents=True, exist_ok=True)

    model = {
        "model_id": (existing_model.get("model_id") if existing_model else "target_clone"),
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": feature_names,
        "model_type": model_type,
        "training_mode": "lite" if lite_mode else "heavy",
        "mode": mode,
        "weighted": sample_weight is not None,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_roc_auc": val_roc,
        "threshold": threshold,
        # main accuracy metric is validation performance when available
        "accuracy": val_acc,
        "num_samples": int(labels.shape[0]) + (int(existing_model.get("num_samples", 0)) if existing_model else 0),
        "last_event_id": int(last_event_id),
        "feature_importance": feature_importance,
        "mean": feature_mean.astype(np.float32).tolist(),
        "std": feature_std.astype(np.float32).tolist(),
    }
    model["feature_flags"] = feature_flags
    if risk_parity:
        model["risk_parity_symbols"] = list(risk_parity.keys())
        model["risk_parity_weights"] = [float(risk_parity[s]) for s in risk_parity]
        model["risk_covariance_symbols"] = cov_symbols
        model["risk_covariance_matrix"] = cov_matrix.tolist()
    if student_coef is not None and student_inter is not None:
        model["student_coefficients"] = student_coef
        model["student_intercept"] = student_inter
        model["teacher_accuracy"] = val_acc
        if student_val_acc is not None:
            model["student_accuracy"] = student_val_acc
    model["half_life_days"] = float(decay_half_life or 0.0)
    if weight_decay_info:
        model["weight_decay"] = weight_decay_info
    if ae_info:
        model["autoencoder"] = ae_info
    if regime_info is not None and reg_centers is not None:
        model["regime_centers"] = reg_centers.astype(float).tolist()
        model["regime_feature_names"] = regime_info.get("feature_names", [])
    if data_commits:
        model["data_commit"] = ",".join(sorted(set(data_commits)))
    if data_checksums:
        model["data_checksum"] = ",".join(sorted(set(data_checksums)))
    if calendar_events:
        model["calendar_events"] = [
            [dt.isoformat(), float(imp)] for dt, imp in calendar_events
        ]
        model["event_window"] = float(event_window)
    if encoder is not None:
        model["encoder_weights"] = encoder.get("weights")
        model["encoder_window"] = encoder.get("window")
        if "centers" in encoder:
            model["encoder_centers"] = encoder.get("centers")
    if enc_model is not None:
        w = enc_model.layer.weight.detach().cpu().numpy().tolist()
        model["encoder_weights"] = w
        model["encoder_window"] = enc_model.layer.weight.shape[1]
        model["encoder_dim"] = enc_model.layer.weight.shape[0]
        model["encoder_onnx"] = "encoder.onnx"
    if best_trial is not None and study is not None:
        model["bayes_best_params"] = best_trial.params
        model["bayes_best_score"] = float(best_trial.value)
        model["bayes_study"] = {"n_trials": len(study.trials)}
        model["bayes_history"] = [
            {"params": t.params, "value": float(t.value)} for t in study.trials
        ]
    if symbol_embeddings:
        model["symbol_embeddings"] = symbol_embeddings
    if graph_params:
        model["graph"] = {
            "symbols": graph_params.get("symbols", []),
            "edge_index": graph_params.get("edge_index", []),
            "edge_weight": graph_params.get("edge_weight", []),
            "embedding_dim": graph_params.get("embedding_dim", 0),
            "metrics": graph_params.get("metrics", {}),
        }
    model["hourly_thresholds"] = hourly_thresholds
    if threshold_map:
        thr_syms = sorted(threshold_map.keys())
        model["threshold_symbols"] = thr_syms
        flat_thr: List[float] = []
        for s in thr_syms:
            vals = threshold_map[s]
            if len(vals) < 24:
                vals = vals + [threshold] * (24 - len(vals))
            flat_thr.extend(vals[:24])
        model["threshold_table"] = flat_thr
    if calibration is not None:
        model["calibration_method"] = calibration
        if calibration == "sigmoid":
            model["calibration_coef"] = cal_coef
            model["calibration_intercept"] = cal_inter
    if stack_models:
        model["stack_models"] = stack_models

    if model_type == "logreg":
        model["coefficients"] = clf.coef_[0].astype(np.float32).tolist()
        model["intercept"] = float(clf.intercept_[0])
        model["classes"] = [int(c) for c in clf.classes_]
    elif model_type == "bayes_logreg":
        model["coefficients"] = clf.coef_.astype(np.float32).tolist()
        model["intercept"] = float(clf.intercept_)
        model["coef_variances"] = coef_variances.astype(np.float32).tolist()
        model["noise_variance"] = float(noise_variance)
        model["classes"] = [0, 1]
    elif model_type == "xgboost":
        # approximate tree ensemble with linear model for MQL4 export
        logit_p = np.log(train_proba_raw / (1.0 - train_proba_raw + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].astype(np.float32).tolist()
        model["intercept"] = float(coef[0])

        # lookup probabilities per trading hour for simple export
        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour_sin" in f:
                f["hour_sin"] = math.sin(2 * math.pi * h / 24)
            if "hour_cos" in f:
                f["hour_cos"] = math.cos(2 * math.pi * h / 24)
            X_h = vec.transform([f])[:, selected_indices]
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "lgbm":
        # approximate boosting model with linear regression for export
        logit_p = np.log(train_proba_raw / (1.0 - train_proba_raw + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].astype(np.float32).tolist()
        model["intercept"] = float(coef[0])

        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour_sin" in f:
                f["hour_sin"] = math.sin(2 * math.pi * h / 24)
            if "hour_cos" in f:
                f["hour_cos"] = math.cos(2 * math.pi * h / 24)
            X_h = vec.transform([f])[:, selected_indices]
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "catboost":
        # approximate boosting model with linear regression for export
        logit_p = np.log(train_proba_raw / (1.0 - train_proba_raw + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].astype(np.float32).tolist()
        model["intercept"] = float(coef[0])

        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour_sin" in f:
                f["hour_sin"] = math.sin(2 * math.pi * h / 24)
            if "hour_cos" in f:
                f["hour_cos"] = math.cos(2 * math.pi * h / 24)
            X_h = vec.transform([f])[:, selected_indices]
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "stack":
        logit_p = np.log(train_proba_raw / (1.0 - train_proba_raw + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].astype(np.float32).tolist()
        model["intercept"] = float(coef[0])
    elif model_type == "nn":
        if hasattr(clf, "get_weights"):
            weights = [w.tolist() for w in clf.get_weights()]
        else:
            weights = [
                clf.coefs_[0].tolist(),
                clf.intercepts_[0].tolist(),
                clf.coefs_[1].tolist(),
                clf.intercepts_[1].tolist(),
            ]
        model["nn_weights"] = weights
        model["hidden_size"] = len(weights[1]) if weights else 0
    elif model_type == "lstm":
        weights = [w.tolist() for w in clf.get_weights()]
        model["lstm_weights"] = weights
        model["sequence_length"] = sequence_length
        model["hidden_size"] = len(weights[1]) // 4 if weights else 0
    elif model_type == "transformer":
        weights = [w.tolist() for w in clf.get_weights()]
        model["transformer_weights"] = weights
        model["sequence_length"] = sequence_length
    elif model_type == "tft":
        try:
            import torch
            state = clf.state_dict()
            enc_w = state.get("encoder.weight")
            dec_w = state.get("decoder.weight")
            if enc_w is not None:
                model["encoder_weights"] = enc_w.detach().cpu().numpy().tolist()
            if dec_w is not None:
                model["decoder_weights"] = dec_w.detach().cpu().numpy().tolist()
            model["onnx_file"] = "model.onnx"
            try:
                dummy = torch.zeros(1, sequence_length, len(feature_names))
                torch.onnx.export(
                    clf,
                    dummy,
                    out_dir / "model.onnx",
                    opset_version=13,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}},
                )
            except Exception as exc:  # pragma: no cover - optional export
                logging.warning("ONNX export failed: %s", exc)
        except Exception:  # pragma: no cover - torch errors
            pass

    if regress_sl_tp:
        model["sl_coefficients"] = sl_coef.astype(np.float32).tolist()
        model["sl_intercept"] = float(sl_inter)
        model["tp_coefficients"] = tp_coef.astype(np.float32).tolist()
        model["tp_intercept"] = float(tp_inter)

    if regress_lots:
        model["lot_coefficients"] = lot_coef.astype(np.float32).tolist()
        model["lot_intercept"] = float(lot_inter)

    # Optional RL refinement
    if rl_finetune and HAS_SB3 and "coefficients" in model and "intercept" in model:
        try:
            temp_model = out_dir / "model_supervised.json"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(temp_model, "w") as f_tmp:
                json.dump(model, f_tmp)
            rl_out = out_dir / "rl_tmp"
            cmd = [
                sys.executable,
                str(Path(__file__).with_name("train_rl_agent.py")),
                "--data-dir",
                str(data_dir),
                "--out-dir",
                str(rl_out),
                "--algo",
                "qlearn",
                "--training-steps",
                "20",
                "--start-model",
                str(temp_model),
            ]
            if compress_model:
                cmd.append("--compress-model")
            subprocess.run(cmd, check=True)
            rl_model_path = rl_out / (
                "model.json.gz" if compress_model else "model.json"
            )
            open_rl = gzip.open if compress_model else open
            with open_rl(rl_model_path, "rt") as f_rl:
                rl_model = json.load(f_rl)
            if "coefficients" in rl_model and "intercept" in rl_model:
                model["coefficients"] = rl_model["coefficients"]
                model["intercept"] = rl_model["intercept"]
            if "q_weights" in rl_model:
                model["q_weights"] = rl_model["q_weights"]
            if "q_intercepts" in rl_model:
                model["q_intercepts"] = rl_model["q_intercepts"]
            model["rl_steps"] = rl_model.get("training_steps")
            model["rl_reward"] = rl_model.get("avg_reward")
        except Exception as exc:  # pragma: no cover - optional RL errors
            logging.warning("RL refinement failed: %s", exc)
        finally:
            try:
                temp_model.unlink()
            except Exception:
                pass

    # Train simple entry and exit models from raw logs
    try:
        df_logs, _, _ = _load_logs(data_dir, flight_uri=flight_uri)
        df_logs.columns = [c.lower() for c in df_logs.columns]
        # Entry model: predict order_type on OPEN actions
        if not df_logs.empty:
            df_open = df_logs[df_logs["action"] == "OPEN"].dropna(subset=["order_type"])
            if not df_open.empty:
                entry_feats = df_open[[
                    "book_imbalance",
                    "spread",
                    "book_bid_vol",
                    "book_ask_vol",
                ]].fillna(0.0)
                entry_target = pd.to_numeric(df_open["order_type"], errors="coerce").fillna(0).astype(int)
                try:
                    entry_clf = LogisticRegression(max_iter=200)
                    entry_clf.fit(entry_feats, entry_target)
                    model["entry_coefficients"] = entry_clf.coef_[0].astype(np.float32).tolist()
                    model["entry_intercept"] = float(entry_clf.intercept_[0])
                    model["entry_threshold"] = 0.5
                except Exception as exc:
                    logging.warning("entry model training failed: %s", exc)
            # Exit model: classify positive profit on CLOSE actions
            df_close = df_logs[df_logs["action"] == "CLOSE"].dropna(subset=["sl_hit_dist", "tp_hit_dist", "profit"])
            if not df_close.empty:
                exit_feats = df_close[["sl_hit_dist", "tp_hit_dist", "profit"]].fillna(0.0)
                exit_target = (pd.to_numeric(df_close["profit"], errors="coerce") > 0).astype(int)
                try:
                    exit_clf = LogisticRegression(max_iter=200)
                    exit_clf.fit(exit_feats, exit_target)
                    model["exit_coefficients"] = exit_clf.coef_[0].astype(np.float32).tolist()
                    model["exit_intercept"] = float(exit_clf.intercept_[0])
                    model["exit_threshold"] = 0.5
                except Exception as exc:
                    logging.warning("exit model training failed: %s", exc)
    except Exception as exc:
        logging.warning("failed to train entry/exit models: %s", exc)

    model_path = out_dir / ("model.json.gz" if compress_model else "model.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    open_func = gzip.open if compress_model else open
    with open_func(model_path, "wt") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {model_path}")
    if model_type != "tft":
        _export_onnx(clf, model.get("feature_names", []), out_dir)

    if "coefficients" in model and "intercept" in model:
        w = np.array(model["coefficients"], dtype=float)
        b = float(model["intercept"])
        init = {
            "weights": [
                (w / 2.0).tolist(),
                (-w / 2.0).tolist(),
            ],
            "intercepts": [b / 2.0, -b / 2.0],
            "feature_names": model.get("feature_names", []),
        }
        with open(out_dir / "policy_init.json", "w") as f:
            json.dump(init, f, indent=2)
        print(f"Initial policy written to {out_dir / 'policy_init.json'}")

    print(f"Validation accuracy: {val_acc:.3f}")


def main():
    resources = detect_resources()
    print(json.dumps(resources, indent=2))
    span = tracer.start_span("train_target_clone")
    ctx = span.get_span_context()
    logger.info("start training", extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id})
    p = argparse.ArgumentParser(
        description=(
            "Train model from exported features. Automatically detects hardware resources, "
            "including CPU frequency (cpu_mhz). Transformer and TFT models are only "
            "enabled when cpu_mhz is at least 2500 MHz."
        )
    )
    default_data_dir = DEFAULT_DATA_HOME / "data"
    default_out_dir = DEFAULT_DATA_HOME
    p.add_argument('--data-dir', default=str(default_data_dir))
    p.add_argument('--out-dir', default=str(default_out_dir))
    p.add_argument('--flight-uri', default=os.environ.get('FLIGHT_URI'),
                   help='Arrow Flight server URI (default from FLIGHT_URI)')
    p.add_argument('--sma-window', type=int, default=5)
    p.add_argument('--rsi-period', type=int, default=14)
    p.add_argument(
        '--higher-timeframes',
        help='comma separated higher timeframes e.g. H1,H4',
    )
    p.add_argument('--calendar-file', help='CSV file with columns time,impact for events')
    p.add_argument('--event-window', type=float, default=60.0, help='minutes around events to flag')
    p.add_argument('--volatility-file', help='JSON file with precomputed volatility')
    p.add_argument('--grid-search', action='store_true', help='enable grid search with cross-validation')
    p.add_argument('--c-values', type=float, nargs='*')
    p.add_argument('--sequence-length', type=int, default=5, help='sequence length for LSTM/transformer models')
    p.add_argument('--n-estimators', type=int, default=100, help='number of boosting rounds')
    p.add_argument('--learning-rate', type=float, default=0.1, help='learning rate for boosted trees')
    p.add_argument('--max-depth', type=int, default=3, help='tree depth for boosting models')
    p.add_argument('--incremental', action='store_true', help='update existing model.json')
    p.add_argument('--start-event-id', type=int, default=0, help='only load rows with event_id greater than this value from SQLite logs')
    p.add_argument('--resume', action='store_true', help='resume from last processed event_id in existing model.json')
    p.add_argument('--no-cache', action='store_true', help='recompute features even if cached')
    p.add_argument('--corr-symbols', help='comma separated correlated symbol pairs e.g. EURUSD:USDCHF')
    p.add_argument('--corr-window', type=int, default=5, help='window for correlation calculations')
    p.add_argument('--symbol-graph', help='JSON file describing symbol correlation graph')
    p.add_argument(
        '--bayes-steps',
        type=int,
        default=0,
        help='number of Bayesian optimization steps to tune model type, threshold and features',
    )
    p.add_argument('--encoder-file', help='JSON file with pretrained encoder weights')
    p.add_argument('--regress-sl-tp', action='store_true', help='learn SL/TP distance regressors')
    p.add_argument('--regress-lots', action='store_true', help='learn lot size regressor')
    p.add_argument('--early-stop', action='store_true', help='enable early stopping for neural nets')
    p.add_argument('--calibration', choices=['sigmoid', 'isotonic'], help='probability calibration method')
    p.add_argument('--stack', help='comma separated list of model types to stack')
    p.add_argument('--prune-threshold', type=float, default=0.0, help='drop features with SHAP importance below this value')
    p.add_argument('--prune-warn', type=float, default=0.5, help='warn if more than this fraction of features are pruned')
    p.add_argument('--compress-model', action='store_true', help='write model.json.gz')
    p.add_argument('--regime-model', help='JSON file with precomputed regime centers')
    p.add_argument('--regime-json', help='regime detection JSON produced by detect_regime.py')
    p.add_argument('--moe', action='store_true', help='train mixture-of-experts model per symbol')
    p.add_argument('--federated-server', help='URL of federated averaging server')
    p.add_argument('--use-encoder', action='store_true', help='apply pretrained contrastive encoder')
    p.add_argument(
        '--uncertain-file',
        help='CSV with labeled uncertain decisions to emphasize during training',
    )
    p.add_argument(
        '--uncertain-weight',
        type=float,
        default=2.0,
        help='sample weight multiplier for labeled uncertainties',
    )
    p.add_argument(
        '--half-life-days',
        type=float,
        help='half-life in days for sample weight decay',
    )
    p.add_argument('--replay-file', help='CSV of decision replay divergences')
    p.add_argument('--replay-weight', type=float, default=3.0, help='sample weight multiplier for replay corrections')
    args = p.parse_args()
    with trace.use_span(span, end_on_exit=True):
        global START_EVENT_ID
        if args.resume:
            START_EVENT_ID = _read_last_event_id(Path(args.out_dir))
        else:
            START_EVENT_ID = args.start_event_id
        if args.volatility_file:
            import json
            with open(args.volatility_file) as f:
                vol_data = json.load(f)
        else:
            vol_data = None
        if args.calendar_file:
            events = _load_calendar(Path(args.calendar_file))
        else:
            events = None
        if args.corr_symbols:
            corr_map = {}
            for p in args.corr_symbols.split(','):
                if ':' in p:
                    base, peer = p.split(':', 1)
                    corr_map.setdefault(base, []).append(peer)
        else:
            corr_map = None
        if args.symbol_graph:
            symbol_graph = Path(args.symbol_graph)
        else:
            symbol_graph = None
        if args.higher_timeframes:
            higher_tfs = [tf.strip() for tf in args.higher_timeframes.split(',') if tf.strip()]
        else:
            higher_tfs = None
        train(
            Path(args.data_dir),
            Path(args.out_dir),
            sma_window=args.sma_window,
            rsi_period=args.rsi_period,
            higher_timeframes=higher_tfs,
            volatility_series=vol_data,
            grid_search=args.grid_search or None,
            c_values=args.c_values,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            incremental=args.incremental,
            sequence_length=args.sequence_length,
            corr_map=corr_map,
            corr_window=args.corr_window,
            bayes_steps=args.bayes_steps or None,
            regress_sl_tp=args.regress_sl_tp,
            regress_lots=args.regress_lots,
            early_stop=args.early_stop,
            encoder_file=Path(args.encoder_file) if args.encoder_file else None,
            cache_features=not args.no_cache,
            calendar_events=events,
            event_window=args.event_window,
            calibration=args.calibration,
            stack_models=[s.strip() for s in args.stack.split(',')] if args.stack else None,
            prune_threshold=args.prune_threshold,
            prune_warn=args.prune_warn,
            compress_model=args.compress_model,
            regime_model_file=(
                Path(args.regime_json)
                if args.regime_json
                else Path(args.regime_model)
                if args.regime_model
                else None
            ),
            moe=args.moe,
            flight_uri=args.flight_uri,
            use_encoder=args.use_encoder,
            uncertain_file=Path(args.uncertain_file) if args.uncertain_file else None,
            uncertain_weight=args.uncertain_weight,
            decay_half_life=args.half_life_days,
            news_sentiment_file=Path(args.news_sentiment_file) if args.news_sentiment_file else None,
            rl_finetune=resources["enable_rl"],
            symbol_graph=symbol_graph,
            replay_file=Path(args.replay_file) if args.replay_file else None,
            replay_weight=args.replay_weight,
        )
        if args.federated_server:
            model_path = Path(args.out_dir) / (
                "model.json.gz" if args.compress_model else "model.json"
            )
            sync_with_server(model_path, args.federated_server)
        logger.info("training complete", extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id})


if __name__ == '__main__':
    main()
