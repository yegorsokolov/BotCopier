#!/usr/bin/env python3
"""Incrementally update a model from streaming trade events.

The trainer is designed to run continuously.  It tails
``logs/trades_raw.csv`` or subscribes to an Arrow Flight stream.  After each batch it updates
an :class:`~sklearn.linear_model.SGDClassifier` using :meth:`partial_fit` and
persists the coefficients to ``model.json`` for downstream consumers.

Hardware capabilities are sampled via :func:`detect_resources` to determine an
appropriate throttling level.  On lightweight VPS hosts the trainer yields
the CPU more aggressively, preventing it from overwhelming the terminal.
During processing the current load is checked with
``psutil.cpu_percent`` and the worker sleeps when the threshold is exceeded.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Iterable, List

from pydantic import ValidationError

from botcopier.config.settings import (
    DataConfig,
    TrainingConfig,
    load_settings,
    save_params,
)
from botcopier.metrics import (
    ERROR_COUNTER,
    TRADE_COUNTER,
    observe_latency,
    start_metrics_server,
)
from botcopier.models.registry import load_params
from botcopier.models.schema import ModelParams
from botcopier.utils.random import set_seed
from automl.controller import AutoMLController

try:  # prefer systemd journal if available
    from systemd.journal import JournalHandler

    logging.basicConfig(handlers=[JournalHandler()], level=logging.INFO)
except ImportError:  # pragma: no cover - fallback to file logging
    logging.basicConfig(filename="online_trainer.log", level=logging.INFO)

from collections import deque

import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier

try:  # detect resources to adapt behaviour on weaker hardware
    if __package__:
        from .train_target_clone import detect_resources  # type: ignore
    else:  # pragma: no cover - script executed directly
        from train_target_clone import detect_resources  # type: ignore
except ImportError:  # pragma: no cover - detection optional
    detect_resources = None  # type: ignore

try:  # drift metrics utilities
    if __package__:
        from .drift_monitor import _compute_metrics, _update_model
    else:  # pragma: no cover - script executed directly
        from drift_monitor import _compute_metrics, _update_model  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _compute_metrics = _update_model = None  # type: ignore

try:  # optional graph embedding support
    from graph_dataset import GraphDataset, compute_gnn_embeddings

    _HAS_TG = True
except ImportError:  # pragma: no cover
    GraphDataset = None  # type: ignore
    compute_gnn_embeddings = None  # type: ignore
    _HAS_TG = False

try:  # optional feast support for feature parity
    from feast import FeatureStore  # type: ignore

    from botcopier.feature_store.feast_repo.feature_views import FEATURE_COLUMNS

    _HAS_FEAST = True
except Exception:  # pragma: no cover - optional
    FeatureStore = None  # type: ignore
    FEATURE_COLUMNS = []  # type: ignore
    _HAS_FEAST = False

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

try:  # optional sequential drift detection
    if __package__:
        from .sequential_drift import PageHinkley
    else:  # pragma: no cover - executed as script
        from sequential_drift import PageHinkley  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PageHinkley = None  # type: ignore

try:  # on-disk ring buffer for raw ticks
    if __package__:
        from .shm_ring import TRADE_MSG, ShmRing
    else:  # pragma: no cover - executed as script
        from shm_ring import TRADE_MSG, ShmRing  # type: ignore
except Exception:  # pragma: no cover - ring buffer optional
    ShmRing = None  # type: ignore
    TRADE_MSG = 0  # type: ignore

resource = Resource.create(
    {"service.name": os.getenv("OTEL_SERVICE_NAME", "online_trainer")}
)
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

logger_provider = LoggerProvider(resource=resource)
if endpoint:
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint))
    )
set_logger_provider(logger_provider)
otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)


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
otel_handler.setFormatter(JsonFormatter())
logger.addHandler(otel_handler)
logger.setLevel(logging.INFO)

try:  # optional systemd notification support
    from systemd import daemon
except ImportError:  # pragma: no cover - systemd not installed
    daemon = None


def _sd_notify_ready() -> None:
    if daemon is not None:
        daemon.sd_notify("READY=1")


def _start_watchdog_thread() -> None:
    if daemon is None:
        return
    try:
        usec = int(os.getenv("WATCHDOG_USEC", "0"))
    except ValueError:
        usec = 0
    interval = usec / 2_000_000 if usec else 30

    def _loop() -> None:
        while True:
            time.sleep(interval)
            try:
                daemon.sd_notify("WATCHDOG=1")
            except OSError:
                pass

    threading.Thread(target=_loop, daemon=True).start()


class OnlineTrainer:
    """Manage incremental updates and model persistence."""

    def __init__(
        self,
        model_path: Path | str = Path("model.json"),
        batch_size: int = 32,
        run_generator: bool = True,
        lr: float = 0.01,
        lr_decay: float = 1.0,
        seed: int = 0,
        online_model: str = "sgd",
        *,
        tick_buffer_path: Path | str | None = None,
        tick_buffer_size: int = 100_000,
        controller: AutoMLController | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.run_generator = run_generator
        self.lr = lr
        self.lr_decay = lr_decay
        self.seed = seed
        self.lr_history: List[float] = []
        self.online_model = online_model
        if online_model == "confidence_weighted":
            from botcopier.models.registry import ConfidenceWeighted

            self.clf = ConfidenceWeighted()
            self.model_type = "confidence_weighted"
        else:
            self.clf = SGDClassifier(
                loss="log_loss", learning_rate="adaptive", eta0=self.lr
            )
            self.model_type = "logreg"
        self.feature_names: List[str] = list(FEATURE_COLUMNS) if _HAS_FEAST else []
        self.feature_flags: Dict[str, bool] = {}
        self._prev_coef: List[float] | None = None
        self.training_mode = "lite"
        self.cpu_threshold = 80.0
        self.sleep_seconds = 3.0
        self.half_life_days = 0.0
        self.weight_decay: Dict[str, Any] | None = None
        self.recent_probs: deque[float] = deque(maxlen=1000)
        self.calib_scores: deque[float] = deque(maxlen=1000)
        self.calib_labels: deque[int] = deque(maxlen=1000)
        self.calibrator: Any | None = None
        self.conformal_lower: float | None = None
        self.conformal_upper: float | None = None
        self.gnn_state: Dict[str, Any] | None = None
        self.meta_weights: List[float] | None = None
        self.adapt_log: List[Dict[str, List[float]]] = []
        self.tick_buffer_path = (
            Path(tick_buffer_path)
            if tick_buffer_path is not None
            else Path("data/live_ticks.parquet")
        )
        self.tick_buffer_size = tick_buffer_size
        self.controller: AutoMLController | None = controller
        self.prev_accuracy: float | None = None
        if _HAS_FEAST:
            repo = Path(__file__).resolve().parents[1] / "feature_store" / "feast_repo"
            self.store = FeatureStore(repo_path=str(repo))
        else:
            self.store = None
        self.graph_dataset: GraphDataset | None = None
        # sequential drift detector on rolling feature statistics
        self.drift_detector = PageHinkley() if PageHinkley is not None else None
        self.last_drift_metric: float = 0.0
        self.last_drift_detected: bool = False
        self.drift_events: int = 0
        if self.model_path.exists():
            self._load()
        elif detect_resources:
            try:
                res = detect_resources()
                self.training_mode = res.get("mode", self.training_mode)
                self.feature_flags["order_book"] = res.get("heavy_mode", False)
            except (OSError, RuntimeError, ValueError):
                pass
        self.feature_flags.setdefault("order_book", self.training_mode not in ("lite",))
        self._apply_mode()

    def _apply_mode(self) -> None:
        if self.training_mode == "lite":
            self.cpu_threshold = 50.0
            self.sleep_seconds = 6.0
        else:
            self.cpu_threshold = 80.0
            self.sleep_seconds = 3.0

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Restore coefficients from ``model.json`` if present."""
        try:
            params = load_params(self.model_path)
        except (OSError, ValidationError):
            return
        data = params.model_dump()
        self.training_mode = data.get("mode") or data.get("training_mode", "lite")
        self.feature_names = data.get("feature_names", [])
        self.feature_flags = data.get("feature_flags", {})
        if "order_book" not in self.feature_flags:
            self.feature_flags["order_book"] = self.training_mode not in ("lite",)
        self.model_type = data.get("model_type", self.model_type)
        coef = data.get("coefficients")
        intercept = data.get("intercept")
        variance = data.get("variance")
        bias_var = data.get("bias_variance")
        self.half_life_days = float(data.get("half_life_days", 0.0))
        self.weight_decay = data.get("weight_decay")
        self.conformal_lower = data.get("conformal_lower")
        self.conformal_upper = data.get("conformal_upper")
        self.gnn_state = data.get("gnn_state")
        meta_section = data.get("meta")
        if isinstance(meta_section, dict):
            self.meta_weights = [float(w) for w in meta_section.get("weights", [])]
        self.adapt_log = data.get("adaptation_log", [])
        calib = data.get("calibration")
        if isinstance(calib, dict):
            if calib.get("method") == "isotonic":
                self.calibrator = None
            elif {"coef", "intercept"} <= calib.keys():
                self.calibrator = LogisticRegression()
                self.calibrator.classes_ = np.array([0, 1])
                self.calibrator.coef_ = np.array([[calib["coef"]]])
                self.calibrator.intercept_ = np.array([calib["intercept"]])
                self.calibrator.n_features_in_ = 1
        if self.gnn_state and _HAS_TG and Path("symbol_graph.json").exists():
            try:
                self.graph_dataset = GraphDataset(Path("symbol_graph.json"))
            except (OSError, ValueError):
                self.graph_dataset = None
        if coef is None and self.meta_weights:
            n = len(self.meta_weights)
            self.feature_names = data.get("feature_names", self.feature_names)
            if not self.feature_names:
                self.feature_names = [f"f{i}" for i in range(n)]
            self.clf.partial_fit(np.zeros((1, n)), [0], classes=np.array([0, 1]))
            self.clf.coef_ = np.array([self.meta_weights])
            self.clf.intercept_ = np.array([0.0])
            self._prev_coef = list(self.meta_weights) + [0.0]
        elif self.feature_names and coef is not None and intercept is not None:
            if self.model_type == "confidence_weighted":
                from botcopier.models.registry import ConfidenceWeighted

                self.clf = ConfidenceWeighted()
                self.clf.w = np.array(coef, dtype=float)
                self.clf.b = float(intercept)
                self.clf.sigma = (
                    np.array(variance, dtype=float)
                    if variance is not None
                    else np.ones(len(coef), dtype=float)
                )
                self.clf.bias_sigma = float(bias_var) if bias_var is not None else 1.0
                self.clf.classes_ = np.array([0, 1])
                self._prev_coef = list(coef) + [self.clf.b]
            else:
                n = len(self.feature_names)
                self.clf.partial_fit(np.zeros((1, n)), [0], classes=np.array([0, 1]))
                self.clf.coef_ = np.array([coef])
                self.clf.intercept_ = np.array([intercept])
                self._prev_coef = list(coef) + [intercept]
        self._apply_mode()

    def _save(self) -> None:
        payload = {
            "feature_names": self.feature_names,
            "training_mode": self.training_mode,
            "mode": self.training_mode,
            "feature_flags": self.feature_flags,
            "model_type": self.model_type,
        }
        if self.model_type == "confidence_weighted":
            payload["coefficients"] = (
                self.clf.w.tolist() if getattr(self.clf, "w", None) is not None else []
            )
            payload["intercept"] = float(getattr(self.clf, "b", 0.0))
            payload["variance"] = (
                self.clf.sigma.tolist()
                if getattr(self.clf, "sigma", None) is not None
                else []
            )
            payload["bias_variance"] = float(getattr(self.clf, "bias_sigma", 1.0))
        else:
            payload["coefficients"] = self.clf.coef_[0].tolist()
            payload["intercept"] = float(self.clf.intercept_[0])
        if self.half_life_days:
            payload["half_life_days"] = self.half_life_days
        if self.weight_decay:
            payload["weight_decay"] = self.weight_decay
        if self.conformal_lower is not None:
            payload["conformal_lower"] = self.conformal_lower
        if self.conformal_upper is not None:
            payload["conformal_upper"] = self.conformal_upper
        if self.gnn_state:
            payload["gnn_state"] = self.gnn_state
        if self.calibrator is not None:
            if isinstance(self.calibrator, IsotonicRegression):
                payload["calibration"] = {
                    "method": "isotonic",
                    "x": self.calibrator.X_thresholds_.tolist(),
                    "y": self.calibrator.y_thresholds_.tolist(),
                }
            else:
                payload["calibration"] = {
                    "coef": float(self.calibrator.coef_[0][0]),
                    "intercept": float(self.calibrator.intercept_[0]),
                }
        try:
            params = load_params(self.model_path)
            existing = params.model_dump()
        except (OSError, ValidationError):
            existing = {}
        existing.update(payload)
        if self.meta_weights is not None:
            existing.setdefault("meta", {})["weights"] = [
                float(w) for w in self.meta_weights
            ]
        if self.adapt_log:
            existing["adaptation_log"] = self.adapt_log
        existing.setdefault("metadata", {})["seed"] = self.seed
        to_hash = dict(existing)
        to_hash.pop("model_hash", None)
        try:
            hash_val = hashlib.sha256(
                json.dumps(to_hash, sort_keys=True).encode()
            ).hexdigest()
            existing["model_hash"] = hash_val
        except Exception:  # pragma: no cover - hashing should not fail
            pass
        self.model_path.write_text(ModelParams(**existing).model_dump_json())

    # ------------------------------------------------------------------
    # Incremental training
    # ------------------------------------------------------------------
    def _ensure_features(self, keys: Iterable[str]) -> None:
        if _HAS_FEAST:
            return
        if not self.feature_flags.get("order_book", False):
            keys = [k for k in keys if not k.startswith("book_")]
        new_feats = [k for k in keys if k not in self.feature_names and k != "y"]
        if not new_feats:
            return
        self.feature_names.extend(sorted(new_feats))
        if hasattr(self.clf, "coef_"):
            n = len(self.feature_names)
            coef = np.zeros((1, n))
            coef[:, : self.clf.coef_.shape[1]] = self.clf.coef_
            self.clf.coef_ = coef
        elif hasattr(self.clf, "w") and getattr(self.clf, "w") is not None:
            add = len(new_feats)
            self.clf.w = np.concatenate([self.clf.w, np.zeros(add)])
            self.clf.sigma = np.concatenate(
                [self.clf.sigma, np.ones(add) / getattr(self.clf, "r", 1.0)]
            )

    def _vectorise(self, batch: List[Dict[str, Any]]):
        if (
            self.gnn_state
            and self.graph_dataset is not None
            and compute_gnn_embeddings is not None
        ):
            try:
                df_batch = pd.DataFrame(batch)
                emb_map, _ = compute_gnn_embeddings(
                    df_batch, self.graph_dataset, state_dict=self.gnn_state
                )
                if emb_map:
                    emb_dim = len(next(iter(emb_map.values())))
                    for rec in batch:
                        sym = str(rec.get("symbol", ""))
                        for i in range(emb_dim):
                            rec[f"graph_emb{i}"] = emb_map.get(sym, [0.0] * emb_dim)[i]
            except (RuntimeError, ValueError):
                pass
        if _HAS_FEAST and self.store is not None:
            entity_rows = [{"symbol": rec.get("symbol", "")} for rec in batch]
            feature_refs = [f"trade_features:{f}" for f in self.feature_names]
            feat_dict = self.store.get_online_features(
                features=feature_refs, entity_rows=entity_rows
            ).to_dict()
            X = [
                [feat_dict[f][i] for f in self.feature_names] for i in range(len(batch))
            ]
        else:
            for rec in batch:
                self._ensure_features(rec.keys())
            X = [[float(rec.get(f, 0.0)) for f in self.feature_names] for rec in batch]
        y = [int(rec["y"]) for rec in batch]
        return np.asarray(X), np.asarray(y)

    def _log_validation(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            preds = self.clf.predict(X)
            acc = float(np.mean(preds == y))
        except (NotFittedError, ValueError):
            acc = 0.0
        conf = None
        if hasattr(self.clf, "confidence_score"):
            try:
                conf = float(np.mean(self.clf.confidence_score(X)))
            except Exception:
                conf = None
        log = {"event": "validation", "size": len(y), "accuracy": acc}
        if conf is not None:
            log["confidence"] = conf
        logger.info(log)
        return acc

    def _handle_regime_shift(self) -> None:
        action = "reset"
        if os.getenv("ONLINE_TRAINER_FULL_RETRAIN"):
            action = "full_retrain"
            base = Path(__file__).resolve().parent
            try:
                subprocess.Popen([sys.executable, str(base / "auto_retrain.py")])
            except (OSError, subprocess.SubprocessError):
                logger.exception("failed to start full retrain")
        else:
            self.clf = SGDClassifier(loss="log_loss")
            self._prev_coef = None
        logger.info({"event": "regime_shift", "action": action})

    def _check_drift(self, value: float) -> None:
        """Run sequential drift detector, store metric and handle regime shifts."""
        metric = 0.0
        detected = False
        if self.drift_detector is not None:
            detected = self.drift_detector.update(value)
            metric = getattr(self.drift_detector, "_cum", 0.0) - getattr(
                self.drift_detector, "_min_cum", 0.0
            )
            if detected:
                self._handle_regime_shift()
                self.drift_events += 1
        self.last_drift_metric = float(metric)
        self.last_drift_detected = bool(detected)

    def _append_buffer(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        try:
            df = pd.DataFrame(records)
            path = self.tick_buffer_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    existing = pd.read_parquet(path)
                except Exception:
                    existing = pd.DataFrame()
                df = pd.concat([existing, df], ignore_index=True).tail(
                    self.tick_buffer_size
                )
            df.to_parquet(path, index=False)
        except Exception:
            pass

    def _reselect_architecture(self) -> None:
        if self.controller is None:
            return
        try:
            action, _ = self.controller.sample_action()
        except Exception:
            return
        feats, model_name = action
        if feats:
            self.feature_names = list(feats)
        if model_name != self.model_type:
            if model_name == "confidence_weighted":
                try:
                    from botcopier.models.registry import ConfidenceWeighted

                    self.clf = ConfidenceWeighted()
                except Exception:
                    self.clf = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=self.lr)
            else:
                self.clf = SGDClassifier(
                    loss="log_loss", learning_rate="adaptive", eta0=self.lr
                )
            self.model_type = model_name
        self._prev_coef = None

    def update(self, batch: List[Dict[str, Any]]) -> bool:
        with observe_latency("train_batch"):
            try:
                with tracer.start_as_current_span("train_batch") as span:
                    X, y = self._vectorise(batch)
                    mean_val = float(X.mean())
                    self._check_drift(mean_val)
                    # configure current learning rate
                    if hasattr(self.clf, "eta0"):
                        self.clf.eta0 = self.lr
                    old_coef = (
                        self.clf.coef_[0].copy()
                        if hasattr(self.clf, "coef_") and self.clf.coef_.size
                        else None
                    )
                    if not hasattr(self.clf, "classes_"):
                        self.clf.partial_fit(X, y, classes=np.array([0, 1]))
                    else:
                        self.clf.partial_fit(X, y)
                    if old_coef is not None:
                        new_coef = self.clf.coef_[0].copy()
                        self.adapt_log.append(
                            {"old": old_coef.tolist(), "new": new_coef.tolist()}
                        )
                        self.meta_weights = new_coef.tolist()
                    lr_used = getattr(self.clf, "eta0", self.lr)
                    self.lr_history.append(lr_used)
                    try:
                        raw_probs = self.clf.predict_proba(X)[:, 1]
                        self.calib_scores.extend(raw_probs.tolist())
                        self.calib_labels.extend(y.tolist())
                        if (
                            len(self.calib_labels) >= 2
                            and len(set(self.calib_labels)) > 1
                        ):
                            scores = np.array(self.calib_scores)
                            labels = np.array(self.calib_labels)
                            try:
                                self.calibrator = IsotonicRegression(
                                    out_of_bounds="clip"
                                ).fit(scores, labels)
                            except ValueError:
                                self.calibrator = None
                        if self.calibrator is not None:
                            if isinstance(self.calibrator, IsotonicRegression):
                                probs1 = self.calibrator.predict(raw_probs)
                            else:
                                probs1 = self.calibrator.predict_proba(
                                    raw_probs.reshape(-1, 1)
                                )[:, 1]
                        else:
                            probs1 = raw_probs
                        probs = np.column_stack([1 - probs1, probs1])
                        self.recent_probs.extend(probs1.tolist())
                        arr = np.fromiter(self.recent_probs, dtype=float)
                        self.conformal_lower = float(np.quantile(arr, 0.05))
                        self.conformal_upper = float(np.quantile(arr, 0.95))
                    except ValueError:
                        pass
                    if self.model_type == "confidence_weighted":
                        coef = self.clf.w.tolist()
                        intercept = float(self.clf.b)
                    else:
                        coef = self.clf.coef_[0].tolist()
                        intercept = float(self.clf.intercept_[0])
                    prev = self._prev_coef
                    self._prev_coef = coef + [intercept]
                    changed = prev != self._prev_coef
                    self._save()
                    acc = self._log_validation(X, y)
                    if self.controller is None and self.feature_names:
                        try:
                            self.controller = AutoMLController(
                                self.feature_names,
                                {"sgd": 1, "confidence_weighted": 2},
                                model_path=self.model_path,
                            )
                        except Exception:
                            self.controller = None
                    if self.controller is not None:
                        try:
                            self.controller.update(
                                (tuple(self.feature_names), self.model_type), acc
                            )
                            if (
                                self.prev_accuracy is not None
                                and acc < self.prev_accuracy - 0.2
                            ):
                                self._reselect_architecture()
                            self.prev_accuracy = acc
                        except Exception:
                            pass
                    ctx = span.get_span_context()
                    log = {
                        "event": "batch_update",
                        "size": len(batch),
                        "coefficients_changed": changed,
                        "lr": lr_used,
                        "drift_metric": self.last_drift_metric,
                        "drift": self.last_drift_detected,
                    }
                    conf = getattr(self.clf, "last_batch_confidence", None)
                    if conf is not None:
                        log["confidence"] = conf
                    logger.info(
                        log,
                        extra={"trace_id": ctx.trace_id, "span_id": ctx.span_id},
                    )
                    # decay learning rate for next batch
                    self.lr *= self.lr_decay
            except Exception:
                ERROR_COUNTER.labels(type="update").inc()
                raise
        TRADE_COUNTER.inc(len(batch))
        return changed

    # ------------------------------------------------------------------
    # Drift monitoring
    # ------------------------------------------------------------------
    def start_drift_monitor(
        self,
        baseline_file: Path,
        recent_file: Path,
        *,
        log_dir: Path,
        out_dir: Path,
        files_dir: Path,
        threshold: float = 0.2,
        interval: float = 300.0,
    ) -> None:
        """Periodically compute drift metrics and retrain if necessary."""
        if _compute_metrics is None or _update_model is None:
            logger.warning("drift_monitor unavailable")
            return

        def _loop() -> None:
            while True:
                try:
                    metrics = _compute_metrics(baseline_file, recent_file)
                    retrain = max(metrics.values()) > threshold
                    if retrain:
                        method = max(metrics, key=metrics.get)
                        base = Path(__file__).resolve().parent
                        try:
                            subprocess.run(
                                [
                                    sys.executable,
                                    str(base / "auto_retrain.py"),
                                    "--log-dir",
                                    str(log_dir),
                                    "--out-dir",
                                    str(out_dir),
                                    "--files-dir",
                                    str(files_dir),
                                    "--baseline-file",
                                    str(baseline_file),
                                    "--recent-file",
                                    str(recent_file),
                                    "--drift-method",
                                    method,
                                    "--drift-threshold",
                                    str(threshold),
                                ],
                                check=True,
                            )
                            self._load()
                        except (OSError, subprocess.SubprocessError):
                            logger.exception("drift monitoring failed")
                            self._handle_regime_shift()
                        _update_model(self.model_path, metrics, True)
                    else:
                        _update_model(self.model_path, metrics, False)
                except Exception:
                    logger.exception("drift monitoring failed")
                time.sleep(interval)

        threading.Thread(target=_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------
    async def consume_ticks(
        self,
        tick_stream: AsyncIterable[Dict[str, Any]],
        ring_path: Path,
        ring_size: int = 1_000_000,
    ) -> None:
        """Subscribe to a live tick stream and persist raw ticks in a ring buffer."""
        if ShmRing is None:
            raise RuntimeError("shm ring unavailable")
        ring_path = Path(ring_path)
        ring = (
            ShmRing.open(str(ring_path))
            if ring_path.exists()
            else ShmRing.create(str(ring_path), ring_size)
        )
        batch: List[Dict[str, Any]] = []
        buffer_records: List[Dict[str, Any]] = []
        try:
            async for tick in tick_stream:
                buffer_records.append(tick)
                try:
                    ring.push(TRADE_MSG, json.dumps(tick).encode())
                except Exception:
                    pass
                if "y" not in tick and "label" not in tick:
                    if len(buffer_records) >= self.batch_size:
                        self._append_buffer(buffer_records)
                        buffer_records.clear()
                    continue
                tick["y"] = tick["y"] if "y" in tick else tick.get("label")
                batch.append(tick)
                if len(batch) >= self.batch_size:
                    load = psutil.cpu_percent(interval=None)
                    if load > self.cpu_threshold:
                        await asyncio.sleep(self.sleep_seconds)
                    self.update(batch)
                    self._append_buffer(buffer_records)
                    batch.clear()
                    buffer_records.clear()
            if batch:
                load = psutil.cpu_percent(interval=None)
                if load > self.cpu_threshold:
                    await asyncio.sleep(self.sleep_seconds)
                self.update(batch)
                self._append_buffer(buffer_records)
                batch.clear()
                buffer_records.clear()
        finally:
            ring.close()

    def tail_csv(self, path: Path) -> None:
        """Continuously follow ``path`` for new rows."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pos = 0
        batch: List[Dict[str, Any]] = []
        while True:
            load = psutil.cpu_percent(interval=None)
            if load > self.cpu_threshold:
                time.sleep(self.sleep_seconds)
            if path.exists():
                with path.open() as f:
                    f.seek(pos)
                    reader = csv.DictReader(f)
                    for row in reader:
                        pos = f.tell()
                        if "y" not in row and "label" not in row:
                            continue
                        row["y"] = row.get("y") or row.get("label")
                        batch.append(row)
                        if len(batch) >= self.batch_size:
                            load = psutil.cpu_percent(interval=None)
                            if load > self.cpu_threshold:
                                time.sleep(self.sleep_seconds)
                            self.update(batch)
                            batch.clear()
            if batch:
                load = psutil.cpu_percent(interval=None)
                if load > self.cpu_threshold:
                    time.sleep(self.sleep_seconds)
                self.update(batch)
                batch.clear()
            time.sleep(1.0)

    async def consume_flight(self, host: str, port: int, path: str = "trades") -> None:
        """Subscribe to an Arrow Flight stream of trade events asynchronously."""
        try:  # pragma: no cover - optional dependency
            import pyarrow.flight as flight
        except ImportError as exc:  # pragma: no cover - pyarrow missing
            raise RuntimeError("pyarrow is required for Flight consumption") from exc

        client = flight.FlightClient(f"grpc://{host}:{port}")
        ticket = flight.Ticket(path.encode())
        offset = 0
        batch: List[Dict[str, Any]] = []
        try:
            while True:
                load = psutil.cpu_percent(interval=None)
                if load > self.cpu_threshold:
                    await asyncio.sleep(self.sleep_seconds)
                    continue
                try:
                    table = await asyncio.to_thread(
                        lambda: client.do_get(ticket).read_all()
                    )
                except (flight.FlightError, OSError):
                    await asyncio.sleep(1.0)
                    continue
                if table.num_rows > offset:
                    for row in table.slice(offset).to_pylist():
                        if "y" not in row and "label" not in row:
                            continue
                        row["y"] = row.get("y") or row.get("label")
                        batch.append(row)
                        if len(batch) >= self.batch_size:
                            load = psutil.cpu_percent(interval=None)
                            if load > self.cpu_threshold:
                                await asyncio.sleep(self.sleep_seconds)
                            self.update(batch)
                            batch.clear()
                    offset = table.num_rows
                if batch:
                    load = psutil.cpu_percent(interval=None)
                    if load > self.cpu_threshold:
                        await asyncio.sleep(self.sleep_seconds)
                    self.update(batch)
                    batch.clear()
                await asyncio.sleep(1.0)
        finally:
            client.close()

    async def consume_websocket(
        self, url: str, ring_path: Path, ring_size: int = 1_000_000
    ) -> None:
        """Subscribe to ticks from a WebSocket source."""
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("websockets is required for WebSocket consumption") from exc

        async with websockets.connect(url) as ws:
            async def _gen():
                async for msg in ws:
                    try:
                        yield json.loads(msg)
                    except Exception:
                        continue

            await self.consume_ticks(_gen(), ring_path, ring_size)

    async def consume_kafka(
        self,
        topic: str,
        bootstrap_servers: str,
        ring_path: Path,
        group_id: str = "online-trainer",
        ring_size: int = 1_000_000,
    ) -> None:
        """Subscribe to ticks from a Kafka topic."""
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("aiokafka is required for Kafka consumption") from exc

        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode()),
        )
        await consumer.start()
        try:
            async def _gen():
                async for msg in consumer:
                    yield msg.value

            await self.consume_ticks(_gen(), ring_path, ring_size)
        finally:
            await consumer.stop()


async def run(data_cfg: DataConfig, train_cfg: TrainingConfig) -> None:
    """Run the online trainer with ``data_cfg`` and ``train_cfg`` asynchronously."""

    set_seed(train_cfg.random_seed)
    save_params(data_cfg, train_cfg)
    trainer = OnlineTrainer(
        train_cfg.model,
        train_cfg.batch_size,
        lr=train_cfg.lr,
        lr_decay=train_cfg.lr_decay,
        seed=train_cfg.random_seed,
        online_model=train_cfg.online_model,
    )
    start_metrics_server(train_cfg.metrics_port)
    _sd_notify_ready()
    _start_watchdog_thread()
    if (
        data_cfg.baseline_file
        and data_cfg.recent_file
        and data_cfg.log_dir
        and data_cfg.out_dir
        and data_cfg.files_dir
    ):
        trainer.start_drift_monitor(
            data_cfg.baseline_file,
            data_cfg.recent_file,
            log_dir=data_cfg.log_dir,
            out_dir=data_cfg.out_dir,
            files_dir=data_cfg.files_dir,
            threshold=train_cfg.drift_threshold,
            interval=train_cfg.drift_interval,
        )
    if data_cfg.csv:
        await asyncio.to_thread(trainer.tail_csv, data_cfg.csv)
    else:
        await trainer.consume_flight(
            train_cfg.flight_host, train_cfg.flight_port, train_cfg.flight_path
        )


async def async_main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Online incremental trainer")
    p.add_argument("--csv", help="Path to trades_raw.csv to follow")
    p.add_argument("--flight-host", help="Arrow Flight host")
    p.add_argument("--flight-port", type=int, help="Arrow Flight port")
    p.add_argument("--model")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float, help="Initial learning rate")
    p.add_argument(
        "--lr-decay", type=float, help="Multiplicative learning rate decay per batch"
    )
    p.add_argument("--flight-path", help="Flight path name")
    p.add_argument(
        "--online-model",
        choices=["sgd", "confidence_weighted"],
        help="Online model to use",
    )
    p.add_argument("--baseline-file", help="Baseline CSV for drift monitoring")
    p.add_argument("--recent-file", help="Recent CSV for drift monitoring")
    p.add_argument("--log-dir", help="Log directory for retrain")
    p.add_argument("--out-dir", help="Output directory for retrain")
    p.add_argument("--files-dir", help="Files directory for retrain")
    p.add_argument(
        "--drift-threshold", type=float, help="Drift threshold triggering retrain"
    )
    p.add_argument("--drift-interval", type=float, help="Seconds between drift checks")
    p.add_argument("--metrics-port", type=int, help="Prometheus metrics port")
    p.add_argument("--random-seed", type=int)
    args = p.parse_args(argv)
    data_cfg, train_cfg, _ = load_settings(vars(args))
    await run(data_cfg, train_cfg)


def main(argv: List[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
