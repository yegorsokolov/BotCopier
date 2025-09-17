from __future__ import annotations

import functools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast

import typer

from botcopier.exceptions import BotCopierError
from botcopier.training.pipeline import train as train_pipeline
from config.settings import (
    DataConfig,
    ExecutionConfig,
    TrainingConfig,
    load_settings,
    save_params,
)

from ..scripts.drift_monitor import run as run_drift_monitor
from ..scripts.evaluation import evaluate as eval_predictions
from ..scripts.online_trainer import run as run_online_trainer

app = typer.Typer(help="BotCopier unified command line interface")


F = TypeVar("F", bound=Callable[..., Any])


def error_handler(func: F) -> F:
    """Decorator for CLI commands to provide structured error handling."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        try:
            return func(*args, **kwargs)
        except BotCopierError as exc:
            logging.error(
                json.dumps(
                    {
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "symbol": exc.symbol,
                        "timestamp": getattr(exc, "timestamp", datetime.utcnow()),
                    }
                )
            )
            raise typer.Exit(code=1)
        except Exception as exc:  # pragma: no cover - unexpected
            logging.error(
                json.dumps(
                    {
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    }
                )
            )
            raise typer.Exit(code=1)

    return cast(F, wrapper)


@app.callback()
@error_handler
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to optional config file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    use_gpu: bool = typer.Option(False, help="Enable GPU execution"),
    trace: bool = typer.Option(False, help="Enable tracing"),
    trace_exporter: str = typer.Option("otlp", help="Tracing exporter"),
    profile: bool = typer.Option(False, help="Enable profiling"),
) -> None:
    """Configure global options for all commands."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    if config:
        data_cfg, train_cfg, exec_cfg = load_settings(path=config)
    else:
        data_cfg, train_cfg, exec_cfg = (
            DataConfig(),
            TrainingConfig(),
            ExecutionConfig(),
        )
    exec_updates: dict[str, Any] = {}
    if use_gpu:
        exec_updates["use_gpu"] = use_gpu
    if trace:
        exec_updates["trace"] = trace
    if trace_exporter:
        exec_updates["trace_exporter"] = trace_exporter
    if profile:
        exec_updates["profile"] = profile
    if exec_updates:
        exec_cfg = exec_cfg.model_copy(update=exec_updates)
    ctx.obj = {
        "config": {
            "data": data_cfg,
            "training": train_cfg,
            "execution": exec_cfg,
        }
    }


def _cfg(ctx: typer.Context) -> tuple[DataConfig, TrainingConfig, ExecutionConfig]:
    if ctx.obj and "config" in ctx.obj:
        cfg = ctx.obj["config"]
        return (
            cfg.get("data", DataConfig()),
            cfg.get("training", TrainingConfig()),
            cfg.get("execution", ExecutionConfig()),
        )
    return DataConfig(), TrainingConfig(), ExecutionConfig()


@app.command("train")
@error_handler
def train(
    ctx: typer.Context,
    data_dir: Optional[Path] = typer.Argument(
        None, help="Directory containing training logs"
    ),
    out_dir: Optional[Path] = typer.Argument(
        None, help="Output directory for the model"
    ),
    model_type: Optional[str] = typer.Option(None, help="Model type"),
    cache_dir: Optional[Path] = typer.Option(None, help="Optional cache directory"),
    features: Optional[list[str]] = typer.Option(
        None,
        "--feature",
        "-f",
        help="Enable feature plugin; can be passed multiple times",
    ),
    orderbook_features: bool = typer.Option(
        False, help="Include order book derived features"
    ),
    csd_features: bool = typer.Option(
        False, help="Include cross spectral density features"
    ),
    random_seed: Optional[int] = typer.Option(None, help="Random seed"),
    hrp_allocation: bool = typer.Option(
        False, help="Compute hierarchical risk parity allocation"
    ),
    strategy_search: bool = typer.Option(
        False, help="Run neural strategy search instead of standard training"
    ),
    reuse_controller: bool = typer.Option(
        False,
        help="Reuse previously learned AutoML controller policy",
    ),
) -> None:
    """Train a model from trade logs."""
    data_cfg, train_cfg, exec_cfg = _cfg(ctx)
    if data_dir:
        data_cfg = data_cfg.model_copy(update={"data": data_dir})
    if out_dir:
        data_cfg = data_cfg.model_copy(update={"out": out_dir})
    if model_type:
        train_cfg = train_cfg.model_copy(update={"model_type": model_type})
    if cache_dir:
        train_cfg = train_cfg.model_copy(update={"cache_dir": cache_dir})
    if features is not None:
        train_cfg = train_cfg.model_copy(update={"features": list(features)})
    if orderbook_features:
        feats = set(train_cfg.features or [])
        feats.add("orderbook")
        train_cfg = train_cfg.model_copy(update={"features": list(feats)})
    if csd_features:
        feats = set(train_cfg.features or [])
        feats.add("csd")
        train_cfg = train_cfg.model_copy(update={"features": list(feats)})
    if random_seed is not None:
        train_cfg = train_cfg.model_copy(update={"random_seed": random_seed})
    if hrp_allocation:
        train_cfg = train_cfg.model_copy(update={"hrp_allocation": hrp_allocation})
    if strategy_search:
        train_cfg = train_cfg.model_copy(update={"strategy_search": strategy_search})
    if reuse_controller:
        train_cfg = train_cfg.model_copy(update={"reuse_controller": reuse_controller})
    ctx.obj["config"] = {
        "data": data_cfg,
        "training": train_cfg,
        "execution": exec_cfg,
    }
    if data_cfg.data is None or data_cfg.out is None:
        raise typer.BadParameter("data_dir and out_dir must be provided")
    snapshot = save_params(data_cfg, train_cfg, exec_cfg)
    config_hash = snapshot.digest
    train_pipeline(
        Path(data_cfg.data),
        Path(data_cfg.out),
        model_type=train_cfg.model_type,
        cache_dir=train_cfg.cache_dir,
        model_json=train_cfg.model,
        features=train_cfg.features,
        random_seed=train_cfg.random_seed,
        hrp_allocation=train_cfg.hrp_allocation,
        strategy_search=train_cfg.strategy_search,
        reuse_controller=train_cfg.reuse_controller,
        config_hash=config_hash,
        config_snapshot=snapshot.as_dict(),
    )


@app.command("evaluate")
@error_handler
def evaluate(
    ctx: typer.Context,
    pred_file: Optional[Path] = typer.Argument(None, help="CSV file with predictions"),
    actual_log: Optional[Path] = typer.Argument(
        None, help="CSV log with actual trades"
    ),
    window: Optional[int] = typer.Option(None, help="Matching window in seconds"),
    model_json: Optional[Path] = typer.Option(
        None, help="Optional model.json for additional metrics"
    ),
    eval_hooks: Optional[str] = typer.Option(
        None, help="Comma separated list of evaluation hooks"
    ),
) -> None:
    """Evaluate predictions against actual trade outcomes."""
    data_cfg, train_cfg, exec_cfg = _cfg(ctx)
    if pred_file:
        data_cfg = data_cfg.model_copy(update={"pred_file": pred_file})
    if actual_log:
        data_cfg = data_cfg.model_copy(update={"actual_log": actual_log})
    if window is not None:
        train_cfg = train_cfg.model_copy(update={"window": window})
    if model_json:
        train_cfg = train_cfg.model_copy(update={"model": model_json})
    if eval_hooks is not None:
        hooks_list = [h.strip() for h in eval_hooks.split(",") if h.strip()]
        train_cfg = train_cfg.model_copy(update={"eval_hooks": hooks_list})
    ctx.obj["config"] = {
        "data": data_cfg,
        "training": train_cfg,
        "execution": exec_cfg,
    }
    if data_cfg.pred_file is None or data_cfg.actual_log is None:
        raise typer.BadParameter("pred_file and actual_log must be provided")
    save_params(data_cfg, train_cfg, exec_cfg)
    stats = eval_predictions(
        Path(data_cfg.pred_file),
        Path(data_cfg.actual_log),
        train_cfg.window,
        train_cfg.model,
        hooks=train_cfg.eval_hooks or None,
    )
    typer.echo(json.dumps(stats, indent=2))


@app.command("online-train")
@error_handler
def online_train(
    ctx: typer.Context,
    csv: Optional[Path] = typer.Option(None, help="Path to trades_raw.csv"),
    flight_host: Optional[str] = typer.Option(None, help="Arrow Flight host"),
    flight_port: Optional[int] = typer.Option(None, help="Arrow Flight port"),
    model: Optional[Path] = typer.Option(None, help="Path to model.json"),
    batch_size: Optional[int] = typer.Option(None, help="Training batch size"),
    lr: Optional[float] = typer.Option(None, help="Initial learning rate"),
    lr_decay: Optional[float] = typer.Option(
        None, help="Multiplicative learning rate decay per batch"
    ),
    flight_path: Optional[str] = typer.Option(None, help="Arrow Flight path"),
    baseline_file: Optional[Path] = typer.Option(
        None, help="Baseline CSV for drift monitoring"
    ),
    recent_file: Optional[Path] = typer.Option(
        None, help="Recent CSV for drift monitoring"
    ),
    log_dir: Optional[Path] = typer.Option(None, help="Log directory for retrain"),
    out_dir: Optional[Path] = typer.Option(None, help="Output directory for retrain"),
    files_dir: Optional[Path] = typer.Option(None, help="Files directory for retrain"),
    drift_threshold: Optional[float] = typer.Option(
        None, help="Drift threshold triggering retrain"
    ),
    drift_interval: Optional[float] = typer.Option(
        None, help="Seconds between drift checks"
    ),
    orderbook_features: bool = typer.Option(
        False, help="Include order book derived features"
    ),
    csd_features: bool = typer.Option(
        False, help="Include cross spectral density features"
    ),
) -> None:
    """Continuously update a model from streaming trade events."""
    data_cfg, train_cfg, exec_cfg = _cfg(ctx)
    updates_data = {
        k: v
        for k, v in {
            "csv": csv,
            "baseline_file": baseline_file,
            "recent_file": recent_file,
            "log_dir": log_dir,
            "out_dir": out_dir,
            "files_dir": files_dir,
        }.items()
        if v is not None
    }
    updates_train: dict[str, Any] = {
        k: v
        for k, v in {
            "model": model,
            "batch_size": batch_size,
            "lr": lr,
            "lr_decay": lr_decay,
            "flight_host": flight_host,
            "flight_port": flight_port,
            "flight_path": flight_path,
            "drift_threshold": drift_threshold,
            "drift_interval": drift_interval,
        }.items()
        if v is not None
    }
    if orderbook_features:
        feats = set(train_cfg.features or [])
        feats.add("orderbook")
        updates_train["features"] = list(feats)
    if csd_features:
        feats = set(train_cfg.features or [])
        feats.add("csd")
        updates_train["features"] = list(feats)
    if updates_data:
        data_cfg = data_cfg.model_copy(update=updates_data)
    if updates_train:
        train_cfg = train_cfg.model_copy(update=updates_train)
    ctx.obj["config"] = {
        "data": data_cfg,
        "training": train_cfg,
        "execution": exec_cfg,
    }
    save_params(data_cfg, train_cfg, exec_cfg)
    import asyncio

    asyncio.run(run_online_trainer(data_cfg, train_cfg))


@app.command("drift-monitor")
@error_handler
def drift_monitor(
    ctx: typer.Context,
    baseline_file: Optional[Path] = typer.Option(None, help="Baseline CSV file"),
    recent_file: Optional[Path] = typer.Option(None, help="Recent CSV file"),
    drift_threshold: Optional[float] = typer.Option(0.2, help="Drift threshold"),
    model_json: Optional[Path] = typer.Option(
        Path("model.json"), help="Path to model.json"
    ),
    log_dir: Optional[Path] = typer.Option(None, help="Log directory for retrain"),
    out_dir: Optional[Path] = typer.Option(None, help="Output directory for retrain"),
    files_dir: Optional[Path] = typer.Option(None, help="Files directory for retrain"),
    drift_scores: Optional[Path] = typer.Option(
        None, help="Optional path to write per-feature drift scores"
    ),
    flag_file: Optional[Path] = typer.Option(
        None, help="Optional file to touch when drift exceeds threshold"
    ),
) -> None:
    """Compute feature drift metrics and trigger retraining when needed."""
    data_cfg, train_cfg, exec_cfg = _cfg(ctx)
    updates_data = {
        k: v
        for k, v in {
            "baseline_file": baseline_file,
            "recent_file": recent_file,
            "log_dir": log_dir,
            "out_dir": out_dir,
            "files_dir": files_dir,
            "drift_scores": drift_scores,
            "flag_file": flag_file,
        }.items()
        if v is not None
    }
    if updates_data:
        data_cfg = data_cfg.model_copy(update=updates_data)
    if drift_threshold is not None:
        train_cfg = train_cfg.model_copy(update={"drift_threshold": drift_threshold})
    if model_json:
        train_cfg = train_cfg.model_copy(update={"model": model_json})
    ctx.obj["config"] = {
        "data": data_cfg,
        "training": train_cfg,
        "execution": exec_cfg,
    }
    if (
        data_cfg.baseline_file is None
        or data_cfg.recent_file is None
        or data_cfg.log_dir is None
        or data_cfg.out_dir is None
        or data_cfg.files_dir is None
    ):
        raise typer.BadParameter(
            "baseline_file, recent_file, log_dir, out_dir and files_dir must be provided"
        )
    save_params(data_cfg, train_cfg, exec_cfg)
    run_drift_monitor(
        baseline_file=data_cfg.baseline_file,
        recent_file=data_cfg.recent_file,
        drift_threshold=train_cfg.drift_threshold,
        model_json=train_cfg.model,
        log_dir=data_cfg.log_dir,
        out_dir=data_cfg.out_dir,
        files_dir=data_cfg.files_dir,
        drift_scores=data_cfg.drift_scores,
        flag_file=data_cfg.flag_file,
    )


@app.command("serve-model")
@error_handler
def serve_model(
    host: str = typer.Option("0.0.0.0", help="Host interface for the FastAPI app"),
    port: int = typer.Option(8000, help="HTTP port for serving predictions"),
    metrics_port: int = typer.Option(8004, help="Prometheus metrics port"),
) -> None:
    """Run the FastAPI service that serves the distilled model."""

    import uvicorn

    from botcopier.metrics import start_metrics_server
    from botcopier.scripts import serve_model as serve_module

    start_metrics_server(metrics_port)
    uvicorn.run(serve_module.app, host=host, port=port)


@app.command("flight-server")
@error_handler
def flight_server(
    host: str = typer.Option("0.0.0.0", help="Host interface for the Flight server"),
    port: int = typer.Option(8815, help="Arrow Flight port"),
    data_dir: Path = typer.Option(
        Path("flight_logs"), help="Directory to persist batches"
    ),
) -> None:
    """Start the Arrow Flight server that mirrors trades and metrics."""

    from botcopier.scripts.flight_server import FlightServer

    server = FlightServer(host, port, str(data_dir))
    try:
        server.serve()
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        typer.echo("Shutting down Flight server", err=True)
        server.shutdown()


__all__ = ["app", "error_handler"]
