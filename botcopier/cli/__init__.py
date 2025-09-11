from __future__ import annotations

import functools
import importlib.util
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, cast

import typer
import yaml
from pydantic_settings import BaseSettings

from botcopier.config.settings import DataConfig, TrainingConfig, save_params
from botcopier.exceptions import BotCopierError
from botcopier.training.pipeline import train as train_pipeline

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


def _load_config(path: Path) -> tuple[DataConfig, TrainingConfig]:
    """Load configuration from ``path`` and return data/training sections."""

    if path.suffix in {".yml", ".yaml", ".json"}:
        raw = yaml.safe_load(path.read_text()) or {}
        return DataConfig(**raw.get("data", {})), TrainingConfig(
            **raw.get("training", {})
        )

    spec = importlib.util.spec_from_file_location("_botcopier_config", path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data_kwargs: dict[str, Any] = {}
        train_kwargs: dict[str, Any] = {}
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseSettings):
                if obj.__name__ == "DataConfig":
                    data_kwargs = obj().model_dump()
                elif obj.__name__ == "TrainingConfig":
                    train_kwargs = obj().model_dump()
        return DataConfig(**data_kwargs), TrainingConfig(**train_kwargs)
    raise typer.BadParameter(f"Unable to load configuration from {path}")


@app.callback()
@error_handler
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to optional config file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
) -> None:
    """Configure global options for all commands."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    if config:
        data_cfg, train_cfg = _load_config(config)
    else:
        data_cfg, train_cfg = DataConfig(), TrainingConfig()
    ctx.obj = {"config": {"data": data_cfg, "training": train_cfg}}


def _cfg(ctx: typer.Context) -> tuple[DataConfig, TrainingConfig]:
    if ctx.obj and "config" in ctx.obj:
        cfg = ctx.obj["config"]
        return cfg.get("data", DataConfig()), cfg.get("training", TrainingConfig())
    return DataConfig(), TrainingConfig()


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
    random_seed: Optional[int] = typer.Option(None, help="Random seed"),
) -> None:
    """Train a model from trade logs."""
    data_cfg, train_cfg = _cfg(ctx)
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
    if random_seed is not None:
        train_cfg = train_cfg.model_copy(update={"random_seed": random_seed})
    ctx.obj["config"] = {"data": data_cfg, "training": train_cfg}
    if data_cfg.data is None or data_cfg.out is None:
        raise typer.BadParameter("data_dir and out_dir must be provided")
    save_params(data_cfg, train_cfg)
    train_pipeline(
        Path(data_cfg.data),
        Path(data_cfg.out),
        model_type=train_cfg.model_type,
        cache_dir=train_cfg.cache_dir,
        features=train_cfg.features,
        random_seed=train_cfg.random_seed,
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
) -> None:
    """Evaluate predictions against actual trade outcomes."""
    data_cfg, train_cfg = _cfg(ctx)
    if pred_file:
        data_cfg = data_cfg.model_copy(update={"pred_file": pred_file})
    if actual_log:
        data_cfg = data_cfg.model_copy(update={"actual_log": actual_log})
    if window is not None:
        train_cfg = train_cfg.model_copy(update={"window": window})
    if model_json:
        train_cfg = train_cfg.model_copy(update={"model": model_json})
    ctx.obj["config"] = {"data": data_cfg, "training": train_cfg}
    if data_cfg.pred_file is None or data_cfg.actual_log is None:
        raise typer.BadParameter("pred_file and actual_log must be provided")
    save_params(data_cfg, train_cfg)
    stats = eval_predictions(
        Path(data_cfg.pred_file),
        Path(data_cfg.actual_log),
        train_cfg.window,
        train_cfg.model,
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
) -> None:
    """Continuously update a model from streaming trade events."""
    data_cfg, train_cfg = _cfg(ctx)
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
    updates_train = {
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
    if updates_data:
        data_cfg = data_cfg.model_copy(update=updates_data)
    if updates_train:
        train_cfg = train_cfg.model_copy(update=updates_train)
    ctx.obj["config"] = {"data": data_cfg, "training": train_cfg}
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
    data_cfg, train_cfg = _cfg(ctx)
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
    ctx.obj["config"] = {"data": data_cfg, "training": train_cfg}
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
    save_params(data_cfg, train_cfg)
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


__all__ = ["app", "error_handler"]
