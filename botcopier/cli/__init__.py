from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from pydantic_settings import BaseSettings

from botcopier.training.pipeline import train as train_pipeline

from ..scripts.drift_monitor import run as run_drift_monitor
from ..scripts.evaluation import evaluate as eval_predictions
from ..scripts.online_trainer import run as run_online_trainer

app = typer.Typer(help="BotCopier unified command line interface")


def _load_config(path: Path) -> dict[str, Any]:
    """Load configuration from ``path``.

    Python modules are imported and any ``BaseSettings`` subclasses are
    instantiated, while YAML/JSON files are parsed via ``yaml.safe_load``.
    """

    if path.suffix in {".yml", ".yaml", ".json"}:
        return yaml.safe_load(path.read_text()) or {}

    spec = importlib.util.spec_from_file_location("_botcopier_config", path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config: dict[str, Any] = {}
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, BaseSettings):
                config.update(obj().model_dump())
        return config
    raise typer.BadParameter(f"Unable to load configuration from {path}")


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to optional config file"
    ),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
) -> None:
    """Configure global options for all commands."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    ctx.obj = {"config": _load_config(config) if config else {}}


def _cfg(ctx: typer.Context) -> dict[str, Any]:
    return ctx.obj.get("config", {}) if ctx.obj else {}


@app.command("train")
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
) -> None:
    """Train a model from trade logs."""
    cfg = _cfg(ctx)
    data_dir = data_dir or cfg.get("data_dir")
    out_dir = out_dir or cfg.get("out_dir")
    model_type = model_type or cfg.get("model_type", "logreg")
    cache_dir = cache_dir or cfg.get("cache_dir")
    if data_dir is None or out_dir is None:
        raise typer.BadParameter("data_dir and out_dir must be provided")
    train_pipeline(
        Path(data_dir), Path(out_dir), model_type=model_type, cache_dir=cache_dir
    )


@app.command("evaluate")
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
    cfg = _cfg(ctx)
    pred_file = pred_file or cfg.get("pred_file")
    actual_log = actual_log or cfg.get("actual_log")
    window = window if window is not None else cfg.get("window", 60)
    model_json = model_json or cfg.get("model_json")
    if pred_file is None or actual_log is None:
        raise typer.BadParameter("pred_file and actual_log must be provided")
    stats = eval_predictions(Path(pred_file), Path(actual_log), window, model_json)
    typer.echo(json.dumps(stats, indent=2))


@app.command("online-train")
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
    cfg = _cfg(ctx)
    csv = csv or cfg.get("csv")
    flight_host = flight_host or cfg.get("flight_host")
    flight_port = flight_port or cfg.get("flight_port")
    model = model or cfg.get("model")
    batch_size = batch_size or cfg.get("batch_size")
    lr = lr or cfg.get("lr")
    lr_decay = lr_decay or cfg.get("lr_decay")
    flight_path = flight_path or cfg.get("flight_path")
    baseline_file = baseline_file or cfg.get("baseline_file")
    recent_file = recent_file or cfg.get("recent_file")
    log_dir = log_dir or cfg.get("log_dir")
    out_dir = out_dir or cfg.get("out_dir")
    files_dir = files_dir or cfg.get("files_dir")
    drift_threshold = drift_threshold or cfg.get("drift_threshold")
    drift_interval = drift_interval or cfg.get("drift_interval")
    run_online_trainer(
        csv=csv,
        flight_host=flight_host,
        flight_port=flight_port,
        model=model,
        batch_size=batch_size,
        lr=lr,
        lr_decay=lr_decay,
        flight_path=flight_path,
        baseline_file=baseline_file,
        recent_file=recent_file,
        log_dir=log_dir,
        out_dir=out_dir,
        files_dir=files_dir,
        drift_threshold=drift_threshold,
        drift_interval=drift_interval,
    )


@app.command("drift-monitor")
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
    cfg = _cfg(ctx)
    baseline_file = baseline_file or cfg.get("baseline_file")
    recent_file = recent_file or cfg.get("recent_file")
    drift_threshold = (
        drift_threshold
        if drift_threshold is not None
        else cfg.get("drift_threshold", 0.2)
    )
    model_json = model_json or cfg.get("model_json", Path("model.json"))
    log_dir = log_dir or cfg.get("log_dir")
    out_dir = out_dir or cfg.get("out_dir")
    files_dir = files_dir or cfg.get("files_dir")
    drift_scores = drift_scores or cfg.get("drift_scores")
    flag_file = flag_file or cfg.get("flag_file")
    if baseline_file is None or recent_file is None:
        raise typer.BadParameter("baseline_file and recent_file must be provided")
    run_drift_monitor(
        baseline_file=baseline_file,
        recent_file=recent_file,
        drift_threshold=drift_threshold,
        model_json=model_json,
        log_dir=log_dir,
        out_dir=out_dir,
        files_dir=files_dir,
        drift_scores=drift_scores,
        flag_file=flag_file,
    )


__all__ = ["app"]
