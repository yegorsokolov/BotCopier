from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from botcopier.training.pipeline import train as train_pipeline
from ..scripts.evaluation import evaluate as eval_predictions
from ..scripts.online_trainer import run as run_online_trainer
from ..scripts.drift_monitor import run as run_drift_monitor

app = typer.Typer(help="BotCopier unified command line interface")


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to optional config file"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="Logging level"
    ),
) -> None:
    """Configure global options for all commands."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    ctx.obj = {"config": config}


@app.command("train")
def train(
    data_dir: Path = typer.Argument(..., help="Directory containing training logs"),
    out_dir: Path = typer.Argument(..., help="Output directory for the model"),
    model_type: str = typer.Option("logreg", help="Model type"),
    cache_dir: Optional[Path] = typer.Option(None, help="Optional cache directory"),
) -> None:
    """Train a model from trade logs."""
    train_pipeline(data_dir, out_dir, model_type=model_type, cache_dir=cache_dir)


@app.command("evaluate")
def evaluate(
    pred_file: Path = typer.Argument(..., help="CSV file with predictions"),
    actual_log: Path = typer.Argument(..., help="CSV log with actual trades"),
    window: int = typer.Option(60, help="Matching window in seconds"),
    model_json: Optional[Path] = typer.Option(
        None, help="Optional model.json for additional metrics"
    ),
) -> None:
    """Evaluate predictions against actual trade outcomes."""
    stats = eval_predictions(pred_file, actual_log, window, model_json)
    typer.echo(json.dumps(stats, indent=2))


@app.command("online-train")
def online_train(
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
    baseline_file: Path = typer.Option(..., help="Baseline CSV file"),
    recent_file: Path = typer.Option(..., help="Recent CSV file"),
    drift_threshold: float = typer.Option(0.2, help="Drift threshold"),
    model_json: Path = typer.Option(Path("model.json"), help="Path to model.json"),
    log_dir: Path = typer.Option(..., help="Log directory for retrain"),
    out_dir: Path = typer.Option(..., help="Output directory for retrain"),
    files_dir: Path = typer.Option(..., help="Files directory for retrain"),
    drift_scores: Optional[Path] = typer.Option(
        None, help="Optional path to write per-feature drift scores"
    ),
    flag_file: Optional[Path] = typer.Option(
        None, help="Optional file to touch when drift exceeds threshold"
    ),
) -> None:
    """Compute feature drift metrics and trigger retraining when needed."""
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
