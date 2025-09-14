"""CLI wrapper around :mod:`botcopier.features.indicator_discovery`."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from botcopier.features.indicator_discovery import evolve_indicators


def main(
    csv_file: Path,
    model: Path = Path("model.json"),
    target: Optional[str] = None,
    n_components: int = 3,
) -> None:
    """Evolve symbolic indicators from ``csv_file`` and update ``model``."""

    df = pd.read_csv(csv_file)
    feature_cols = [c for c in df.columns if c != target]
    y = df[target] if target is not None else None
    evolve_indicators(df[feature_cols], feature_cols, y, model, n_components=n_components)


if __name__ == "__main__":
    typer.run(main)
