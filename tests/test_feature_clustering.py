import json
from pathlib import Path

import numpy as np
import pandas as pd

from botcopier.training.pipeline import train


def _make_dataset(path: Path, n: int = 120) -> Path:
    rng = np.random.default_rng(0)
    spread = rng.normal(size=n)
    volume = spread + 5.0 + rng.normal(scale=0.01, size=n)
    hour = np.arange(n) % 24
    label = (spread > 0).astype(float)
    df = pd.DataFrame(
        {"label": label, "spread": spread, "volume": volume, "hour": hour}
    )
    df.to_csv(path, index=False)
    return path


def test_feature_clustering_reduces_features_and_preserves_accuracy(
    tmp_path: Path,
) -> None:
    data = _make_dataset(tmp_path / "data.csv")

    out_baseline = tmp_path / "baseline"
    train(data, out_baseline, cluster_correlation=1.0)
    model_baseline = json.loads((out_baseline / "model.json").read_text())
    n_features_baseline = len(model_baseline["feature_names"])
    acc_baseline = model_baseline["cv_accuracy"]

    out_clustered = tmp_path / "clustered"
    train(data, out_clustered, cluster_correlation=0.9)
    model_clustered = json.loads((out_clustered / "model.json").read_text())
    n_features_clustered = len(model_clustered["feature_names"])
    acc_clustered = model_clustered["cv_accuracy"]

    assert n_features_clustered < n_features_baseline
    assert abs(acc_baseline - acc_clustered) < 0.05
    assert model_clustered.get("feature_clusters")
