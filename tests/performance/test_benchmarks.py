from pathlib import Path

from botcopier.data.loading import _load_logs
from botcopier.features.engineering import _extract_features


def test_load_logs_benchmark(benchmark):
    """Benchmark the log loading routine."""
    data_file = Path("tests/fixtures/trades_small.csv")
    benchmark(lambda: _load_logs(data_file))


def test_feature_engineering_benchmark(benchmark):
    """Benchmark feature extraction."""
    data_file = Path("tests/fixtures/trades_small.csv")
    df, feature_cols, _ = _load_logs(data_file)
    benchmark(lambda: _extract_features(df.copy(), feature_names=list(feature_cols)))
