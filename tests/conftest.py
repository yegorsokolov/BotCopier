import pathlib

ALLOWED = {
    "test_generate.py",
    "test_extra_price_features.py",
    "test_generate_mql4_from_model.py",
    "test_batch_backtest.py",
    "test_full_pipeline.py",
    "performance",
    "test_benchmarks.py",
    "test_nats_publisher_async.py",
    "test_online_trainer_async.py",
    "property",
    "test_invariants.py",
}


def pytest_ignore_collect(path, config):
    if pathlib.Path(str(path)).name not in ALLOWED:
        return True
