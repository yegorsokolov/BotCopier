import importlib.util
import pathlib
import sys
import time

from metrics.aggregator import get_aggregated_metrics, reset_metrics

# Import the orchestrator directly from file to avoid package name clashes
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "strategy_orchestrator", REPO_ROOT / "scripts" / "strategy_orchestrator.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)  # type: ignore[attr-defined]
run_strategies = module.run_strategies


def test_multiple_strategies_run_concurrently():
    reset_metrics()

    def strat_a(data):
        time.sleep(0.2)
        return {"a": data}

    def strat_b(data):
        time.sleep(0.2)
        return {"b": data}

    market_data = [1, 2]

    start = time.time()
    run_strategies({"A": strat_a, "B": strat_b}, market_data)
    duration = time.time() - start

    aggregated = get_aggregated_metrics()
    assert len(aggregated["A"]) == len(market_data)
    assert len(aggregated["B"]) == len(market_data)
    # Two strategies with 0.2s per item would take >0.8s sequentially
    assert duration < 0.7
