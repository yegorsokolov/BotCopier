import importlib.util
import json
import multiprocessing
import os
import pathlib
import sys
import time
import urllib.request

import pytest

from metrics.aggregator import get_aggregated_metrics, reset_metrics
from metrics.server import start_http_server

# Import the orchestrator directly from file to avoid package name clashes
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "strategy_orchestrator", REPO_ROOT / "scripts" / "strategy_orchestrator.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)  # type: ignore[attr-defined]
run_strategies = module.run_strategies


@pytest.mark.parametrize("use_shm", [False, True])
def test_multiple_strategies_run_concurrently(use_shm):
    reset_metrics()

    def strat_a(data):
        time.sleep(0.2)
        return {"a": data}

    def strat_b(data):
        time.sleep(0.2)
        return {"b": data}

    market_data = [1, 2]

    start = time.time()
    run_strategies({"A": strat_a, "B": strat_b}, market_data, use_shm=use_shm)
    duration = time.time() - start

    aggregated = get_aggregated_metrics()
    assert len(aggregated["A"]) == len(market_data)
    assert len(aggregated["B"]) == len(market_data)
    # Two strategies with 0.2s per item would take >0.8s sequentially
    assert duration < 0.7


@pytest.mark.parametrize(
    ("market_data_factory", "use_shm"),
    [
        (lambda: ([1, 2, 3], [1, 2, 3]), False),
        (lambda: ([1, 2, 3], [1, 2, 3]), True),
        (lambda: ([1, 2, 3], (item for item in [1, 2, 3])), False),
    ],
)
def test_all_strategies_receive_complete_market_data(market_data_factory, use_shm):
    expected, market_data = market_data_factory()
    reset_metrics()

    def strat_a(data):
        return {"a": data}

    def strat_b(data):
        return {"b": data}

    run_strategies({"A": strat_a, "B": strat_b}, market_data, use_shm=use_shm)

    aggregated = get_aggregated_metrics()
    assert len(aggregated["A"]) == len(expected)
    assert len(aggregated["B"]) == len(expected)


def test_strategy_restarts_on_crash():
    reset_metrics()
    flag = multiprocessing.Value("i", 1)

    def strat(data, flag=flag):
        if flag.value:
            flag.value = 0
            os._exit(1)
        return {"c": data}

    market_data = [1, 2]
    run_strategies({"C": strat}, market_data, max_restarts=1)
    aggregated = get_aggregated_metrics()
    assert len(aggregated["C"]) == len(market_data)


def test_metrics_http_endpoint():
    reset_metrics()

    def strat(data):
        return {"m": data}

    server, thread = start_http_server(port=0)
    port = server.server_address[1]
    run_strategies({"M": strat}, [1])
    resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics")
    body = json.load(resp)
    assert "M" in body
    server.shutdown()
    thread.join()
