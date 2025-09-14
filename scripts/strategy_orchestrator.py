"""Orchestrate running multiple strategies in isolated processes.

Each strategy is executed in its own ``multiprocessing.Process`` and receives
market data via a ``multiprocessing.Queue``. Metrics produced by the strategies
are funnelled back through another queue and recorded with
:mod:`metrics.aggregator`.

The orchestrator can be imported and the :func:`run_strategies` function used in
unit tests. When executed as a script it expects strategy specifications in the
form ``module:function`` and a path to a JSON lines file containing market data
entries.
"""
from __future__ import annotations

import argparse
import importlib
import json
from multiprocessing import Process, Queue
from typing import Callable, Dict, Iterable, Any, Tuple

from metrics.aggregator import add_metric, get_aggregated_metrics


StrategyFn = Callable[[Any], Any]


def _strategy_worker(strategy_id: str, fn: StrategyFn, data_q: Queue, metric_q: Queue) -> None:
    """Consume market data from ``data_q`` and push metrics to ``metric_q``."""
    for item in iter(data_q.get, None):
        try:
            metric = fn(item)
            if metric is not None:
                metric_q.put((strategy_id, metric))
        except Exception as exc:  # pragma: no cover - defensive
            metric_q.put((strategy_id, {"error": str(exc)}))
    metric_q.put((strategy_id, None))  # signal completion


def run_strategies(strategies: Dict[str, StrategyFn], market_data: Iterable[Any]) -> Dict[str, list]:
    """Run ``strategies`` concurrently over ``market_data``.

    Parameters
    ----------
    strategies:
        Mapping of strategy identifiers to callables.
    market_data:
        Iterable of market data items to broadcast to all strategies.
    Returns
    -------
    dict
        Aggregated metrics keyed by strategy identifier.
    """
    data_queues: Dict[str, Queue] = {sid: Queue() for sid in strategies}
    metric_q: Queue = Queue()
    procs: list[Process] = []

    for sid, fn in strategies.items():
        p = Process(target=_strategy_worker, args=(sid, fn, data_queues[sid], metric_q))
        p.start()
        procs.append(p)

    # Broadcast market data to each strategy
    for item in market_data:
        for q in data_queues.values():
            q.put(item)
    for q in data_queues.values():
        q.put(None)  # terminate signal

    finished = 0
    total = len(strategies)
    while finished < total:
        sid, metric = metric_q.get()
        if metric is None:
            finished += 1
        else:
            add_metric(sid, metric)

    for p in procs:
        p.join()

    return get_aggregated_metrics()


def _parse_strategy(spec: str) -> Tuple[str, StrategyFn]:
    """Parse ``module:function`` strategy specifier."""
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    return spec, fn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run strategies concurrently")
    parser.add_argument("strategies", nargs="+", help="Strategy specs in module:function form")
    parser.add_argument("--data", required=True, help="Path to JSON lines market data file")
    args = parser.parse_args(argv)

    strategies = dict(_parse_strategy(spec) for spec in args.strategies)
    with open(args.data, "r", encoding="utf-8") as fh:
        market_data = [json.loads(line) for line in fh]

    aggregated = run_strategies(strategies, market_data)
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
