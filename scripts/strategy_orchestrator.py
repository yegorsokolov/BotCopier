"""Orchestrate running multiple strategies in isolated processes.

This module dispatches market data to strategy functions running in
subprocesses.  Data can be shared via ``multiprocessing.Queue`` or a
``multiprocessing.shared_memory.ShareableList`` for zero-copy
broadcasting.  Metrics produced by strategies are funnelled back
through a queue and recorded with :mod:`metrics.aggregator`.

An optional HTTP server exposes the aggregated metrics so that external
systems can monitor strategy performance while the orchestrator is
running.
"""

from __future__ import annotations

import argparse
import importlib
import json
import threading
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import ShareableList
from typing import Any, Callable, Dict, Iterable, Tuple

from metrics.aggregator import add_metric, get_aggregated_metrics
from metrics.server import start_http_server


StrategyFn = Callable[[Any], Any]


def _strategy_worker_queue(strategy_id: str, fn: StrategyFn, data_q: Queue, metric_q: Queue) -> None:
    """Consume market data from ``data_q`` and push metrics to ``metric_q``."""
    for item in iter(data_q.get, None):
        try:
            metric = fn(item)
            if metric is not None:
                metric_q.put((strategy_id, metric))
        except Exception as exc:  # pragma: no cover - defensive
            metric_q.put((strategy_id, {"error": str(exc)}))
    metric_q.put((strategy_id, None))  # signal completion


def _strategy_worker_shm(strategy_id: str, fn: StrategyFn, shm_name: str, length: int, metric_q: Queue) -> None:
    """Read market data from shared memory and report metrics."""
    sl = ShareableList(name=shm_name)
    try:
        for i in range(length):
            item = json.loads(sl[i])
            try:
                metric = fn(item)
                if metric is not None:
                    metric_q.put((strategy_id, metric))
            except Exception as exc:  # pragma: no cover - defensive
                metric_q.put((strategy_id, {"error": str(exc)}))
    finally:
        sl.shm.close()
    metric_q.put((strategy_id, None))


def run_strategies(
    strategies: Dict[str, StrategyFn],
    market_data: Iterable[Any],
    *,
    use_shm: bool = False,
    max_restarts: int = 0,
    serve_metrics: bool = False,
    metrics_host: str = "127.0.0.1",
    metrics_port: int = 8000,
) -> Dict[str, list]:
    """Run ``strategies`` concurrently over ``market_data``.

    Parameters
    ----------
    strategies:
        Mapping of strategy identifiers to callables.
    market_data:
        Iterable of market data items to broadcast to all strategies.
    use_shm:
        If ``True`` market data is placed in a
        :class:`~multiprocessing.shared_memory.ShareableList` and read by
        workers. Otherwise a ``multiprocessing.Queue`` is used.
    max_restarts:
        Number of times a crashing strategy will be restarted.
    serve_metrics:
        If ``True`` expose aggregated metrics via an HTTP endpoint for
        monitoring.
    metrics_host / metrics_port:
        Address for the metrics HTTP server when ``serve_metrics`` is
        enabled.
    """

    server = None
    server_thread: threading.Thread | None = None
    if serve_metrics:
        server, server_thread = start_http_server(metrics_host, metrics_port)

    metric_q: Queue = Queue()
    procs: Dict[str, Process] = {}
    restarts: Dict[str, int] = {sid: 0 for sid in strategies}

    if use_shm:
        data_list = [json.dumps(item) for item in market_data]
        sl = ShareableList(data_list)

        def start_proc(sid: str, fn: StrategyFn) -> None:
            p = Process(target=_strategy_worker_shm, args=(sid, fn, sl.shm.name, len(sl), metric_q))
            p.start()
            procs[sid] = p

        for sid, fn in strategies.items():
            start_proc(sid, fn)
    else:
        cached_market_data = market_data if isinstance(market_data, list) else list(market_data)
        data_queues: Dict[str, Queue] = {}

        def start_proc(sid: str, fn: StrategyFn) -> None:
            q = Queue()
            data_queues[sid] = q
            p = Process(target=_strategy_worker_queue, args=(sid, fn, q, metric_q))
            p.start()
            procs[sid] = p
            for item in cached_market_data:
                q.put(item)
            q.put(None)

        for sid, fn in strategies.items():
            start_proc(sid, fn)

    finished: Dict[str, bool] = {sid: False for sid in strategies}
    total = len(strategies)
    while sum(finished.values()) < total:
        try:
            sid, metric = metric_q.get(timeout=0.1)
        except Exception:
            # Check for crashed processes with no sentinel
            for sid, p in procs.items():
                if not finished[sid] and not p.is_alive():
                    if p.exitcode and restarts[sid] < max_restarts:
                        restarts[sid] += 1
                        start_proc(sid, strategies[sid])
                    else:
                        finished[sid] = True
            continue
        if metric is None:
            p = procs[sid]
            p.join()
            if p.exitcode and restarts[sid] < max_restarts:
                restarts[sid] += 1
                start_proc(sid, strategies[sid])
            else:
                finished[sid] = True
        else:
            add_metric(sid, metric)

    if serve_metrics and server and server_thread:
        server.shutdown()
        server_thread.join()

    if use_shm:
        sl.shm.close()
        sl.shm.unlink()

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
    parser.add_argument("--use-shm", action="store_true", help="Stream market data via shared memory")
    parser.add_argument("--serve-metrics", action="store_true", help="Expose metrics via an HTTP endpoint")
    parser.add_argument("--metrics-port", type=int, default=8000, help="Port for the metrics HTTP server")
    args = parser.parse_args(argv)

    strategies = dict(_parse_strategy(spec) for spec in args.strategies)
    with open(args.data, "r", encoding="utf-8") as fh:
        market_data = [json.loads(line) for line in fh]

    aggregated = run_strategies(
        strategies,
        market_data,
        use_shm=args.use_shm,
        serve_metrics=args.serve_metrics,
        metrics_port=args.metrics_port,
    )
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

