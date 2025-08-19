# Hardware-aware Training

The training utilities automatically inspect the host machine and adapt the
model to available resources.  :mod:`scripts.train_target_clone` samples RAM,
CPU frequency, free disk space and GPU memory before any feature extraction
occurs.  Based on these metrics a **mode** is selected and recorded in
``model.json`` together with a ``feature_flags`` section.

Modes roughly correspond to the following configurations:

| Mode     | Description |
|----------|-------------|
| ``lite`` | Minimal feature set without order‑book inputs.  Designed for low
memory VPS hosts. |
| ``standard``/``heavy`` | Enables technical indicators and order‑book
features when sufficient CPU/GPU resources are present. |
| ``deep``/``rl`` | Allows transformer or reinforcement learning refinements on
powerful hardware. |

Downstream components read these flags to stay in sync:

* :mod:`scripts.generate_mql4_from_model.py` skips embedding order‑book
  functions unless ``feature_flags.order_book`` is ``true``.
* :mod:`scripts.online_trainer.py` filters unsupported features and passes the
  appropriate ``--lite-mode`` flag when regenerating Expert Advisors.

This metadata is persisted in ``model.json`` so that models trained on one
machine can be safely deployed on another with different capabilities.

## Risk-Parity Allocation

Training also estimates a covariance matrix across traded symbols and derives
``risk_parity_weights``.  These weights scale position sizes so that each
symbol contributes equally to portfolio risk, moderating overall exposure
when one market becomes unusually volatile.  The resulting symbol list,
weights and covariance matrix are exported in ``model.json`` and embedded
into generated strategies so live trading mirrors the balanced allocation.

## Observability

Runtime services emit structured logs to the systemd journal when available. Inspect them with:

```
sudo journalctl -u stream-listener.service -f
sudo journalctl -u metrics-collector.service -f
```

Without systemd the services fall back to local ``*.log`` files.

When :mod:`scripts.metrics_collector` is launched with ``--prom-port`` it exposes Prometheus gauges and counters on ``/metrics``. Add the endpoint to your scrape configuration:

```
scrape_configs:
  - job_name: botcopier
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:8001']
```

Grafana dashboards can then be built on top of the Prometheus data. Import ``docs/metrics_dashboard.json`` for a starter layout showing CPU/memory usage, Arrow Flight queue depth and error counters.

To visualise OpenTelemetry traces in Jaeger run the all-in-one image with OTLP ingestion enabled:

```
docker run --rm -p 4318:4318 -p 16686:16686 \
  jaegertracing/all-in-one:latest --collector.otlp.enabled=true
```

Point services at the collector by setting:

```
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

Traces will then appear in the Jaeger UI at ``http://localhost:16686``.

## Active Learning

Strategies exported from this project can highlight trades where the model is
unsure of the correct action. When the absolute difference between the model
probability and the trade threshold is below ``UncertaintyMargin`` a snapshot of
the current feature vector is written to ``uncertain_decisions.csv`` along with
an empty ``label`` column ready for later annotation.

These records can be labeled offline with::

    python scripts/label_uncertain.py

The resulting ``uncertain_decisions_labeled.csv`` is supplied to
``scripts/train_target_clone.py`` via ``--uncertain-file``. During training the
script multiplies the weight of these rows (configurable with
``--uncertain-weight``) so the newly labeled examples influence the next model
more heavily.

This loop—log uncertain decisions, label them and retrain—provides a lightweight
form of active learning that incrementally improves the strategy where it
previously hesitated.

## Decision Replay

When migrating to a more capable VM or heavier model it can be helpful to
re‑evaluate historical trades.  Use ``scripts/replay_decisions.py`` to recompute
probabilities from an archived ``decisions.csv`` against a new ``model.json``.
The script writes ``divergences.csv`` by default and can tag each mismatch with
an explicit sample weight::

    python scripts/replay_decisions.py decisions.csv model.json --weight 2

Feed this back into training to emphasise corrections and scale them further::

    python scripts/train_target_clone.py --replay-file divergences.csv --replay-weight 3

Exported experts also accept ``ReplayDecisions=true``.  When enabled, the EA
scans ``DecisionLogFile`` at start-up and prints any discrepancies between the
old and new probabilities, providing immediate feedback after upgrades.

## Bandit Router

``scripts/bandit_router.py`` exposes a lightweight HTTP service that chooses
which model to trade based on historical win rates.  Each tick the strategy
queries ``/choose`` to obtain the next regime index and then reports trade
outcomes back to ``/reward`` with ``{"model": idx, "reward": 1|0}``.

Rewards are accumulated using Thompson Sampling or UCB and persisted to
``bandit_state.json`` so exploration survives restarts.  Delete the file to
reset the router.

To keep the router running automatically create a systemd unit such as::

    [Unit]
    Description=Model bandit router
    After=network.target

    [Service]
    ExecStart=/usr/bin/python3 /path/to/scripts/bandit_router.py --models 3
    Restart=always

    [Install]
    WantedBy=multi-user.target

On simpler setups an ``@reboot`` cron entry achieves the same effect::

    @reboot /usr/bin/python3 /path/to/scripts/bandit_router.py --models 3 >> bandit.log 2>&1

