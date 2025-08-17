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

## Observability

Runtime services emit structured logs to the systemd journal when available. Inspect them with:

```
sudo journalctl -u stream-listener.service -f
sudo journalctl -u metrics-collector.service -f
```

Without systemd the services fall back to local ``*.log`` files.

When :mod:`scripts.metrics_collector` is launched with ``--prom-port`` it exposes Prometheus gauges and counters. Add the endpoint to your scrape configuration:

```
scrape_configs:
  - job_name: botcopier
    static_configs:
      - targets: ['localhost:8001']
```

Grafana dashboards can then be built on top of the Prometheus data. Import ``docs/metrics_dashboard.json`` for a starter layout showing CPU/memory usage, Arrow Flight queue depth and error counters.

