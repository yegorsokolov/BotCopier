# Observability

## Viewing logs

The Python services write structured logs to the systemd journal when available. Socket send failures and disk write errors are reported here so they can be alerted on. On Ubuntu use `journalctl` to inspect them:

```bash
sudo journalctl -u stream-listener.service -f
sudo journalctl -u metrics-collector.service -f
sudo journalctl -u online-trainer.service -f
```

If systemd is unavailable, the services fall back to local `*.log` files in the working directory.

## Scraping metrics

`metrics_collector.py` exposes Prometheus metrics. Start the service with `--prom-port` and then scrape:

```bash
curl http://localhost:8001/metrics
```

Add the endpoint to your Prometheus configuration:

```yaml
scrape_configs:
  - job_name: botcopier
    static_configs:
      - targets: ['localhost:8001']
```

CPU and memory utilisation metrics are collected using `psutil` and exported alongside bot performance gauges. Arrow Flight queue depth is reported via `bot_metric_queue_depth` and `bot_trade_queue_depth` gauges so backpressure on the message bus is visible.

## Grafana dashboard

Import `metrics_dashboard.json` from the `docs/` directory into Grafana to visualise the Prometheus data. The dashboard includes CPU and memory usage, Arrow Flight queue depth and error counters for socket and file write failures.

## OpenTelemetry exporter

`otel-exporter.yaml` provides a minimal OTLP exporter configuration that forwards traces, metrics and logs to a collector running on `localhost:4318`.
