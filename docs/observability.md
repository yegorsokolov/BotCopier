# Observability

## Viewing logs

The Python services write structured logs to the systemd journal when available. On Ubuntu use `journalctl` to inspect them:

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

CPU and memory utilisation metrics are collected using `psutil` and exported alongside bot performance gauges.
