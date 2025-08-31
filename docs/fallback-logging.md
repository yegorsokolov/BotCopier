# Fallback Logging

Observer_TBot retries sending trade and metric messages to the Flight server. After a configurable number of failed attempts (default 5), messages are redirected to a local fallback log.

The bot first tries to forward the original CSV/JSON entry to `systemd` via `systemd-cat`. If `systemd` is unavailable, the entry is appended to `~/.local/share/botcopier/logs/trades_fallback.log` or `metrics_fallback.log`.

Metrics include a `fallback_logging` flag so downstream monitoring can detect when the exporter is writing to the fallback log instead of Flight.

During startup, any pending messages are recovered from write-ahead logs (WAL). Each WAL record carries a checksum; corrupt entries are skipped and a warning is printed. After a message is successfully sent to the Flight server, a checkpoint marker is written so that if the process stops mid-replay, restart will resume from the last confirmed record.

## Viewing logs on Ubuntu

On an Ubuntu VPS, fallback entries written to journald can be retrieved with:

```bash
sudo journalctl -t botcopier-trades
sudo journalctl -t botcopier-metrics
```

Replace the tag with the desired stream. If journald is not available, review the log files in `~/.local/share/botcopier/logs/`.
