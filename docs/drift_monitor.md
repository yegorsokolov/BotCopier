# Automated Drift Checks

`scripts/drift_monitor.py` computes Population Stability Index (PSI) and
Kolmogorov–Smirnov (KS) statistics between a baseline feature log and a recent
sample.  When either metric exceeds a threshold the monitor can trigger
`auto_retrain.py` and touch a flag file so other services know drift occurred.
The computed drift metrics are written to both `model.json` and
`evaluation.json` (using `drift_` prefixes) so subsequent tooling can inspect
them.

`scripts/drift_service.py` runs the monitor in a simple loop and is suitable for
`systemd` or container based deployments.  `promote_best_models.py` will skip any
model whose `drift_psi` or `drift_ks` in `evaluation.json` exceed the
`--max-drift` threshold when publishing models.

## Cron

Check for drift each hour and promote a new model when necessary:

```cron
0 * * * * /usr/bin/python3 /opt/BotCopier/scripts/auto_retrain.py \
  --log-dir /opt/BotCopier/logs --out-dir /opt/BotCopier/models \
  --baseline-file /opt/BotCopier/logs/baseline.csv \
  --recent-file /opt/BotCopier/logs/recent.csv \
  --drift-threshold 0.2 >> /var/log/botcopier/retrain.log 2>&1
```

## systemd

Create `/etc/systemd/system/auto-retrain.service`:

```ini
[Unit]
Description=Retrain model on feature drift

[Service]
Type=oneshot
WorkingDirectory=/opt/BotCopier
ExecStart=/usr/bin/python3 scripts/auto_retrain.py \
  --log-dir logs --out-dir models \
  --baseline-file logs/baseline.csv --recent-file logs/recent.csv \
  --drift-threshold 0.2
Environment=PYTHONUNBUFFERED=1
```

Then schedule it with `/etc/systemd/system/auto-retrain.timer`:

```ini
[Unit]
Description=Hourly drift check

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

Enable the timer with:

```
sudo systemctl enable --now auto-retrain.timer
```

The service computes drift between `baseline.csv` and `recent.csv`, retrains the
model when the metric exceeds the threshold and writes the computed
`drift_metric` into both `model.json` and `evaluation.json` for traceability.
