# Automated Drift Checks

`scripts/auto_retrain.py` can watch feature logs for population drift. When
PSI or KS statistics exceed a threshold the script retrains the model using
`train_target_clone.py` and publishes the result.

## Cron

Check for drift each hour and promote a new model when necessary:

```cron
0 * * * * /usr/bin/python3 /opt/BotCopier/scripts/auto_retrain.py \
  --log-dir /opt/BotCopier/logs --out-dir /opt/BotCopier/models \
  --files-dir /opt/MT4/MQL4/Files \
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
  --log-dir logs --out-dir models --files-dir /opt/MT4/MQL4/Files \
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
`drift_metric` into `model.json` for traceability.
