# MT4 Observer + Learner

This project provides a skeleton framework for creating an "Observer" Expert Advisor (EA) that monitors other bots trading in the same MetaTrader 4 account.  It logs trade activity, exports data for learning and can generate candidate strategy EAs based on that information.

The EA records trade openings and closings using the `OnTradeTransaction` callback when available. Platforms without this callback fall back to scanning orders each tick so logs remain accurate.

## Directory Layout

- `experts/` – MQL4 source files.
  - `Observer_TBot.mq4` – main observer EA.
  - `StrategyTemplate.mq4` – template used for generated strategies.
    Trade volume is automatically scaled between `MinLots` and `MaxLots`
    based on the model probability. Stop management can move the
    stop loss to break-even after `BreakEvenPips` profit and optionally
    trail by `TrailingPips`.
  - `model_interface.mqh` – shared structures.
- `scripts/` – helper Python scripts.
  - `train_target_clone.py` – trains a model from exported logs.
  - `generate_mql4_from_model.py` – renders a new EA from a trained model description.
  - `evaluate_predictions.py` – basic log evaluation utility.
  - `promote_best_models.py` – selects top models by metric and copies them to a best directory.
  - `plot_metrics.py` – plot metric history using Matplotlib.
  - `plot_feature_importance.py` – display SHAP feature importances saved in `model.json`.
  - `grpc_log_service.py` – gRPC server receiving trade and metric logs.
- `models/` – location for generated models.
- `config.json` – example configuration file.

## Installation

1. Install MetaTrader 4 on a Windows machine or VPS.
2. Copy the contents of `experts/` to your terminal's `MQL4\Experts` folder.
3. Copy the `scripts/` directory somewhere accessible with Python 3 installed.
4. Restart the MT4 terminal and compile `Observer_TBot.mq4` using MetaEditor.
5. Attach `Observer_TBot` to a single chart and adjust the extern inputs as needed
   (magic numbers to observe, log directory, lot bounds, etc.).

## Workflow

1. Run `Observer_TBot` inside MetaTrader to capture trade activity in `logs/`.
2. Train a model from the collected logs:
   ```bash
   python scripts/train_target_clone.py --data-dir logs --out-dir models
   ```
3. Generate an EA from the trained model:
   ```bash
   python scripts/generate_mql4_from_model.py models/model.json experts
   ```
4. Evaluate the model against new trades:
   ```bash
   python scripts/evaluate_predictions.py predictions.csv logs/trades.csv
   ```
5. Promote the top performing models:
   ```bash
   python scripts/promote_best_models.py models --dest models/best
   ```

## Dashboard

A small web dashboard can display trades, metrics and training progress in real time.

1. Install dependencies if not already done:
   ```bash
   pip install -r requirements.txt
   ```
2. Set an API token used for authentication:
   ```bash
   export DASHBOARD_API_TOKEN="your-token"
   ```
3. Start the dashboard server:
   ```bash
   python dashboard/server.py
   ```
4. Launch the stream listener and forward events to the dashboard:
   ```bash
   python scripts/stream_listener.py --ws-url ws://localhost:8000 --api-token "$DASHBOARD_API_TOKEN"
   ```
5. Open <http://localhost:8000> in a browser and supply the token when prompted to view live updates.



### DVC Pipeline

The `observer_logs/` and `models/` directories are tracked with [DVC](https://dvc.org). A basic workflow:

```bash
# reproduce training and upload logs
dvc repro train_pipeline
# push artifacts and metadata
dvc push
# verify data hashes
dvc status
```
## External Training

Exported logs can be processed by the Python scripts.  A typical workflow is:

```bash
python train_target_clone.py --data-dir "C:\\path\\to\\observer_logs" --out-dir models
python generate_mql4_from_model.py models/model.json experts
```

The training step saves mean absolute SHAP values for each feature under
`feature_importance` in `model.json`. Visualise these importances with:

```bash
python plot_feature_importance.py models/model.json
```

Multiple models can be supplied to build a simple ensemble. Feature names are
merged and the generated EA averages the probabilities from each model:

```bash
python generate_mql4_from_model.py models/model_a.json models/model_b.json experts
```

Pass `--model-type catboost` to train a CatBoost model when the `catboost`
package is installed.

Pass ``--regress-sl-tp`` to also fit linear models predicting stop loss and take
profit distances. The coefficients are saved in ``model.json`` and generated
strategies will automatically place orders with these predicted values.

To automatically drop weak predictors, provide ``--prune-threshold``. Features
whose mean absolute SHAP value falls below this threshold are removed and the
classifier is refit on the reduced set. A warning is emitted when pruning
eliminates more than the fraction specified via ``--prune-warn``.

Hyperparameters, model type, decision threshold and feature selection can be
optimised automatically when `optuna` is installed. If `optuna` is missing the
script trains with default parameters and continues. The best trial's
parameters and validation score are saved to `model.json` along with a summary
of the study:

```bash
pip install optuna
python scripts/train_target_clone.py --data-dir "C:\\path\\to\\observer_logs" --out-dir models --optuna-trials 50
```

Optuna explores learning rate, tree depth or regularisation strength depending
on the chosen model type.

For ongoing training simply rerun the script with the ``--incremental`` flag and
the latest log directory. The generated MQ4 file name will include the training
timestamp so previous versions remain intact:

```bash
python train_target_clone.py --data-dir "C:\\path\\to\\observer_logs" --out-dir models --incremental
python generate_mql4_from_model.py models/model.json experts
```

Pass ``--cache-features`` to reuse the previously extracted feature matrix when
running incrementally. This avoids reprocessing large log files as long as the
configured features match the cached ``feature_names``.

When ``--incremental`` is used with the default ``logreg`` model, training
updates the existing classifier in place by calling ``partial_fit`` on batches
of new samples. The class vector ``[0, 1]`` is stored in ``model.json`` so
subsequent runs can keep learning from where the last run left off.

### Lite Mode

Very large log datasets can exhaust memory when processed all at once. Use
``--lite-mode`` to stream feature batches and train an ``SGDClassifier``
incrementally via ``partial_fit``. Heavy extras such as higher time‑frame
indicators and SHAP importance calculations are disabled to minimise resource
usage. Peak memory consumption becomes roughly proportional to the batch size
(``_load_logs`` reads 50k rows per batch by default). Batches of 10k–50k rows
usually provide a good balance between memory usage and convergence speed.

Compile the generated MQ4 file and the observer will begin evaluating predictions from that model.

## Reinforcement Learning Fine-Tuning

Supervised models can be further tuned with reinforcement learning. The
``train_rl_agent.py`` helper loads an existing ``model.json`` and fine-tunes it
against historical trade logs using a DQN or PPO agent from
``stable-baselines3`` when available.

```bash
pip install stable-baselines3 pandas
python scripts/train_rl_agent.py --data-dir logs --out-dir models --start-model models/model.json --algo ppo --training-steps 1000
```

The script writes an updated ``model.json`` containing the RL-enhanced weights
along with ``training_steps`` and ``avg_reward`` metadata.

## Model Reloading

Strategies built from ``StrategyTemplate.mq4`` can reload their parameters
without recompilation. After new training completes run:

```bash
python scripts/publish_model.py models/model.json /path/to/MT4/MQL4/Files
```

Set the ``ReloadModelInterval`` input on the EA to the desired number of
seconds. When the file ``model.json`` in the terminal's ``Files`` directory is
updated the strategy will automatically load the new coefficients and continue
trading with minimal downtime.

## Automated Promotion

The ``promote_best_models.py`` helper can be scheduled to run periodically,
copying the highest-scoring models to ``models/best`` and publishing the top
one to the terminal's ``Files`` directory. An example cron entry running every
30 minutes:

```cron
*/30 * * * * /path/to/BotCopier/scripts/promote_and_publish.sh
```

Set ``ReloadModelInterval`` on the EA so it automatically reloads the published
model when ``model.json`` changes.

## Automatic Retraining

The ``auto_retrain.py`` helper monitors the latest metrics produced by the
observer.  When the win rate falls below a threshold or drawdown exceeds a
limit, the script trains a new model with ``train_target_clone.py`` using only
events after the previously processed ``last_event_id``.  The marker is stored
alongside the trained model so subsequent runs continue from the most recent
data.  After training it validates the model with ``backtest_strategy.py`` and
publishes the updated ``model.json`` only if the backtest shows an improvement
over the live metrics.

Example cron job running every 15 minutes:

```cron
*/15 * * * * /path/to/BotCopier/scripts/auto_retrain.py --log-dir /path/to/observer_logs --out-dir /path/to/BotCopier/models --files-dir /path/to/MT4/MQL4/Files --win-rate-threshold 0.4 --drawdown-threshold 0.2
```

Example systemd service and timer:

```ini
# /etc/systemd/system/botcopier-auto-retrain.service
[Unit]
Description=BotCopier auto retrain

[Service]
Type=simple
ExecStart=/usr/bin/python /path/to/BotCopier/scripts/auto_retrain.py --log-dir /path/to/observer_logs --out-dir /path/to/BotCopier/models --files-dir /path/to/MT4/MQL4/Files --win-rate-threshold 0.4 --drawdown-threshold 0.2
```

```ini
# /etc/systemd/system/botcopier-auto-retrain.timer
[Unit]
Description=Run BotCopier auto retrain every 15 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
Unit=botcopier-auto-retrain.service

[Install]
WantedBy=timers.target
```

Enable with ``systemctl enable --now botcopier-auto-retrain.timer``.

## Metrics Tracking

During operation the EA records a summary line for each tracked model in
`observer_logs/metrics.csv`. Each entry contains the time of capture, the model
identifier (its magic number), the rolling win rate, average profit, trade
count, drawdown and Sharpe ratio calculated over the last
`MetricsRollingDays` days. Old entries beyond `MetricsDaysToKeep` days are
pruned automatically.
The ``plot_metrics.py`` script can be used to visualise these values.

## Maintenance

Logs are written to the directory specified by the EA parameter `LogDirectoryName` (default `observer_logs`).  Periodically archive or clean this directory to avoid large disk usage.  Models placed in the `models/best` folder can be retained for future analysis.
Trade events are stored in a small in-memory buffer before being flushed to `trades_raw.csv` on each timer tick or when the buffer reaches `LogBufferSize` lines.  Set `EnableDebugLogging` to `true` to enable verbose output and force immediate writes for easier debugging.
Metrics entries older than the number of days specified by `MetricsDaysToKeep` (default 30) are removed automatically during log export.

### Run Metadata

On start-up the observer EA writes `run_info.json` in the directory specified by `LogDirectoryName`.  The file records the `CommitHash` input, model version, broker, list of tracked symbols and the MT4 build.

The `scripts/stream_listener.py` helper writes a corresponding `run_info.json` under `logs/` the first time it processes a message.  It captures the host operating system, Python version and available Python libraries.

After completing a trial run commit both `run_info.json` files along with the generated logs so the environment can be reproduced later.

## Real-time Streaming

On start-up the observer EA tries to stream each trade event and periodic
metric summary over gRPC. If the channel cannot be reached it
falls back to writing a SQLite database ``trades_raw.sqlite`` and finally to a
CSV file ``trades_raw.csv``. Configure the destination via the ``GrpcHost``
and ``GrpcPort`` inputs (default ``127.0.0.1:50051``).

Run the ``grpc_log_service.py`` helper to capture RPC messages into CSV files:

```bash
python scripts/grpc_log_service.py --trade-out trades.csv --metrics-out metrics.csv
```

For persistent storage you can instead log directly to a SQLite database using ``sqlite_log_service.py``:

```bash
python scripts/sqlite_log_service.py --db stream.db
```

Query the logs later with the ``sqlite3`` command line tool, for example:

```bash
sqlite3 stream.db "SELECT COUNT(*) FROM logs;"
```

Start ``Observer_TBot`` with the same host and port settings and the CSV will be
populated as trades occur.  If the connection is lost, the EA automatically
attempts to reconnect so streaming can resume without manual intervention.

To forward metric snapshots without creating ``metrics.csv`` set ``StreamMetricsOnly``
to ``true``. Use the ``metrics_collector.py`` helper to store these messages
in a SQLite database:

```bash
python scripts/metrics_collector.py --db metrics.db --http-port 8080 \
    --prometheus-port 8000
```

The collector exposes an HTTP endpoint returning the most recent metrics as JSON:

```bash
curl http://127.0.0.1:8080/metrics?limit=10
```

When ``--prometheus-port`` is supplied the collector also exposes a Prometheus
endpoint at ``/metrics``. Configure Prometheus to scrape it and optionally
import the sample Grafana dashboard under ``grafana/metrics_dashboard.json`` to
visualise win rate, drawdown and error counters. A minimal Prometheus scrape
configuration:

```yaml
scrape_configs:
  - job_name: "bot_metrics"
    static_configs:
      - targets: ["localhost:8000"]
```

To capture newline-delimited JSON from the EA directly, run the
``stream_listener.py`` helper which writes trade events to
``logs/trades_raw.csv`` and metrics to ``logs/metrics.csv``. Each message must
include a matching ``schema_version`` field (default ``1.0`` or overridden via
the ``SCHEMA_VERSION`` environment variable).

All helper scripts are instrumented with [OpenTelemetry](https://opentelemetry.io/)
and export traces using the OTLP protocol.  Point
``OTEL_EXPORTER_OTLP_ENDPOINT`` at a Jaeger or Zipkin collector to correlate
events across components.  The ``trace_id`` of each trade event is also included
in the ``LogTrade`` JSON emitted by the EA for end-to-end tracing.

After each trading session ``upload_logs.py`` packages ``trades_raw.csv``,
``metrics.csv`` and ``model.json`` together with a ``manifest.json`` into a
single ``run_<timestamp>.tar.gz`` archive. The archive is committed to the
repository and the original files are removed locally. The script uses the
``GITHUB_TOKEN`` environment variable for authentication.

### Environment Variables

* ``SCHEMA_VERSION`` – expected message schema version for ``stream_listener.py``.
* ``GITHUB_TOKEN`` – personal access token with ``repo`` scope used by
  ``upload_logs.py`` to push commits.
* ``OTEL_EXPORTER_OTLP_ENDPOINT`` – URL of the OTLP collector (Jaeger or Zipkin).
* ``OTEL_EXPORTER_OTLP_HEADERS`` – optional headers for the OTLP exporter.
* ``OTEL_EXPORTER_OTLP_CERTIFICATE`` – TLS certificate for secure OTLP.
* ``OTEL_SERVICE_NAME`` – override the default service name reported in traces.
* ``OTEL_EXPORTER_JAEGER_AGENT_HOST``/``OTEL_EXPORTER_JAEGER_AGENT_PORT`` –
  send traces to a Jaeger agent instead of OTLP.

### Viewing Traces

`Observer_TBot.mq4`, `stream_listener.py` and the training scripts emit
OpenTelemetry spans that include the trace and span IDs.  Point the Python
components at a collector by setting either ``OTEL_EXPORTER_OTLP_ENDPOINT`` or
``OTEL_EXPORTER_JAEGER_AGENT_HOST``/``OTEL_EXPORTER_JAEGER_AGENT_PORT``.  The
observer EA accepts an ``OtelEndpoint`` input which should reference the same
collector.  A minimal Jaeger setup can be started locally with:

```bash
docker run -it --rm -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one
```

With the environment variables configured the scripts print the active
``trace_id`` and ``span_id``.  The `stream_listener.py` also records these
identifiers alongside each event in ``logs/trades_raw.csv`` and
``logs/metrics.csv`` so spans from the EA, log listener and training phases can
be correlated in Jaeger's web UI.

### Cron Job Example

```cron
0 0 * * * cd /path/to/BotCopier && GITHUB_TOKEN=XXXX python scripts/upload_logs.py
```

## Tick History Export

Historical tick data can be exported for all symbols that appear in the account
history using the ``TickHistoryExporter`` script.  Copy
``experts/TickHistoryExporter.mq4`` to your ``MQL4\Scripts`` directory and run
it from the terminal.  CSV files named ``ticks_SYMBOL.csv`` will be written to
the ``Files`` folder inside ``OutDir`` for the period spanning your trading
activity.  The helper script ``analyze_ticks.py`` can then compute basic
statistics from these files:

```bash
python scripts/analyze_ticks.py observer_logs/ticks_EURUSD.csv
```

## Strategy Backtesting

Evaluate a generated strategy against historical tick data using
``scripts/backtest_strategy.py``. The script extracts the ``MagicNumber`` and
threshold from the MQ4 file, simulates trades on the tick series and writes a
JSON report containing win rate, profit factor, drawdown and Sharpe ratio. When
``--metrics-file`` is supplied the results are also appended to a
``metrics.csv`` file for comparison with live trading:

```bash
python scripts/backtest_strategy.py experts/MyStrategy.mq4 \
    observer_logs/ticks_EURUSD.csv --report backtest.json \
    --metrics-file metrics.csv
```


## Running Tests

Install the Python requirements and run `pytest` from the repository root. At a
minimum `numpy`, `scikit-learn` and `pytest` are needed. The `xgboost`,
`lightgbm`, `catboost` and TensorFlow packages are optional. When these
libraries are unavailable the training scripts emit a warning and fall back to a
lightweight classifier such as logistic regression. `stable-baselines3` can be
installed to experiment with PPO, DQN, A2C or DDPG agents. Continuous-action
methods like DDPG require an environment using a `Box` action space – the
included discrete environment must be adapted for such algorithms:

```bash
pip install numpy scikit-learn pytest xgboost lightgbm catboost stable-baselines3 shap
pytest
```

## Feature Importance Visualization

Training stores SHAP-based feature importances in `model.json`. Create a bar chart with Matplotlib:

```python
import json
import matplotlib.pyplot as plt

with open("models/model.json") as f:
    data = json.load(f)

fi = data.get("feature_importance", {})
plt.bar(range(len(fi)), list(fi.values()), tick_label=list(fi.keys()))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

## Troubleshooting

- Ensure the MT4 terminal has permission to write files in `MQL4\Files`.
- When running Python scripts, verify the paths to log files and models are correct.
- Use the Experts and Journal tabs inside MT4 for additional debugging information.

## Debugging Tips

- Set `EnableDebugLogging` to `true` to print socket status and feature values.
- Review the Experts and Journal tabs in MT4 to see these messages.

This repository contains only minimal placeholder code to get started.  Extend the MQL4 and Python modules to implement full learning and cloning functionality.

## License

This project is licensed under the [MIT License](LICENSE).

