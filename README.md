# MT4 Observer + Learner

This project provides a skeleton framework for creating an "Observer" Expert Advisor (EA) that monitors other bots trading in the same MetaTrader 4 account.  It logs trade activity, exports data for learning and can generate candidate strategy EAs based on that information.

The EA records trade openings, closings and order modifications through the `OnTradeTransaction` callback, updating an internal ticket-state map so changes are captured immediately. `OnTick` is reserved for light housekeeping such as CPU-load sampling.

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
  - `train_target_clone.py` – trains a model from exported logs. Its
    `detect_resources()` helper reports available CPU, memory, GPU and free disk
    space via `disk_gb`.  When resources are constrained (e.g. <5 GB free disk or
    low RAM) the script falls back to a lightweight logistic-regression path.
    On more powerful machines it automatically enables heavy indicators such as
    SMA, RSI, MACD and ATR and may select deeper transformer models.  The chosen
    mode and feature flags are written to `model.json` for reproducibility.
  - `generate_mql4_from_model.py` – renders a new EA from a trained model description.
  - `evaluate_predictions.py` – evaluate predictions producing accuracy,
    Sharpe/Sortino ratios and expectancy.
  - `promote_best_models.py` – selects top models by metric (e.g. Sharpe or
    Sortino) and copies them to a best directory.
  - `plot_metrics.py` – plot metric history using Matplotlib.
  - `plot_feature_importance.py` – display SHAP feature importances saved in `model.json`.
  - `nats_stream.py` – proxy that publishes trade and metric events to NATS JetStream.
  - `nats_publisher.py` – send a single encoded event to NATS JetStream.
  - `nats_consumer.py` – persist NATS events to CSV and/or SQLite.
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
   On Linux, logs default to `$XDG_DATA_HOME/botcopier/logs` (or `~/.local/share/botcopier/logs` if `XDG_DATA_HOME` is unset).
2. Train a model from the collected logs:
   ```bash
   python scripts/train_target_clone.py --data-dir logs --out-dir models
   ```
   To add rolling correlations or price ratios with peer symbols include
   pairs via ``--corr-symbols``:

   ```bash
   python scripts/train_target_clone.py --data-dir logs --out-dir models \
       --corr-symbols EURUSD:USDCHF
   ```
   The script inspects CPU, RAM, GPU and disk space and enables advanced
   indicators or deep models only when sufficient resources are available.
   On constrained VPS instances it defaults to a lean configuration with a
   small model and minimal features, so manual `--use-foo` flags are no longer
   required. Extracted features are cached in `out_dir/features.parquet` along
   with the `feature_names` and `last_event_id`. When these match the existing
   model, subsequent runs reuse the cache and skip feature generation for faster
   reproducible training. Use the `--half-life-days` flag to weight recent trades
   more heavily via an exponential decay where each sample weight is
   `0.5 ** (age_days / half_life_days)` (set to `0` to disable). The script also
   checks for class imbalance and fits `LogisticRegression` with
   `class_weight='balanced'` when necessary. The selected decay half-life and
   class-weighting strategy are stored in `model.json` so future training runs and online
   updates apply the same weighting automatically.
3. Generate an EA from the trained model:
   ```bash
   python scripts/generate_mql4_from_model.py models/model.json experts
   ```
4. Evaluate the model against new trades:
   ```bash
   python scripts/evaluate_predictions.py predictions.csv logs/trades.csv
   ```
   The JSON summary now includes Sharpe and Sortino ratios along with
   expectancy to aid model comparison.
5. Promote the top performing models:
   ```bash
   python scripts/promote_best_models.py models --dest models/best
   ```

## Modular Structure

The Python tooling is organised into separate modules for clarity and reuse:

- `scripts/features.py` implements `_extract_features` for transforming raw trade rows into feature dictionaries.
- `scripts/model_fitting.py` exposes `fit_logistic_regression` which trains a simple classifier on feature matrices.
- `scripts/evaluation.py` contains evaluation helpers such as `evaluate` for comparing predictions to actual trades.

Each module has accompanying unit tests and they can be composed for end-to-end training and analysis.

## Rollback and Safety Limits

`scripts/bandit_router.py` routes between multiple generated models and
persists win/loss counts in `bandit_state.json`. Removing or replacing this
file rolls the router back to a fresh state if behaviour degrades. The EA
continues to enforce its own risk controls through inputs like `MinLots`,
`MaxLots`, `BreakEvenPips` and `TrailingPips` so model selection errors remain
bounded.

## Online Training

`scripts/online_trainer.py` keeps a model updated as new trades arrive.  It can
tail ``logs/trades_raw.csv`` or consume newline-delimited JSON records from a
socket and applies :func:`sklearn.linear_model.SGDClassifier.partial_fit`
after each batch.  When the coefficients change the updated ``model.json`` is
written and ``generate_mql4_from_model.py`` is invoked to rebuild the strategy
source.  Set ``ReloadModelInterval`` on the EA so it periodically reloads the
file from the terminal's ``Files`` directory and continues trading with the new
parameters.

## Federated Experience Buffer

The `scripts/federated_buffer.py` module implements a lightweight gRPC
service that allows multiple learners to share experience batches without
revealing individual trades. Each client compresses its local experience
tuples and uploads them to a central server. The server aggregates the
data using secure averaging with a small amount of Gaussian noise and
returns the merged buffer to clients.

Run the buffer server:

```bash
python scripts/federated_buffer.py server --address 127.0.0.1:50051
```

Clients can periodically sync their buffers during training by providing
the server address:

```bash
python scripts/train_rl_agent.py --data-dir logs --out-dir models \
    --federated-server 127.0.0.1:50051 --sync-interval 10
```

This setup helps keep individual trade data confidential while enabling a
shared experience pool for more robust models.

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

`uvloop` can be installed (`pip install uvloop` or `pip install '.[uvloop]'`) to accelerate asyncio event loops. `scripts/stream_listener.py` and `scripts/metrics_collector.py` use it automatically when available to reduce queue handling latency.

### Arrow Flight Logging

`Observer_TBot` streams trade and metric events over [Apache Arrow Flight](https://arrow.apache.org/) using schemas defined in `schemas/trades.py` and `schemas/metrics.py`. The EA falls back to a write‑ahead log when the Flight server is unavailable. Start the in-memory server and point clients to it:

```bash
python scripts/flight_server.py
python scripts/stream_listener.py --flight-host 127.0.0.1 --flight-port 8815
```

`stream_listener.py` validates each incoming record batch against the schema and appends it to a local Parquet dataset under `$FLIGHT_DATA_DIR` (default `data/`). `train_target_clone.py` accepts `--flight-uri` (defaulting to `$FLIGHT_URI`) and the dashboard pre-loads data when `FLIGHT_URI` is set.

#### Latency Benchmark

Sending 500 trade events as individual JSON posts took ~0.53s while uploading the same data as a single Arrow Flight batch finished in ~0.002s on this machine.

### Consuming Cap'n Proto Messages

The observer now serialises trade and metric events using
[Cap'n Proto](https://capnproto.org/).  The snippet below demonstrates how to
subscribe to the ``trades`` stream over NATS JetStream and decode messages:

```python
import asyncio
import capnp
import nats

from proto import trade_capnp
from scripts.stream_listener import _decode_event

async def main():
    nc = await nats.connect("nats://127.0.0.1:4222")
    js = nc.jetstream()
    prev = bytearray()

    async def handler(msg):
        _, body = _decode_event(msg.data, prev)
        trade = trade_capnp.TradeEvent.from_bytes(body)
        print(trade.symbol, trade.price)
        await msg.ack()

    await js.subscribe("trades", cb=handler)
    await asyncio.Event().wait()

asyncio.run(main())
```

Replace ``trade_capnp`` with ``metrics_capnp`` and subscribe to the
``metrics`` subject to process metric updates in the same fashion.

## Tracing and Logging

The observer, stream listener and training scripts emit OpenTelemetry traces and JSON logs. Configure exports via environment variables:

- `OTEL_SERVICE_NAME` – service name reported to the collector.
- `OTEL_EXPORTER_OTLP_ENDPOINT` – OTLP HTTP endpoint (e.g. `http://localhost:4318/v1`).
- `OTEL_EXPORTER_JAEGER_AGENT_HOST` and `OTEL_EXPORTER_JAEGER_AGENT_PORT` – send traces to a Jaeger agent.

Collect logs locally with:

```bash
python scripts/log_collector.py
```

Sample Jaeger deployment:

```bash
docker run --rm -it \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 6831:6831/udp -p 6832:6832/udp \
  -p 5778:5778 -p 16686:16686 \
  -p 14268:14268 -p 14250:14250 \
  -p 9411:9411 jaegertracing/all-in-one:1.47
```

To visualise end-to-end request flows via OTLP, enable the Jaeger collector's
OTLP HTTP listener and point the services to it:

```bash
docker run --rm -it \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 4318:4318 -p 16686:16686 \
  jaegertracing/all-in-one:1.47

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1
python scripts/stream_listener.py
python scripts/metrics_collector.py --db metrics.db
```

To run a Zipkin collector for trace visualisation instead:

```
docker run --rm -d -p 9411:9411 openzipkin/zipkin
```

The provided Python utilities emit log lines including ``trace_id`` and ``span_id`` fields for easy correlation with traces.

## Prometheus and Grafana

Expose runtime metrics with Prometheus by starting the collector with a port:

```bash
python scripts/metrics_collector.py --db metrics.db --prom-port 9000
```

Prometheus can then scrape the endpoint using a configuration like:

```yaml
# grafana/prometheus.yml
scrape_configs:
  - job_name: bot_metrics
    static_configs:
      - targets: ['localhost:9000']
```

On Ubuntu the tools are available via APT:

```bash
sudo apt-get install prometheus grafana
sudo cp grafana/prometheus.yml /etc/prometheus/prometheus.yml
sudo systemctl enable --now prometheus grafana-server
```

Prometheus will now collect metrics such as CPU load, book refresh interval,
file write errors and socket errors from the collector. Import
`grafana/metrics_dashboard.json` into Grafana to visualise win rate, drawdown and
these new metrics.

### Anomaly Monitoring

```
python scripts/anomaly_monitor.py --db metrics.db --metric win_rate --method ewma --threshold 3
```

The ``--threshold`` flag controls the alert sensitivity. For EWMA it represents the number of standard deviations from the moving average; for Isolation Forest it maps to the ``contamination`` parameter. Alerts are printed to stdout and optionally emailed when ``--email`` is supplied.




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

`train_target_clone.py` now selects an appropriate model and enabled features
based on a hardware probe, so manual `--model-type` flags are no longer needed.

The detected `mode` and feature flags are stored in `model.json`. Downstream
tools read these to mirror the training environment automatically. For example,
`generate_mql4_from_model.py` enables `--lite-mode` when the model was trained in
lite mode, and `scripts/online_trainer.py` throttles CPU usage more aggressively
on constrained VPS hosts. This hardware-aware workflow lets Ubuntu deployments
scale model complexity and feature usage without extra flags.

Pass ``--regress-sl-tp`` to also fit linear models predicting stop loss and take
profit distances. The coefficients are saved in ``model.json`` and generated
strategies will automatically place orders with these predicted values.

To automatically drop weak predictors, provide ``--prune-threshold``. Features
whose mean absolute SHAP value falls below this threshold are removed and the
classifier is refit on the reduced set. A warning is emitted when pruning
eliminates more than the fraction specified via ``--prune-warn``.

Hyperparameters, model type, decision threshold and feature selection can be
optimised automatically with Bayesian methods when `optuna` is installed. If
`optuna` is missing the script trains with default parameters and continues. The
best trial's parameters and validation score are saved to `model.json` along
with a summary of the search:

```bash
pip install optuna
python scripts/train_target_clone.py --data-dir "C:\\path\\to\\observer_logs" --out-dir models --bayes-steps 50
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

Feature extraction results are cached to ``features.parquet`` and reused on
subsequent runs when the ``feature_names`` and ``last_event_id`` in
``model.json`` match. Use ``--no-cache`` to force recomputation if needed.

### Calendar events

If a ``calendar.csv`` file is present alongside exported logs or supplied via
``--calendar-file``, the training script augments each trade with nearby
economic event information. The CSV requires columns ``time`` (parseable
timestamp), ``impact`` (numeric severity) and an optional ``id`` identifying the
event. Features ``event_flag``, ``event_impact`` and ``calendar_event_id`` (with
one-hot ``event_id_*`` entries) are added for events occurring within the
``--event-window`` minutes around each trade. The generated Expert Advisor uses
``CalendarEventIdAt()`` to expose these IDs via ``GetFeature()``.

When ``--incremental`` is used with the default ``logreg`` model, training
updates the existing classifier in place by calling ``partial_fit`` on batches
of new samples. The class vector ``[0, 1]`` is stored in ``model.json`` so
subsequent runs can keep learning from where the last run left off.

### Lite Mode

Very large log datasets can exhaust memory when processed all at once. On
machines with limited RAM or disk space the training script automatically
streams feature batches and trains an ``SGDClassifier`` incrementally via
``partial_fit``. Heavy extras such as higher time‑frame indicators and SHAP
importance calculations are disabled to minimise resource usage. Peak memory
consumption becomes roughly proportional to the batch size (``_load_logs`` reads
50k rows per batch by default). Batches of 10k–50k rows usually provide a good
balance between memory usage and convergence speed.

Compile the generated MQ4 file and the observer will begin evaluating predictions from that model.

## Reinforcement Learning Fine-Tuning

Supervised models can be further tuned with reinforcement learning. The
``train_rl_agent.py`` helper loads an existing ``model.json`` and fine-tunes it
against historical trade logs using a DQN, C51, QR-DQN or PPO agent from
``stable-baselines3`` (and ``sb3-contrib`` for distributional variants) when
available. Multi-step returns and prioritized replay can be enabled with
``--n-step`` and ``--replay-*`` options, and live metrics may be POSTed to a
feedback service via ``--metrics-url`` for continual fine-tuning.

Sample configurations using prioritized replay buffers are provided in
``configs/prioritized_c51.json`` and ``configs/prioritized_qr_dqn.json``.

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
observer. It can also compare recent feature distributions against a baseline
using Population Stability Index (PSI) or Kolmogorov–Smirnov (KS) statistics.
When the win rate falls below a threshold, drawdown exceeds a limit, or drift
is above ``--drift-threshold``, the script trains a new model with
``train_target_clone.py`` using only events after the previously processed
``last_event_id``.  The marker is stored alongside the trained model so
subsequent runs continue from the most recent data.  After training it
validates the model with ``backtest_strategy.py`` and publishes the updated
``model.json`` only if the backtest shows an improvement over the live metrics.
The calculated drift metric is saved into ``model.json`` for audit purposes.

Example cron job running every 15 minutes:

```cron
*/15 * * * * /path/to/BotCopier/scripts/auto_retrain.py --log-dir /path/to/observer_logs --out-dir /path/to/BotCopier/models --files-dir /path/to/MT4/MQL4/Files --win-rate-threshold 0.4 --drawdown-threshold 0.2 --baseline-file baseline.csv --recent-file recent.csv --drift-threshold 0.2 --uncertain-file /path/to/observer_logs/uncertain_decisions_labeled.csv --uncertain-weight 3.0
```

Example systemd service and timer:

```ini
# /etc/systemd/system/botcopier-auto-retrain.service
[Unit]
Description=BotCopier auto retrain

[Service]
Type=simple
ExecStart=/usr/bin/python /path/to/BotCopier/scripts/auto_retrain.py --log-dir /path/to/observer_logs --out-dir /path/to/BotCopier/models --files-dir /path/to/MT4/MQL4/Files --win-rate-threshold 0.4 --drawdown-threshold 0.2 --uncertain-file /path/to/observer_logs/uncertain_decisions_labeled.csv --uncertain-weight 3.0
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

### Resume Behaviour

The observer keeps track of the last processed event id in
`model_online.json`.  After flushing trade or metric buffers this file is
updated atomically with the current `last_event_id` so that progress can be
resumed after a restart.  On start-up the counter is validated against any
existing log files to avoid duplicating event ids.  When logs are exported the
state file is copied alongside the rotated log files and should be included in
any uploads.

### Run Metadata

On start-up the observer EA writes `run_info.json` in the directory specified by `LogDirectoryName`.  The file records the `CommitHash` input, model version, broker, list of tracked symbols and the MT4 build.

The `scripts/stream_listener.py` helper writes a corresponding `run_info.json` under `logs/` the first time it processes a message.  It captures the host operating system, Python version and available Python libraries.

After completing a trial run commit both `run_info.json` files along with the generated logs so the environment can be reproduced later.

## Real-time Streaming

`Observer_TBot` serialises trade events and periodic metric summaries using
Protobuf and forwards them to a local proxy. The proxy
(`scripts/nats_stream.py`) publishes each payload to NATS JetStream subjects
``trades`` and ``metrics``. The first byte of every message is the schema
version, currently ``1``. Publishers and consumers compare this byte and log a
warning if it does not match their expected version.

Start the proxy and listeners:

```bash
python scripts/nats_stream.py
python scripts/stream_listener.py
python scripts/metrics_collector.py --db metrics.db --http-port 8080 \
    --prometheus-port 8000
```

Publish a single event from a JSON description:

```bash
python scripts/nats_publisher.py trade trade.json
```

Consume events and store them for training:

```bash
python scripts/nats_consumer.py --trades-csv trades.csv --metrics-sqlite metrics.db
```

`stream_listener.py` appends trade events to ``logs/trades_raw.csv`` and metric
updates to ``logs/metrics.csv``. ``metrics_collector.py`` stores snapshots in a
SQLite database and exposes them over HTTP:

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
  During initialisation the observer EA writes a small JSON ``hello`` packet
  containing this version to the shared memory ring. ``stream_listener.py``
  verifies the packet before processing events and exits on mismatch.
  Third-party producers should perform the same handshake and include their
  ``schema_version`` before streaming events; messages with mismatched versions
  are logged and ignored.
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
- For services managed by systemd, view logs with `journalctl -u <service>`. If
  the journal isn't available, check the corresponding `.log` file in the
  project directory.

### Trace Correlation

Traces exported via OpenTelemetry can be inspected to investigate anomalies or
specific trades. First locate the `trace_id` in `logs/metrics.csv` or
`logs/trades_raw.csv`, then query the collector:

```bash
TRACE_ID="your-trace-id"
curl -s "http://localhost:16686/api/traces/$TRACE_ID" | jq '.data[0]'
```

Filter JSON logs for the same trace to see related events:

```bash
rg $TRACE_ID logs/*.json
```

Matching `trace_id` values let you follow a decision from the logs to its span
in Jaeger or Zipkin for deeper analysis.

## Ubuntu Setup and Services

Run `scripts/setup_ubuntu.sh` on an Ubuntu host to install system packages,
Python dependencies and initialise a Wine prefix for MetaTrader. The script
also installs and enables [Chrony](https://chrony.tuxfamily.org/) so the system
clock remains synchronised. You can review synchronisation details in the
output of `chronyc tracking`:

```bash
./scripts/setup_ubuntu.sh
```

The script installs and enables an `online-trainer.service` unit so the
incremental trainer starts automatically at boot. Disable or re-enable it with:

```bash
sudo systemctl disable --now online-trainer.service
sudo systemctl enable --now online-trainer.service
```

Example systemd unit files are provided under `docs/systemd/` for running
`stream_listener.py`, `metrics_collector.py` and the online trainer. The
trainer tails `logs/trades_raw.csv`, applies `partial_fit` on new rows and
rewrites `model.json` so running Expert Advisors reload updated weights. After
copying the listener and collector units to `/etc/systemd/system/`, enable the
services with:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stream-listener.service metrics-collector.service
```

Logs for these services can be viewed with:

```bash
journalctl -u stream-listener.service -u metrics-collector.service -u online-trainer.service
```

Adjust the `WorkingDirectory` in the unit files to match where the repository
is located (e.g. `/opt/BotCopier`).

### Watchdog

The example units enable systemd's watchdog by setting `WatchdogSec=60`. Each
Python service notifies systemd when it has finished initialising and continues
to send heartbeat messages so the manager can restart the service if it stops
responding. The ping interval is derived from the `WATCHDOG_USEC` environment
variable provided by systemd. Tune the watchdog by changing `WatchdogSec` in the
unit files.

## Debugging Tips

- Set `EnableDebugLogging` to `true` to print socket status and feature values.
- Review the Experts and Journal tabs in MT4 to see these messages.

This repository contains only minimal placeholder code to get started.  Extend the MQL4 and Python modules to implement full learning and cloning functionality.

## License

This project is licensed under the [MIT License](LICENSE).

