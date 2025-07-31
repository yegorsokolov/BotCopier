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
- `models/` – location for generated models.
- `config.json` – example configuration file.

## Installation

1. Install MetaTrader 4 on a Windows machine or VPS.
2. Copy the contents of `experts/` to your terminal's `MQL4\Experts` folder.
3. Copy the `scripts/` directory somewhere accessible with Python 3 installed.
4. Restart the MT4 terminal and compile `Observer_TBot.mq4` using MetaEditor.
5. Attach `Observer_TBot` to a single chart and adjust the extern inputs as needed
   (magic numbers to observe, log directory, lot bounds, etc.).

## External Training

Exported logs can be processed by the Python scripts.  A typical workflow is:

```bash
python train_target_clone.py --data-dir "C:\\path\\to\\observer_logs" --out-dir models
python generate_mql4_from_model.py models/model.json experts
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

Hyperparameters can be optimised automatically when `optuna` is installed:

```bash
pip install optuna
python train_target_clone.py --data-dir "C:\\path\\to\\observer_logs" --out-dir models --optuna-trials 50
```

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

Compile the generated MQ4 file and the observer will begin evaluating predictions from that model.

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

The ``auto_retrain.py`` helper watches the most recent entries in
``metrics.csv`` and triggers a new training run when the rolling win rate drops
below a chosen threshold. After training completes the updated model is
published to the terminal's ``Files`` directory.

An example cron job running every 15 minutes:

```cron
*/15 * * * * /path/to/BotCopier/scripts/auto_retrain.py --log-dir /path/to/observer_logs --out-dir /path/to/BotCopier/models --files-dir /path/to/MT4/MQL4/Files --win-rate-threshold 0.4
```

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

## Real-time Streaming

When `EnableSocketLogging` is enabled the observer EA emits each trade event and
periodic metric summary as newline separated JSON over a TCP socket. Run the
``socket_log_service.py`` helper to capture these messages into a CSV file. The
service now uses ``asyncio`` to handle multiple connections concurrently:

```bash
python scripts/socket_log_service.py --out stream.csv
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
python scripts/metrics_collector.py --db metrics.db --http-port 8080
```

The collector exposes an HTTP endpoint returning the most recent metrics as JSON:

```bash
curl http://127.0.0.1:8080/metrics?limit=10
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

## Running Tests

Install the Python requirements and run `pytest` from the repository root. At a
minimum `numpy`, `scikit-learn` and `pytest` are needed.  The `xgboost`,
`lightgbm` and `catboost` packages are optional if you want to train XGBoost,
LightGBM or CatBoost models.
`stable-baselines3` can be
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

