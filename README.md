# MT4 Observer + Learner

This project provides a skeleton framework for creating an "Observer" Expert Advisor (EA) that monitors other bots trading in the same MetaTrader 4 account.  It logs trade activity, exports data for learning and can generate candidate strategy EAs based on that information.

The EA records trade openings and closings using the `OnTradeTransaction` callback when available. Platforms without this callback fall back to scanning orders each tick so logs remain accurate.

## Directory Layout

- `experts/` – MQL4 source files.
  - `Observer_TBot.mq4` – main observer EA.
  - `StrategyTemplate.mq4` – template used for generated strategies.
  - `model_interface.mqh` – shared structures.
- `scripts/` – helper Python scripts.
  - `train_target_clone.py` – trains a model from exported logs.
  - `generate_mql4_from_model.py` – renders a new EA from a trained model description.
  - `evaluate_predictions.py` – basic log evaluation utility.
  - `promote_best_models.py` – selects top models by metric and copies them to a best directory.
- `models/` – location for generated models.
- `config.json` – example configuration file.

## Installation

1. Install MetaTrader 4 on a Windows machine or VPS.
2. Copy the contents of `experts/` to your terminal's `MQL4\Experts` folder.
3. Copy the `scripts/` directory somewhere accessible with Python 3 installed.
4. Restart the MT4 terminal and compile `Observer_TBot.mq4` using MetaEditor.
5. Attach `Observer_TBot` to a single chart and adjust the extern inputs as needed (magic numbers to observe, log directory, etc.).

## External Training

Exported logs can be processed by the Python scripts.  A typical workflow is:

```bash
python train_target_clone.py --data-dir "C:\\path\\to\\observer_logs" --out-dir models
python generate_mql4_from_model.py models/model.json experts
```

Compile the generated MQ4 file and the observer will begin evaluating predictions from that model.

## Metrics Tracking

During operation the EA records a summary line for each tracked model in
`observer_logs/metrics.csv`. Each entry contains the time of capture, the model
identifier (its magic number), the hit rate and the profit factor calculated
over the last `MetricsRollingDays` days. Old entries beyond
`MetricsDaysToKeep` days are pruned automatically.

## Maintenance

Logs are written to the directory specified by the EA parameter `LogDirectoryName` (default `observer_logs`).  Periodically archive or clean this directory to avoid large disk usage.  Models placed in the `models/best` folder can be retained for future analysis.
Trade events are stored in a small in-memory buffer before being flushed to `trades_raw.csv` on each timer tick or when the buffer reaches `LogBufferSize` lines.  Set `EnableDebugLogging` to `true` to enable verbose output and force immediate writes for easier debugging.
Metrics entries older than the number of days specified by `MetricsDaysToKeep` (default 30) are removed automatically during log export.

## Real-time Streaming

When `EnableSocketLogging` is enabled the observer EA emits each trade event as a JSON
line over a TCP socket. The helper script ``stream_listener.py`` can convert these
messages into a CSV log in real time:

```bash
python scripts/stream_listener.py --out stream.csv
```

Attach ``Observer_TBot`` in MT4 with the same host and port parameters and the CSV
will be populated as trades occur.

## Running Tests

Install the Python requirements and run `pytest` from the repository root:

```bash
pytest
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

