# BotCopier

Python utilities for observing live trading activity and exercising strategy logic without relying on MetaTrader.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Observer

Collect trade events with the observer:

```bash
python observer.py
```

### Strategy runner

Execute strategies against the recorded data:

```bash
python strategy_runner.py --model path/to/model.json
```

### Flight server

Start the Arrow Flight server to accept trade and metric batches and persist
them to SQLite and Parquet:

```bash
python scripts/flight_server.py --host 0.0.0.0 --port 8815
```

Clients such as ``Observer_TBot`` connect to this server.  When the server
cannot be reached, the observer writes CSV lines to
``trades_fallback.csv`` or ``metrics_fallback.csv`` so that no data is lost.

## Testing

Run the unit tests:

```bash
pytest
```

## Memory usage

The log loading helpers in `scripts/train_target_clone.py` and
`scripts/model_fitting.py` accept a `chunk_size` argument. Providing a positive
value streams DataFrame chunks instead of materialising the entire log in
memory, enabling training on machines with limited RAM.

## License

This project is licensed under the [MIT License](LICENSE).
