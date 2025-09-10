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

## Model distillation

The training pipeline can fit a high-capacity transformer model and then
distil it into a simple logistic regression.  After the transformer is trained
on rolling feature windows, its probabilities for each training sample are used
as soft targets for a linear student.  The distilled coefficients and teacher
evaluation metrics are saved in ``model.json`` and are embedded into
``StrategyTemplate.mq4`` by ``scripts/generate_mql4_from_model.py`` for use in
MetaTrader.

## Memory usage

The log loading helpers in the training pipeline (`botcopier.cli train`) and
`scripts/model_fitting.py` accept a `chunk_size` argument. Providing a positive
value streams DataFrame chunks instead of materialising the entire log in
memory, enabling training on machines with limited RAM.

## License

This project is licensed under the [MIT License](LICENSE).
