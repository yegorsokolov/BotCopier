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

## Testing

Run the unit tests:

```bash
pytest
```

## License

This project is licensed under the [MIT License](LICENSE).
