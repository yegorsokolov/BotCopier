# BotCopier

Python utilities for observing live trading activity and exercising strategy logic without relying on MetaTrader.

## Installation

Install the package and its dependencies:

```bash
pip install .
```

Optionally include GPU extras:

```bash
pip install .[gpu]
```

## Usage

After installation, several command line tools are available:

- `botcopier-analyze-ticks` – compute metrics from exported tick history.
- `botcopier-flight-server` – start an Arrow Flight server for trade and metric batches.
- `botcopier-online-trainer` – continuously update a model from streaming events.
- `botcopier-serve-model` – expose the distilled model via a FastAPI service.

Example:

```bash
botcopier-serve-model --host 0.0.0.0 --port 8000
```

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
``StrategyTemplate.mq4`` by ``botcopier/scripts/generate_mql4_from_model.py`` for use in
MetaTrader.

## Memory usage

The log loading helpers in the training pipeline (`botcopier.cli train`) and
`scripts/model_fitting.py` accept a `chunk_size` argument. Providing a positive
value streams DataFrame chunks instead of materialising the entire log in
memory, enabling training on machines with limited RAM.

## License

This project is licensed under the [MIT License](LICENSE).
