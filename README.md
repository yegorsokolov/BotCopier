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

The package exposes a consolidated Typer CLI under the ``botcopier`` command.
Inspect the available subcommands with ``botcopier --help``. Common workflows
include:

```bash
# Train a lightweight model against the bundled sample data
botcopier train notebooks/data ./artifacts --model-type logreg --random-seed 7

# Evaluate predictions against the sample trade log
botcopier evaluate notebooks/data/predictions.csv notebooks/data/trades_raw.csv --window 900

# Launch the online trainer in streaming mode
botcopier online-train --csv notebooks/data/trades_raw.csv --model ./models/latest/model.json

# Generate summary metrics from exported tick history
botcopier analyze-ticks notebooks/data/ticks.csv --interval daily
```

## Documentation and notebooks

The MkDocs site, including automatic API reference pages powered by
``mkdocstrings``, lives under ``docs/``. Build the site locally with live reload:

```bash
mkdocs serve
```

Example notebooks with pre-stripped outputs are provided in ``notebooks/``.
Execute them with Jupyter to reproduce the getting started flow.

## Testing

Run the unit tests:

```bash
pytest
```

Run ``pre-commit run --all-files`` to apply formatting and ensure notebook
outputs are stripped before pushing changes.

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

## Contributing

See [docs/contributing.md](docs/contributing.md) for development workflow and
testing expectations. New contributions should document CLI additions and update
the notebooks when appropriate.

## License

This project is licensed under the [MIT License](LICENSE).
