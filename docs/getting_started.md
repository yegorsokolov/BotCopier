# Getting Started

Follow these steps to set up the BotCopier project locally and exercise the
Typer-based CLI against the bundled sample data.

## Prerequisites

* Python 3.10 or newer (matching the ``pyproject.toml`` requirement).
* A virtual environment (``python -m venv .venv``) is strongly recommended.
* Optional: Docker if you plan to run the observability stack defined in
  ``docs/docker-compose.yml``.

## Installation

1. Clone the repository and change into the directory.
2. Create and activate a virtual environment.
3. Install the project in editable mode together with the developer extras:
   ```bash
   pip install -e .
   ```
4. Install the documentation toolchain and notebook helper used in the
   repository:
   ```bash
   pip install "mkdocs>=1.5" "mkdocstrings[python]" "pymdown-extensions" \
               "plantuml-markdown" nbstripout
   ```
5. Enable the ``pre-commit`` hooks (formatters, static analysis, ``nbstripout``):
   ```bash
   pre-commit install
   ```

Training runs snapshot the dependency versions to ``dependencies.txt`` inside
each output directory. Reinstall from that file to reproduce historic runs:

```bash
pip install -r dependencies.txt
```

## First run with the sample data

The ``notebooks/data`` directory ships with a minimal ``trades_raw.csv`` and a
matching ``predictions.csv`` file. Use them to try the CLI locally:

```bash
# Train a lightweight model using the Typer CLI
botcopier train notebooks/data ./artifacts --model-type logreg --random-seed 7

# Inspect the produced model card
head -n 20 ./artifacts/model_card.md
```

Evaluate the bundled predictions against the same log:

```bash
botcopier evaluate notebooks/data/predictions.csv notebooks/data/trades_raw.csv --window 900
```

Both commands write their configuration snapshot to ``params.yaml`` and print a
JSON metrics summary so you can track changes between runs.

When you have exported raw tick history from MetaTrader, compute quick summary
statistics directly from the CLI:

```bash
botcopier analyze-ticks notebooks/data/ticks.csv --interval hourly
```

## Explore the notebooks

Open the notebooks in ``notebooks/`` for an executable walk-through of the same
workflow. ``nbstripout`` is configured in ``.pre-commit-config.yaml`` to keep the
outputs clean when you save the notebooks.

## Documentation

Build the documentation locally with live reload:

```bash
mkdocs serve
```

CI invokes ``mkdocs build --strict`` on every pull request; run it locally
before pushing to catch warnings:

```bash
mkdocs build --strict
```

## Running tests

Execute the automated tests to ensure everything is working:

```bash
pytest
```

For a broader smoke test, run the integration suite and property-based tests
when time permits:

```bash
pytest -m integration
HYPOTHESIS_MAX_EXAMPLES=25 pytest tests/property
```
