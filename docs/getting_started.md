# Getting Started

Follow these steps to set up the BotCopier project locally and run a complete
pipeline against the bundled sample data.

## Prerequisites

* Python 3.9 or newer.
* A virtual environment (``venv`` or ``conda``) is strongly recommended.
* Optional: Docker if you plan to run the observability stack defined in
  ``docs/docker-compose.yml``.

## Installation

1. Clone the repository and change into the directory.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install the documentation toolchain (needed for MkDocs builds):
   ```bash
   pip install "mkdocs>=1.5" "mkdocstrings[python]" "pymdown-extensions"
   ```
   Training runs snapshot the environment to ``dependencies.txt`` in their output
   directories. Reinstall from this file to reproduce the exact package
   versions:
   ```bash
   pip install -r dependencies.txt
   ```

## Using the Sample Data

A compact CSV sample is provided at ``tests/fixtures/trades_small.csv``. Create a
working directory with the expected filenames:

```bash
mkdir -p /tmp/botcopier-sample
cp tests/fixtures/trades_small.csv /tmp/botcopier-sample/trades_raw.csv
```

You can now execute the training pipeline end-to-end:

```bash
python -m botcopier.training.pipeline /tmp/botcopier-sample ./artifacts \
  --model-type logreg \
  --metrics accuracy f1 \
  --random-seed 123
```

The pipeline writes model checkpoints, evaluation metrics, and a model card into
``./artifacts``. Inspect ``artifacts/model_card.md`` for a human readable
summary of the run.

## Documentation

Build the documentation locally with:
```bash
mkdocs serve
```
This launches a local server with live reload. To produce the static site that
is uploaded in CI, run ``mkdocs build``.

## Running Tests

Execute the test suite to ensure everything is working:
```bash
pytest
```
Consider running ``pytest -m slow`` periodically to exercise the more
computationally expensive integration tests.
