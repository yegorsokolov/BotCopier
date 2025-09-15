# End-to-End Pipeline Tutorial

This tutorial walks through the entire BotCopier pipeline using the bundled
sample data. You will load raw trades, engineer features, train a model, and
publish the resulting artefacts.

## Prerequisites

* A working environment configured according to
  [Getting Started](getting_started.md).
* Sample data copied into ``/tmp/botcopier-sample`` as described below.

## 1. Prepare the Data Directory

The training pipeline expects a directory containing ``trades_raw.csv`` (and
optionally ``depth_raw.csv``). Copy the sample CSV provided with the repository:

```bash
mkdir -p /tmp/botcopier-sample
cp tests/fixtures/trades_small.csv /tmp/botcopier-sample/trades_raw.csv
```

## 2. Run the Training Pipeline

Invoke the pipeline with an output directory for artefacts:

```bash
python -m botcopier.training.pipeline /tmp/botcopier-sample ./tutorial-run \
  --model-type logreg \
  --metrics accuracy f1 sharpe \
  --random-seed 7 \
  --trials 5
```

Key artefacts produced in ``./tutorial-run``:

* ``model.json`` – serialised model parameters registered for deployment.
* ``metrics.json`` – summary of evaluation metrics requested via ``--metrics``.
* ``model_card.md`` – human-readable description of the experiment.
* ``dependencies.txt`` – frozen Python environment used during the run.

## 3. Inspect Feature Engineering Outputs

Intermediate feature sets are cached under ``./tutorial-run/cache`` when a cache
backend is configured. The tutorial run uses the in-memory cache, but you can
persist features by exporting the ``TRAINING_CACHE_DIR`` environment variable:

```bash
export TRAINING_CACHE_DIR=./tutorial-run/cache
python -m botcopier.training.pipeline /tmp/botcopier-sample ./tutorial-run --reuse-controller
```

With caching enabled subsequent runs reuse feature computations and focus on
model search.

## 4. Promote the Model to the Registry

Successful runs automatically write the model artefacts into the registry path
configured by ``TRAINING_MODEL`` (defaults to ``model.json`` in the output
folder). To promote the model to a shared registry, copy the artefact into the
``models/`` directory or publish it to your remote storage bucket:

```bash
cp ./tutorial-run/model.json models/tutorial_logreg.json
```

Update ``params.yaml`` or environment variables to point inference services to
this artefact.

## 5. Verify the Deployment Pipeline

Run the lightweight inference smoke test to ensure the exported model can be
loaded:

```bash
python -m botcopier.models.registry --list
python -m botcopier.models.registry --load models/tutorial_logreg.json
```

These commands use the API definitions referenced in ``docs/api.md``, ensuring
mkdocstrings-generated documentation stays aligned with the actual code.

## Troubleshooting Tips

* Use ``--profile`` to enable cProfile if you need performance diagnostics.
* Set ``--trace`` and ``--trace-exporter=jaeger`` to integrate with the
  OpenTelemetry collector defined in ``docs/otel-collector.yaml``.
* Increase ``--trials`` for a more exhaustive Optuna search when working with
  larger datasets.
