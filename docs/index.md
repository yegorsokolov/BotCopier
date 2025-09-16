# BotCopier Documentation

Welcome to the BotCopier documentation. This site covers the system architecture,
module APIs, CLI usage, and troubleshooting tips. The navigation mirrors the
recommended onboarding flow:

* Start with the [Getting Started guide](getting_started.md) to install
  dependencies and run the Typer-based ``botcopier`` CLI against the bundled
  sample data.
* Browse the executable walkthroughs in the [example notebooks](notebooks.md).
* Review [Architecture](architecture.md) and [Data Flow](data_flow.md) to
  understand how the ingestion, training, and deployment services interact.
* Dive into the [API reference](api.md) generated automatically via
  ``mkdocstrings``.

See [Model Serving](serve_model.md) for running the FastAPI prediction service.

```python
>>> from botcopier.training.pipeline import detect_resources
>>> isinstance(detect_resources(), dict)
True

```
