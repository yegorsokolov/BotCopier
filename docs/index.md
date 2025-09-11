# BotCopier Documentation

Welcome to the BotCopier documentation. This site covers the system architecture, module APIs, and troubleshooting tips.

For an overview of how data moves through the platform, see the [architecture](architecture.md) page. API details for core modules live in the [reference](api.md).

See [Model Serving](serve_model.md) for running the FastAPI prediction service.

```python
>>> from botcopier.training.pipeline import detect_resources
>>> isinstance(detect_resources(), dict)
True

```
