# Model Serving

`botcopier.scripts.serve_model` runs a FastAPI application that loads `model.json` and exposes a `/predict` endpoint.
The endpoint accepts a batch of feature vectors and returns probabilities from the model.

## Run locally

```bash
botcopier-serve-model --host 0.0.0.0 --port 8000 --metrics-port 8004
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[0, 0, 0, 0], [1, 1, 1, 1]]}'
```

## Docker

A Docker target named `serve-model` is provided for deployment:

```bash
docker build -f Dockerfile.ubuntu --target serve-model -t botcopier-model .
docker run -p 8000:8000 botcopier-model
```

Metrics are exposed on `/metrics` at the configured `--metrics-port`.
