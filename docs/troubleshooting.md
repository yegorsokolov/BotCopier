# Troubleshooting

## Common Issues

### Missing Logs
Ensure the log directory is writable and that agents emit telemetry. Verify network paths to the OTel collector.

### Build Failures
Run `pre-commit run --all-files` and `pytest` locally before pushing. For documentation errors, execute `mkdocs build` to see detailed warnings.

### Deployment Errors
Confirm that the target machine has the required dependencies listed in `requirements.txt` and that model files exist.

