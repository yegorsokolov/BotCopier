# Testing

Run the full test suite with [`pytest`](https://docs.pytest.org/):

```bash
pytest
```

Build the documentation to validate examples and markdown:
```bash
mkdocs build
```

## Viewing model card artifacts

Training runs write a `model_card.md` summarising parameters and evaluation
metrics. Continuous integration uploads these cards as workflow artifacts. To
inspect them, open the relevant GitHub Actions run and download the
**model-card** artifact from the *Artifacts* section.
