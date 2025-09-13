# Model Explanations

`botcopier.scripts.explain_model` generates feature attribution reports using
three techniques:

- **SHAP** values (falling back to coefficient based estimates when SHAP is not
  available)
- **Integrated Gradients** for linear models
- **Permutation importance** based on sklearn

The training pipeline automatically runs this script after fitting a model. A
Markdown report together with an HTML version is stored under
`reports/explanations/` inside the output directory. The relative path to the
report is also recorded in `model.json` under the key `explanation_report`.

To run the script standalone on an existing `model.json` and training data:

```bash
python -m botcopier.scripts.explain_model model.json trades.csv reports/explanations.md
```

The input CSV must contain a label column (prefixed with `label`) and the feature
columns expected by the model.
