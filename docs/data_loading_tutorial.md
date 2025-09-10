# Data Loading and Feature Engineering

This tutorial demonstrates how to load trade logs and derive features.

```python
>>> from pathlib import Path
>>> from botcopier.data.loading import _load_logs
>>> from botcopier.features.engineering import _extract_features
>>> df, feature_cols, _ = _load_logs(Path('tests/fixtures/trades_small.csv'))
>>> bool(len(df))
True
>>> _, cols, *_ = _extract_features(df, feature_names=list(feature_cols))
>>> set(cols) == set(feature_cols)
True
>>> 
```

