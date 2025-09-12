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

## Memory considerations

When ``lite_mode`` is ``False`` the loader inspects the file size before
reading. Files larger than a configurable threshold are accessed using a
memory-mapped or ``pyarrow.dataset`` reader. This avoids loading the entire
dataset into RAM but may incur a slight performance penalty for random access.
For smaller files the data is read eagerly into memory which provides faster
subsequent access at the cost of peak memory usage.

