# Feature Plugins

BotCopier exposes a simple registry so that additional feature extraction
functions can be plugged in at runtime.

## Writing a plugin

Create a module on the Python path and register a function with
`botcopier.features.registry.register_feature`:

```python
from botcopier.features.registry import register_feature

def my_features(df, feature_names, **kwargs):
    # add columns to df and append their names
    df["ones"] = 1.0
    feature_names.append("ones")
    return df, feature_names, {}, {}

register_feature("custom", my_features)
# or as a decorator:
# @register_feature("custom")
# def my_features(...):
#     ...
```

Third-party packages can expose features via Python entry points.  In your
``pyproject.toml`` add:

```
[project.entry-points."botcopier.features"]
custom = "my_pkg.my_module:my_features"
```

BotCopier will automatically discover such plugins when they are enabled.
Enable a plugin by passing ``--feature custom`` on the CLI or by listing it
under ``training.features`` in a configuration file.

## Metric plugins

Classification metrics share the same ergonomics. Implement a callable that
accepts ``(y_true, probas, profits=None)`` and register it via
``metrics.registry.register_metric``:

```python
from metrics.registry import register_metric

@register_metric("fancy")
def fancy_metric(y_true, probas, profits=None):
    return 42.0
```

Expose the metric from a third-party package with an entry point:

```
[project.entry-points."botcopier.metrics"]
fancy = "my_pkg.metrics:fancy_metric"
```

Metrics can be selected when training with ``--metric fancy`` or by setting
``training.metrics`` in a configuration file.

Built-in evaluation metrics such as ``accuracy``, ``precision``, ``recall``,
``profit``, ``sharpe_ratio``, ``sortino_ratio``, ``max_drawdown`` and ``var_95``
depend on the optional scientific stack (``numpy``, ``pandas`` and
``scikit-learn``).  When these packages are not installed the evaluation module
raises ``ImportError("optional dependencies not installed")`` rather than
attempting to run with partial functionality.

## Evaluation hooks

Evaluation hooks extend the JSON payload returned by the ``evaluate`` command.
They are simple callables that receive a mutable context dictionary. Register
hooks with ``botcopier.eval.hooks.register_hook`` or publish them via the
``botcopier.eval_hooks`` entry point group:

```python
from botcopier.eval import hooks

@hooks.register_hook("alpha")
def add_alpha(ctx):
    ctx.setdefault("stats", {})["alpha"] = 1.0
```

Or from a plugin package:

```
[project.entry-points."botcopier.eval_hooks"]
alpha = "my_pkg.hooks:add_alpha"
```

Enable hooks on the CLI with ``--eval-hooks alpha`` or set ``training.eval_hooks``
in configuration.
