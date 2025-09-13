# Feature Plugins

BotCopier exposes a simple registry so that additional feature extraction
functions can be plugged in at runtime.

## Writing a plugin

Create a module on the Python path and register a function with
`botcopier.features.plugins.register_feature`:

```python
from botcopier.features.plugins import register_feature

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
