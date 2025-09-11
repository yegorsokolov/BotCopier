# Feature Plugins

BotCopier exposes a simple registry so that additional feature extraction
functions can be plugged in at runtime.

## Writing a plugin

Create a module on the Python path and decorate a function with
`botcopier.features.registry.register_feature`:

```python
from botcopier.features.registry import register_feature

@register_feature("custom")
def my_features(df, feature_names, **kwargs):
    # add columns to df and append their names
    df["ones"] = 1.0
    feature_names.append("ones")
    return df, feature_names, {}, {}
```

Enable the plugin when training by passing ``--feature custom`` on the CLI or by
listing it under ``training.features`` in a configuration file.
