# Evaluation Hooks

BotCopier exposes a lightweight hook system for extending evaluation.  Hooks
are simple functions that accept a mutable context dictionary.  They may compute
additional metrics or perform side effects such as uploading results.

```python
from botcopier.eval import hooks

@hooks.register_hook("my-metric")
def my_metric(ctx):
    ctx.setdefault("stats", {})["my_metric"] = 42
```

Hooks can be executed via the CLI by listing them with `--eval-hooks` or by
setting the `eval_hooks` field in `TrainingConfig`.

```bash
botcopier evaluate preds.csv trades.csv --eval-hooks precision,sharpe
```

Each hook receives the same context object, allowing them to share results.
They are executed sequentially in the order provided.

Third-party packages can expose hooks via the `botcopier.eval_hooks` entry
point group:

```
[project.entry-points."botcopier.eval_hooks"]
precision_plus = "my_pkg.hooks:precision_plus"
```

When listed on the CLI the hook will be discovered and executed alongside the
built-in ones.
