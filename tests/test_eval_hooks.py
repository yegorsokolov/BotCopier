import sys
import types

from importlib.metadata import EntryPoint

from botcopier.eval import hooks


def test_hooks_run_in_order_and_share_context():
    original = hooks._REGISTRY.copy()
    try:
        order = []

        @hooks.register_hook("first")
        def _first(ctx):
            order.append("first")
            ctx["value"] = [1]

        @hooks.register_hook("second")
        def _second(ctx):
            order.append("second")
            ctx["value"].append(2)

        ctx = {}
        hooks.dispatch_hooks(["first", "second"], ctx)

        assert order == ["first", "second"]
        assert ctx["value"] == [1, 2]
    finally:
        hooks._REGISTRY = original


def test_entry_point_hook_executes(monkeypatch):
    original = hooks._REGISTRY.copy()
    try:
        module = types.ModuleType("thirdparty_eval_hook")

        def plugin(ctx):
            ctx.setdefault("stats", {})["thirdparty"] = 99

        module.plugin = plugin
        sys.modules["thirdparty_eval_hook"] = module

        ep = EntryPoint(
            name="thirdparty",
            value="thirdparty_eval_hook:plugin",
            group="botcopier.eval_hooks",
        )
        monkeypatch.setattr(
            hooks,
            "entry_points",
            lambda group=None: [ep] if group == "botcopier.eval_hooks" else [],
        )

        hooks.load_entry_point_hooks(["thirdparty"])
        ctx = {"stats": {}}
        hooks.dispatch_hooks(["thirdparty"], ctx)
        assert ctx["stats"]["thirdparty"] == 99
    finally:
        hooks._REGISTRY = original
        sys.modules.pop("thirdparty_eval_hook", None)
