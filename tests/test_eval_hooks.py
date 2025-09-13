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
