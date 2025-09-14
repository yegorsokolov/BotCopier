import json
from pathlib import Path

from scripts.promote_strategy import promote


def _write_returns(path: Path, returns):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(r) for r in returns))


def test_promote_keeps_failing_model(tmp_path: Path):
    shadow = tmp_path / "shadow"
    live = tmp_path / "live"
    metrics_dir = tmp_path / "metrics"
    registry = tmp_path / "models" / "active.json"

    bad_model = shadow / "bad"
    _write_returns(bad_model / "oos.csv", [0.1, -0.6, 0.05])

    promote(shadow, live, metrics_dir, registry, max_drawdown=0.2, max_risk=0.2)

    assert (bad_model).exists(), "failing model should remain in shadow"
    assert not (live / "bad").exists()
    data = json.loads((metrics_dir / "risk.json").read_text())
    assert "bad" in data
    assert "bad" not in json.loads(registry.read_text())


def test_promote_moves_successful_model(tmp_path: Path):
    shadow = tmp_path / "shadow"
    live = tmp_path / "live"
    metrics_dir = tmp_path / "metrics"
    registry = tmp_path / "models" / "active.json"

    good_model = shadow / "good"
    _write_returns(good_model / "oos.csv", [0.05, 0.04, 0.03])

    promote(shadow, live, metrics_dir, registry, max_drawdown=0.2, max_risk=0.2)

    assert not (shadow / "good").exists(), "successful model moved from shadow"
    assert (live / "good").exists()
    reg = json.loads(registry.read_text())
    assert reg["good"] == str(live / "good")
    data = json.loads((metrics_dir / "risk.json").read_text())
    assert "good" in data


def test_promote_rejects_budget_overuse(tmp_path: Path):
    shadow = tmp_path / "shadow"
    live = tmp_path / "live"
    metrics_dir = tmp_path / "metrics"
    registry = tmp_path / "models" / "active.json"

    model = shadow / "over_budget"
    _write_returns(model / "oos.csv", [0.6, 0.6])

    promote(
        shadow,
        live,
        metrics_dir,
        registry,
        max_drawdown=1.0,
        max_risk=1.0,
        budget_limit=1.0,
    )

    assert (model).exists()
    report = json.loads((metrics_dir / "risk.json").read_text())
    assert report["over_budget"]["budget_utilisation"] > 1.0


def test_promote_rejects_order_mismatch(tmp_path: Path):
    shadow = tmp_path / "shadow"
    live = tmp_path / "live"
    metrics_dir = tmp_path / "metrics"
    registry = tmp_path / "models" / "active.json"

    model = shadow / "bad_orders"
    _write_returns(model / "oos.csv", [0.1, 0.1])
    orders_file = model / "orders.csv"
    orders_file.parent.mkdir(parents=True, exist_ok=True)
    orders_file.write_text("market\nlimit\n")

    promote(
        shadow,
        live,
        metrics_dir,
        registry,
        max_drawdown=1.0,
        max_risk=1.0,
        allowed_order_types=["market"],
    )

    assert (model).exists()
    report = json.loads((metrics_dir / "risk.json").read_text())
    assert report["bad_orders"]["order_type_compliance"] < 1.0
