import json
from pathlib import Path

from scripts import promote_strategy

promote = promote_strategy.promote


def _write_returns(path: Path, returns):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(r) for r in returns))


def _write_slippage(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(v) for v in values))


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
    assert data["bad"]["reasons"]
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
    assert not data["good"]["reasons"]


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
    assert not (live / "over_budget").exists()
    report = json.loads((metrics_dir / "risk.json").read_text())
    assert report["over_budget"]["budget_utilisation"] > 1.0
    assert "budget" in report["over_budget"]["reasons"]
    assert "over_budget" not in json.loads(registry.read_text())


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
    assert not (live / "bad_orders").exists()
    report = json.loads((metrics_dir / "risk.json").read_text())
    assert report["bad_orders"]["order_type_compliance"] < 1.0
    assert "order_type" in report["bad_orders"]["reasons"]
    assert "bad_orders" not in json.loads(registry.read_text())


def test_promote_rejects_high_risk(tmp_path: Path):
    shadow = tmp_path / "shadow"
    live = tmp_path / "live"
    metrics_dir = tmp_path / "metrics"
    registry = tmp_path / "models" / "active.json"

    model = shadow / "too_risky"
    # Large volatility but drawdown within allowed limit
    _write_returns(model / "oos.csv", [0.3, -0.3, 0.3, -0.3])

    promote(
        shadow,
        live,
        metrics_dir,
        registry,
        max_drawdown=0.4,
        max_risk=0.2,
        budget_limit=10.0,
    )

    assert model.exists()
    assert not (live / "too_risky").exists()
    report = json.loads((metrics_dir / "risk.json").read_text())
    assert report["too_risky"]["risk"] > 0.2
    assert "risk" in report["too_risky"]["reasons"]


def test_promote_logs_additional_metrics(tmp_path: Path):
    shadow = tmp_path / "shadow"
    live = tmp_path / "live"
    metrics_dir = tmp_path / "metrics"
    registry = tmp_path / "models" / "active.json"

    model = shadow / "metrics_model"
    _write_returns(model / "oos.csv", [0.1, -0.2, 0.05, -0.4, 0.02])
    _write_slippage(model / "slippage.csv", [0.1, 0.2, 0.1])

    promote(
        shadow,
        live,
        metrics_dir,
        registry,
        max_drawdown=1.0,
        max_risk=1.0,
    )

    report = json.loads((metrics_dir / "risk.json").read_text())
    metrics = report["metrics_model"]
    assert "var_95" in metrics
    assert "volatility_spikes" in metrics
    assert "slippage_mean" in metrics
    assert "slippage_std" in metrics
