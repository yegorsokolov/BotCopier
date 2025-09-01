import json
from pathlib import Path

from scripts.meta_strategy_audit import aggregate_decisions, update_bandit_state


def _write_logs(tmpdir: Path) -> tuple[Path, Path]:
    decisions = tmpdir / "decisions.csv"
    trades = tmpdir / "trades_raw.csv"
    decisions.write_text(
        """event_id;timestamp;model_version;action;probability;sl_dist;tp_dist;model_idx;regime;chosen;risk_weight;variance;lots_predicted;executed_model_idx;features;trace_id;span_id
1;2024-01-01T00:00:00;v1;buy;0.8;5;10;0;0;1;1;0.1;0;0;;t;s
1;2024-01-01T00:00:00;v1;shadow;0.3;5;10;1;0;0;1;0.1;0;0;;t;s
2;2024-01-01T00:01:00;v1;sell;0.2;5;10;0;0;1;1;0.1;0;0;;t;s
2;2024-01-01T00:01:00;v1;shadow;0.7;5;10;1;0;0;1;0.1;0;0;;t;s
"""
    )
    trades.write_text("decision_id,profit\n1,10\n2,-5\n")
    return decisions, trades


def test_aggregate_and_bandit(tmp_path: Path) -> None:
    decisions, trades = _write_logs(tmp_path)
    metrics = aggregate_decisions(decisions, trades, threshold=0.5)
    by_model = {m["model_idx"]: m for m in metrics}
    assert by_model[0]["chosen"]["profit"] == 5.0
    assert by_model[0]["chosen"]["win_rate"] == 0.5
    assert by_model[1]["shadow"]["profit"] == -5.0

    state_file = tmp_path / "bandit.json"
    update_bandit_state(metrics, state_file)
    state = json.loads(state_file.read_text())
    assert state["total"][0] == by_model[0]["chosen"]["trades"]
    assert state["wins"][0] == by_model[0]["chosen"]["wins"]
