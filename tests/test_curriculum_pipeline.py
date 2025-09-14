import json
from pathlib import Path
import sys
import types

import pytest

np = pytest.importorskip("numpy")


def _make_dataset(path: Path) -> None:
    rows = ["event_time,profit,volume,spread,hour,symbol\n"]
    levels = [0.01, 0.05, 0.1]
    signs_per_level = [[1, -1] * 3, [1] * 6, [1, -1] * 3]
    idx = 0
    for level, amp in enumerate(levels):
        for sign in signs_per_level[level]:
            profit = sign * amp
            volume = sign * amp * 100
            spread = amp * 0.1
            rows.append(f"{idx},{profit},{volume},{spread},{level},EURUSD\n")
            idx += 1
    path.write_text("".join(rows))


def test_curriculum_logging(tmp_path: Path) -> None:
    data_file = tmp_path / "trades_raw.csv"
    _make_dataset(data_file)
    out_dir = tmp_path / "out"
    # Avoid heavy optional dependencies during import
    dummy = types.ModuleType("indicator_discovery")
    dummy.evolve_indicators = lambda *a, **k: None
    sys.modules.setdefault("botcopier.features.indicator_discovery", dummy)
    from botcopier.training.pipeline import train
    train(
        data_file,
        out_dir,
        n_splits=2,
        cv_gap=1,
        param_grid=[{}],
        curriculum_threshold=1e-6,
        curriculum_steps=3,
        cluster_correlation=1.0,
    )
    model = json.loads((out_dir / "model.json").read_text())
    assert "curriculum" in model
    assert "curriculum_final" in model
    assert model["curriculum_final"] == model["curriculum"][-1]
