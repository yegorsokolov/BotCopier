from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.online_trainer import async_main


@pytest.mark.asyncio
async def test_async_main_uses_csv(tmp_path: Path):
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text("a,b,y\n1,0,1\n")
    model_path = tmp_path / "model.json"
    called = {}

    def fake_tail(path):
        called["path"] = path

    with patch(
        "scripts.online_trainer.OnlineTrainer.tail_csv", side_effect=fake_tail
    ) as mock_tail, patch("botcopier.config.settings.save_params"):
        await async_main(["--csv", str(csv_path), "--model", str(model_path)])
        assert called["path"] == csv_path
        assert mock_tail.called
