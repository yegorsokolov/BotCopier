import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from scripts.nats_publisher import async_main


@pytest.mark.asyncio
async def test_async_main_publishes(tmp_path: Path):
    data = {"event_id": 1}
    event_file = tmp_path / "trade.json"
    event_file.write_text(json.dumps(data))

    class FakeNC:
        def __init__(self):
            self.js = type("JS", (), {"publish": AsyncMock()})()

        def jetstream(self):
            return self.js

        async def close(self):
            pass

    fake_nc = FakeNC()

    with patch("scripts.nats_publisher.nats.connect", return_value=fake_nc) as mock_conn:
        await async_main([
            "trade",
            str(event_file),
            "--servers",
            "nats://test",
            "--schema-version",
            "2",
        ])
        mock_conn.assert_called_once()
        fake_nc.js.publish.assert_awaited_once()
