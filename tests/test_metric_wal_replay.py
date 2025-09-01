from pathlib import Path


class MetricLogger:
    """Minimal logger emulating WAL behaviour for metrics."""

    def __init__(self, base: Path) -> None:
        self.metrics_file = base / "metrics.csv"
        self.wal_file = base / "metrics.wal"
        self.buffer: list[str] = []
        self.metric_retry_count = 0

    def append(self, line: str) -> None:
        self.buffer.append(line)

    def flush(self, fail: bool = False) -> None:
        if self.wal_file.exists() and not fail:
            with self.wal_file.open() as wf, self.metrics_file.open("a") as mf:
                for line in wf:
                    mf.write(line)
            self.wal_file.unlink()
            self.metric_retry_count = 0

        if not self.buffer:
            return

        if fail:
            with self.wal_file.open("a") as wf:
                for line in self.buffer:
                    wf.write(line + "\n")
            self.buffer.clear()
            self.metric_retry_count += 1
            return

        with self.metrics_file.open("a") as mf:
            for line in self.buffer:
                mf.write(line + "\n")
        self.buffer.clear()
        self.metric_retry_count = 0


def test_metric_wal_replay(tmp_path: Path) -> None:
    logger = MetricLogger(tmp_path)

    logger.append("m1")
    logger.flush(fail=True)
    assert logger.metric_retry_count == 1
    assert not logger.metrics_file.exists()
    assert logger.wal_file.read_text().splitlines() == ["m1"]

    logger.append("m2")
    logger.flush()

    assert logger.metric_retry_count == 0
    assert not logger.wal_file.exists()
    assert logger.metrics_file.read_text().splitlines() == ["m1", "m2"]

