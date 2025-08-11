from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shm_ring import ShmRing, TRADE_MSG, METRIC_MSG


def test_ring_roundtrip(tmp_path):
    path = tmp_path / "ring"
    ring = ShmRing.create(str(path), 1024)
    assert ring.push(TRADE_MSG, b"abc")
    assert ring.push(METRIC_MSG, b"xyz")
    t = ring.pop()
    assert t is not None
    t_type, t_payload = t
    assert t_type == TRADE_MSG
    assert bytes(t_payload) == b"abc"
    del t_payload, t
    m = ring.pop()
    assert m is not None
    m_type, m_payload = m
    assert m_type == METRIC_MSG
    assert bytes(m_payload) == b"xyz"
    del m_payload, m
    assert ring.pop() is None
    ring.close()
