import time
from threading import Thread

import pyarrow as pa
import pyarrow.flight as flight

from scripts.flight_server import FlightServer
from schemas import METRIC_SCHEMA, TRADE_SCHEMA, DECISION_SCHEMA


def test_flight_server_roundtrip():
    server = FlightServer(port=0)
    t = Thread(target=server.serve, daemon=True)
    t.start()
    # wait for server to bind a port
    while server.port == 0:
        time.sleep(0.01)

    client = flight.FlightClient(f"grpc://127.0.0.1:{server.port}")

    # send a metrics batch
    m_desc = flight.FlightDescriptor.for_path("metrics")
    m_writer, m_reader = client.do_put(m_desc, METRIC_SCHEMA)
    m_batch = pa.record_batch([
        pa.array([1]),
        pa.array(["2024-01-01T00:00:00"]),
        pa.array([1]),
        pa.array([0.5]),
        pa.array([0.1]),
        pa.array([10]),
        pa.array([0.2]),
        pa.array([1.0]),
        pa.array([0]),
        pa.array([0]),
        pa.array([5]),
    ], schema=METRIC_SCHEMA)
    m_writer.write_batch(m_batch)
    ack_buf = m_reader.read()
    assert ack_buf.to_pybytes() == b"2024-01-01T00:00:00"
    m_writer.close()

    # send a trade batch
    t_desc = flight.FlightDescriptor.for_path("trades")
    t_batch = pa.record_batch([
        pa.array([1]),  # schema_version
        pa.array([1]),  # event_id
        pa.array(["trace"]),
        pa.array(["2024-01-01T00:00:00"]),
        pa.array(["2024-01-01T00:00:00"]),
        pa.array(["2024-01-01T00:00:00"]),
        pa.array(["OPEN"]),
        pa.array([1]),
        pa.array([1]),
        pa.array(["src"]),
        pa.array(["EURUSD"]),
        pa.array([0]),
        pa.array([1.0]),
        pa.array([1.0]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array(["c"]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array([0]),
        pa.array([""]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array([0.0]),
        pa.array([0]),
        pa.array([0]),  # decision_id
        pa.array([""]),  # exit_reason
        pa.array([0]),  # duration_sec
    ], schema=TRADE_SCHEMA)
    t_writer, t_reader = client.do_put(t_desc, TRADE_SCHEMA)

    t_writer.write_batch(t_batch)
    t_ack_buf = t_reader.read()
    assert t_ack_buf.to_pybytes() == b"1"
    t_writer.close()

    # send a decision batch
    d_desc = flight.FlightDescriptor.for_path("decisions")
    d_batch = pa.record_batch([
        pa.array([1]),  # schema_version
        pa.array([1]),  # event_id
        pa.array(["2024-01-01T00:00:00"]),
        pa.array(["v1"]),
        pa.array(["buy"]),
        pa.array([0.5]),
        pa.array([1.0]),
        pa.array([1.0]),
        pa.array([0]),
        pa.array([0]),
        pa.array([1]),
        pa.array([0.1]),
        pa.array([0.2]),
        pa.array([0.1]),
        pa.array([0]),
        pa.array(["f"]),
        pa.array(["trace"]),
        pa.array(["span"]),
    ], schema=DECISION_SCHEMA)
    d_writer, d_reader = client.do_put(d_desc, DECISION_SCHEMA)
    d_writer.write_batch(d_batch)
    d_ack_buf = d_reader.read()
    assert d_ack_buf.to_pybytes() == b"1"
    d_writer.close()

    # verify metrics
    m_info = client.get_flight_info(m_desc)
    m_reader = client.do_get(m_info.endpoints[0].ticket)
    m_table = m_reader.read_all()
    assert m_table.num_rows == 1

    # verify trades
    t_info = client.get_flight_info(t_desc)
    t_reader = client.do_get(t_info.endpoints[0].ticket)
    t_table = t_reader.read_all()
    assert t_table.num_rows == 1

    # verify decisions
    d_info = client.get_flight_info(d_desc)
    d_reader = client.do_get(d_info.endpoints[0].ticket)
    d_table = d_reader.read_all()
    assert d_table.num_rows == 1

    server.shutdown()
    t.join(timeout=1)
