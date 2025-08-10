import time
from threading import Thread

import pyarrow as pa
import pyarrow.flight as flight

from scripts.flight_server import FlightServer
from schemas import METRIC_SCHEMA, TRADE_SCHEMA


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
    m_writer, _ = client.do_put(m_desc, METRIC_SCHEMA)
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
    m_writer.close()

    # send a trade batch
    t_desc = flight.FlightDescriptor.for_path("trades")
    t_writer, _ = client.do_put(t_desc, TRADE_SCHEMA)
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
    ], schema=TRADE_SCHEMA)
    t_writer.write_batch(t_batch)
    t_writer.close()

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

    server.shutdown()
    t.join(timeout=1)
