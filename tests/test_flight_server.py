import time
from threading import Thread

import pyarrow as pa
import pyarrow.flight as flight

from scripts.flight_server import FlightServer
from schemas import METRIC_SCHEMA


def test_flight_server_roundtrip():
    server = FlightServer(port=0)
    t = Thread(target=server.serve, daemon=True)
    t.start()
    # wait for server to bind a port
    while server.port == 0:
        time.sleep(0.01)

    client = flight.FlightClient(f"grpc://127.0.0.1:{server.port}")
    desc = flight.FlightDescriptor.for_path("metrics")
    writer, _ = client.do_put(desc, METRIC_SCHEMA)
    batch = pa.record_batch([
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
    writer.write_batch(batch)
    writer.close()

    info = client.get_flight_info(desc)
    reader = client.do_get(info.endpoints[0].ticket)
    table = reader.read_all()
    assert table.num_rows == 1

    server.shutdown()
    t.join(timeout=1)
