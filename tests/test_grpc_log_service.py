import socket
from pathlib import Path

import grpc

from proto import trade_event_pb2, metric_event_pb2
from scripts import grpc_log_service


def test_grpc_log_service(tmp_path: Path):
    host = "127.0.0.1"
    srv_sock = socket.socket()
    srv_sock.bind((host, 0))
    port = srv_sock.getsockname()[1]
    srv_sock.close()

    trade_out = tmp_path / "trades.csv"
    metrics_out = tmp_path / "metrics.csv"
    server = grpc_log_service.create_server(host, port, trade_out, metrics_out)
    server.start()
    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        log_trade = channel.unary_unary(
            "/tbot.LogService/LogTrade",
            request_serializer=lambda x: x.SerializeToString(),
        )
        log_metrics = channel.unary_unary(
            "/tbot.LogService/LogMetrics",
            request_serializer=lambda x: x.SerializeToString(),
        )

        trade = trade_event_pb2.TradeEvent(
            event_id=1,
            event_time="t",
            broker_time="b",
            local_time="l",
            action="OPEN",
            ticket=1,
            magic=2,
            source="mt4",
            symbol="EURUSD",
            order_type=0,
            lots=0.1,
            price=1.2345,
        )
        log_trade(trade)

        metrics = metric_event_pb2.MetricEvent(
            time="t",
            magic=2,
            win_rate=0.5,
            avg_profit=1.0,
            trade_count=1,
            drawdown=0.1,
            sharpe=1.2,
            file_write_errors=0,
            socket_errors=0,
            book_refresh_seconds=5,
            var_breach_count=0,
        )
        log_metrics(metrics)
    finally:
        server.stop(0)

    assert trade_out.exists()
    assert metrics_out.exists()
    assert "EURUSD" in trade_out.read_text()
    assert "0.5" in metrics_out.read_text()
