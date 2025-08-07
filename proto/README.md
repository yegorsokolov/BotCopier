# Protobuf Schemas

This directory defines the message formats used for communication between the
MQL4 expert advisor and Python tools. The schema is versioned to allow safe
changes over time.

## Messages

- **TradeEvent** (`trade_event.proto`)
  - `event_id`: unique sequential identifier
  - `event_time`, `broker_time`, `local_time`: ISO8601 timestamps
  - `action`: event type string
  - `symbol`, `price`, `slippage`, etc. capture trade details
- **Metrics** (`metrics.proto`)
  - High level performance metrics such as `win_rate` and `drawdown`.
- **ObserverMessage** (`observer.proto`)
  - Envelope that carries either a `TradeEvent` or `Metrics` message along with
    a `schema_version` field.

## Regenerating Bindings

Run the helper script to regenerate Python and C++ bindings:

```bash
./scripts/gen_protos.sh
```

This requires `protoc` and the Protobuf Python package to be installed.
