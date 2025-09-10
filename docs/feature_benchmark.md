# Feature extraction benchmark

Benchmark on 20k synthetic log rows (`open`, `high`, `low`, `close`, `event_time`, `symbol`, `price`).

| implementation | time (s) |
| -------------- | -------- |
| pandas loops (before) | 0.0978 |
| pandas vectorised | 0.0306 |
| polars vectorised | 0.0275 |

Times measured on container hardware using `time.perf_counter`.
