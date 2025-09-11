#!/usr/bin/env python3
"""Convert CSV logs to Parquet using pyarrow."""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.csv as pc
import pyarrow.parquet as pq

from botcopier.data.schema import SCHEMAS


def convert_csv_to_parquet(csv_file: Path, *, schema: str) -> Path:
    """Convert ``csv_file`` to Parquet using the named schema."""
    table = pc.read_csv(csv_file)
    out_file = csv_file.with_suffix(".parquet")
    sch = SCHEMAS.get(schema)
    if sch is not None:
        table = table.cast(sch, safe=False)
    pq.write_table(table, out_file)
    return out_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert CSV logs to Parquet")
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument(
        "--schema", choices=SCHEMAS.keys(), default="trades", help="Schema name"
    )
    args = parser.parse_args(argv)
    out = convert_csv_to_parquet(args.csv, schema=args.schema)
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
