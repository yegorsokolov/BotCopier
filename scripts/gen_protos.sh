#!/usr/bin/env bash
set -euo pipefail
PROTO_DIR="$(dirname "$0")/../proto"
python -m grpc_tools.protoc -I"$PROTO_DIR" \
  --python_out="$PROTO_DIR" \
  --grpc_python_out="$PROTO_DIR" \
  "$PROTO_DIR"/*.proto
