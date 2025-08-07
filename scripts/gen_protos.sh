#!/usr/bin/env bash
set -euo pipefail
PROTO_DIR="$(dirname "$0")/../proto"
protoc -I"$PROTO_DIR" --python_out="$PROTO_DIR" --cpp_out="$PROTO_DIR" "$PROTO_DIR"/*.proto
