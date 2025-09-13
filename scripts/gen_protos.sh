#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(dirname "$0")"
bash "$SCRIPT_DIR/../botcopier/scripts/gen_protos.sh"
