#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root
REPO_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_DIR"

sudo apt-get update
sudo apt-get install -y build-essential python3 python3-pip wine64

python3 -m pip install --upgrade pip
pip3 install --no-cache-dir -r requirements.txt

# Create Wine prefix for MetaTrader
: "${WINEPREFIX:=$HOME/.wine_mt4}"
export WINEPREFIX
mkdir -p "$WINEPREFIX"
wineboot --init
