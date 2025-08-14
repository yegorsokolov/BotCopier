#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[setup_ubuntu] $*"
}

# Navigate to repository root
REPO_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_DIR"

log "Updating package list"
sudo apt-get update

log "Installing system packages"
sudo apt-get install -y build-essential python3 python3-pip wine64 \
  protobuf-compiler libcapnp-dev nats-server git

log "Enabling and starting nats-server"
sudo systemctl enable nats-server && sudo systemctl start nats-server

log "Upgrading pip"
python3 -m pip install --upgrade pip

log "Installing Python dependencies"
pip3 install --no-cache-dir -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  log "CUDA-capable GPU detected, installing onnxruntime-gpu"
  pip3 install --no-cache-dir onnxruntime-gpu
else
  log "No CUDA-capable GPU detected; skipping onnxruntime-gpu installation"
fi

# Create Wine prefix for MetaTrader
log "Creating Wine prefix for MetaTrader"
: "${WINEPREFIX:=$HOME/.wine_mt4}"
export WINEPREFIX
mkdir -p "$WINEPREFIX"
wineboot --init

log "Installing online trainer service"
sudo cp docs/systemd/online-trainer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now online-trainer.service
