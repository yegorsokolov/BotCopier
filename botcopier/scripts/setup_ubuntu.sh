#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[setup_ubuntu] $*"
}

# Navigate to repository root
REPO_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_DIR"

log "Updating package list"
mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
log "Detected ${mem_kb} kB of total memory"
SWAP_GB=${SWAP_GB:-0}
if [ "$SWAP_GB" -eq 0 ] && [ "$mem_kb" -lt 2097152 ]; then
  SWAP_GB=2
fi
if [ "$SWAP_GB" -gt 0 ]; then
  sudo fallocate -l ${SWAP_GB}G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  log "Activated ${SWAP_GB}G swapfile"
fi
sudo apt-get update

log "Installing system packages"
sudo apt-get install -y build-essential python3 python3-pip wine64 \
  protobuf-compiler libcapnp-dev nats-server git

log "Installing and enabling Chrony for time synchronization"
sudo apt-get install -y chrony
sudo systemctl enable --now chrony
log "chrony service status: $(systemctl is-active chrony)"
log "Chrony tracking summary"
chronyc tracking || true

log "Enabling and starting nats-server"
sudo systemctl enable nats-server && sudo systemctl start nats-server

log "Upgrading pip"
python3 -m pip install --upgrade pip

log "Installing Python dependencies"
pip3 install --no-cache-dir -r requirements.txt
# pyarrow is required for Arrow Flight support
pip3 install --no-cache-dir pyarrow

log "Detecting hardware resources and training mode"
python3 - <<'PY' | tee hardware.log | while read line; do log "$line"; done
from botcopier.training.pipeline import detect_resources
import json
res = detect_resources()
print(json.dumps(res))
print("mode=" + res.get("mode", "unknown"))
print("training_mode=" + ("lite" if res["lite_mode"] else "heavy"))
PY

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
log "online-trainer.service status: $(systemctl is-active online-trainer.service)"

log "Installing stream listener and metrics collector services"
sudo cp docs/systemd/stream-listener.service docs/systemd/metrics-collector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stream-listener.service
log "stream-listener.service status: $(systemctl is-active stream-listener.service)"
sudo systemctl enable --now metrics-collector.service
log "metrics-collector.service status: $(systemctl is-active metrics-collector.service)"

log "Installing Flight server service"
sudo cp docs/systemd/flight-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now flight-server.service
log "flight-server.service status: $(systemctl is-active flight-server.service)"
