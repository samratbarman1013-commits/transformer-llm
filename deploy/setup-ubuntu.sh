#!/usr/bin/env bash
# One-shot setup for the Transformer API server on a fresh Ubuntu VM
# (works on Oracle Cloud Always Free ARM, or any Ubuntu 22.04/24.04 server).
#
# On your PC:   scp -r pc-kit youruser@SERVER_IP:/tmp/ && put model files in server-data/
# On the server: put the kit at /opt/transformer-api, then run this script from inside it.
set -e
cd "$(dirname "$0")/.."

echo "[1/4] Installing Python venv tools..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip curl

echo "[2/4] Creating venv + installing dependencies (CPU torch, ARM64 wheels included)..."
python3 -m venv .venv
.venv/bin/pip install --upgrade -q pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r server/requirements.txt

echo "[3/4] Installing systemd service..."
sudo cp deploy/transformer-api.service /etc/systemd/system/
echo "  >> EDIT THE API KEY:  sudo nano /etc/systemd/system/transformer-api.service"
echo "  >> (set TRANSFORMER_API_KEY to a long random string, then rerun this script)"
sudo systemctl daemon-reload
sudo systemctl enable transformer-api

echo "[4/4] Opening port 8000 in the firewall (Oracle images ship strict iptables)..."
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || sudo iptables-save | sudo tee /etc/iptables/rules.v4 >/dev/null 2>&1 || true

echo
echo "Done. Start with:  sudo systemctl start transformer-api"
echo "Test with:         curl http://localhost:8000/api/health"
echo "Remember to also open port 8000 in your cloud Security List / firewall rules."
