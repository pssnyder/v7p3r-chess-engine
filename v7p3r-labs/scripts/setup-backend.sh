#!/bin/bash
# Backend Setup Script - Run on VM after uploading backend code
# Sets up Python environment and systemd service

set -e

echo "========================================="
echo "v7p3r-labs Backend Setup"
echo "========================================="
echo ""

cd /opt/v7p3r-labs/backend

# Create virtual environment
echo "Creating Python virtual environment..."
python3.12 -m venv venv

# Activate and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pull Docker images
echo ""
echo "Pulling engine Docker images..."
VERSIONS=("12.6" "14.1" "16.1" "17.7" "18.4")
for VERSION in "${VERSIONS[@]}"; do
    echo "Pulling v7p3r:$VERSION..."
    docker pull gcr.io/rts-labs-f3981/v7p3r:$VERSION
done

# Create systemd service
echo ""
echo "Creating systemd service..."
sudo tee /etc/systemd/system/v7p3r-api.service > /dev/null <<EOF
[Unit]
Description=v7p3r-labs FastAPI Backend
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/v7p3r-labs/backend
Environment="PATH=/opt/v7p3r-labs/backend/venv/bin"
ExecStart=/opt/v7p3r-labs/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable v7p3r-api
sudo systemctl start v7p3r-api

# Wait for service to start
sleep 3

# Check service status
echo ""
echo "Service status:"
sudo systemctl status v7p3r-api --no-pager

# Test health endpoint
echo ""
echo "Testing API health endpoint..."
curl http://localhost:8080/ || echo "Warning: Health check failed"

echo ""
echo "========================================="
echo "Backend setup complete!"
echo ""
echo "Useful commands:"
echo "  View logs: sudo journalctl -u v7p3r-api -f"
echo "  Restart: sudo systemctl restart v7p3r-api"
echo "  Stop: sudo systemctl stop v7p3r-api"
echo "  Status: sudo systemctl status v7p3r-api"
echo "========================================="
