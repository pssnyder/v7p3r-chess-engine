#!/bin/bash
# VM Setup Script - Run on fresh e2-micro instance
# Installs Docker, Python, and prepares environment for backend deployment

set -e

echo "========================================="
echo "v7p3r-labs VM Setup"
echo "========================================="
echo ""

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo ""
echo "Installing Docker..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Set up Docker repository
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Start and enable Docker
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER

# Install Python 3.12
echo ""
echo "Installing Python 3.12..."
sudo apt-get install -y python3.12 python3.12-venv python3-pip

# Configure Docker for GCR
echo ""
echo "Configuring Docker authentication for GCR..."
gcloud auth configure-docker gcr.io

# Create application directory
echo ""
echo "Creating application directory..."
sudo mkdir -p /opt/v7p3r-labs
sudo chown $USER:$USER /opt/v7p3r-labs

echo ""
echo "========================================="
echo "VM setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload backend code: gcloud compute scp --recurse backend/ v7p3r-labs-vm:/opt/v7p3r-labs/backend --zone=us-central1-a"
echo "2. SSH into VM: gcloud compute ssh v7p3r-labs-vm --zone=us-central1-a"
echo "3. Run backend setup: cd /opt/v7p3r-labs/backend && ./setup-backend.sh"
echo "========================================="
