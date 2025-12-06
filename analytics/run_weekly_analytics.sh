#!/bin/bash
# V7P3R Weekly Analytics - Shell Script
# Runs Docker Compose to generate weekly analytics reports

set -e

echo "========================================"
echo "V7P3R Weekly Analytics"
echo "========================================"
echo ""
echo "Starting Docker Compose..."
echo "This will analyze the last 7 days of games."
echo ""

cd "$(dirname "$0")"

if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

docker-compose up --build

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "SUCCESS: Analytics complete!"
    echo "========================================"
    echo ""
    echo "Reports saved to: analytics_reports/"
    echo ""
else
    echo ""
    echo "========================================"
    echo "ERROR: Analytics failed!"
    echo "========================================"
    echo ""
    echo "Review logs above for details."
    exit 1
fi
