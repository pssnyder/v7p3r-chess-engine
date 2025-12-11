#!/bin/bash
# V7P3R v17.1.1 Production Deployment Script
# Date: December 10, 2025

echo "=== V7P3R v17.1.1 Deployment to Lichess Production VM ==="
echo ""

# Fix gcloud Python issue
echo "Step 1: Setting gcloud to use Python 3.13..."
export CLOUDSDK_PYTHON="C:/Users/patss/AppData/Local/Programs/Python/Python313/python.exe"
echo "✓ CLOUDSDK_PYTHON set to Python 3.13"
echo ""

# Check current version on VM
echo "Step 2: Checking current engine version on VM..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production ls -la /lichess-bot/engines/v7p3r/ | head -10"
echo ""

# Check if v17.1.1 backup exists
echo "Step 3: Checking for v17.1.1 backup..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production ls -la /lichess-bot/engines/ | grep v7p3r"
echo ""

# Restore v17.1.1 (swap current with backup if needed)
echo "Step 4: Restoring v17.1.1 code..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  # Check if backup exists
  if sudo docker exec v7p3r-production [ -d /lichess-bot/engines/v7p3r.backup ]; then
    echo 'Backup exists, swapping with current...'
    sudo docker exec v7p3r-production bash -c 'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.temp && mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r && mv /lichess-bot/engines/v7p3r.temp /lichess-bot/engines/v7p3r.backup'
  else
    echo 'No backup found, current version should already be v17.1.1'
  fi
  
  # Verify files
  sudo docker exec v7p3r-production ls -la /lichess-bot/engines/v7p3r/
"
echo ""

# Update config.yml to version 17.1
echo "Step 5: Updating config.yml to version 17.1..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
  sudo docker exec v7p3r-production sed -i 's/name: \"V7P3R.*/name: \"V7P3R v17.1\"/' /lichess-bot/config.yml
  echo 'Config updated, verifying...'
  sudo docker exec v7p3r-production grep 'name:' /lichess-bot/config.yml | grep V7P3R
"
echo ""

# Restart the bot
echo "Step 6: Restarting bot..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker restart v7p3r-production"
echo "✓ Bot restarted, waiting 15 seconds for startup..."
sleep 15
echo ""

# Verify engine version
echo "Step 7: Verifying engine version..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker exec v7p3r-production bash -c 'echo uci | python /lichess-bot/engines/v7p3r/v7p3r_uci.py 2>/dev/null | grep \"id name\"'"
echo ""

# Check bot logs
echo "Step 8: Checking bot logs (last 30 lines)..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="sudo docker logs v7p3r-production --tail 30"
echo ""

echo "=== Deployment Complete ==="
echo "Monitor first 5 games at: https://lichess.org/@/V7P3R"
echo ""
