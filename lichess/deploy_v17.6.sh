#!/bin/bash
# V7P3R v17.6 GCP Deployment Script
# Deploys v17.6 to production Lichess bot on GCP

set -e

PROJECT_ID="v7p3r-lichess-bot"
VERSION="v17.6"
TARBALL="v17.6-src.tar.gz"
VM_NAME="v7p3r-production-bot"
ZONE="us-central1-a"
CONTAINER="v7p3r-production"

echo "=========================================="
echo "V7P3R v17.6 GCP Deployment"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Version: $VERSION"
echo "VM: $VM_NAME"
echo "Zone: $ZONE"
echo "Container: $CONTAINER"
echo "=========================================="
echo ""

# Set GCP project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"
echo ""

# Check if tarball exists
if [ ! -f "lichess/engines/$TARBALL" ]; then
    echo "Error: Tarball not found at lichess/engines/$TARBALL"
    exit 1
fi

echo "Step 1: Uploading v17.6 source to GCP VM..."
gcloud compute scp "lichess/engines/$TARBALL" "v7p3r@$VM_NAME:/tmp/$TARBALL" --zone="$ZONE" --project="$PROJECT_ID"
echo "✓ Upload complete"
echo ""

echo "Step 2: Backing up current engine and deploying v17.6..."
gcloud compute ssh "v7p3r@$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
    echo 'Copying tarball to container...'
    sudo docker cp /tmp/$TARBALL $CONTAINER:/tmp/
    
    echo 'Backing up current engine...'
    sudo docker exec $CONTAINER bash -c 'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup-\$(date +%Y%m%d-%H%M%S)'
    
    echo 'Creating new engine directory...'
    sudo docker exec $CONTAINER mkdir -p /lichess-bot/engines/v7p3r
    
    echo 'Extracting v17.6 source...'
    sudo docker exec $CONTAINER bash -c 'cd /lichess-bot/engines/v7p3r && tar -xzf /tmp/$TARBALL --strip-components=1'
    
    echo 'Setting permissions...'
    sudo docker exec $CONTAINER chmod +x /lichess-bot/engines/v7p3r/*.py
    
    echo 'Cleaning up...'
    sudo docker exec $CONTAINER rm /tmp/$TARBALL
    rm /tmp/$TARBALL
    
    echo 'Restarting container...'
    sudo docker restart $CONTAINER
"
echo "✓ Deployment complete"
echo ""

echo "Step 3: Waiting for container to restart (10 seconds)..."
sleep 10
echo ""

echo "Step 4: Verifying deployment..."
gcloud compute ssh "v7p3r@$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
    echo 'Container status:'
    sudo docker ps | grep $CONTAINER
    echo ''
    echo 'Recent logs:'
    sudo docker logs --tail 20 $CONTAINER
"
echo ""

echo "=========================================="
echo "✓ V7P3R v17.6 Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Monitor first few games for stability"
echo "2. Check logs: gcloud compute ssh v7p3r@$VM_NAME --zone=$ZONE --project=$PROJECT_ID --command='sudo docker logs -f $CONTAINER'"
echo "3. Wait 48-72 hours for analytics data"
echo ""
echo "Rollback command (if needed):"
echo "  gcloud compute ssh v7p3r@$VM_NAME --zone=$ZONE --project=$PROJECT_ID --command='sudo docker exec $CONTAINER bash -c \"rm -rf /lichess-bot/engines/v7p3r && mv /lichess-bot/engines/v7p3r.backup-* /lichess-bot/engines/v7p3r\" && sudo docker restart $CONTAINER'"
echo ""
