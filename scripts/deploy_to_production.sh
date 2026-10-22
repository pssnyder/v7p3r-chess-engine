#!/bin/bash
# V7P3R Chess Engine - Production Deployment Script
# Usage: ./deploy_to_production.sh <version>
# Example: ./deploy_to_production.sh v17.8.0

set -e  # Exit on error

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./deploy_to_production.sh <version>"
    echo "Example: ./deploy_to_production.sh v17.8.0"
    exit 1
fi

echo "=========================================="
echo "V7P3R Production Deployment"
echo "Version: $VERSION"
echo "Date: $(date)"
echo "=========================================="
echo

# Pre-deployment checks
echo "Step 1: Pre-Deployment Validation"
echo "-----------------------------------"
echo "Checking CHANGELOG.md for version entry..."
if grep -q "\[$VERSION\]" CHANGELOG.md; then
    echo "✅ Version found in CHANGELOG.md"
else
    echo "❌ Version not found in CHANGELOG.md"
    echo "Please update CHANGELOG.md before deployment"
    exit 1
fi

echo "Checking deployment_log.json..."
if grep -q "\"version\": \"$VERSION\"" deployment_log.json; then
    echo "✅ Version found in deployment_log.json"
else
    echo "❌ Version not found in deployment_log.json"
    echo "Please update deployment_log.json before deployment"
    exit 1
fi

echo "Checking git tag..."
if git tag | grep -q "^$VERSION$"; then
    echo "✅ Git tag exists"
else
    echo "⚠️  Git tag not found - creating annotated tag..."
    read -p "Enter tag message: " TAG_MSG
    git tag -a "$VERSION" -m "$TAG_MSG"
    echo "✅ Tag created"
fi

echo
read -p "Pre-deployment checks passed. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 1
fi

# Create deployment package
echo
echo "Step 2: Creating Deployment Package"
echo "------------------------------------"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEPLOY_DIR="lichess/engines/V7P3R_${VERSION}_${TIMESTAMP}"
mkdir -p "$DEPLOY_DIR"

echo "Copying source files to $DEPLOY_DIR..."
cp -r src/ "$DEPLOY_DIR/"
cp requirements.txt "$DEPLOY_DIR/"
cp README.md "$DEPLOY_DIR/"

echo "Creating tarball..."
cd "$DEPLOY_DIR"
tar -czf "../${VERSION}-src.tar.gz" src/
cd ../../..

echo "✅ Deployment package created: lichess/engines/${VERSION}-src.tar.gz"

# Upload to GCP
echo
echo "Step 3: Uploading to GCP VM"
echo "---------------------------"
echo "Uploading tarball to v7p3r-production-bot..."
gcloud compute scp "lichess/engines/${VERSION}-src.tar.gz" v7p3r-production-bot:/home/patss/ \
    --zone=us-central1-a

echo "Verifying upload..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a \
    --command="ls -lh /home/patss/${VERSION}-src.tar.gz"

echo "✅ Upload complete"

# Backup current version
echo
echo "Step 4: Backing Up Current Production Version"
echo "----------------------------------------------"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
echo "Creating backup with timestamp: $BACKUP_DATE..."

gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    # Ensure backup directory exists
    mkdir -p /home/patss/backups
    
    # Create backup inside container
    sudo docker exec v7p3r-production bash -c \
        'tar -czf /tmp/v7p3r_backup_${BACKUP_DATE}.tar.gz /lichess-bot/engines/v7p3r'
    
    # Copy backup out of container
    sudo docker cp v7p3r-production:/tmp/v7p3r_backup_${BACKUP_DATE}.tar.gz /home/patss/backups/
    
    # Verify backup
    ls -lh /home/patss/backups/v7p3r_backup_${BACKUP_DATE}.tar.gz
"

echo "✅ Backup created: v7p3r_backup_${BACKUP_DATE}.tar.gz"

# Deploy new version
echo
echo "Step 5: Deploying New Version"
echo "------------------------------"
read -p "Ready to deploy $VERSION to production? This will restart the bot. (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled. Backup preserved on VM."
    exit 1
fi

echo "Deploying $VERSION to container..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    # Copy tarball into container
    sudo docker cp ${VERSION}-src.tar.gz v7p3r-production:/tmp/
    
    # Move current version to backup location inside container
    sudo docker exec v7p3r-production bash -c \
        'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup'
    
    # Create new engine directory
    sudo docker exec v7p3r-production mkdir -p /lichess-bot/engines/v7p3r
    
    # Extract new version
    sudo docker exec v7p3r-production bash -c \
        'cd /lichess-bot/engines/v7p3r && tar -xzf /tmp/${VERSION}-src.tar.gz --strip-components=1'
    
    # Verify extraction
    echo 'Verifying deployment...'
    sudo docker exec v7p3r-production ls -la /lichess-bot/engines/v7p3r/
"

echo "✅ Files deployed"

# Restart bot
echo
echo "Step 6: Restarting Bot"
echo "----------------------"
echo "Restarting Docker container..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a \
    --command="sudo docker restart v7p3r-production"

echo "Waiting for container to start (10 seconds)..."
sleep 10

# Verify deployment
echo
echo "Step 7: Verifying Deployment"
echo "----------------------------"
echo "Checking Docker logs..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a \
    --command="sudo docker logs v7p3r-production --tail 20"

echo
echo "Verifying engine version..."
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    sudo docker exec v7p3r-production bash -c \
        'echo uci | python /lichess-bot/engines/v7p3r/v7p3r_uci.py | grep \"id name\"'
"

echo
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Version: $VERSION"
echo "Deployed: $(date)"
echo "Backup: /home/patss/backups/v7p3r_backup_${BACKUP_DATE}.tar.gz"
echo
echo "Next Steps:"
echo "1. Monitor first 5 games on Lichess web interface"
echo "2. Check Docker logs for errors: gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command='sudo docker logs v7p3r-production -f'"
echo "3. Watch for unusual move times or blunders"
echo "4. Update deployment_log.json status to 'production' if stable after 24 hours"
echo
echo "Rollback Command (if needed):"
echo "gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command=\"sudo docker exec v7p3r-production bash -c 'rm -rf /lichess-bot/engines/v7p3r && mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r' && sudo docker restart v7p3r-production\""
echo "=========================================="
