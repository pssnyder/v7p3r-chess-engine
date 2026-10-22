#!/bin/bash
# ============================================================================
# V7P3R Universal Version Deployment Script
# ============================================================================
# Purpose: Deploy ANY V7P3R version to GCP production using the established
#          5-minute manual container update method
# Usage: ./deploy_version.sh <version> [environment]
#   Example: ./deploy_version.sh 17.5 production
#   Example: ./deploy_version.sh 17.5 test
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT="s:/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine"
LICHESS_DIR="$PROJECT_ROOT/lichess"
GCP_PROJECT="v7p3r-lichess-bot"
GCP_INSTANCE="v7p3r-production-bot"
GCP_ZONE="us-central1-a"
CONTAINER_NAME="v7p3r-production"

# ============================================================================
# Parse Arguments
# ============================================================================
if [ -z "$1" ]; then
    echo "ERROR: Version number required"
    echo "Usage: $0 <version> [environment]"
    echo "Example: $0 17.5 production"
    exit 1
fi

VERSION="$1"
ENVIRONMENT="${2:-production}"  # Default to production

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(production|test)$ ]]; then
    echo "ERROR: Environment must be 'production' or 'test'"
    exit 1
fi

echo "======================================================================"
echo "V7P3R Universal Version Deployment"
echo "======================================================================"
echo "Version: v$VERSION"
echo "Environment: $ENVIRONMENT"
echo "Project Root: $PROJECT_ROOT"
echo "GCP Instance: $GCP_INSTANCE"
echo "Container: $CONTAINER_NAME"
echo "======================================================================"
echo ""

# ============================================================================
# Step 1: Verify Source Files Exist
# ============================================================================
echo "Step 1: Verifying source files..."

SOURCE_DIR="$PROJECT_ROOT/src"
if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Check for required engine files
REQUIRED_FILES=(
    "v7p3r.py"
    "v7p3r_uci.py"
    "v7p3r_fast_evaluator.py"
    "v7p3r_bitboard_evaluator.py"
    "v7p3r_openings_v161.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$SOURCE_DIR/$file" ]; then
        echo "ERROR: Required file not found: $SOURCE_DIR/$file"
        exit 1
    fi
done

echo "[OK] All required source files found"
echo ""

# ============================================================================
# Step 2: Verify Version in Code
# ============================================================================
echo "Step 2: Verifying version numbers in code..."

# Check v7p3r_uci.py for version
if ! grep -q "V7P3R v$VERSION" "$SOURCE_DIR/v7p3r_uci.py"; then
    echo "WARNING: Version v$VERSION not found in v7p3r_uci.py"
    echo "Current version string:"
    grep "id name" "$SOURCE_DIR/v7p3r_uci.py" || echo "  (not found)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "[OK] Version verified in UCI interface"
fi

# Check v7p3r.py for version
if ! grep -q "v$VERSION" "$SOURCE_DIR/v7p3r.py" | head -5; then
    echo "WARNING: Version v$VERSION not clearly found in v7p3r.py header"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "[OK] Version verified in engine"
fi

echo ""

# ============================================================================
# Step 3: Create Timestamped Deployment Directory
# ============================================================================
echo "Step 3: Creating timestamped deployment directory..."

TIMESTAMP=$(date +%Y%m%d)
DEPLOY_DIR="$LICHESS_DIR/engines/V7P3R_v${VERSION}_${TIMESTAMP}"

if [ -d "$DEPLOY_DIR" ]; then
    echo "[INFO] Directory already exists: $DEPLOY_DIR"
    echo "       Using existing directory"
else
    mkdir -p "$DEPLOY_DIR/src"
    echo "[OK] Created: $DEPLOY_DIR"
fi

echo ""

# ============================================================================
# Step 4: Copy Engine Files
# ============================================================================
echo "Step 4: Copying engine files to deployment directory..."

for file in "${REQUIRED_FILES[@]}"; do
    cp "$SOURCE_DIR/$file" "$DEPLOY_DIR/src/"
    echo "  Copied: $file"
done

echo "[OK] All engine files copied"
echo ""

# ============================================================================
# Step 5: Update Config Greetings
# ============================================================================
echo "Step 5: Updating config.yml greetings..."

CONFIG_FILE="$LICHESS_DIR/config.yml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Backup config
cp "$CONFIG_FILE" "$CONFIG_FILE.backup"

# Update greeting messages
sed -i "s/V7P3R v[0-9.]\+/V7P3R v$VERSION/g" "$CONFIG_FILE"

if grep -q "V7P3R v$VERSION" "$CONFIG_FILE"; then
    echo "[OK] Config greetings updated to v$VERSION"
else
    echo "WARNING: Failed to update config greetings"
fi

echo ""

# ============================================================================
# Step 6: Create Deployment Tarball
# ============================================================================
echo "Step 6: Creating deployment tarball..."

TARBALL_NAME="v${VERSION}-src.tar.gz"
cd "$DEPLOY_DIR"
tar -czf "../$TARBALL_NAME" src/

if [ -f "../$TARBALL_NAME" ]; then
    TARBALL_SIZE=$(du -h "../$TARBALL_NAME" | cut -f1)
    echo "[OK] Created: $TARBALL_NAME ($TARBALL_SIZE)"
else
    echo "ERROR: Failed to create tarball"
    exit 1
fi

cd - > /dev/null
echo ""

# ============================================================================
# Step 7: Upload to GCP (Production Only)
# ============================================================================
if [ "$ENVIRONMENT" == "production" ]; then
    echo "Step 7: Uploading to GCP instance..."
    
    # Check if gcloud is available
    if ! command -v gcloud &> /dev/null; then
        echo "ERROR: gcloud CLI not found. Please install Google Cloud SDK"
        exit 1
    fi
    
    # Upload tarball
    TARBALL_PATH="$LICHESS_DIR/engines/$TARBALL_NAME"
    gcloud compute scp "$TARBALL_PATH" "$GCP_INSTANCE:" --project="$GCP_PROJECT" --zone="$GCP_ZONE"
    
    if [ $? -eq 0 ]; then
        echo "[OK] Tarball uploaded to GCP"
    else
        echo "ERROR: Failed to upload tarball"
        exit 1
    fi
    
    echo ""
    
    # ========================================================================
    # Step 8: Deploy to Container
    # ========================================================================
    echo "Step 8: Deploying to container..."
    
    gcloud compute ssh "$GCP_INSTANCE" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command="
        cd ~
        echo '  Copying tarball to container...'
        sudo docker cp $TARBALL_NAME $CONTAINER_NAME:/tmp/
        echo '  Backing up current engine...'
        sudo docker exec $CONTAINER_NAME bash -c 'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup 2>/dev/null || true'
        echo '  Creating new engine directory...'
        sudo docker exec $CONTAINER_NAME mkdir -p /lichess-bot/engines/v7p3r
        echo '  Extracting new engine files...'
        sudo docker exec $CONTAINER_NAME bash -c 'cd /lichess-bot/engines/v7p3r && tar -xzf /tmp/$TARBALL_NAME --strip-components=1'
        echo '  Setting permissions...'
        sudo docker exec $CONTAINER_NAME bash -c 'chmod +x /lichess-bot/engines/v7p3r/*.py'
        echo '  Cleaning up...'
        sudo docker exec $CONTAINER_NAME rm /tmp/$TARBALL_NAME
        rm $TARBALL_NAME
        echo '  Restarting container...'
        sudo docker restart $CONTAINER_NAME
        echo '  Done!'
    "
    
    if [ $? -eq 0 ]; then
        echo "[OK] Deployment complete"
    else
        echo "ERROR: Deployment failed"
        exit 1
    fi
    
    echo ""
    
    # ========================================================================
    # Step 9: Update Config on GCP
    # ========================================================================
    echo "Step 9: Updating config on GCP..."
    
    gcloud compute scp "$CONFIG_FILE" "$GCP_INSTANCE:config.yml" --project="$GCP_PROJECT" --zone="$GCP_ZONE"
    
    if [ $? -eq 0 ]; then
        echo "[OK] Config uploaded to GCP"
        # Move to correct location for volume mount
        gcloud compute ssh "$GCP_INSTANCE" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command="sudo mv ~/config.yml /home/v7p3r/config.yml"
        echo "[OK] Config moved to volume mount location"
        gcloud compute ssh "$GCP_INSTANCE" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command="sudo docker restart $CONTAINER_NAME"
        echo "[OK] Container restarted with new config"
    else
        echo "WARNING: Failed to update config (bot will use cached version)"
    fi
    
    echo ""
    
    # ========================================================================
    # Step 10: Verify Deployment
    # ========================================================================
    echo "Step 10: Verifying deployment..."
    echo "Waiting 15 seconds for container startup..."
    sleep 15
    
    echo ""
    echo "Container Status:"
    gcloud compute ssh "$GCP_INSTANCE" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command="sudo docker ps --filter name=$CONTAINER_NAME"
    
    echo ""
    echo "Recent Logs (last 30 lines):"
    gcloud compute ssh "$GCP_INSTANCE" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command="sudo docker logs $CONTAINER_NAME --tail=30"
    
    echo ""
    
else
    echo "Step 7-10: Skipped (test environment)"
    echo ""
    echo "[INFO] Test deployment prepared at: $DEPLOY_DIR"
    echo "[INFO] Tarball created at: $LICHESS_DIR/engines/$TARBALL_NAME"
fi

# ============================================================================
# Step 11: Update CHANGELOG
# ============================================================================
echo "Step 11: Updating CHANGELOG..."

CHANGELOG="$LICHESS_DIR/CHANGELOG.md"
TODAY=$(date +%Y-%m-%d)

# Create backup
cp "$CHANGELOG" "$CHANGELOG.backup"

# Add new entry at the top of the 2025 section
# Find the line with "## 2025 Deployment History" and insert after it
awk -v version="$VERSION" -v date="$TODAY" '
/^## 2025 Deployment History/ {
    print $0
    print ""
    print "### " date
    print "- **Version**: v" version
    print "- **Status**: DEPLOYED"
    print "- **Environment**: production"
    print "- **Deployment Method**: Automated deployment script"
    print "- **Notes**: Deployed via universal deployment script"
    print ""
    next
}
{ print }
' "$CHANGELOG" > "$CHANGELOG.tmp" && mv "$CHANGELOG.tmp" "$CHANGELOG"

echo "[OK] CHANGELOG updated"
echo ""

# ============================================================================
# Deployment Summary
# ============================================================================
echo "======================================================================"
echo "DEPLOYMENT SUMMARY"
echo "======================================================================"
echo "[OK] Version v$VERSION prepared for $ENVIRONMENT"
echo "[OK] Deployment directory: $DEPLOY_DIR"
echo "[OK] Tarball: $LICHESS_DIR/engines/$TARBALL_NAME"
echo "[OK] Config updated with v$VERSION greetings"
echo "[OK] CHANGELOG updated"

if [ "$ENVIRONMENT" == "production" ]; then
    echo "[OK] Deployed to GCP: $GCP_INSTANCE"
    echo "[OK] Container: $CONTAINER_NAME"
    echo ""
    echo "Next Steps:"
    echo "  1. Verify bot online: https://lichess.org/@/v7p3r_bot"
    echo "  2. Check first game greeting shows 'V7P3R v$VERSION'"
    echo "  3. Monitor first 10 games for issues"
    echo "  4. Run analytics after 50 games:"
    echo "     cd $PROJECT_ROOT/analytics"
    echo "     python weekly_analysis.py --days 3"
else
    echo ""
    echo "Test Deployment Ready:"
    echo "  - Files prepared in: $DEPLOY_DIR"
    echo "  - To deploy to production, run:"
    echo "    $0 $VERSION production"
fi

echo ""
echo "Rollback Instructions (if needed):"
echo "  gcloud compute ssh $GCP_INSTANCE --project=$GCP_PROJECT --zone=$GCP_ZONE --command=\""
echo "    sudo docker exec $CONTAINER_NAME bash -c 'rm -rf /lichess-bot/engines/v7p3r'"
echo "    sudo docker exec $CONTAINER_NAME bash -c 'mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r'"
echo "    sudo docker restart $CONTAINER_NAME"
echo "  \""
echo "======================================================================"
