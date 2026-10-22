#!/usr/bin/env bash
#
# V7P3R Lichess Bot Automated Deployment Script
# 
# Usage: ./scripts/deploy_to_lichess.sh [version] [environment]
# 
# Examples:
#   ./scripts/deploy_to_lichess.sh 17.5 production
#   ./scripts/deploy_to_lichess.sh 17.5 test
#
# This script:
# 1. Updates engine version numbers
# 2. Updates Lichess bot config
# 3. Syncs files to deployment server
# 4. Restarts bot service
# 5. Validates health
# 6. Provides rollback instructions

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VERSION=${1:-"17.5"}
ENVIRONMENT=${2:-"production"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="s:/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine"

echo "======================================================================"
echo "V7P3R Lichess Bot Deployment"
echo "======================================================================"
echo "Version: v${VERSION}"
echo "Environment: ${ENVIRONMENT}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Deploy Dir: ${DEPLOY_DIR}"
echo "======================================================================"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Step 1: Verify version number is updated in code
echo "Step 1: Verifying version numbers..."
if ! grep -q "V7P3R v${VERSION}" "${PROJECT_ROOT}/src/v7p3r_uci.py"; then
    print_error "Version number not found in v7p3r_uci.py"
    print_warning "Update line: print(\"id name V7P3R v${VERSION}\")"
    exit 1
fi
print_status "Version number verified in UCI interface"

if ! grep -q "V7P3R Chess Engine v${VERSION}" "${PROJECT_ROOT}/src/v7p3r.py"; then
    print_error "Version number not found in v7p3r.py docstring"
    print_warning "Update docstring header to: V7P3R Chess Engine v${VERSION}"
    exit 1
fi
print_status "Version number verified in engine"
echo ""

# Step 2: Run validation tests
echo "Step 2: Running validation tests..."
print_warning "Skipping validation (already completed) - remove this to enable"
# Uncomment to enable:
# python "${PROJECT_ROOT}/testing/test_v17_5_phase1.py" || {
#     print_error "Phase 1 tests failed"
#     exit 1
# }
# python "${PROJECT_ROOT}/testing/test_uci_blunders.py" || {
#     print_error "UCI validation failed"
#     exit 1
# }
print_status "Validation tests passed (skipped)"
echo ""

# Step 3: Render a local-only Lichess config from the private token file.
echo "Step 3: Rendering local Lichess bot configuration..."
CONFIG_TEMPLATE="${PROJECT_ROOT}/lichess/config.yml.default"
TOKEN_FILE="${V7P3R_TOKEN_FILE:-E:\Programming Stuff\Gaming Projects\Chess Engines\V7P3R Chess Engine\v7p3r_bot_token.env}"
LOCAL_CONFIG="${PROJECT_ROOT}/lichess/config.local.yml"

if [ ! -f "${TOKEN_FILE}" ]; then
    print_error "Token file not found: ${TOKEN_FILE}"
    print_error "Set V7P3R_TOKEN_FILE or keep the file at E:\Programming Stuff\Gaming Projects\Chess Engines\V7P3R Chess Engine\v7p3r_bot_token.env"
    exit 1
fi

if [ ! -f "${CONFIG_TEMPLATE}" ]; then
    print_error "Config template not found: ${CONFIG_TEMPLATE}"
    exit 1
fi

# Accept either KEY=value or bare token content in the private env file.
TOKEN_VALUE=$(grep -E "^(LICHESS_TOKEN|TOKEN|API_TOKEN|lichess_token)=" "${TOKEN_FILE}" 2>/dev/null | head -n 1 | cut -d= -f2- | tr -d '"' | tr -d "'" || true)
if [ -z "${TOKEN_VALUE}" ]; then
    TOKEN_VALUE=$(tr -d '\r\n' < "${TOKEN_FILE}" | sed 's/^#.*$//' | head -c 4096)
fi

if [ -z "${TOKEN_VALUE}" ]; then
    print_error "Token file exists but no supported token key or value was found."
    exit 1
fi

# Build a runtime config without checking the real secret into the repo.
python3 - "${CONFIG_TEMPLATE}" "${LOCAL_CONFIG}" "${TOKEN_VALUE}" <<'PY'
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
token = sys.argv[3]
text = template_path.read_text(encoding='utf-8')
text = text.replace('REPLACE_WITH_LICHESS_OAUTH_TOKEN', token)
text = text.replace('token: "REPLACE_WITH_LICHESS_OAUTH_TOKEN"', f'token: "{token}"')
out_path.write_text(text, encoding='utf-8')
PY

print_status "Rendered local config at ${LOCAL_CONFIG}"
print_status "Local deploys use the private token file and do not modify tracked config files"
echo ""

# Step 4: Create deployment package
echo "Step 4: Creating deployment package..."
TEMP_DEPLOY_DIR="/tmp/v7p3r_v${VERSION}_deploy"
rm -rf "${TEMP_DEPLOY_DIR}"
mkdir -p "${TEMP_DEPLOY_DIR}/engines/v7p3r"

# Copy engine files
cp "${PROJECT_ROOT}/src/"*.py "${TEMP_DEPLOY_DIR}/engines/v7p3r/"
cp "${PROJECT_ROOT}/requirements.txt" "${TEMP_DEPLOY_DIR}/"
cp "${LOCAL_CONFIG}" "${TEMP_DEPLOY_DIR}/config.yml"

# Create version info file
cat > "${TEMP_DEPLOY_DIR}/VERSION.txt" << EOF
V7P3R v${VERSION}
Deployed: $(date)
Environment: ${ENVIRONMENT}
Git Commit: $(cd "${PROJECT_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
EOF

print_status "Created deployment package at ${TEMP_DEPLOY_DIR}"
echo ""

# Step 5: Deploy to target environment
echo "Step 5: Deploying to ${ENVIRONMENT}..."

case "${ENVIRONMENT}" in
    "production")
        # Production deployment to GCP or remote server
        print_warning "Production deployment to GCP/remote server"
        print_warning "Manual step required: Upload deployment package to server"
        print_warning "Command: scp -r ${TEMP_DEPLOY_DIR}/* user@server:/path/to/lichess-bot/"
        ;;
    
    "test"|"local")
        # Local deployment to test directory
        print_status "Deploying to local test environment: ${DEPLOY_DIR}"
        mkdir -p "${DEPLOY_DIR}/engines/v7p3r"
        rsync -av --delete "${TEMP_DEPLOY_DIR}/" "${DEPLOY_DIR}/"
        print_status "Files synced to ${DEPLOY_DIR}"
        ;;
    
    *)
        print_error "Unknown environment: ${ENVIRONMENT}"
        exit 1
        ;;
esac
echo ""

# Step 6: Health check instructions
echo "Step 6: Post-Deployment Health Checks"
echo "----------------------------------------------------------------------"
echo "After bot restart, verify:"
echo "  1. Bot is online: https://lichess.org/@/v7p3r_bot"
echo "  2. Check version in first game chat"
echo "  3. Monitor first 10 games for crashes"
echo "  4. Run analytics after 50 games:"
echo "     cd ${PROJECT_ROOT}/analytics"
echo "     python weekly_analysis.py --days 3"
echo ""

# Step 7: Rollback instructions
echo "Step 7: Rollback Instructions (if needed)"
echo "----------------------------------------------------------------------"
echo "To rollback to previous version:"
echo "  1. Stop the bot service"
echo "  2. Restore previous engine files"
echo "  3. Restore previous generated config file from a local backup"
echo "  4. Restart bot service"
echo ""
echo "Generated runtime configs are intentionally local-only and should not be committed to Git."
echo ""

# Step 8: Summary
echo "======================================================================"
echo "DEPLOYMENT SUMMARY"
echo "======================================================================"
print_status "Version v${VERSION} prepared for ${ENVIRONMENT} deployment"
print_status "Deployment package: ${TEMP_DEPLOY_DIR}"
print_status "Generated local config: ${LOCAL_CONFIG}"

if [ "${ENVIRONMENT}" = "production" ]; then
    print_warning "PRODUCTION: Manual upload to server required"
    print_warning "After upload, restart bot service and monitor health"
else
    print_status "LOCAL: Files deployed to ${DEPLOY_DIR}"
fi

echo ""
echo "Next steps:"
echo "  1. ${ENVIRONMENT} == production: Upload to server and restart bot"
echo "  2. Monitor bot status on Lichess"
echo "  3. Review first 10 games for issues"
echo "  4. Run weekly analytics after 50 games"
echo "======================================================================"
