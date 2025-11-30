#!/bin/bash
# Quick Start Script - Test V7P3R Analytics Locally
# This script helps you test the analytics system before deploying to GCP

set -e

echo "=========================================="
echo "V7P3R Analytics - Local Quick Start"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python
echo -n "Checking Python installation... "
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi
echo -e "${GREEN}✓${NC}"

# Check Stockfish
echo -n "Checking Stockfish installation... "
if ! command -v stockfish &> /dev/null; then
    echo -e "${YELLOW}⚠ Stockfish not found in PATH${NC}"
    echo ""
    echo "Please install Stockfish:"
    echo "  Ubuntu/Debian: sudo apt install stockfish"
    echo "  macOS: brew install stockfish"
    echo "  Windows: Download from https://stockfishchess.org/download/"
    echo ""
    read -p "Enter full path to Stockfish executable: " STOCKFISH_PATH
    if [ ! -f "$STOCKFISH_PATH" ]; then
        echo -e "${RED}✗ Stockfish not found at $STOCKFISH_PATH${NC}"
        exit 1
    fi
else
    STOCKFISH_PATH=$(which stockfish)
    echo -e "${GREEN}✓ Found at $STOCKFISH_PATH${NC}"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check for test PGN
echo ""
echo "Checking for test PGN files..."
if [ ! -d "test_pgns" ] || [ -z "$(ls -A test_pgns/*.pgn 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠ No test PGN files found${NC}"
    echo ""
    echo "You can:"
    echo "  1. Download games from GCP (requires gcloud configured)"
    echo "  2. Copy PGN files manually to test_pgns/"
    echo "  3. Skip this test"
    echo ""
    read -p "Download recent games from GCP? (y/n): " DOWNLOAD
    
    if [ "$DOWNLOAD" = "y" ]; then
        mkdir -p test_pgns
        echo "Downloading games..."
        python3 game_collector.py test_pgns 7
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Games downloaded${NC}"
        else
            echo -e "${RED}✗ Download failed${NC}"
            exit 1
        fi
    else
        echo "Skipping game download. To test manually:"
        echo "  1. Place PGN files in test_pgns/"
        echo "  2. Run: python3 v7p3r_analytics.py $STOCKFISH_PATH test_pgns/game.pgn"
        exit 0
    fi
fi

# Count PGN files
PGN_COUNT=$(find test_pgns -name "*.pgn" | wc -l)
echo -e "${GREEN}✓ Found $PGN_COUNT PGN files${NC}"

# Run test analysis
echo ""
echo "=========================================="
echo "Running Test Analysis"
echo "=========================================="
echo ""

# Analyze first game
FIRST_PGN=$(find test_pgns -name "*.pgn" | head -1)
if [ -n "$FIRST_PGN" ]; then
    echo "Analyzing sample game: $(basename $FIRST_PGN)"
    echo ""
    python3 v7p3r_analytics.py "$STOCKFISH_PATH" "$FIRST_PGN"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Single game analysis successful${NC}"
    else
        echo -e "${RED}✗ Analysis failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ No PGN files to analyze${NC}"
    exit 1
fi

# Run full pipeline
echo ""
echo "=========================================="
echo "Running Full Pipeline"
echo "=========================================="
echo ""
read -p "Run full pipeline on all test games? (y/n): " RUN_PIPELINE

if [ "$RUN_PIPELINE" = "y" ]; then
    python3 weekly_pipeline.py \
        --stockfish "$STOCKFISH_PATH" \
        --work-dir ./test_workspace \
        --days-back 7
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Pipeline completed successfully!${NC}"
        echo ""
        echo "Reports generated:"
        ls -lh test_workspace/reports/
        echo ""
        echo "View markdown report:"
        LATEST_MD=$(ls -t test_workspace/reports/*.md | head -1)
        echo "  cat $LATEST_MD"
        echo ""
        echo "View JSON report:"
        LATEST_JSON=$(ls -t test_workspace/reports/*.json | head -1)
        echo "  cat $LATEST_JSON"
    else
        echo -e "${RED}✗ Pipeline failed${NC}"
        exit 1
    fi
fi

# Test email delivery
echo ""
echo "=========================================="
echo "Email Delivery Test (Optional)"
echo "=========================================="
echo ""
read -p "Test email delivery? Requires SendGrid API key (y/n): " TEST_EMAIL

if [ "$TEST_EMAIL" = "y" ]; then
    read -p "SendGrid API Key: " SENDGRID_API_KEY
    read -p "Recipient Email: " TO_EMAIL
    
    export SENDGRID_API_KEY
    export TO_EMAIL
    
    LATEST_MD=$(ls -t test_workspace/reports/*.md | head -1)
    LATEST_JSON=$(ls -t test_workspace/reports/*.json | head -1)
    
    if [ -f "$LATEST_MD" ] && [ -f "$LATEST_JSON" ]; then
        python3 email_delivery.py "$LATEST_MD" "$LATEST_JSON"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Email sent successfully!${NC}"
        else
            echo -e "${RED}✗ Email delivery failed${NC}"
        fi
    else
        echo -e "${RED}✗ Report files not found${NC}"
    fi
fi

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review the generated reports in test_workspace/reports/"
echo "  2. Customize analysis parameters in v7p3r_analytics.py"
echo "  3. Deploy to GCP: ./deploy_gcp.sh"
echo ""
echo "For help: cat README.md"
echo ""
