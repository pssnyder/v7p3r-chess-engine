#!/bin/bash

# V7P3R Cloud Game Records Download Script
# Downloads game records from production GCP instance

PROJECT_ID="v7p3r-lichess-bot"
INSTANCE_NAME="v7p3r-production-bot"
ZONE="us-central1-a"
CONTAINER_NAME="v7p3r-production"

echo "🎯 V7P3R Cloud Game Records Download"
echo "===================================="
echo ""

# Local directories
LOCAL_CLOUD_TEMP="./cloud_download_temp"
LOCAL_RECORDS="./game_records"

# Set project
gcloud config set project $PROJECT_ID

# Create temp directory
mkdir -p "$LOCAL_CLOUD_TEMP"

echo "📁 Local directories:"
echo "  Cloud temp: $LOCAL_CLOUD_TEMP"
echo "  Local records: $LOCAL_RECORDS"
echo ""

echo "🔍 Checking cloud instance status..."
if ! gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &> /dev/null; then
    echo "❌ Production instance '$INSTANCE_NAME' not found!"
    echo "   Check your instance name and zone settings"
    exit 1
fi

INSTANCE_STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)" 2>/dev/null)
if [ "$INSTANCE_STATUS" != "RUNNING" ]; then
    echo "❌ Instance is not running (status: $INSTANCE_STATUS)"
    echo "   Start it with: gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
    exit 1
fi

echo "✅ Instance is running"

echo ""
echo "🎮 Checking bot status and game records..."

# Check container status and game count
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    echo '🐳 Container status:'
    sudo docker ps --filter name=$CONTAINER_NAME --format 'table {{.Names}}\t{{.Status}}'
    echo ''
    echo '📊 Game records on cloud:'
    if [ -d /home/v7p3r/game_records ]; then
        game_count=\$(find /home/v7p3r/game_records -name '*.pgn' 2>/dev/null | wc -l)
        echo \"  Total PGN files: \$game_count\"
        echo \"  Recent files:\"
        ls -la /home/v7p3r/game_records/*.pgn 2>/dev/null | tail -5 || echo '  No PGN files found'
    else
        echo '  Game records directory not found'
    fi
    echo ''
    echo '📅 Most recent game dates:'
    find /home/v7p3r/game_records -name '*.pgn' -exec grep -l 'UTCDate' {} \; 2>/dev/null | head -5 | xargs grep 'UTCDate' | tail -5 || echo '  No dates found'
" 2>/dev/null

echo ""
echo "📥 Downloading game records from cloud..."

# Download game records from cloud instance
echo "Command: gcloud compute scp --recurse $INSTANCE_NAME:/home/v7p3r/game_records/* \"$LOCAL_CLOUD_TEMP/\" --zone=$ZONE"

if gcloud compute scp --recurse $INSTANCE_NAME:/home/v7p3r/game_records/* "$LOCAL_CLOUD_TEMP/" --zone=$ZONE 2>/dev/null; then
    echo "✅ Download successful!"
    
    # Count downloaded files
    downloaded_count=$(find "$LOCAL_CLOUD_TEMP" -name "*.pgn" 2>/dev/null | wc -l)
    echo "📈 Downloaded $downloaded_count PGN files"
    
    if [ $downloaded_count -gt 0 ]; then
        echo ""
        echo "📋 Downloaded files:"
        ls -la "$LOCAL_CLOUD_TEMP"/*.pgn 2>/dev/null | head -10
        
        if [ $downloaded_count -gt 10 ]; then
            echo "... and $((downloaded_count - 10)) more files"
        fi
        
        echo ""
        echo "🎯 Next step: Integrate with local records"
        echo "Command: python \"s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/scripts/integrate_cloud_records.py\""
    else
        echo "⚠️  No PGN files found in download"
        echo "Check if the bot has played games recently"
    fi
else
    echo "❌ Download failed!"
    echo ""
    echo "🔧 Troubleshooting options:"
    echo "1. Check if instance is accessible:"
    echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    echo ""
    echo "2. Check game records directory on instance:"
    echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command=\"ls -la /home/v7p3r/game_records/\""
    echo ""
    echo "3. Alternative download method (if directory structure is different):"
    echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command=\"find /home -name '*.pgn' 2>/dev/null\""
    exit 1
fi

echo ""
echo "📊 Current local status before integration:"
if [ -d "$LOCAL_RECORDS" ]; then
    local_count=$(find "$LOCAL_RECORDS" -name "*.pgn" 2>/dev/null | wc -l)
    echo "  Local PGN files: $local_count"
else
    echo "  Local records directory not found"
fi

echo ""
echo "✅ Cloud download complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Integrate cloud records with local records:"
echo "   cd \"s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine\""
echo "   python scripts/integrate_cloud_records.py"
echo ""
echo "2. Analyze complete dataset:"
echo "   python scripts/analyze_local_lichess_records.py \"./game_records\""
echo ""
echo "3. Clean up temp directory after integration:"
echo "   rm -rf \"$LOCAL_CLOUD_TEMP\""