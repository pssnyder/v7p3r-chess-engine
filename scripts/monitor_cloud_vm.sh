#!/bin/bash

# V7P3R Cloud VM Monitoring and Game Records Download
# Connects to cloud VM, checks bot status, and downloads new game records

# Configuration - UPDATE THESE WITH YOUR ACTUAL VM DETAILS
VM_USER="your-username"
VM_HOST="your-vm-ip-or-hostname" 
VM_KEY_PATH="~/.ssh/your-vm-key"  # If using SSH key
REMOTE_BOT_DIR="/lichess-bot"
LOCAL_RECORDS_DIR="s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/cloud_records"

echo "🎯 V7P3R Cloud VM Monitor & Records Download"
echo "============================================"

# Create local directories
mkdir -p "$LOCAL_RECORDS_DIR"
mkdir -p "$LOCAL_RECORDS_DIR/game_records"
mkdir -p "$LOCAL_RECORDS_DIR/logs"

echo "📅 $(date)"
echo ""

# Function to check if VM is accessible
check_vm_connection() {
    echo "🔍 Checking VM connection..."
    
    # Try to connect to VM (update with your actual connection method)
    if command -v ssh >/dev/null 2>&1; then
        # SSH connection example
        ssh -i "$VM_KEY_PATH" -o ConnectTimeout=10 "$VM_USER@$VM_HOST" "echo 'Connection successful'" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✅ VM connection successful"
            return 0
        else
            echo "❌ Cannot connect to VM"
            return 1
        fi
    else
        echo "⚠️  SSH not available, skipping connection test"
        return 0
    fi
}

# Function to check bot status
check_bot_status() {
    echo "🤖 Checking V7P3R bot status..."
    
    # Example commands to check bot status (adjust for your setup)
    echo "Commands to run on VM:"
    echo "  docker ps | grep v7p3r"
    echo "  tail -n 10 $REMOTE_BOT_DIR/logs/*.log"
    echo "  ls -la $REMOTE_BOT_DIR/game_records/*.pgn | tail -5"
    echo ""
    
    # If you have direct access, uncomment and modify:
    # ssh -i "$VM_KEY_PATH" "$VM_USER@$VM_HOST" "docker ps | grep v7p3r"
    # ssh -i "$VM_KEY_PATH" "$VM_USER@$VM_HOST" "tail -n 10 $REMOTE_BOT_DIR/logs/*.log"
}

# Function to download game records
download_records() {
    echo "📥 Downloading game records..."
    
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_dir="$LOCAL_RECORDS_DIR/backup_$timestamp"
    
    mkdir -p "$backup_dir"
    
    echo "📁 Download target: $backup_dir"
    
    # Example download commands (adjust for your setup)
    echo "Download commands to run:"
    echo "  scp -i $VM_KEY_PATH -r $VM_USER@$VM_HOST:$REMOTE_BOT_DIR/game_records/* $backup_dir/"
    echo "  scp -i $VM_KEY_PATH -r $VM_USER@$VM_HOST:$REMOTE_BOT_DIR/logs/* $LOCAL_RECORDS_DIR/logs/"
    echo ""
    
    # For manual copy-paste:
    echo "🔗 Manual download commands:"
    echo "rsync -av -e \"ssh -i $VM_KEY_PATH\" $VM_USER@$VM_HOST:$REMOTE_BOT_DIR/game_records/ $backup_dir/"
    echo ""
    
    # If you have direct access, uncomment and modify:
    # scp -i "$VM_KEY_PATH" -r "$VM_USER@$VM_HOST:$REMOTE_BOT_DIR/game_records/*" "$backup_dir/"
    
    echo "✅ Download commands prepared"
}

# Function to analyze new records
analyze_new_records() {
    echo "📊 Analyzing downloaded records..."
    
    local latest_backup=$(ls -t "$LOCAL_RECORDS_DIR"/backup_* 2>/dev/null | head -1)
    
    if [ -n "$latest_backup" ] && [ -d "$latest_backup" ]; then
        echo "🔍 Latest backup: $latest_backup"
        
        # Count new games
        local pgn_count=$(find "$latest_backup" -name "*.pgn" 2>/dev/null | wc -l)
        echo "📈 PGN files found: $pgn_count"
        
        if [ $pgn_count -gt 0 ]; then
            echo "🎯 Run analysis:"
            echo "python scripts/analyze_local_lichess_records.py \"$latest_backup\""
        fi
    else
        echo "⚠️  No backup directory found"
    fi
}

# Function to check for recent activity
check_recent_activity() {
    echo "📈 Recent activity check..."
    
    echo "Manual commands to check activity on VM:"
    echo "  ls -la $REMOTE_BOT_DIR/game_records/*.pgn | tail -10"
    echo "  grep 'Result' $REMOTE_BOT_DIR/game_records/v7p3r_bot* | tail -10"
    echo "  docker logs v7p3r-bot-container | tail -20"
    echo ""
}

# Main execution
main() {
    echo "🚀 Starting V7P3R cloud monitoring..."
    
    # Comment out functions you don't want to run automatically
    check_vm_connection
    check_bot_status
    check_recent_activity
    download_records
    analyze_new_records
    
    echo ""
    echo "📋 Next Steps:"
    echo "1. Update VM connection details in this script"
    echo "2. Run the provided download commands manually"
    echo "3. Analyze downloaded records with provided Python script"
    echo "4. Monitor bot performance and rating changes"
    echo ""
    echo "🎯 For V13 development:"
    echo "1. Look for tactical failure patterns in lost games"
    echo "2. Identify most common opening weaknesses"
    echo "3. Track rating progression over time"
    echo "4. Note opponent types and strengths"
}

# Show usage if no parameters
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --download-only    Only run download functions"
    echo "  --analyze-only     Only run analysis functions"  
    echo "  --check-only       Only check VM status"
    echo "  --help            Show this help message"
    echo ""
    echo "Before first use:"
    echo "1. Update VM_USER, VM_HOST, and VM_KEY_PATH variables"
    echo "2. Ensure SSH access to your cloud VM"
    echo "3. Verify paths to lichess-bot directory on VM"
    exit 0
fi

# Handle command line options
case "$1" in
    --download-only)
        download_records
        analyze_new_records
        ;;
    --analyze-only)
        analyze_new_records
        ;;
    --check-only)
        check_vm_connection
        check_bot_status
        check_recent_activity
        ;;
    *)
        main
        ;;
esac