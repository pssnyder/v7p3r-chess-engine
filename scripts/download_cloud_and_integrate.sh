#!/bin/bash

# V7P3R Cloud Records Download and Integration Script
# Downloads from cloud VM and merges with existing records

echo "🎯 V7P3R Cloud Records Download and Integration"
echo "=============================================="

# Configuration
LICHESS_DIR="s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine"
CLOUD_TEMP_DIR="${LICHESS_DIR}/cloud_download_temp"
MAIN_RECORDS_DIR="${LICHESS_DIR}/game_records"

# Create temporary directory for cloud download
mkdir -p "$CLOUD_TEMP_DIR"

echo "📁 Directories:"
echo "  Cloud temp: $CLOUD_TEMP_DIR"
echo "  Main records: $MAIN_RECORDS_DIR"
echo ""

# Show current status
echo "📊 Current status after local merge:"
cd "$LICHESS_DIR"
python -c "
import os
from pathlib import Path

records_dir = Path('game_records')
if records_dir.exists():
    pgn_files = list(records_dir.glob('*.pgn'))
    total_games = 0
    for pgn_file in pgn_files:
        try:
            with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                total_games += content.count('[Event ')
        except:
            pass
    
    print(f'📈 Current merged records:')
    print(f'  PGN files: {len(pgn_files)}')
    print(f'  Total games: {total_games}')
else:
    print('❌ Game records directory not found')
"
echo ""

echo "🔍 Cloud Download Instructions:"
echo ""
echo "STEP 1: Connect to your cloud VM and check bot status"
echo "======================================================"
echo ""
echo "Replace 'YOUR_VM_DETAILS' with your actual cloud VM connection info:"
echo ""
echo "# For Google Cloud:"
echo "gcloud compute ssh your-instance-name --zone=your-zone"
echo ""
echo "# For AWS:"
echo "ssh -i your-key.pem ec2-user@your-instance-ip"
echo ""
echo "# For Azure:"
echo "ssh azureuser@your-instance-ip"
echo ""
echo "# General SSH:"
echo "ssh user@your-vm-ip"
echo ""

echo "STEP 2: Check bot status on VM"
echo "==============================="
echo ""
echo "Once connected to VM, run these commands:"
echo ""
echo "# Check if bot is running"
echo "docker ps | grep v7p3r"
echo ""
echo "# Check recent games"
echo "ls -la /lichess-bot/game_records/*.pgn | tail -10"
echo ""
echo "# Count total games on VM"
echo "find /lichess-bot/game_records -name '*.pgn' -exec grep -l 'Event' {} \; | wc -l"
echo ""
echo "# Check most recent game dates"
echo "grep 'UTCDate' /lichess-bot/game_records/*.pgn | tail -5"
echo ""

echo "STEP 3: Download from VM to local temp directory"
echo "================================================"
echo ""
echo "Run ONE of these commands from your LOCAL machine:"
echo ""
echo "# Using SCP (replace connection details):"
echo "scp -r user@your-vm-ip:/lichess-bot/game_records/* \"$CLOUD_TEMP_DIR/\""
echo ""
echo "# Using rsync (recommended):"
echo "rsync -av user@your-vm-ip:/lichess-bot/game_records/ \"$CLOUD_TEMP_DIR/\""
echo ""
echo "# For Google Cloud:"
echo "gcloud compute scp --recurse your-instance-name:/lichess-bot/game_records/* \"$CLOUD_TEMP_DIR/\" --zone=your-zone"
echo ""
echo "# Using Docker copy if running locally:"
echo "docker cp v7p3r-bot-container:/lichess-bot/game_records/. \"$CLOUD_TEMP_DIR/\""
echo ""

echo "STEP 4: After download, run the integration"
echo "==========================================="
echo ""
echo "After you've downloaded to the temp directory, run:"
echo ""
echo "cd \"s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine\""
echo "python scripts/integrate_cloud_records.py"
echo ""

# Create the integration script
cat > "s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/scripts/integrate_cloud_records.py" << 'EOF'
#!/usr/bin/env python3
"""
V7P3R Cloud Records Integration
Integrates downloaded cloud records with existing merged records
"""

import os
import shutil
from pathlib import Path
import hashlib
from datetime import datetime

def integrate_cloud_records():
    """Integrate cloud records with existing merged records"""
    
    # Paths
    base_dir = Path("s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine")
    cloud_temp_dir = base_dir / "cloud_download_temp"
    main_records_dir = base_dir / "game_records"
    backup_dir = base_dir / f"pre_cloud_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🎯 V7P3R Cloud Records Integration")
    print("=" * 50)
    print(f"Cloud temp: {cloud_temp_dir}")
    print(f"Main records: {main_records_dir}")
    print(f"Backup: {backup_dir}")
    print()
    
    # Check if cloud temp directory exists and has files
    if not cloud_temp_dir.exists():
        print(f"❌ Cloud temp directory not found: {cloud_temp_dir}")
        print("Please download cloud records first using the instructions above.")
        return
    
    cloud_files = list(cloud_temp_dir.glob("*.pgn"))
    if len(cloud_files) == 0:
        print(f"❌ No PGN files found in: {cloud_temp_dir}")
        print("Please check the download was successful.")
        return
    
    print(f"📁 Found {len(cloud_files)} PGN files in cloud temp directory")
    
    # Create backup
    if main_records_dir.exists():
        print("📦 Creating backup of current records...")
        try:
            shutil.copytree(main_records_dir, backup_dir)
            print(f"✅ Backup created: {backup_dir}")
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            return
    
    # Count games before integration
    games_before = 0
    if main_records_dir.exists():
        for pgn_file in main_records_dir.glob("*.pgn"):
            try:
                with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    games_before += content.count('[Event ')
            except:
                pass
    
    print(f"📊 Games before integration: {games_before}")
    
    # Process each cloud file
    files_integrated = 0
    files_created = 0
    games_added = 0
    
    for cloud_file in cloud_files:
        filename = cloud_file.name
        target_file = main_records_dir / filename
        
        print(f"🔄 Processing: {filename}")
        
        # Count games in cloud file
        try:
            with open(cloud_file, 'r', encoding='utf-8', errors='ignore') as f:
                cloud_content = f.read()
                cloud_games = cloud_content.count('[Event ')
            print(f"  Cloud file games: {cloud_games}")
        except Exception as e:
            print(f"  ❌ Error reading cloud file: {e}")
            continue
        
        if target_file.exists():
            # Merge with existing file
            try:
                with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                    target_content = f.read()
                
                target_games_before = target_content.count('[Event ')
                
                # Simple merge - append if content not already present
                if cloud_content.strip() not in target_content:
                    with open(target_file, 'a', encoding='utf-8') as f:
                        if target_content and not target_content.endswith('\n'):
                            f.write('\n')
                        f.write(cloud_content)
                    
                    # Count games after merge
                    with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                        merged_content = f.read()
                        target_games_after = merged_content.count('[Event ')
                    
                    games_added_this_file = target_games_after - target_games_before
                    games_added += games_added_this_file
                    print(f"  ✅ Merged (+{games_added_this_file} games)")
                    files_integrated += 1
                else:
                    print(f"  ⏭️  No new content to merge")
                    
            except Exception as e:
                print(f"  ❌ Error merging: {e}")
        else:
            # Create new file
            try:
                main_records_dir.mkdir(exist_ok=True)
                shutil.copy2(cloud_file, target_file)
                games_added += cloud_games
                print(f"  ✅ Created new file ({cloud_games} games)")
                files_created += 1
            except Exception as e:
                print(f"  ❌ Error creating file: {e}")
    
    # Count total games after integration
    games_after = 0
    for pgn_file in main_records_dir.glob("*.pgn"):
        try:
            with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                games_after += content.count('[Event ')
        except:
            pass
    
    print()
    print("📈 Integration Summary:")
    print(f"  Cloud files processed: {len(cloud_files)}")
    print(f"  Files integrated: {files_integrated}")
    print(f"  Files created: {files_created}")
    print(f"  Games before: {games_before}")
    print(f"  Games after: {games_after}")
    print(f"  Games added: {games_after - games_before}")
    print()
    
    # Clean up temp directory option
    print("🧹 Cleanup:")
    print(f"  Cloud temp directory: {cloud_temp_dir}")
    print(f"  You can safely delete the temp directory after verification")
    print()
    
    print("✅ Integration complete!")
    print()
    print("🎯 Next steps:")
    print("1. Run analysis on complete dataset")
    print("2. Identify patterns for V13 development")
    print("3. Plan tactical improvements based on recent games")

if __name__ == "__main__":
    integrate_cloud_records()
EOF

echo ""
echo "📋 Quick Commands Summary:"
echo "========================="
echo ""
echo "# 1. Connect to VM and check status"
echo "ssh user@your-vm-ip"
echo ""
echo "# 2. Download records (choose one method)"
echo "rsync -av user@your-vm-ip:/lichess-bot/game_records/ \"$CLOUD_TEMP_DIR/\""
echo ""
echo "# 3. Integrate downloaded records"
echo "cd \"s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine\""
echo "python scripts/integrate_cloud_records.py"
echo ""
echo "# 4. Analyze complete dataset"
echo "python scripts/analyze_local_lichess_records.py \"$MAIN_RECORDS_DIR\""
echo ""

echo "✅ Ready for cloud download and integration!"