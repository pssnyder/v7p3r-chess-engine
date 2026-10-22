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
    backup_dir = base_dir / f"pre_cloud_integration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
