#!/usr/bin/env python3
"""
V7P3R Game Records Merger
Safely merges game_records_local with game_records without data loss
Handles duplicates and preserves all existing games
"""

import os
import shutil
from pathlib import Path
import hashlib
import re
from datetime import datetime

def calculate_file_hash(filepath):
    """Calculate MD5 hash of file content for duplicate detection"""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None

def count_games_in_pgn(filepath):
    """Count number of games in a PGN file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return content.count('[Event ')
    except Exception as e:
        print(f"Error counting games in {filepath}: {e}")
        return 0

def extract_game_info(filepath):
    """Extract basic info about games in PGN file"""
    games_info = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Find all events
            events = re.findall(r'\[Event "([^"]+)"\]', content)
            dates = re.findall(r'\[UTCDate "([^"]+)"\]', content)
            sites = re.findall(r'\[Site "([^"]+)"\]', content)
            
            for i in range(len(events)):
                info = {
                    'event': events[i] if i < len(events) else 'Unknown',
                    'date': dates[i] if i < len(dates) else 'Unknown',
                    'site': sites[i] if i < len(sites) else 'Unknown'
                }
                games_info.append(info)
                
    except Exception as e:
        print(f"Error extracting game info from {filepath}: {e}")
    
    return games_info

def merge_pgn_files(source_file, target_file):
    """Merge two PGN files, avoiding duplicate games"""
    try:
        # Read source content
        with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
            source_content = f.read()
        
        # Read target content if it exists
        target_content = ""
        if os.path.exists(target_file):
            with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
                target_content = f.read()
        
        # Simple duplicate detection - check if source content is already in target
        if source_content.strip() not in target_content:
            # Append source to target
            with open(target_file, 'a', encoding='utf-8') as f:
                if target_content and not target_content.endswith('\n'):
                    f.write('\n')
                f.write(source_content)
            return True
        else:
            print(f"  Content already exists in target file")
            return False
            
    except Exception as e:
        print(f"Error merging {source_file} to {target_file}: {e}")
        return False

def merge_game_records():
    """Main function to merge game records directories"""
    
    # Paths
    base_dir = Path("s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine")
    source_dir = base_dir / "game_records_local"
    target_dir = base_dir / "game_records"
    backup_dir = base_dir / f"game_records_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🎯 V7P3R Game Records Merger")
    print("=" * 50)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Backup: {backup_dir}")
    print()
    
    # Verify directories exist
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    if not target_dir.exists():
        print(f"❌ Target directory not found: {target_dir}")
        return
    
    # Create backup of target directory
    print("📦 Creating backup of target directory...")
    try:
        shutil.copytree(target_dir, backup_dir)
        print(f"✅ Backup created: {backup_dir}")
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return
    
    # Get list of files to merge
    source_files = list(source_dir.glob("*.pgn"))
    print(f"📁 Found {len(source_files)} PGN files in source directory")
    
    if len(source_files) == 0:
        print("⚠️  No PGN files found in source directory")
        return
    
    # Statistics
    total_games_before = 0
    total_games_after = 0
    files_merged = 0
    files_created = 0
    files_skipped = 0
    
    # Count games before merge
    for target_file in target_dir.glob("*.pgn"):
        total_games_before += count_games_in_pgn(target_file)
    
    print(f"📊 Games in target before merge: {total_games_before}")
    print()
    
    # Process each source file
    for source_file in source_files:
        filename = source_file.name
        target_file = target_dir / filename
        
        print(f"🔄 Processing: {filename}")
        
        source_games = count_games_in_pgn(source_file)
        print(f"  Source games: {source_games}")
        
        if target_file.exists():
            target_games_before = count_games_in_pgn(target_file)
            print(f"  Target games (before): {target_games_before}")
            
            # Merge files
            if merge_pgn_files(source_file, target_file):
                target_games_after = count_games_in_pgn(target_file)
                print(f"  Target games (after): {target_games_after}")
                print(f"  ✅ Merged (+{target_games_after - target_games_before} games)")
                files_merged += 1
            else:
                print(f"  ⏭️  Skipped (no new content)")
                files_skipped += 1
        else:
            # Copy new file
            try:
                shutil.copy2(source_file, target_file)
                print(f"  ✅ Created new file ({source_games} games)")
                files_created += 1
            except Exception as e:
                print(f"  ❌ Failed to copy: {e}")
        
        print()
    
    # Count total games after merge
    for target_file in target_dir.glob("*.pgn"):
        total_games_after += count_games_in_pgn(target_file)
    
    # Summary
    print("📈 Merge Summary:")
    print(f"  Files processed: {len(source_files)}")
    print(f"  Files merged: {files_merged}")
    print(f"  Files created: {files_created}")
    print(f"  Files skipped: {files_skipped}")
    print(f"  Games before: {total_games_before}")
    print(f"  Games after: {total_games_after}")
    print(f"  Games added: {total_games_after - total_games_before}")
    print()
    
    print("✅ Merge complete!")
    print(f"📦 Backup available at: {backup_dir}")
    print()
    
    # Show what's now in the merged directory
    print("📁 Merged directory contents:")
    target_files = list(target_dir.glob("*.pgn"))
    print(f"  Total PGN files: {len(target_files)}")
    
    # Show some statistics
    opponent_counts = {}
    for target_file in target_files:
        filename = target_file.name
        if " vs. " in filename:
            opponent = filename.split(" vs. ")[1].replace(".pgn", "")
            games = count_games_in_pgn(target_file)
            opponent_counts[opponent] = opponent_counts.get(opponent, 0) + games
    
    if opponent_counts:
        print("📊 Games by opponent (top 10):")
        sorted_opponents = sorted(opponent_counts.items(), key=lambda x: x[1], reverse=True)
        for opponent, count in sorted_opponents[:10]:
            print(f"  {opponent}: {count} games")
    
    print()
    print("🎯 Next steps:")
    print("1. Download latest records from cloud VM")
    print("2. Run this merger again with cloud records")
    print("3. Analyze complete dataset for V13 development")

if __name__ == "__main__":
    merge_game_records()