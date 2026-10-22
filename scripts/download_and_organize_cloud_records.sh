#!/bin/bash

# V7P3R Cloud Game Records Download and Organization Script
# Safely downloads game records from cloud VM and organizes them locally
# Maintains existing records while adding new V12.6 data

echo "🎯 V7P3R Cloud Game Records Download and Organization"
echo "=================================================="

# Configuration
LOCAL_GAME_DIR="s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records"
LICHESS_RECORDS_DIR="${LOCAL_GAME_DIR}/Lichess_V7P3R_Bot"
V12_6_DIR="${LICHESS_RECORDS_DIR}/v12.6_era"
CLOUD_BACKUP_DIR="${LICHESS_RECORDS_DIR}/cloud_backup_$(date +%Y%m%d_%H%M%S)"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p "${V12_6_DIR}"
mkdir -p "${CLOUD_BACKUP_DIR}"
mkdir -p "${LICHESS_RECORDS_DIR}/organized_by_date"
mkdir -p "${LICHESS_RECORDS_DIR}/organized_by_opponent"

echo "✅ Directories created:"
echo "   - V12.6 era: ${V12_6_DIR}"
echo "   - Cloud backup: ${CLOUD_BACKUP_DIR}"

# Note: Replace with actual cloud VM details when available
echo ""
echo "🚨 CLOUD DOWNLOAD INSTRUCTIONS:"
echo "Replace the following placeholders with your actual cloud VM details:"
echo ""
echo "# Example cloud download command (UPDATE WITH REAL DETAILS):"
echo "# scp -r user@your-cloud-vm:/lichess-bot/game_records/* \"${CLOUD_BACKUP_DIR}/\""
echo "# rsync -av user@your-cloud-vm:/lichess-bot/game_records/ \"${CLOUD_BACKUP_DIR}/\""
echo ""
echo "For GCP/AWS/Azure instances, use appropriate connection method:"
echo "# gcloud compute scp --recurse instance-name:/lichess-bot/game_records/* \"${CLOUD_BACKUP_DIR}/\""
echo "# aws ec2-instance-connect ssh --instance-id i-1234567890abcdef0 --os-user ec2-user"
echo ""

# Function to organize downloaded records
organize_records() {
    local source_dir="$1"
    
    echo "📊 Organizing game records from: ${source_dir}"
    
    if [ ! -d "${source_dir}" ]; then
        echo "❌ Source directory does not exist: ${source_dir}"
        return 1
    fi
    
    # Count PGN files
    pgn_count=$(find "${source_dir}" -name "*.pgn" | wc -l)
    echo "📈 Found ${pgn_count} PGN files to organize"
    
    if [ $pgn_count -eq 0 ]; then
        echo "⚠️  No PGN files found in source directory"
        return 0
    fi
    
    # Organize by date and opponent
    for pgn_file in "${source_dir}"/*.pgn; do
        if [ -f "$pgn_file" ]; then
            filename=$(basename "$pgn_file")
            
            # Extract date from PGN content if possible
            if grep -q "UTCDate" "$pgn_file"; then
                date_str=$(grep "UTCDate" "$pgn_file" | head -1 | sed 's/.*"\([^"]*\)".*/\1/' | tr '.' '-')
                date_dir="${LICHESS_RECORDS_DIR}/organized_by_date/${date_str}"
                mkdir -p "$date_dir"
                cp "$pgn_file" "$date_dir/"
            fi
            
            # Extract opponent from filename
            if [[ $filename =~ "vs" ]]; then
                opponent=$(echo "$filename" | sed 's/.*vs\.\? *\([^-\.]*\).*/\1/' | tr ' ' '_')
                opponent_dir="${LICHESS_RECORDS_DIR}/organized_by_opponent/${opponent}"
                mkdir -p "$opponent_dir"
                cp "$pgn_file" "$opponent_dir/"
            fi
            
            # Copy to V12.6 era directory
            cp "$pgn_file" "${V12_6_DIR}/"
        fi
    done
    
    echo "✅ Organization complete!"
    echo "📂 Files organized in:"
    echo "   - By date: ${LICHESS_RECORDS_DIR}/organized_by_date/"
    echo "   - By opponent: ${LICHESS_RECORDS_DIR}/organized_by_opponent/"
    echo "   - V12.6 era: ${V12_6_DIR}/"
}

# Organize existing local records if they exist
if [ -d "s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine/game_records" ]; then
    echo "📁 Organizing existing local Lichess records..."
    organize_records "s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine/game_records"
fi

# Generate analysis commands
echo ""
echo "🔍 ANALYSIS COMMANDS (run after cloud download):"
echo ""
echo "# Count games by opponent:"
echo "find \"${LICHESS_RECORDS_DIR}/organized_by_opponent\" -name \"*.pgn\" | wc -l"
echo ""
echo "# List most frequent opponents:"
echo "ls \"${LICHESS_RECORDS_DIR}/organized_by_opponent\" | head -10"
echo ""
echo "# Count games by date:"
echo "find \"${LICHESS_RECORDS_DIR}/organized_by_date\" -name \"*.pgn\" | wc -l"
echo ""
echo "# Extract win/loss statistics (example):"
echo "grep -h \"Result\" \"${V12_6_DIR}\"/*.pgn | sort | uniq -c"
echo ""

# Create a simple analysis script
cat > "${LICHESS_RECORDS_DIR}/analyze_v12_6_performance.py" << 'EOF'
#!/usr/bin/env python3
"""
V7P3R v12.6 Performance Analysis Script
Analyzes PGN files to extract performance metrics
"""

import os
import re
import chess.pgn
from collections import defaultdict, Counter
from pathlib import Path

def analyze_pgn_directory(directory):
    """Analyze all PGN files in directory"""
    results = {
        'total_games': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'by_opponent': defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0}),
        'by_time_control': defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0}),
        'by_opening': defaultdict(int),
        'rating_changes': [],
        'game_lengths': []
    }
    
    pgn_files = list(Path(directory).glob('*.pgn'))
    print(f"Analyzing {len(pgn_files)} PGN files...")
    
    for pgn_file in pgn_files:
        try:
            with open(pgn_file, 'r') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Basic game info
                    results['total_games'] += 1
                    
                    # Result analysis
                    result = game.headers.get('Result', '1/2-1/2')
                    white = game.headers.get('White', '')
                    black = game.headers.get('Black', '')
                    
                    # Determine if V7P3R was white or black
                    v7p3r_color = None
                    opponent = None
                    if 'v7p3r' in white.lower():
                        v7p3r_color = 'white'
                        opponent = black
                    elif 'v7p3r' in black.lower():
                        v7p3r_color = 'black'
                        opponent = white
                    
                    if v7p3r_color:
                        # Count wins/losses/draws
                        if (result == '1-0' and v7p3r_color == 'white') or (result == '0-1' and v7p3r_color == 'black'):
                            results['wins'] += 1
                            results['by_opponent'][opponent]['wins'] += 1
                        elif (result == '0-1' and v7p3r_color == 'white') or (result == '1-0' and v7p3r_color == 'black'):
                            results['losses'] += 1
                            results['by_opponent'][opponent]['losses'] += 1
                        else:
                            results['draws'] += 1
                            results['by_opponent'][opponent]['draws'] += 1
                    
                    # Time control analysis
                    time_control = game.headers.get('TimeControl', 'unknown')
                    if v7p3r_color:
                        if (result == '1-0' and v7p3r_color == 'white') or (result == '0-1' and v7p3r_color == 'black'):
                            results['by_time_control'][time_control]['wins'] += 1
                        elif (result == '0-1' and v7p3r_color == 'white') or (result == '1-0' and v7p3r_color == 'black'):
                            results['by_time_control'][time_control]['losses'] += 1
                        else:
                            results['by_time_control'][time_control]['draws'] += 1
                    
                    # Opening analysis
                    opening = game.headers.get('Opening', 'Unknown')
                    results['by_opening'][opening] += 1
                    
                    # Game length
                    moves = list(game.mainline_moves())
                    results['game_lengths'].append(len(moves))
                    
                    # Rating changes
                    if v7p3r_color == 'white':
                        rating_diff = game.headers.get('WhiteRatingDiff', '')
                    else:
                        rating_diff = game.headers.get('BlackRatingDiff', '')
                    
                    if rating_diff and rating_diff not in ['', '?']:
                        try:
                            results['rating_changes'].append(int(rating_diff))
                        except ValueError:
                            pass
                            
        except Exception as e:
            print(f"Error processing {pgn_file}: {e}")
    
    return results

def print_analysis(results):
    """Print formatted analysis results"""
    total = results['total_games']
    if total == 0:
        print("No games found!")
        return
    
    print(f"\n🎯 V7P3R v12.6 Performance Analysis")
    print(f"=" * 50)
    print(f"Total Games: {total}")
    print(f"Wins: {results['wins']} ({results['wins']/total*100:.1f}%)")
    print(f"Losses: {results['losses']} ({results['losses']/total*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/total*100:.1f}%)")
    
    if results['rating_changes']:
        avg_rating_change = sum(results['rating_changes']) / len(results['rating_changes'])
        print(f"Average Rating Change: {avg_rating_change:+.1f}")
    
    if results['game_lengths']:
        avg_length = sum(results['game_lengths']) / len(results['game_lengths'])
        print(f"Average Game Length: {avg_length:.1f} moves")
    
    # Top opponents
    print(f"\n📊 Top Opponents:")
    opponent_totals = [(opp, sum(stats.values())) for opp, stats in results['by_opponent'].items()]
    for opponent, total_games in sorted(opponent_totals, key=lambda x: x[1], reverse=True)[:10]:
        stats = results['by_opponent'][opponent]
        win_rate = stats['wins'] / total_games * 100 if total_games > 0 else 0
        print(f"  {opponent}: {total_games} games, {win_rate:.1f}% win rate")
    
    # Top openings
    print(f"\n♟️  Most Common Openings:")
    for opening, count in sorted(results['by_opening'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {opening}: {count} games ({count/total*100:.1f}%)")

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    results = analyze_pgn_directory(directory)
    print_analysis(results)
EOF

chmod +x "${LICHESS_RECORDS_DIR}/analyze_v12_6_performance.py"

echo "📊 Created analysis script: ${LICHESS_RECORDS_DIR}/analyze_v12_6_performance.py"
echo ""
echo "🎯 TO RUN ANALYSIS AFTER DOWNLOAD:"
echo "python \"${LICHESS_RECORDS_DIR}/analyze_v12_6_performance.py\" \"${V12_6_DIR}\""
echo ""
echo "✅ Setup complete! Download your cloud records and run the analysis."