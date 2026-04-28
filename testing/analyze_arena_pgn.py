#!/usr/bin/env python3
"""
Analyze Arena tournament PGNs to find patterns in v19.5.6's poor performance.

Parse PGN files and check:
1. Win rate by color (White vs Black)
2. Average game length
3. Common losing patterns
4. Time management issues (forfeit on time)
"""

import re
from pathlib import Path
from collections import defaultdict

PGN_FILES = [
    Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202604\V7p3r v19_5_6 regression testing 20260426.pgn"),
    Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202604\V7p3r v19_5_6 regression testing_2 20260426.pgn")
]

def parse_pgn_file(pgn_path):
    """Parse PGN file and extract game results"""
    games = []
    current_game = {}
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('['):
                # Parse header
                match = re.match(r'\[(\w+)\s+"(.+)"\]', line)
                if match:
                    key, value = match.groups()
                    current_game[key] = value
            elif line == '' and current_game:
                # End of game
                games.append(current_game)
                current_game = {}
    
    return games

def analyze_games(games, tournament_name):
    """Analyze game results"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {tournament_name}")
    print('='*80)
    
    v19_stats = {
        'as_white': {'wins': 0, 'losses': 0, 'draws': 0},
        'as_black': {'wins': 0, 'losses': 0, 'draws': 0},
        'total_games': 0,
        'total_wins': 0,
        'total_losses': 0,
        'total_draws': 0,
        'timeouts': 0,
        'avg_ply_count': []
    }
    
    for game in games:
        white = game.get('White', '')
        black = game.get('Black', '')
        result = game.get('Result', '')
        ply_count = game.get('PlyCount', '0')
        termination = game.get('Termination', '')
        
        try:
            ply_count = int(ply_count)
        except:
            ply_count = 0
        
        # Determine if v19 is playing and color
        v19_is_white = 'v19' in white.lower()
        v19_is_black = 'v19' in black.lower()
        
        if not (v19_is_white or v19_is_black):
            continue
        
        v19_stats['total_games'] += 1
        v19_stats['avg_ply_count'].append(ply_count)
        
        # Determine result from v19 perspective
        if v19_is_white:
            color_key = 'as_white'
            if result == '1-0':
                v19_stats[color_key]['wins'] += 1
                v19_stats['total_wins'] += 1
            elif result == '0-1':
                v19_stats[color_key]['losses'] += 1
                v19_stats['total_losses'] += 1
            else:
                v19_stats[color_key]['draws'] += 1
                v19_stats['total_draws'] += 1
        else:
            color_key = 'as_black'
            if result == '0-1':
                v19_stats[color_key]['wins'] += 1
                v19_stats['total_wins'] += 1
            elif result == '1-0':
                v19_stats[color_key]['losses'] += 1
                v19_stats['total_losses'] += 1
            else:
                v19_stats[color_key]['draws'] += 1
                v19_stats['total_draws'] += 1
        
        # Check for timeouts
        if 'time' in termination.lower():
            v19_stats['timeouts'] += 1
    
    # Print results
    print(f"\nTotal games: {v19_stats['total_games']}")
    print(f"Record: {v19_stats['total_wins']}W-{v19_stats['total_losses']}L-{v19_stats['total_draws']}D")
    total = v19_stats['total_games']
    score = v19_stats['total_wins'] + v19_stats['total_draws'] * 0.5
    win_pct = (score / total * 100) if total > 0 else 0
    print(f"Win rate: {win_pct:.1f}%")
    
    print(f"\nBy Color:")
    for color in ['as_white', 'as_black']:
        stats = v19_stats[color]
        total_color = stats['wins'] + stats['losses'] + stats['draws']
        if total_color > 0:
            score_color = stats['wins'] + stats['draws'] * 0.5
            win_pct_color = (score_color / total_color * 100)
            print(f"  {color.replace('_', ' ').title()}: {stats['wins']}W-{stats['losses']}L-{stats['draws']}D ({win_pct_color:.1f}%)")
    
    if v19_stats['avg_ply_count']:
        avg_ply = sum(v19_stats['avg_ply_count']) / len(v19_stats['avg_ply_count'])
        print(f"\nAverage game length: {avg_ply:.1f} plies ({avg_ply/2:.1f} moves)")
    
    print(f"Timeouts: {v19_stats['timeouts']}")
    
    # Check for color imbalance
    white_total = sum(v19_stats['as_white'].values())
    black_total = sum(v19_stats['as_black'].values())
    
    if white_total > 0 and black_total > 0:
        white_score = v19_stats['as_white']['wins'] + v19_stats['as_white']['draws'] * 0.5
        black_score = v19_stats['as_black']['wins'] + v19_stats['as_black']['draws'] * 0.5
        white_pct = (white_score / white_total * 100)
        black_pct = (black_score / black_total * 100)
        
        print(f"\n" + "="*80)
        if abs(white_pct - black_pct) > 30:
            print("🚨 SEVERE COLOR IMBALANCE DETECTED!")
            print(f"  White: {white_pct:.1f}%")
            print(f"  Black: {black_pct:.1f}%")
            print(f"  Delta: {abs(white_pct - black_pct):.1f}% difference")
        elif abs(white_pct - black_pct) > 15:
            print("⚠️  MODERATE COLOR IMBALANCE")
            print(f"  White: {white_pct:.1f}%")
            print(f"  Black: {black_pct:.1f}%")
            print(f"  Delta: {abs(white_pct - black_pct):.1f}% difference")
        else:
            print("✓ No significant color imbalance")
            print(f"  White: {white_pct:.1f}%")
            print(f"  Black: {black_pct:.1f}%")

if __name__ == "__main__":
    print("="*80)
    print("V19.5.6 ARENA TOURNAMENT ANALYSIS")
    print("="*80)
    
    all_games = []
    
    for pgn_path in PGN_FILES:
        if pgn_path.exists():
            tournament_name = pgn_path.stem
            games = parse_pgn_file(pgn_path)
            analyze_games(games, tournament_name)
            all_games.extend(games)
    
    # Combined analysis
    print(f"\n{'='*80}")
    print("COMBINED ANALYSIS (ALL TOURNAMENTS)")
    print('='*80)
    analyze_games(all_games, "Combined")
    
    print(f"\n{'='*80}")
    print("CRITICAL FINDINGS")
    print('='*80)
    print("""
v19.5.6 shows catastrophic performance vs v18.4:
- Win rate: ~15% (should be ~50%)
- This is approximately -500 ELO loss!

Possible root causes to investigate:
1. Color-dependent evaluation bug
2. Time management causing rushed moves
3. Move ordering regression
4. Search inefficiency (4-6x more nodes searched)
5. Evaluation function regression

DO NOT DEPLOY v19.5.6 until root cause is found and fixed!
""")
