#!/usr/bin/env python3
"""
Quick V7P3R Local Lichess Records Analysis
Analyzes existing PGN files to provide immediate performance baseline
"""

import os
import re
from pathlib import Path

def analyze_local_records():
    """Analyze local lichess records for immediate baseline"""
    
    records_dir = Path("s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine/game_records")
    
    if not records_dir.exists():
        print("❌ Local records directory not found")
        return
    
    pgn_files = list(records_dir.glob("*.pgn"))
    
    print(f"🎯 V7P3R Local Lichess Records Analysis")
    print(f"=" * 50)
    print(f"📁 Directory: {records_dir}")
    print(f"📈 PGN Files Found: {len(pgn_files)}")
    
    total_games = 0
    v7p3r_wins = 0
    v7p3r_losses = 0
    v7p3r_draws = 0
    ongoing_games = 0
    
    opponents = {}
    time_controls = {}
    openings = {}
    ratings = []
    
    for pgn_file in pgn_files:
        with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Count games in this file
            games_in_file = content.count('[Event ')
            total_games += games_in_file
            
            # Extract results
            results = re.findall(r'\[Result "([^"]+)"\]', content)
            whites = re.findall(r'\[White "([^"]+)"\]', content)
            blacks = re.findall(r'\[Black "([^"]+)"\]', content)
            
            for i, result in enumerate(results):
                if i < len(whites) and i < len(blacks):
                    white = whites[i]
                    black = blacks[i]
                    
                    # Determine V7P3R color and opponent
                    v7p3r_color = None
                    opponent = None
                    
                    if 'v7p3r' in white.lower():
                        v7p3r_color = 'white'
                        opponent = black
                    elif 'v7p3r' in black.lower():
                        v7p3r_color = 'black'
                        opponent = white
                    
                    if v7p3r_color:
                        # Count results for V7P3R
                        if result == '*':
                            ongoing_games += 1
                        elif (result == '1-0' and v7p3r_color == 'white') or (result == '0-1' and v7p3r_color == 'black'):
                            v7p3r_wins += 1
                        elif (result == '0-1' and v7p3r_color == 'white') or (result == '1-0' and v7p3r_color == 'black'):
                            v7p3r_losses += 1
                        elif result == '1/2-1/2':
                            v7p3r_draws += 1
                        
                        # Track opponents
                        if opponent:
                            opponents[opponent] = opponents.get(opponent, 0) + 1
            
            # Extract time controls
            time_control_matches = re.findall(r'\[TimeControl "([^"]+)"\]', content)
            for tc in time_control_matches:
                time_controls[tc] = time_controls.get(tc, 0) + 1
            
            # Extract openings
            opening_matches = re.findall(r'\[Opening "([^"]+)"\]', content)
            for opening in opening_matches:
                openings[opening] = openings.get(opening, 0) + 1
            
            # Extract V7P3R ratings
            white_elos = re.findall(r'\[WhiteElo "(\d+)"\]', content)
            black_elos = re.findall(r'\[BlackElo "(\d+)"\]', content)
            
            for i in range(len(whites)):
                if i < len(whites) and 'v7p3r' in whites[i].lower() and i < len(white_elos):
                    ratings.append(int(white_elos[i]))
                elif i < len(blacks) and 'v7p3r' in blacks[i].lower() and i < len(black_elos):
                    ratings.append(int(black_elos[i]))
    
    # Calculate statistics
    completed_games = v7p3r_wins + v7p3r_losses + v7p3r_draws
    
    print(f"\n📊 Game Statistics:")
    print(f"Total Games: {total_games}")
    print(f"Completed Games: {completed_games}")
    print(f"Ongoing Games: {ongoing_games}")
    
    if completed_games > 0:
        print(f"\n🎯 V7P3R Performance:")
        print(f"Wins: {v7p3r_wins} ({v7p3r_wins/completed_games*100:.1f}%)")
        print(f"Losses: {v7p3r_losses} ({v7p3r_losses/completed_games*100:.1f}%)")
        print(f"Draws: {v7p3r_draws} ({v7p3r_draws/completed_games*100:.1f}%)")
        
        score = v7p3r_wins + (v7p3r_draws * 0.5)
        score_percentage = score / completed_games * 100
        print(f"Score: {score}/{completed_games} ({score_percentage:.1f}%)")
    
    if ratings:
        print(f"\n🏆 Rating Analysis:")
        print(f"Rating samples: {len(ratings)}")
        print(f"Current rating range: {min(ratings)} - {max(ratings)}")
        print(f"Average rating: {sum(ratings)/len(ratings):.0f}")
    
    # Top opponents
    if opponents:
        print(f"\n👥 Most Frequent Opponents:")
        sorted_opponents = sorted(opponents.items(), key=lambda x: x[1], reverse=True)
        for opponent, count in sorted_opponents[:10]:
            print(f"  {opponent}: {count} games")
    
    # Time controls
    if time_controls:
        print(f"\n⏱️  Time Controls:")
        for tc, count in sorted(time_controls.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tc}: {count} games")
    
    # Popular openings
    if openings:
        print(f"\n♟️  Most Common Openings:")
        for opening, count in sorted(openings.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {opening}: {count} games")
    
    print(f"\n📈 Performance Summary:")
    if completed_games > 0:
        if score_percentage > 60:
            print("🟢 Strong performance - above 60% score rate")
        elif score_percentage > 50:
            print("🟡 Solid performance - above 50% score rate") 
        else:
            print("🔴 Below 50% score rate - needs improvement")
    
    print(f"\n🎯 Next Steps:")
    print("1. Download cloud game records for complete analysis")
    print("2. Analyze tactical patterns in lost games")
    print("3. Review opening preparation effectiveness")
    print("4. Plan V13.0 tactical enhancements based on weaknesses")

if __name__ == "__main__":
    analyze_local_records()