#!/usr/bin/env python3
"""Analyze V7P3R v14.2 tournament results"""

import re
from collections import defaultdict

# Read PGN file
with open("Engine Battle 20251107_2.pgn", "r", encoding="utf-8") as f:
    pgn_content = f.read()

# Parse games
games = pgn_content.split("\n\n[Event")
games = ["[Event" + g for g in games[1:]]  # Skip first empty split

# Standings
standings = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "score": 0.0})

# V7P3R v14.2 specific analysis
v7p3r_games = []

for game in games:
    # Extract metadata
    white_match = re.search(r'\[White "([^"]+)"\]', game)
    black_match = re.search(r'\[Black "([^"]+)"\]', game)
    result_match = re.search(r'\[Result "([^"]+)"\]', game)
    
    if not (white_match and black_match and result_match):
        continue
    
    white = white_match.group(1)
    black = black_match.group(1)
    result = result_match.group(1)
    
    # Track V7P3R_v14.2 games
    if "V7P3R_v14.2" in white or "V7P3R_v14.2" in black:
        v7p3r_games.append({
            "white": white,
            "black": black,
            "result": result,
            "game": game
        })
    
    # Update standings
    if result == "1-0":
        standings[white]["wins"] += 1
        standings[white]["score"] += 1.0
        standings[black]["losses"] += 1
    elif result == "0-1":
        standings[black]["wins"] += 1
        standings[black]["score"] += 1.0
        standings[white]["losses"] += 1
    elif result == "1/2-1/2":
        standings[white]["draws"] += 1
        standings[white]["score"] += 0.5
        standings[black]["draws"] += 1
        standings[black]["score"] += 0.5

# Print standings
print("="*80)
print("TOURNAMENT STANDINGS")
print("="*80)
sorted_standings = sorted(standings.items(), key=lambda x: x[1]["score"], reverse=True)
for i, (engine, stats) in enumerate(sorted_standings, 1):
    total_games = stats["wins"] + stats["losses"] + stats["draws"]
    win_pct = (stats["wins"] / total_games * 100) if total_games > 0 else 0
    print(f"{i}. {engine:25s} | Score: {stats['score']:4.1f} | +{stats['wins']} ={stats['draws']} -{stats['losses']} | {win_pct:.1f}%")

# V7P3R_v14.2 head-to-head
print("\n" + "="*80)
print("V7P3R_v14.2 HEAD-TO-HEAD BREAKDOWN")
print("="*80)

h2h = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0})

for game_data in v7p3r_games:
    white = game_data["white"]
    black = game_data["black"]
    result = game_data["result"]
    
    if "V7P3R_v14.2" in white:
        opponent = black
        if result == "1-0":
            h2h[opponent]["wins"] += 1
        elif result == "0-1":
            h2h[opponent]["losses"] += 1
        else:
            h2h[opponent]["draws"] += 1
    else:
        opponent = white
        if result == "0-1":
            h2h[opponent]["wins"] += 1
        elif result == "1-0":
            h2h[opponent]["losses"] += 1
        else:
            h2h[opponent]["draws"] += 1

for opponent, stats in sorted(h2h.items()):
    total = stats["wins"] + stats["losses"] + stats["draws"]
    score = stats["wins"] + stats["draws"] * 0.5
    pct = (score / total * 100) if total > 0 else 0
    print(f"vs {opponent:25s} | +{stats['wins']} ={stats['draws']} -{stats['losses']} | {score}/{total} = {pct:.1f}%")

# Analyze losses to MaterialOpponent
print("\n" + "="*80)
print("V7P3R_v14.2 LOSSES TO MaterialOpponent - ANALYSIS")
print("="*80)

material_losses = [g for g in v7p3r_games if "MaterialOpponent" in (g["white"] + g["black"]) and 
                   ((g["result"] == "1-0" and "MaterialOpponent" in g["white"]) or 
                    (g["result"] == "0-1" and "MaterialOpponent" in g["black"]))]

for i, game_data in enumerate(material_losses, 1):
    print(f"\nLoss #{i}:")
    print(f"  White: {game_data['white']}")
    print(f"  Black: {game_data['black']}")
    print(f"  Result: {game_data['result']}")
    
    # Extract moves
    moves_match = re.search(r'\n\n(1\..+?)(?:\n\n|\Z)', game_data['game'], re.DOTALL)
    if moves_match:
        moves = moves_match.group(1).strip()
        move_list = re.findall(r'\d+\.\s*(\S+)(?:\s+(\S+))?', moves)
        move_count = len(move_list)
        print(f"  Moves: {move_count}")
        
        # Show first 10 moves
        first_moves = " ".join([f"{i+1}.{m[0]}" + (f" {m[1]}" if m[1] else "") 
                                for i, m in enumerate(move_list[:10])])
        print(f"  Opening: {first_moves}...")

print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
v7p3r_stats = standings.get("V7P3R_v14.2", {})
total = v7p3r_stats.get("wins", 0) + v7p3r_stats.get("losses", 0) + v7p3r_stats.get("draws", 0)
print(f"Total games: {total}")
print(f"Overall score: {v7p3r_stats.get('score', 0):.1f}/{total}")
print(f"Win rate: {(v7p3r_stats.get('wins', 0)/total*100) if total > 0 else 0:.1f}%")
print()

# Key observations
material_stats = h2h.get("MaterialOpponent_v1.0", {"wins": 0, "losses": 0, "draws": 0})
material_total = material_stats["wins"] + material_stats["losses"] + material_stats["draws"]
material_score = material_stats["wins"] + material_stats["draws"] * 0.5
print(f"vs MaterialOpponent: {material_score}/{material_total} = {(material_score/material_total*100) if material_total > 0 else 0:.1f}%")
print(f"  Expected: 50%+ (our target)")
print(f"  Result: {'[FAIL]' if material_total > 0 and material_score/material_total < 0.5 else '[PASS]'}")
