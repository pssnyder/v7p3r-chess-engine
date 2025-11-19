#!/usr/bin/env python3
"""
Analyze PositionalOpponent games from the Ultimate Engine Battle tournament

This script analyzes PositionalOpponent's 105 games to understand:
1. Where did it lose? (14.5 losses/draws - which opponents?)
2. What types of positions did it struggle with?
3. Were there material blunders like the Qxf6 issue in V15.0?
4. Was the 81.4% win rate legitimate or exploiting blind spots?
"""

import chess
import chess.pgn
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def analyze_game(game: chess.pgn.Game, opponent: str) -> Dict:
    """Analyze a single PositionalOpponent game"""
    board = game.board()
    result = game.headers.get("Result", "*")
    
    # Determine PositionalOpponent's color
    if game.headers.get("White") == "PositionalOpponent":
        po_color = "white"
        po_result = result
    else:
        po_color = "black"
        # Flip result for black
        if result == "1-0":
            po_result = "0-1"
        elif result == "0-1":
            po_result = "1-0"
        else:
            po_result = result
    
    # Track game metrics
    moves_count = 0
    material_swings = []
    blunders = []
    
    # Parse moves and look for material changes
    prev_material = 39  # Starting material (Q=9, R=5, B=3, N=3, P=1 each)
    
    for node in game.mainline():
        move = node.move
        
        # Get SAN before pushing
        try:
            san = board.san(move)
        except:
            san = str(move)
        
        moves_count += 1
        board.push(move)
        
        # Calculate material balance
        material = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            material += len(board.pieces(piece_type, chess.WHITE)) * [0, 1, 3, 3, 5, 9][piece_type]
            material -= len(board.pieces(piece_type, chess.BLACK)) * [0, 1, 3, 3, 5, 9][piece_type]
        
        # Detect big material swings (potential blunders)
        material_change = abs(material - (prev_material if moves_count > 1 else 0))
        if material_change >= 5:  # Queen or Rook level change
            # Check if it's PositionalOpponent's move
            is_po_move = (moves_count % 2 == 1 and po_color == "white") or \
                        (moves_count % 2 == 0 and po_color == "black")
            
            if is_po_move:
                blunders.append({
                    'move': moves_count,
                    'san': san,
                    'material_change': material_change,
                    'fen': board.fen()
                })
        
        material_swings.append(material)
        prev_material = material
    
    return {
        'opponent': opponent,
        'color': po_color,
        'result': po_result,
        'moves': moves_count,
        'blunders': blunders,
        'material_swings': material_swings
    }


def main():
    pgn_file = "s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Engine Battle 202511/Ultimate Engine Battle 20251108.pgn"
    
    print("=" * 80)
    print("POSITIONALOPPONENT GAME ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing: {pgn_file}")
    
    # Track statistics
    stats = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'total_games': 0,
        'games_by_opponent': defaultdict(list),
        'losses_by_opponent': defaultdict(int),
        'blunders_found': [],
        'loss_games': []
    }
    
    with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            
            # Check if PositionalOpponent is playing
            if "PositionalOpponent" in white or "PositionalOpponent" in black:
                opponent = black if "PositionalOpponent" in white else white
                analysis = analyze_game(game, opponent)
                
                stats['total_games'] += 1
                stats['games_by_opponent'][opponent].append(analysis)
                
                # Track results
                if analysis['result'] == "1-0":
                    stats['wins'] += 1
                elif analysis['result'] == "0-1":
                    stats['losses'] += 1
                    stats['losses_by_opponent'][opponent] += 1
                    stats['loss_games'].append((game, analysis))
                else:
                    stats['draws'] += 1
                
                # Track blunders
                if analysis['blunders']:
                    stats['blunders_found'].extend(analysis['blunders'])
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print('=' * 80)
    print(f"Total games: {stats['total_games']}")
    print(f"Wins: {stats['wins']} ({stats['wins']/stats['total_games']*100:.1f}%)")
    print(f"Losses: {stats['losses']} ({stats['losses']/stats['total_games']*100:.1f}%)")
    print(f"Draws: {stats['draws']} ({stats['draws']/stats['total_games']*100:.1f}%)")
    
    # Losses by opponent
    print(f"\n{'=' * 80}")
    print("LOSSES BY OPPONENT")
    print('=' * 80)
    for opponent, count in sorted(stats['losses_by_opponent'].items(), key=lambda x: x[1], reverse=True):
        total_games = len(stats['games_by_opponent'][opponent])
        print(f"{opponent:30s}: {count}/{total_games} losses ({count/total_games*100:.1f}%)")
    
    # Blunders found
    print(f"\n{'=' * 80}")
    print(f"MATERIAL BLUNDERS DETECTED: {len(stats['blunders_found'])}")
    print('=' * 80)
    if stats['blunders_found']:
        for i, blunder in enumerate(stats['blunders_found'][:10]):  # Show first 10
            print(f"\nBlunder #{i+1}:")
            print(f"  Move: {blunder['move']}")
            print(f"  Move SAN: {blunder['san']}")
            print(f"  Material change: {blunder['material_change']}")
            print(f"  Position: {blunder['fen']}")
    else:
        print("No major material blunders detected (5+ points)")
    
    # Detailed loss analysis
    print(f"\n{'=' * 80}")
    print(f"DETAILED LOSS ANALYSIS (First 5 losses)")
    print('=' * 80)
    
    for idx, (game, analysis) in enumerate(stats['loss_games'][:5]):
        print(f"\n--- Loss #{idx+1} vs {analysis['opponent']} ---")
        print(f"Color: {analysis['color']}")
        print(f"Moves: {analysis['moves']}")
        print(f"Opening: {game.headers.get('Opening', 'Unknown')}")
        
        # Show the game moves
        board = game.board()
        moves_list = []
        for node in game.mainline():
            moves_list.append(board.san(node.move))
            board.push(node.move)
        
        # Show last 10 moves
        print(f"Last 10 moves: {' '.join(moves_list[-10:])}")
        
        if analysis['blunders']:
            print(f"Blunders in this game: {len(analysis['blunders'])}")
            for blunder in analysis['blunders']:
                print(f"  Move {blunder['move']}: {blunder['san']} (material change: {blunder['material_change']})")
    
    # Win rate against top opponents
    print(f"\n{'=' * 80}")
    print("PERFORMANCE VS TOP OPPONENTS")
    print('=' * 80)
    
    top_opponents = ["V7P3R_v14.0", "V7P3R_v14.3", "V7P3R_v14.1", "V7P3R_v12.6", 
                    "CoverageOpponent", "C0BR4_v2.9"]
    
    for opponent in top_opponents:
        if opponent in stats['games_by_opponent']:
            games = stats['games_by_opponent'][opponent]
            wins = sum(1 for g in games if g['result'] == "1-0")
            losses = sum(1 for g in games if g['result'] == "0-1")
            draws = sum(1 for g in games if g['result'] not in ["1-0", "0-1"])
            total = len(games)
            score = wins + 0.5 * draws
            print(f"{opponent:30s}: {score}/{total} ({score/total*100:.1f}%) - W:{wins} D:{draws} L:{losses}")
    
    # Key findings
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS")
    print('=' * 80)
    
    total_losses_draws = stats['losses'] + stats['draws']
    print(f"1. PositionalOpponent lost/drew {total_losses_draws} of {stats['total_games']} games")
    print(f"2. Most losses to: {max(stats['losses_by_opponent'].items(), key=lambda x: x[1])[0] if stats['losses_by_opponent'] else 'None'}")
    print(f"3. Material blunders detected: {len(stats['blunders_found'])}")
    print(f"4. Average game length: {sum(g['moves'] for games in stats['games_by_opponent'].values() for g in games) / stats['total_games']:.1f} moves")
    
    # Check for V15.0 style blunders
    queen_sacs = [b for b in stats['blunders_found'] if b['material_change'] >= 8]
    if queen_sacs:
        print(f"\n⚠️  QUEEN-LEVEL BLUNDERS FOUND: {len(queen_sacs)}")
        print("   PositionalOpponent DOES have the same weakness as V15.0!")
        for qs in queen_sacs[:3]:
            print(f"   - Move {qs['move']}: {qs['san']}")
    else:
        print(f"\n✅ NO QUEEN-LEVEL BLUNDERS (8+ points) detected in sample")
        print("   PositionalOpponent appears to preserve major pieces better than V15.0")


if __name__ == "__main__":
    main()
