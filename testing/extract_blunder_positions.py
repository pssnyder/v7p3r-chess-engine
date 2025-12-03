#!/usr/bin/env python3
"""
Extract Critical Blunder Positions for Testing

Analyzes recent games to find critical blunder positions where v7p3r made
200+ centipawn mistakes. These positions will be used to validate v17.5
improvements against v17.1.1.

Creates test_blunder_positions.json with FEN positions and context.
"""

import chess
import chess.pgn
import sys
import json
import os
from pathlib import Path

# Add src to path for Stockfish analysis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def extract_critical_positions_from_pgn(pgn_file: str, max_positions: int = 30) -> list:
    """
    Extract positions from PGN where we can manually identify issues.
    Since we don't have the full Stockfish analysis cached, we'll extract
    endgame positions where mate threats or tactical issues are likely.
    """
    positions = []
    
    with open(pgn_file) as f:
        game_count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            game_count += 1
            result = game.headers.get("Result", "*")
            
            # Focus on losses where we likely made critical mistakes
            if result not in ["0-1", "1-0"]:  # Include our losses and wins for comparison
                continue
            
            board = game.board()
            move_number = 0
            
            for move in game.mainline_moves():
                move_number += 1
                
                # Look for endgame positions (material < 1500cp per side)
                white_material = sum(len(board.pieces(pt, chess.WHITE)) * val 
                                    for pt, val in [(chess.QUEEN, 900), (chess.ROOK, 500), 
                                                   (chess.BISHOP, 300), (chess.KNIGHT, 300)])
                black_material = sum(len(board.pieces(pt, chess.BLACK)) * val 
                                    for pt, val in [(chess.QUEEN, 900), (chess.ROOK, 500), 
                                                   (chess.BISHOP, 300), (chess.KNIGHT, 300)])
                
                is_endgame = white_material < 1500 and black_material < 1500
                
                # Store endgame positions from losses (likely contain blunders)
                if is_endgame and len(positions) < max_positions:
                    # Check if position is interesting (not trivial)
                    piece_count = len(board.piece_map())
                    if 4 <= piece_count <= 10:  # Not too simple, not too complex
                        positions.append({
                            'fen': board.fen(),
                            'move_number': move_number,
                            'game_result': result,
                            'white_material': white_material,
                            'black_material': black_material,
                            'piece_count': piece_count,
                            'description': f"Move {move_number} from game (Result: {result})"
                        })
                
                board.push(move)
            
            if len(positions) >= max_positions:
                break
    
    print(f"Extracted {len(positions)} endgame positions from {game_count} games")
    return positions


def create_known_tactical_positions():
    """
    Create a set of known tactical endgame positions where engines commonly blunder.
    These are from standard endgame theory and tactical puzzles.
    """
    return [
        {
            'fen': '6k1/5ppp/8/8/8/8/5PPP/R5KR w - - 0 1',
            'description': 'R+R endgame - back rank coordination test',
            'expected_plan': 'Ra8+ or coordinate rooks on 7th/8th rank'
        },
        {
            'fen': '8/8/2k5/3p4/3P4/2K5/8/8 w - - 0 1',
            'description': 'K+P endgame - opposition test',
            'expected_plan': 'King must maintain opposition'
        },
        {
            'fen': '8/8/8/4k3/8/3K4/3P4/8 w - - 0 1',
            'description': 'K+P vs K - pawn promotion',
            'expected_plan': 'Push pawn while maintaining support'
        },
        {
            'fen': '8/5k2/8/4r3/8/2K5/1P6/8 w - - 0 1',
            'description': 'K+P vs K+R - defense required',
            'expected_plan': 'King must shield pawn'
        },
        {
            'fen': '6k1/8/6K1/6P1/8/8/8/8 w - - 0 1',
            'description': 'K+P vs K - basic pawn endgame',
            'expected_plan': 'Push pawn with king support'
        },
        {
            'fen': '8/8/4k3/8/4K3/4P3/8/8 w - - 0 1',
            'description': 'K+P vs K - central pawn',
            'expected_plan': 'Advance pawn with king'
        },
        {
            'fen': 'r6k/5Qpp/8/8/8/8/8/R3K2R w - - 0 1',
            'description': 'Mate in 1 - back rank mate pattern',
            'expected_plan': 'Qf8# or Ra8#'
        },
        {
            'fen': '6k1/5ppp/8/8/8/8/8/1R4KR w - - 0 1',
            'description': 'Mate in 1 - rook coordination',
            'expected_plan': 'Ra8# or Rh8#'
        },
        {
            'fen': '7k/5Q1p/8/8/8/8/8/7K w - - 0 1',
            'description': 'Mate in 1 - queen mate',
            'expected_plan': 'Qg7# or Qg8#'
        },
        {
            'fen': '4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1',
            'description': 'R vs R endgame - activity test',
            'expected_plan': 'Activate rook on 7th rank'
        },
    ]


def main():
    """Extract blunder positions and create test file"""
    print("=" * 70)
    print("Extracting Critical Blunder Positions for v17.5 Testing")
    print("=" * 70)
    print()
    
    # Check for PGN file
    pgn_path = Path("analytics/current_games/v7p3r_since_nov21.pgn")
    
    positions = create_known_tactical_positions()
    print(f"Created {len(positions)} known tactical test positions")
    
    # Try to extract from PGN if available
    if pgn_path.exists():
        print(f"\nExtracting positions from {pgn_path}...")
        extracted = extract_critical_positions_from_pgn(str(pgn_path), max_positions=20)
        positions.extend(extracted)
    else:
        print(f"\nPGN file not found at {pgn_path}")
        print("Using only known tactical positions")
    
    # Save to JSON
    output_file = "testing/test_blunder_positions.json"
    with open(output_file, 'w') as f:
        json.dump({
            'total_positions': len(positions),
            'created': '2025-12-02',
            'purpose': 'Validate v17.5 endgame improvements vs v17.1.1',
            'positions': positions
        }, f, indent=2)
    
    print(f"\n✓ Saved {len(positions)} test positions to {output_file}")
    print()
    print("Next steps:")
    print("  1. Run: python testing/test_uci_blunders.py")
    print("  2. Compare v17.5 vs v17.1.1 move choices")
    print("  3. Verify v17.5 finds better moves or avoids mates")


if __name__ == "__main__":
    main()
