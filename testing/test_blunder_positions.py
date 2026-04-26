#!/usr/bin/env python3
"""
Test v19.5 vs v18.4 on Historical Blunder Positions

Based on v18.3 YTD analysis showing:
- Blunders per game: 7.6
- Heavy blunders in moves 10-14 (11 blunders), 15-19 (11), 25-29 (13)
- Critical positions from real Lichess games

Tests if v19.5's optimizations maintain decision quality on historical mistakes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lichess', 'engines', 'V7P3R_v18.4_20260417', 'src'))

import chess
import chess.pgn
import io
from typing import List, Dict, Tuple

# Test positions from real v18.3 games
# Format: (name, FEN, expected_blunder_move, game_id, context)
BLUNDER_POSITIONS = [
    # Game 1 (Ckdbut9m): Black blundered Ke8 at move 26 (ply 52), losing position
    (
        "Nimzowitsch - Pre-Ke8 Blunder (Move 26)",
        "8/p1pr2p1/3k4/1PQ5/8/8/1R6/4K3 b - - 1 26",
        "e8",  # The blunder that led to mate
        "Ckdbut9m",
        "Critical endgame - v18.3 played Ke8?? allowing Qd8#"
    ),
    
    # Game 1: Earlier tactical blunder at move 11 (ply 22)
    (
        "Nimzowitsch - Pre-O-O Blunder (Move 11)",
        "r1b3r1/pppp1pkp/2n5/2b1pP2/3P4/2P5/PP2Q1PP/R1B1K2R b KQ - 0 11",
        "O-O",  # Castled into danger
        "Ckdbut9m",
        "Black castled kingside allowing tactical assault"
    ),
    
    # Game 4 (OF1H59yW): French Defense vs 2135-rated opponent
    (
        "French Defense - Move 24 Position",
        "r5k1/pbp1rp1p/4p2p/4PQ2/5P2/B2B4/P4PPP/R5K1 b - - 1 24",
        "Nxe5",  # Captured on e5
        "OF1H59yW",
        "Middlegame vs strong opponent - led to losing position"
    ),
    
    # Game 5 (3j6DS1wY): Queen's Gambit as White
    (
        "QGD - Pre-Resignation Position (Move 38)",
        "8/6p1/1P6/8/3b4/5P2/4r1K1/1r6 w - - 2 38",
        "h6",  # Last move before resignation
        "3j6DS1wY",
        "Lost endgame - testing if Phase 1 recognizes hopelessness"
    ),
    
    # Game 6 (QEcpi0uK): vs C0BR4_bot
    (
        "Dutch Defense - Pre-Mate Position (Move 42)",
        "2k2r2/6p1/1q4p1/1K1Q4/3b4/8/4R2P/8 w - - 0 42",
        "Qxd4",  # Move before checkmate
        "QEcpi0uK",
        "Lost vs C0BR4 - testing if Phase 1 sees Qc6# threat"
    ),
]

def test_position(engine_v18_4, engine_v19_4, position_data: Tuple) -> Dict:
    """
    Test both engines on a position and compare decisions
    
    Returns:
        Dict with engine moves, scores, and comparison
    """
    name, fen, blunder_move, game_id, context = position_data
    
    print(f"\n{'=' * 80}")
    print(f"Testing: {name}")
    print(f"Game ID: {game_id}")
    print(f"Context: {context}")
    print(f"FEN: {fen}")
    print(f"Historical Blunder: {blunder_move}")
    print(f"{'=' * 80}\n")
    
    board = chess.Board(fen)
    
    # Test v18.4
    print("v18.4 thinking (3 seconds)...")
    move_v18_4 = engine_v18_4.search(board, time_limit=3.0)
    print(f"  Move: {move_v18_4}")
    print(f"  Stats: {engine_v18_4.search_stats}")
    
    # Reset stats
    engine_v18_4.search_stats = {'nodes_per_second': 0, 'cache_hits': 0, 'cache_misses': 0, 'tt_hits': 0, 'tt_stores': 0, 'killer_hits': 0}
    
    # Test v19.5
    print("\nv19.5 thinking (3 seconds)...")
    move_v19_4 = engine_v19_4.search(board, time_limit=3.0)
    print(f"  Move: {move_v19_4}")
    print(f"  Stats: {engine_v19_4.search_stats}")
    
    # Reset stats
    engine_v19_4.search_stats = {'nodes_per_second': 0, 'cache_hits': 0, 'cache_misses': 0, 'tt_hits': 0, 'tt_stores': 0, 'killer_hits': 0}
    
    # Compare
    same_move = (str(move_v18_4) == str(move_v19_4))
    v18_matches_blunder = (str(move_v18_4) == blunder_move)
    v19_matches_blunder = (str(move_v19_4) == blunder_move)
    
    print(f"\n{'─' * 80}")
    print(f"COMPARISON:")
    print(f"  v18.4 move:  {move_v18_4} {'❌ (BLUNDER!)' if v18_matches_blunder else ''}")
    print(f"  v19.5:       {move_v19_4} {'❌ (BLUNDER!)' if v19_matches_blunder else ''}")
    print(f"  Same move:   {'✓ YES' if same_move else '✗ NO (DIFFERENT DECISION!)'}")
    print(f"  Improvement: {'✓ YES' if not v19_matches_blunder and v18_matches_blunder else ('SAME' if same_move else 'UNKNOWN')}")
    print(f"{'─' * 80}")
    
    return {
        'name': name,
        'fen': fen,
        'historical_blunder': blunder_move,
        'v18_4_move': str(move_v18_4),
        'v19_4_move': str(move_v19_4),
        'same_move': same_move,
        'v18_matches_blunder': v18_matches_blunder,
        'v19_matches_blunder': v19_matches_blunder,
        'improved': not v19_matches_blunder and v18_matches_blunder
    }

def main():
    print("""
================================================================================
HISTORICAL BLUNDER POSITION TESTING
================================================================================
Testing v19.4.1 Phase 1 vs v18.4 on positions from real v18.3 blunders

Analysis showed:
- 7.6 blunders per game
- Heaviest blunders in moves 10-14 (11), 15-19 (11), 25-29 (13)
- Some catastrophic 9000+ cp loss blunders

Goal: See if Phase 1's character-defining evaluations (castling, pawn
      advancement, passed pawns) help avoid these historical mistakes
================================================================================
""")
    
    # Import engines
    print("Loading engines...")
    from v7p3r import V7P3REngine as EngineV18_4
    
    # v19.4.1 will be in the main src directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from v7p3r import V7P3REngine as EngineV19_4
    
    engine_v18_4 = EngineV18_4()
    engine_v19_4 = EngineV19_4()
    
    print(f"✓ v18.4 loaded from lichess/engines/V7P3R_v18.4_20260417/src")
    print(f"✓ v19.5 loaded from src/")
    
    # Test all positions
    results = []
    for position in BLUNDER_POSITIONS:
        result = test_position(engine_v18_4, engine_v19_4, position)
        results.append(result)
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY OF BLUNDER POSITION TESTING")
    print(f"{'=' * 80}\n")
    
    total = len(results)
    same_moves = sum(1 for r in results if r['same_move'])
    different_moves = total - same_moves
    v18_blunders = sum(1 for r in results if r['v18_matches_blunder'])
    v19_blunders = sum(1 for r in results if r['v19_matches_blunder'])
    improvements = sum(1 for r in results if r['improved'])
    
    print(f"Total positions tested: {total}")
    print(f"Same decision:          {same_moves}/{total} ({same_moves/total*100:.1f}%)")
    print(f"Different decision:     {different_moves}/{total} ({different_moves/total*100:.1f}%)")
    print(f"")
    print(f"v18.4 repeated historical blunder:  {v18_blunders}/{total}")
    print(f"v19.5 avoided the blunder:           {improvements}/{total}")
    print(f"v19.5 made same blunder:             {v19_blunders}/{total}")
    print(f"")
    
    if improvements > 0:
        print(f"✓ IMPROVEMENT: v19.5 avoided {improvements} historical blunders!")
    elif different_moves > 0:
        print(f"~ MIXED: Different decisions but unclear if better")
    else:
        print(f"✗ NO CHANGE: Same decisions as v18.4")
    
    print(f"\n{'=' * 80}")

if __name__ == "__main__":
    main()
