#!/usr/bin/env python3
"""
V13.2 vs Traditional Engine Comparison
Head-to-head test of puzzle solver vs traditional minimax
"""

import os
import sys
import time
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from v7p3r import V7P3REngine

def compare_engines():
    """Compare V13.2 Puzzle Solver against traditional search."""
    print("=" * 60)
    print("V13.2 PUZZLE SOLVER vs TRADITIONAL MINIMAX")
    print("=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print()
    
    engine = V7P3REngine()
    
    # Test positions from different game phases
    test_positions = [
        ("Opening", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Sicilian", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("Middlegame", "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQ - 4 6"),
        ("Tactical", "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3"),
    ]
    
    total_traditional_time = 0
    total_puzzle_time = 0
    different_moves = 0
    
    for pos_name, fen in test_positions:
        print(f"\n{pos_name} Position:")
        print(f"FEN: {fen}")
        print("-" * 50)
        
        board = chess.Board(fen)
        
        # Traditional search with detailed timing
        engine.nodes_searched = 0
        start_time = time.time()
        traditional_move = engine.search(board, 2.0)
        traditional_time = time.time() - start_time
        traditional_nodes = engine.nodes_searched
        traditional_nps = int(traditional_nodes / traditional_time) if traditional_time > 0 else 0
        
        # Puzzle solver search with detailed timing  
        engine.nodes_searched = 0
        start_time = time.time()
        puzzle_move = engine.puzzle_search(board, 2.0)
        puzzle_time = time.time() - start_time
        puzzle_nodes = engine.nodes_searched
        puzzle_nps = int(puzzle_nodes / puzzle_time) if puzzle_time > 0 else 0
        
        # Compare results
        move_different = traditional_move != puzzle_move
        if move_different:
            different_moves += 1
        
        total_traditional_time += traditional_time
        total_puzzle_time += puzzle_time
        
        print(f"Traditional: {traditional_move}")
        print(f"  Time: {traditional_time:.2f}s | Nodes: {traditional_nodes:,} | NPS: {traditional_nps:,}")
        print(f"Puzzle Solver: {puzzle_move}")
        print(f"  Time: {puzzle_time:.2f}s | Nodes: {puzzle_nodes:,} | NPS: {puzzle_nps:,}")
        print(f"Speed improvement: {traditional_time/puzzle_time:.1f}x faster" if puzzle_time > 0 else "")
        print(f"Move selection: {'DIFFERENT' if move_different else 'SAME'}")
    
    print("\n" + "=" * 60)
    print("OVERALL COMPARISON RESULTS")
    print("=" * 60)
    print(f"Total traditional time: {total_traditional_time:.2f}s")
    print(f"Total puzzle solver time: {total_puzzle_time:.2f}s")
    print(f"Overall speed improvement: {total_traditional_time/total_puzzle_time:.1f}x")
    print(f"Positions with different moves: {different_moves}/{len(test_positions)}")
    print()
    
    if different_moves > 0:
        print("✓ Puzzle solver shows distinct strategic thinking")
    if total_puzzle_time < total_traditional_time:
        print("✓ Puzzle solver is significantly faster")
    if total_puzzle_time < total_traditional_time * 0.5:
        print("✓ Puzzle solver achieves major performance gains")

def test_move_quality():
    """Test the quality of moves chosen by each approach."""
    print("\n" + "=" * 60)
    print("MOVE QUALITY ANALYSIS")
    print("=" * 60)
    print("Analyzing strategic characteristics of move choices")
    print()
    
    engine = V7P3REngine()
    
    # Known positions with good/poor moves
    test_cases = [
        {
            "name": "Scholar's Mate Defense",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
            "good_moves": ["Qf3", "d3", "Nf3"],  # Attacking or developing
            "poor_moves": ["a3", "h3", "Nh3"]    # Passive or bad
        },
        {
            "name": "Center Opening",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "good_moves": ["e4", "d4", "Nf3"],   # Central or developing
            "poor_moves": ["e3", "a3", "h3"]     # Passive
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"FEN: {test_case['fen']}")
        
        board = chess.Board(test_case['fen'])
        
        # Get moves from both approaches
        traditional_move = engine.search(board, 1.0)
        puzzle_move = engine.puzzle_search(board, 1.0)
        
        # Analyze move quality
        def analyze_move(move, move_type):
            move_san = board.san(move)
            is_good = any(good in move_san for good in test_case['good_moves'])
            is_poor = any(poor in move_san for poor in test_case['poor_moves'])
            
            if is_good:
                quality = "GOOD"
            elif is_poor:
                quality = "POOR"
            else:
                quality = "NEUTRAL"
            
            print(f"  {move_type}: {move_san} ({quality})")
            return quality
        
        trad_quality = analyze_move(traditional_move, "Traditional")
        puzzle_quality = analyze_move(puzzle_move, "Puzzle Solver")
        
        # Compare
        if puzzle_quality == "GOOD" and trad_quality != "GOOD":
            print("  ✓ Puzzle solver chose better move")
        elif trad_quality == "GOOD" and puzzle_quality != "GOOD":
            print("  ○ Traditional chose better move")
        else:
            print("  = Similar move quality")

if __name__ == "__main__":
    try:
        compare_engines()
        test_move_quality()
        
        print("\n" + "=" * 60)
        print("COMPARISON COMPLETE")
        print("=" * 60)
        print("V13.2 Puzzle Solver evaluation complete!")
        print()
        print("Next steps:")
        print("1. Test against V12.6 in actual games")
        print("2. Run Universal Puzzle Analyzer")
        print("3. Deploy for tournament testing")
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()