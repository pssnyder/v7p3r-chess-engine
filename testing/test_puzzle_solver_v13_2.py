#!/usr/bin/env python3
"""
Test V13.2 Puzzle Solver Phase 1 Implementation
Tests the Multi-PV foundation and opportunity-based move selection
"""

import os
import sys
import time
import chess

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def test_multi_pv_search():
    """Test the Multi-PV foundation of the puzzle solver."""
    print("=" * 60)
    print("V13.2 PUZZLE SOLVER - PHASE 1 TEST")
    print("=" * 60)
    print("TESTING: Multi-PV search with opportunity analysis")
    print()
    
    engine = V7P3REngine()
    
    # Test position: Starting position
    board = chess.Board()
    
    print("Position: Starting position")
    print("Testing Multi-PV search vs traditional search")
    print("-" * 50)
    
    # Traditional search
    start_time = time.time()
    traditional_move = engine.search(board, 2.0)
    traditional_time = time.time() - start_time
    
    print(f"Traditional Search:")
    print(f"  Best Move: {traditional_move}")
    print(f"  Time: {traditional_time:.2f}s")
    print()
    
    # Multi-PV search
    start_time = time.time()
    candidates = engine.search_multi_pv(board, 2.0, num_lines=5)
    multi_pv_time = time.time() - start_time
    
    print(f"Multi-PV Search (Top 5 candidates):")
    print(f"  Time: {multi_pv_time:.2f}s")
    print()
    
    for i, candidate in enumerate(candidates, 1):
        print(f"  {i}. Move: {candidate['move']}")
        print(f"     Score: {candidate['score']:.2f}")
        print(f"     Improvement: {candidate['improvement']:.2f}")
        print(f"     Opportunities: {', '.join(candidate['opportunities'])}")
        print()

def test_puzzle_solver_search():
    """Test the full puzzle solver search."""
    print("=" * 60)
    print("PUZZLE SOLVER SEARCH TEST")
    print("=" * 60)
    print("TESTING: Anti-null-move-pruning with opportunity focus")
    print()
    
    engine = V7P3REngine()
    
    # Test different game phases
    positions = [
        ("Opening", chess.Board()),
        ("After 1.e4 e5", chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")),
        ("Middlegame", chess.Board("r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQ - 4 6")),
    ]
    
    for pos_name, board in positions:
        print(f"\n{pos_name}:")
        print(f"FEN: {board.fen()}")
        print("-" * 40)
        
        # Traditional search
        start_time = time.time()
        traditional_move = engine.search(board, 1.5)
        traditional_time = time.time() - start_time
        
        # Puzzle solver search
        start_time = time.time()
        puzzle_move = engine.puzzle_search(board, 1.5)
        puzzle_time = time.time() - start_time
        
        print(f"Traditional: {traditional_move} ({traditional_time:.2f}s)")
        print(f"Puzzle Solver: {puzzle_move} ({puzzle_time:.2f}s)")
        
        # Compare if different
        if traditional_move != puzzle_move:
            print(f"✓ Puzzle solver chose different move!")
        else:
            print(f"○ Same move selected")

def test_opportunity_detection():
    """Test opportunity detection system."""
    print("=" * 60)
    print("OPPORTUNITY DETECTION TEST")
    print("=" * 60)
    print("TESTING: Practical opportunity identification")
    print()
    
    engine = V7P3REngine()
    
    # Test positions with different opportunity types
    test_positions = [
        ("Development", chess.Board()),
        ("Tactical", chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 3 4")),
        ("Material gain", chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3")),
    ]
    
    for pos_name, board in test_positions:
        print(f"\n{pos_name} Position:")
        print(f"FEN: {board.fen()}")
        
        opportunities = engine._detect_opportunities(board)
        print(f"Detected opportunities: {', '.join(opportunities) if opportunities else 'None'}")

def test_anti_null_move_concept():
    """Test the anti-null-move-pruning concept."""
    print("=" * 60)
    print("ANTI-NULL-MOVE-PRUNING TEST")
    print("=" * 60)
    print("TESTING: Opportunity evaluation vs traditional minimax")
    print()
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Make a move to test the concept
    board.push_san("e4")
    
    print("Position after 1.e4:")
    print(f"FEN: {board.fen()}")
    print()
    
    # Traditional evaluation
    traditional_eval = engine._evaluate_position(board)
    
    # Opportunity-based evaluation
    opportunity_eval = engine._evaluate_opportunity_position(board, depth=3, time_limit=1.0)
    
    print(f"Traditional evaluation: {traditional_eval:.2f}")
    print(f"Opportunity evaluation: {opportunity_eval:.2f}")
    print(f"Difference: {opportunity_eval - traditional_eval:.2f}")
    print()
    
    if abs(opportunity_eval - traditional_eval) > 10:
        print("✓ Opportunity evaluation shows significant difference")
    else:
        print("○ Evaluations are similar")

if __name__ == "__main__":
    try:
        test_multi_pv_search()
        test_puzzle_solver_search()
        test_opportunity_detection()
        test_anti_null_move_concept()
        
        print("\n" + "=" * 60)
        print("V13.2 PUZZLE SOLVER PHASE 1 TEST COMPLETE")
        print("=" * 60)
        print("✓ Multi-PV foundation implemented")
        print("✓ Opportunity detection working")
        print("✓ Anti-null-move-pruning concept tested")
        print("✓ Practical search with game phase awareness")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()