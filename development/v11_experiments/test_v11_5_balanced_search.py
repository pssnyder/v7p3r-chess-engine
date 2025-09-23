#!/usr/bin/env python3
"""
V11.5 Balanced Search Test Script
=================================

Test the new balanced search implementation that combines:
- Fast search for simple positions
- Full tactical analysis for critical positions

Expected results:
- NPS: 3,000+ (balanced between speed and accuracy)
- Tactical accuracy: 75%+ (better than 33% fast-only)
"""

import os
import sys
import time
import chess

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def test_balanced_search_performance():
    """Test the balanced search performance and tactical accuracy"""
    print("V11.5 BALANCED SEARCH VALIDATION")
    print("=================================")
    print()
    
    # Initialize engine
    engine = V7P3REngine()
    
    # Test positions for performance measurement
    test_positions = [
        ("Opening", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middlegame", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"),
        ("Tactical", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
        ("Endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    ]
    
    print("=== PERFORMANCE TEST ===")
    total_nodes = 0
    total_time = 0
    
    for position_name, fen in test_positions:
        print(f"--- {position_name} Position ---")
        board = chess.Board(fen)
        
        start_time = time.time()
        engine.nodes_searched = 0
        
        # Search to depth 4 with 3 second time limit
        move, score, search_info = engine.search(board, time_limit=3.0, depth=4)
        
        elapsed = time.time() - start_time
        nodes = engine.nodes_searched
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        
        print(f"Move: {move} | Score: {score}")
        print(f"Nodes: {nodes:,} | Time: {elapsed:.2f}s | NPS: {nps:,}")
        print()
        
        total_nodes += nodes
        total_time += elapsed
    
    avg_nps = int(total_nodes / total_time) if total_time > 0 else 0
    print("=== PERFORMANCE SUMMARY ===")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average NPS: {avg_nps:,}")
    
    if avg_nps >= 3000:
        print("Performance Rating: ✅ GOOD")
    elif avg_nps >= 2000:
        print("Performance Rating: ⚠️ ACCEPTABLE")
    else:
        print("Performance Rating: ❌ NEEDS IMPROVEMENT")
    
    print()
    
    # Test tactical accuracy
    print("=== TACTICAL ACCURACY TEST ===")
    
    tactical_positions = [
        {
            "name": "Fork",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
            "expected": ["d1h5"],
            "description": "Queen fork on king and knight"
        },
        {
            "name": "Pin", 
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2",
            "expected": ["d7d6", "f7f5"],
            "description": "Pin prevention"
        },
        {
            "name": "Check",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
            "expected": ["d1h5"],
            "description": "Checking move with fork"
        }
    ]
    
    correct_moves = 0
    total_tactical_tests = len(tactical_positions)
    
    for test in tactical_positions:
        print(f"--- {test['name']} ---")
        board = chess.Board(test['fen'])
        
        start_time = time.time()
        engine.nodes_searched = 0
        
        move, score, search_info = engine.search(board, time_limit=2.0, depth=4)
        
        elapsed = time.time() - start_time
        nodes = engine.nodes_searched
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        
        print(f"Engine played: {move}")
        print(f"Expected: {test['expected']}")
        print(f"Performance: {nodes:,} nodes, {nps:,} NPS")
        
        if str(move) in test['expected']:
            print("Result: ✅ CORRECT")
            correct_moves += 1
        else:
            print("Result: ❌ INCORRECT")
        print()
    
    tactical_accuracy = (correct_moves / total_tactical_tests) * 100
    print("=== TACTICAL SUMMARY ===")
    print(f"Correct moves: {correct_moves}/{total_tactical_tests}")
    print(f"Tactical accuracy: {tactical_accuracy:.1f}%")
    
    if tactical_accuracy >= 75:
        print("Tactical Rating: ✅ EXCELLENT")
    elif tactical_accuracy >= 60:
        print("Tactical Rating: ⚠️ GOOD")
    else:
        print("Tactical Rating: ❌ NEEDS IMPROVEMENT")
    
    print()
    print("=== FINAL VERDICT ===")
    
    if avg_nps >= 3000 and tactical_accuracy >= 75:
        print("🎉 BALANCED SEARCH SUCCESS!")
        print("Ready for v11.5 build")
    elif avg_nps >= 2000 and tactical_accuracy >= 60:
        print("⚠️ ACCEPTABLE PERFORMANCE")
        print("Consider minor tuning before build")
    else:
        print("❌ NEEDS MORE WORK")
        print("Significant improvements required")
    
    return avg_nps, tactical_accuracy

if __name__ == "__main__":
    test_balanced_search_performance()