#!/usr/bin/env python3
"""
Bitboard Performance Test
Test the new bitboard-based evaluation system performance
"""

import chess
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3RCleanEngine

def test_bitboard_performance():
    """Test bitboard evaluator performance"""
    
    print("⚡ BITBOARD PERFORMANCE TEST")
    print("=" * 40)
    
    engine = V7P3RCleanEngine()
    board = chess.Board()
    
    print("Testing 2-second search with bitboard evaluation...")
    print("Expected: 15,000+ NPS with bitboards")
    
    start_time = time.time()
    move = engine.search(board, 2.0)
    actual_time = time.time() - start_time
    
    nps = engine.nodes_searched / actual_time
    
    print(f"\nResults:")
    print(f"Time: {actual_time:.2f}s")
    print(f"Nodes: {engine.nodes_searched}")
    print(f"NPS: {nps:.0f}")
    print(f"Best move: {move}")
    
    if nps > 15000:
        print(f"🚀 EXCELLENT! Bitboards are working great!")
    elif nps > 10000:
        print(f"✅ GOOD! Significant improvement with bitboards")
    elif nps > 7000:
        print(f"⚡ BETTER! Some improvement but could be faster")
    else:
        print(f"❌ Still slow - bitboards may need optimization")
    
    print(f"\nComparison:")
    print(f"Previous (with complex tactics): ~1,200 NPS")
    print(f"Stubbed tactics: ~7,600 NPS") 
    print(f"Bitboard evaluation: {nps:.0f} NPS")
    
    improvement = nps / 7600
    print(f"Improvement over stubbed version: {improvement:.1f}x")
    
    return nps

def test_multiple_positions():
    """Test bitboard performance on different position types"""
    
    print(f"\n🧪 MULTI-POSITION BITBOARD TEST")
    print("=" * 40)
    
    engine = V7P3RCleanEngine()
    
    positions = [
        ("Starting position", chess.Board()),
        ("Open position", chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4")),
        ("Endgame", chess.Board("8/8/8/3k4/3K4/8/8/8 w - - 0 1")),
        ("Complex middle", chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"))
    ]
    
    total_nps = 0
    
    for name, board in positions:
        print(f"\n📍 {name}:")
        start_time = time.time()
        move = engine.search(board, 1.0)  # 1 second each
        actual_time = time.time() - start_time
        
        nps = engine.nodes_searched / actual_time
        total_nps += nps
        
        print(f"   NPS: {nps:.0f}")
        print(f"   Nodes: {engine.nodes_searched}")
        print(f"   Move: {move}")
        
        engine.new_game()  # Reset for next test
    
    avg_nps = total_nps / len(positions)
    print(f"\n📊 Average NPS across all positions: {avg_nps:.0f}")
    
    return avg_nps

if __name__ == "__main__":
    print("🎯 V10 BITBOARD EVALUATION PERFORMANCE TEST")
    print("=" * 50)
    
    try:
        single_nps = test_bitboard_performance()
        multi_nps = test_multiple_positions()
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"Single position NPS: {single_nps:.0f}")
        print(f"Multi-position average: {multi_nps:.0f}")
        
        if single_nps > 15000:
            print(f"🚀 SUCCESS! Bitboards achieved target performance!")
        elif single_nps > 10000:
            print(f"✅ GOOD! Significant improvement with bitboards")
        else:
            print(f"⚠️  More optimization needed")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Check that bitboard evaluator is properly integrated")
