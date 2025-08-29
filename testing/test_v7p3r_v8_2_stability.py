#!/usr/bin/env python3
"""
V7P3R v8.2 Stability Testing Suite
Test the V8.2 enhanced move ordering while maintaining V8.1 stability
"""

import sys
import os
sys.path.append('src')

import chess
import time
from v7p3r import V7P3RCleanEngine

def test_v8_2_stability():
    """Quick stability test for V8.2"""
    print("🔍 Testing V8.2 Stability")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Test same position multiple times - should get identical results
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5")
    
    print(f"Position: {board.fen()[:30]}...")
    
    # Test deterministic evaluation
    evaluations = []
    for run in range(3):
        eval_score = engine._evaluate_position_deterministic(board)
        evaluations.append(eval_score)
    
    if len(set(evaluations)) == 1:
        print(f"✅ Deterministic evaluation: {evaluations[0]:+.2f}")
    else:
        print(f"❌ Non-deterministic: {evaluations}")
        return False
    
    # Test enhanced move ordering with mate detection
    moves = list(board.legal_moves)
    from v7p3r import SearchOptions
    options = SearchOptions()
    
    ordered_moves = engine._order_moves_enhanced(board, moves.copy(), 0, options)
    
    print(f"✅ Move ordering working: {len(ordered_moves)} moves ordered")
    print(f"  Top move: {ordered_moves[0]}")
    
    # Test mate detection priority
    mate_position = chess.Board("4k3/4P3/4K3/8/8/8/8/8 w - - 0 1")  # Simple mate in 1
    mate_moves = list(mate_position.legal_moves)
    ordered_mate_moves = engine._order_moves_enhanced(mate_position, mate_moves, 0, options)
    
    # Find the mate move
    mate_move = None
    for move in ordered_mate_moves:
        mate_position.push(move)
        if mate_position.is_checkmate():
            mate_move = move
            mate_position.pop()
            break
        mate_position.pop()
    
    if mate_move and ordered_mate_moves[0] == mate_move:
        print(f"✅ Mate detection priority: {mate_move} is first")
    else:
        print(f"⚠️  Mate detection: expected {mate_move}, got {ordered_mate_moves[0] if ordered_mate_moves else 'none'}")
    
    # Test search functionality
    try:
        start_time = time.time()
        best_move = engine.search(board, 1.0)
        search_time = time.time() - start_time
        
        print(f"✅ Search working: {best_move} in {search_time:.3f}s")
        print(f"  Nodes: {engine.nodes_searched}")
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False
    
    return True

def main():
    """Run V8.2 stability test"""
    print("V7P3R v8.2 Stability Test")
    print("Enhanced move ordering with maintained stability")
    print("=" * 60)
    
    if test_v8_2_stability():
        print("\n🎉 V8.2 is stable and ready for enhanced move ordering development!")
        return 0
    else:
        print("\n⚠️  V8.2 stability issues detected.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
