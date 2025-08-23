#!/usr/bin/env python3
"""
Test script for V7P3R v6.2 optimizations
Compares fast search vs traditional search performance
"""

import chess
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REvaluationEngine

def test_search_performance():
    """Test performance of fast vs traditional search"""
    
    # Test positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After e4
        chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 3"),  # Sicilian
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"),  # Four Knights
    ]
    
    depths = [3, 4, 5]
    
    print("V7P3R v6.2 Optimization Performance Test")
    print("=" * 50)
    
    for pos_idx, board in enumerate(test_positions):
        print(f"\nPosition {pos_idx + 1}: {board.fen()[:50]}...")
        
        for depth in depths:
            print(f"\n  Depth {depth}:")
            
            # Test fast search
            engine_fast = V7P3REvaluationEngine(board.copy(), board.turn)
            engine_fast.use_fast_search = True
            engine_fast.depth = depth
            
            start_time = time.time()
            move_fast = engine_fast.search(board, board.turn)
            fast_time = time.time() - start_time
            fast_nodes = engine_fast.nodes_searched
            
            # Test traditional search
            engine_trad = V7P3REvaluationEngine(board.copy(), board.turn)
            engine_trad.use_fast_search = False
            engine_trad.depth = depth
            
            start_time = time.time()
            move_trad = engine_trad.search(board, board.turn)
            trad_time = time.time() - start_time
            trad_nodes = engine_trad.nodes_searched
            
            # Calculate speedup
            speedup = trad_time / fast_time if fast_time > 0 else float('inf')
            node_ratio = trad_nodes / fast_nodes if fast_nodes > 0 else float('inf')
            
            print(f"    Fast Search:  {fast_time:.4f}s, {fast_nodes:,} nodes, move: {move_fast}")
            print(f"    Trad Search:  {trad_time:.4f}s, {trad_nodes:,} nodes, move: {move_trad}")
            print(f"    Speedup:      {speedup:.2f}x time, {node_ratio:.2f}x nodes")
            
            # Check if moves are the same
            if move_fast == move_trad:
                print(f"    Result:       ✓ Same move selected")
            else:
                print(f"    Result:       ⚠ Different moves selected")

def test_time_management():
    """Test aggressive time management"""
    print("\n\nTime Management Test")
    print("=" * 30)
    
    board = chess.Board()
    engine = V7P3REvaluationEngine(board, chess.WHITE)
    
    # Test different time controls
    time_controls = [
        {'wtime': 30000, 'btime': 30000, 'winc': 0, 'binc': 0},      # 30 seconds sudden death
        {'wtime': 180000, 'btime': 180000, 'winc': 2000, 'binc': 2000},  # 3+2 blitz
        {'wtime': 600000, 'btime': 600000, 'winc': 5000, 'binc': 5000},  # 10+5 rapid
        {'movetime': 5000},  # 5 second per move
    ]
    
    for i, tc in enumerate(time_controls):
        allocated = engine._calculate_time_allocation(tc, board)
        print(f"Time Control {i+1}: {tc}")
        print(f"  Allocated: {allocated:.3f} seconds")
        
        # Test actual search with time management
        start_time = time.time()
        move, depth, nodes, search_time = engine.search_with_time_management(board, tc)
        actual_time = time.time() - start_time
        
        print(f"  Actual:    {actual_time:.3f} seconds")
        print(f"  Depth:     {depth}")
        print(f"  Nodes:     {nodes:,}")
        print(f"  Move:      {move}")
        print(f"  Efficiency: {nodes/max(actual_time, 0.001):.0f} nodes/sec")
        print()

def test_fast_evaluation():
    """Test fast evaluation function"""
    print("\nFast Evaluation Test")
    print("=" * 25)
    
    board = chess.Board()
    engine = V7P3REvaluationEngine(board, chess.WHITE)
    
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("8/8/8/8/8/8/8/K7 w - - 0 1"),  # Lone king
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),  # After e5
    ]
    
    for i, pos in enumerate(test_positions):
        print(f"Position {i+1}:")
        
        # Quick evaluation
        start_time = time.time()
        quick_score = engine._quick_evaluate(pos)
        quick_time = time.time() - start_time
        
        # Full evaluation
        start_time = time.time()
        full_score = engine.evaluate_position_from_perspective(pos, chess.WHITE)
        full_time = time.time() - start_time
        
        speedup = full_time / quick_time if quick_time > 0 else float('inf')
        
        print(f"  Quick: {quick_score:.2f} ({quick_time:.6f}s)")
        print(f"  Full:  {full_score:.2f} ({full_time:.6f}s)")
        print(f"  Speedup: {speedup:.1f}x")
        print()

if __name__ == "__main__":
    print("Testing V7P3R v6.2 Optimizations...")
    print("This will take a few moments...\n")
    
    try:
        test_search_performance()
        test_time_management()
        test_fast_evaluation()
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
