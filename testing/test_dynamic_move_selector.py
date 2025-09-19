#!/usr/bin/env python3
"""
Test V7P3R Dynamic Move Selector
Verify that move pruning is working correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_dynamic_move_selector():
    """Test dynamic move selection at different depths"""
    print("TESTING V7P3R DYNAMIC MOVE SELECTOR")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test positions
    positions = [
        ("Opening", chess.Board()),
        ("Complex Middlegame", chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")),
        ("Tactical Position", chess.Board("r2qkb1r/ppp2ppp/2n1bn2/3pp3/3PP3/3B1N2/PPP2PPP/RNBQK2R w KQkq - 4 6"))
    ]
    
    for pos_name, board in positions:
        print(f"\n{pos_name}:")
        print("-" * 30)
        
        legal_moves = list(board.legal_moves)
        print(f"Total legal moves: {len(legal_moves)}")
        
        # Test move selection at different depths
        for depth in range(1, 8):
            # Get ordered moves
            ordered_moves = engine._order_moves_advanced(board, legal_moves, depth)
            
            # Apply dynamic filtering
            if depth >= 3:
                filtered_moves = engine.dynamic_move_selector.filter_moves_by_depth(board, ordered_moves, depth)
            else:
                filtered_moves = ordered_moves
            
            pruning_percent = ((len(ordered_moves) - len(filtered_moves)) / len(ordered_moves)) * 100
            
            print(f"  Depth {depth}: {len(filtered_moves):2d}/{len(ordered_moves):2d} moves ({pruning_percent:4.1f}% pruned)")
            
            # Show first few moves at deeper levels
            if depth >= 5:
                move_names = [str(move) for move in filtered_moves[:5]]
                print(f"    Top moves: {', '.join(move_names)}")
    
    print(f"\n{'='*50}")
    print("DYNAMIC MOVE SELECTOR SPEED TEST")
    print("=" * 50)
    
    # Speed comparison test
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    
    print("Testing search performance with dynamic move selection...")
    
    for depth in range(3, 7):
        print(f"\nDepth {depth}:")
        
        # Test search time
        engine.default_depth = depth
        start_time = time.time()
        
        try:
            move = engine.search(board, time_limit=20.0)
            elapsed = time.time() - start_time
            nodes = engine.nodes_searched
            nps = int(nodes / max(elapsed, 0.001))
            
            print(f"  Search time: {elapsed:6.2f}s")
            print(f"  Nodes: {nodes:8,}")
            print(f"  NPS: {nps:8,}")
            print(f"  Best move: {move}")
            
            if elapsed > 15.0:
                print("  -> Stopping due to time")
                break
                
        except Exception as e:
            print(f"  ERROR: {e}")
            break

if __name__ == "__main__":
    test_dynamic_move_selector()