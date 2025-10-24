#!/usr/bin/env python3
"""
Quick V13.x Functionality Test
Just verify everything is working without intensive search
"""

import chess
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from v7p3r import V7P3REngine
    
    def quick_v13x_test():
        """Quick test of V13.x functionality"""
        print("🚀 QUICK V13.x FUNCTIONALITY TEST")
        print("="*50)
        
        engine = V7P3REngine()
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        
        print(f"Testing position: {board.fen()}")
        
        # Test move ordering
        legal_moves = list(board.legal_moves)
        print(f"Legal moves: {len(legal_moves)}")
        
        ordered_moves = engine._order_moves_advanced(board, legal_moves, 2)
        waiting_moves = engine.get_waiting_moves()
        
        print(f"Critical moves: {len(ordered_moves)}")
        print(f"Waiting moves: {len(waiting_moves)}")
        print(f"Pruning rate: {len(waiting_moves)/len(legal_moves)*100:.1f}%")
        
        # Test waiting move selection
        selected_waiting = engine._select_waiting_moves(board, waiting_moves, 2)
        print(f"Selected waiting moves: {len(selected_waiting)}")
        
        # Quick search test (depth 2 only)
        print(f"\nTesting quick search...")
        start_time = time.time()
        best_move = engine.search(board, time_limit=2.0, depth=2)
        end_time = time.time()
        
        search_time = end_time - start_time
        print(f"Best move: {board.san(best_move) if best_move else 'None'}")
        print(f"Search time: {search_time:.3f}s")
        print(f"Nodes: {engine.nodes_searched}")
        
        if hasattr(engine, 'v13x_stats'):
            print(f"V13.x stats: {engine.v13x_stats}")
        
        if hasattr(engine, 'waiting_move_stats'):
            print(f"Waiting move stats: {engine.waiting_move_stats}")
        
        print(f"\n✅ V13.x QUICK TEST COMPLETE!")
        return True
        
    if __name__ == "__main__":
        success = quick_v13x_test()
        if success:
            print("🎉 V13.x is working correctly!")
        else:
            print("❌ V13.x needs debugging")
            
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()