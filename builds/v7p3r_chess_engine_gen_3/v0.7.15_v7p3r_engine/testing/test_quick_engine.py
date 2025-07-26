# testing/test_quick_engine.py

"""Quick Engine Test - bypasses slow components"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from v7p3r_config import V7P3RConfig
from v7p3r_search import SearchController
from v7p3r_scoring import ScoringSystem

def quick_engine_test():
    """Test core engine functionality without opening book"""
    print("=== Quick Engine Test ===")
    
    try:
        # Test configuration
        print("1. Testing configuration...")
        config = V7P3RConfig("config.json")
        print("Γ£ô Configuration loaded")
        
        # Test scoring system
        print("2. Testing scoring system...")
        scoring = ScoringSystem(config)
        board = chess.Board()
        eval_score = scoring.evaluate_position(board, chess.WHITE)
        print(f"Γ£ô Position evaluation: {eval_score}")
        
        # Test search system
        print("3. Testing search system...")
        search = SearchController(config)
        
        # Test move generation
        print("4. Testing move generation...")
        best_move = search.find_best_move(board, chess.WHITE)
        
        if best_move and best_move in board.legal_moves:
            print(f"Γ£ô Generated legal move: {best_move}")
            
            # Test search stats
            stats = search.get_search_stats()
            print(f"  - Nodes searched: {stats['nodes_searched']}")
            print(f"  - Search time: {stats['search_time']:.3f}s")
            print(f"  - NPS: {stats['nps']:.0f}")
            
            return True
        else:
            print(f"Γ£ù Invalid move: {best_move}")
            return False
            
    except Exception as e:
        print(f"Γ£ù Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_engine_test()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
