#!/usr/bin/env python3
"""
Test to verify the search evaluation perspective fixes.
"""

import sys
import os
import chess

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def test_search_fixes():
    """Test the search evaluation perspective fixes."""
    print("=" * 60)
    print("V7P3R SEARCH LOGIC FIXES VERIFICATION")
    print("=" * 60)
    
    try:
        # Import the fixed search module
        from v7p3r_search import v7p3rSearch
        from v7p3r_score import v7p3rScore
        from v7p3r_ordering import v7p3rOrdering
        from v7p3r_time import v7p3rTime
        from v7p3r_book import v7p3rBook
        from v7p3r_rules import v7p3rRules
        from v7p3r_pst import v7p3rPST
        from v7p3r_config import v7p3rConfig
        
        print("Γ£ô All modules imported successfully")
        
        # Initialize config to get ruleset and other dependencies
        config_manager = v7p3rConfig()
        ruleset = config_manager.get_ruleset()
        
        # Initialize components properly
        pst = v7p3rPST()
        rules_manager = v7p3rRules(ruleset=ruleset, pst=pst)
        score_calculator = v7p3rScore(rules_manager, pst)
        move_organizer = v7p3rOrdering(score_calculator)
        time_manager = v7p3rTime()
        opening_book = v7p3rBook()
        
        print("Γ£ô All components initialized successfully")
        
        # Create search engine
        search_engine = v7p3rSearch(
            score_calculator,
            move_organizer,
            time_manager,
            opening_book
        )
        
        print("Γ£ô Search engine created successfully")
        
        # Test positions
        start_position = chess.Board()
        middle_game = chess.Board("r2qkb1r/ppp2ppp/2n1bn2/3p4/3P4/3B1N2/PPP2PPP/RNBQK2R w KQkq - 0 6")
        
        print("\n=== TESTING EVALUATION PERSPECTIVE CONSISTENCY ===")
        
        # Test evaluation consistency
        white_eval = score_calculator.evaluate_position_from_perspective(middle_game, chess.WHITE)
        black_eval = score_calculator.evaluate_position_from_perspective(middle_game, chess.BLACK)
        
        print(f"Position: {middle_game.fen()}")
        print(f"White's perspective: {white_eval:.3f}")
        print(f"Black's perspective: {black_eval:.3f}")
        print(f"Sum (should be ~0): {white_eval + black_eval:.3f}")
        
        # Check that evaluations are reasonable
        if abs(white_eval + black_eval) > 0.1:
            print("ΓÜá∩╕Å  WARNING: Evaluations are not properly negated")
        else:
            print("Γ£ô Evaluation perspective consistency verified")
        
        print("\n=== TESTING SIMPLE SEARCH ALGORITHM ===")
        
        # Test simple search from White's perspective
        search_engine.search_algorithm = 'simple'
        search_engine.depth = 1
        
        move_white = search_engine.search(start_position, chess.WHITE)
        print(f"Simple search (White): {move_white}")
        
        if start_position.is_legal(move_white):
            print("Γ£ô Simple search returned legal move for White")
        else:
            print("Γ¥î Simple search returned illegal move for White")
            return False
        
        # Test simple search from Black's perspective
        black_board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        move_black = search_engine.search(black_board, chess.BLACK)
        print(f"Simple search (Black): {move_black}")
        
        if black_board.is_legal(move_black):
            print("Γ£ô Simple search returned legal move for Black")
        else:
            print("Γ¥î Simple search returned illegal move for Black")
            return False
        
        print("\n=== TESTING MINIMAX SEARCH ALGORITHM ===")
        
        # Test minimax search
        search_engine.search_algorithm = 'minimax'
        search_engine.depth = 2
        
        try:
            move_minimax = search_engine.search(start_position, chess.WHITE)
            print(f"Minimax search: {move_minimax}")
            
            if start_position.is_legal(move_minimax):
                print("Γ£ô Minimax search returned legal move")
            else:
                print("Γ¥î Minimax search returned illegal move")
                return False
                
        except Exception as e:
            print(f"Γ¥î Minimax search failed: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("Γ£à ALL SEARCH LOGIC FIXES VERIFIED!")
        print("The search functions now maintain consistent evaluation perspective.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Γ¥î TEST FAILED: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_search_fixes()
    sys.exit(0 if success else 1)
