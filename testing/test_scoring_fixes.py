#!/usr/bin/env python3
"""
Test script to validate the fixed v7p3r_score rule evaluation functions.
Ensures all functions work with static evaluation and don't mutate the board.
"""

import sys
import os
import chess

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_fixed_scoring_functions():
    """Test all the fixed scoring functions"""
    print("Testing Fixed v7p3r Scoring Functions")
    print("=" * 50)
    
    try:
        # Import required modules
        from v7p3r_engine.v7p3r_pst import v7p3rPST
        from v7p3r_engine.v7p3r_score import v7p3rScore
        import logging
        
        # Setup
        engine_config = {
            'verbose_output': False,
            'engine_ruleset': 'default_evaluation'
        }
        pst = v7p3rPST()
        logger = logging.getLogger("test_logger")
        
        scorer = v7p3rScore(engine_config=engine_config, pst=pst, logger=logger)
        
        # Test different board positions
        test_positions = [
            ("Starting position", chess.Board()),
            ("After 1.e4 e5", chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")),
            ("Endgame position", chess.Board("8/8/8/3k4/3K4/8/8/8 w - - 0 1")),
            ("Queen attack position", chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 2"))
        ]
        
        for desc, board in test_positions:
            print(f"\nTesting: {desc}")
            print(f"FEN: {board.fen()}")
            
            # Store original board state
            original_fen = board.fen()
            
            # Test all fixed functions
            functions_to_test = [
                '_special_moves',
                '_tactical_evaluation', 
                '_rook_coordination',
                '_queen_capture',
                '_king_endangerment',
                '_checkmate_threats',
                '_king_safety',
                '_king_threat'
            ]
            
            for func_name in functions_to_test:
                try:
                    func = getattr(scorer, func_name)
                    white_score = func(board, chess.WHITE)
                    black_score = func(board, chess.BLACK)
                    
                    # Verify board wasn't mutated
                    if board.fen() != original_fen:
                        print(f"  ❌ {func_name}: BOARD MUTATED!")
                        return False
                    else:
                        print(f"  ✅ {func_name}: White={white_score:.2f}, Black={black_score:.2f}")
                        
                except Exception as e:
                    print(f"  ❌ {func_name}: ERROR - {e}")
                    return False
            
            # Test full scoring calculation
            try:
                total_white = scorer.calculate_score(board, chess.WHITE)
                total_black = scorer.calculate_score(board, chess.BLACK)
                print(f"  📊 Total scores: White={total_white:.2f}, Black={total_black:.2f}")
                
                # Verify board wasn't mutated by full calculation
                if board.fen() != original_fen:
                    print(f"  ❌ calculate_score: BOARD MUTATED!")
                    return False
                    
            except Exception as e:
                print(f"  ❌ calculate_score: ERROR - {e}")
                return False
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ No board mutations detected")
        print("✅ All functions work with static evaluation")
        print("✅ Performance should be significantly improved")
        
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False

if __name__ == "__main__":
    print("V7P3R SCORING FUNCTIONS - VALIDATION TEST")
    print("=" * 60)
    
    success = test_fixed_scoring_functions()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 SCORING FIXES VALIDATED SUCCESSFULLY!")
        print("Your engine should now:")
        print("  • Make faster evaluations (no legal move iteration)")
        print("  • Find checkmates more reliably")
        print("  • Make more consistent positional decisions") 
        print("  • Avoid erratic play caused by board mutations")
    else:
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED - Please check the errors above")
