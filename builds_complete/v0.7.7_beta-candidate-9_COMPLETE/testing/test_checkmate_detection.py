#!/usr/bin/env python3
"""
Test script to validate the v7p3r engine's ability to detect checkmates.
This is critical for avoiding obvious blunders and missed winning opportunities.
"""

import sys
import os
import chess
import time

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_checkmate_detection():
    """Test if the engine can find checkmates in various positions"""
    print("Testing Checkmate Detection")
    print("=" * 50)
    
    try:
        # Import required modules
        from v7p3r_engine.v7p3r import v7p3rEngine
        import logging
        
        # Setup a logger
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Engine configuration optimized for checkmate detection
        engine_config = {
            "search_algorithm": "negamax",
            "depth": 5,
            "max_depth": 8,
            "verbose_output": False,
            "logger": "test_logger"
        }
        
        # Create the engine
        engine = v7p3rEngine(engine_config)
        
        # Test positions with checkmates
        test_positions = [
            {
                "name": "Fool's Mate",
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
                "expected_move": "d8h4",  # Qh4#
                "expected_to_find_mate": True
            },
            {
                "name": "Scholar's Mate",
                "fen": "rnbqkbnr/ppp2ppp/3p4/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4",
                "expected_move": "f3f7",  # Qxf7#
                "expected_to_find_mate": True
            },
            {
                "name": "Back Rank Mate",
                "fen": "5rk1/5ppp/8/8/8/8/8/R6K w - - 0 1",
                "expected_move": "a1a8",  # Ra8#
                "expected_to_find_mate": True
            },
            {
                "name": "Smothered Mate",
                "fen": "6k1/5ppp/8/8/8/8/1N6/K7 w - - 0 1",
                "expected_move": "b2c4",  # Nc4 (leading to Nd6# next move)
                "expected_to_find_mate": True
            }
        ]
        
        successful_tests = 0
        total_tests = len(test_positions)
        
        for i, position in enumerate(test_positions):
            print(f"\nTest {i+1}/{total_tests}: {position['name']}")
            print(f"FEN: {position['fen']}")
            
            # Create the board
            board = chess.Board(position['fen'])
            
            # Time the search
            start_time = time.time()
            best_move = engine.search(board)
            end_time = time.time()
            
            print(f"Engine found move: {best_move}")
            print(f"Search time: {end_time - start_time:.2f} seconds")
            
            # Check if this is the expected move
            expected_move = chess.Move.from_uci(position['expected_move'])
            if best_move == expected_move:
                print(f"Γ£à PASS: Engine found the expected checkmate move: {expected_move}")
                successful_tests += 1
            else:
                print(f"Γ¥î FAIL: Engine did not find the expected move {expected_move}")
                # Make the move and see if it leads to checkmate anyway
                if best_move != chess.Move.null():
                    board_copy = board.copy()
                    board_copy.push(best_move)
                    if board_copy.is_checkmate():
                        print(f"Γ£à However, the move {best_move} also leads to checkmate!")
                        successful_tests += 1
                    else:
                        # Check if it's a strong move anyway
                        board_copy2 = board.copy()
                        board_copy2.push(expected_move)
                        if board_copy2.is_checkmate() and not board_copy.is_checkmate():
                            print(f"ΓÜá∩╕Å The engine missed a direct checkmate!")
                
            print("-" * 40)
        
        # Summary
        print(f"\nSummary: {successful_tests}/{total_tests} tests passed")
        if successful_tests == total_tests:
            print("≡ƒÄë All checkmate tests passed! The engine is detecting checkmates correctly.")
        else:
            print(f"ΓÜá∩╕Å {total_tests - successful_tests} tests failed. The engine still needs improvement in checkmate detection.")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("V7P3R CHECKMATE DETECTION TEST")
    print("=" * 60)
    
    success = test_checkmate_detection()
    
    print("\n" + "=" * 60)
    if success:
        print("≡ƒÄ» CHECKMATE DETECTION VALIDATED SUCCESSFULLY!")
    else:
        print("Γ¥î VALIDATION FAILED - Please check the errors above")
