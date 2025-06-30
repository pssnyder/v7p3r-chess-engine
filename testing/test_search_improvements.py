#!/usr/bin/env python3
"""
Test script to validate improvements to v7p3r_search and v7p3r_score.
Tests the engine's ability to find checkmates and make tactically sound moves.
"""

import sys
import os
import chess
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def setup_test_logger():
    """Setup a logger for the tests"""
    logger = logging.getLogger("test_search_logger")
    logger.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def test_checkmate_detection():
    """Test if the engine can find checkmates in 1, 2, and 3 moves"""
    logger = setup_test_logger()
    logger.info("Testing checkmate detection capabilities")
    
    # Import required modules
    from v7p3r_engine.v7p3r_pst import v7p3rPST
    from v7p3r_engine.v7p3r_score import v7p3rScore
    from v7p3r_engine.v7p3r_search import v7p3rSearch
    from v7p3r_engine.v7p3r_time import v7p3rTime
    from v7p3r_engine.v7p3r_ordering import v7p3rOrdering
    from v7p3r_engine.v7p3r_book import v7p3rBook
    
    # Test positions with known checkmates
    test_positions = [
        {
            "name": "Checkmate in 1 for White",
            "fen": "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
            "best_move": "f7f8",  # Queen checkmates on f8
            "depth": 1,
        },
        {
            "name": "Checkmate in 2 for White",
            "fen": "r1bk3r/ppp2ppp/2p5/4Pn2/5q2/2N5/PPP2PPP/R3KB1R w KQ - 0 13",
            "depth": 3,
        },
        {
            "name": "Tactical win for White",
            "fen": "r1b1kb1r/pp3ppp/2p2n2/q3n3/4P3/P1N2N2/1PP2PPP/R1BQKB1R w KQkq - 0 9",
            "depth": 4,
        },
    ]
    
    # Setup the engine with good search parameters
    engine_config = {
        'engine_search_algorithm': 'negamax',
        'depth': 5,
        'max_depth': 8,
        'verbose_output': False,
        'engine_ruleset': 'default_evaluation'
    }
    
    pst = v7p3rPST()
    scoring_calculator = v7p3rScore(engine_config=engine_config, pst=pst, logger=logger)
    time_manager = v7p3rTime()
    move_organizer = v7p3rOrdering(engine_config, scoring_calculator, logger)
    opening_book = v7p3rBook()

    # Create the search engine
    search_engine = v7p3rSearch(
        engine_config=engine_config,
        scoring_calculator=scoring_calculator,
        move_organizer=move_organizer,
        time_manager=time_manager,
        opening_book=opening_book,
        logger=logger
    )
    
    for test in test_positions:
        logger.info(f"\nTesting position: {test['name']}")
        logger.info(f"FEN: {test['fen']}")
        
        # Set depth for this test
        search_engine.depth = test.get('depth', 3)
        
        # Create the board from FEN
        board = chess.Board(test['fen'])
        
        # Let the engine find the best move
        best_move = search_engine.search(board, board.turn)
        
        logger.info(f"Engine's best move: {best_move}")
        
        # If there's a known best move, check if the engine found it
        if 'best_move' in test:
            expected_move = chess.Move.from_uci(test['best_move'])
            if best_move == expected_move:
                logger.info(f"✓ Engine correctly found the expected move: {expected_move}")
            else:
                logger.info(f"✗ Engine did not find the expected move. Found {best_move} instead of {expected_move}")
        
        # Check if the move is actually good
        if best_move != chess.Move.null():
            board_after_move = board.copy()
            board_after_move.push(best_move)
            
            # Check if it found a checkmate
            if board_after_move.is_checkmate():
                logger.info("✓ The move leads to immediate checkmate!")
            
            # Check if it avoided checkmate
            elif board.is_check() and not board_after_move.is_check():
                logger.info("✓ The move successfully escapes check")
            
            # Check if it captures a piece
            elif board.is_capture(best_move):
                logger.info("✓ The move captures a piece")
            
            # Check if it gives check
            elif board_after_move.is_check():
                logger.info("✓ The move gives check")
            
            logger.info(f"Board after move: {board_after_move.fen()}")
        else:
            logger.info("✗ Engine couldn't find a valid move")
    
    logger.info("\nCheckmate detection test completed")

if __name__ == "__main__":
    print("V7P3R SEARCH AND SCORE VALIDATION TEST")
    print("=" * 60)
    
    test_checkmate_detection()
    
    print("\n" + "=" * 60)
    print("Test completed!")
