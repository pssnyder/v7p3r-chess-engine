"""
Simplified test script for verifying search function evaluation perspectives.
This script tests that search functions return perspective-correct scores.
"""

import chess
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required engine modules
from v7p3r_rules import v7p3rRules
from v7p3r_pst import v7p3rPST
from v7p3r_score import v7p3rScore
from v7p3r_ordering import v7p3rOrdering
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook
from v7p3r_search import v7p3rSearch
from v7p3r_config import v7p3rConfig

def test_simplified_search_perspectives():
    """Test that search functions return consistent perspective-based scores."""
    
    print("Starting simplified search perspectives test")
    
    # Create game configuration
    config_manager = v7p3rConfig()
    ruleset = config_manager.get_ruleset()
    
    # Initialize required components in the correct order
    pst = v7p3rPST()
    rules_manager = v7p3rRules(ruleset=ruleset, pst=pst)
    scoring_calculator = v7p3rScore(rules_manager=rules_manager, pst=pst)
    move_organizer = v7p3rOrdering(scoring_calculator=scoring_calculator)
    time_manager = v7p3rTime()
    opening_book = v7p3rBook()
    
    # Configure all components to disable logging
    try:
        rules_manager.monitoring_enabled = False
        pst.monitoring_enabled = False
        scoring_calculator.monitoring_enabled = False
        move_organizer.monitoring_enabled = False
        time_manager.monitoring_enabled = False
        opening_book.monitoring_enabled = False
    except:
        print("Note: Could not disable monitoring for some components")
        
    # Initialize a single search engine
    search = v7p3rSearch(
        scoring_calculator=scoring_calculator,
        move_organizer=move_organizer,
        time_manager=time_manager,
        opening_book=opening_book
    )
    search.depth = 1  # Use minimal depth to avoid long computation
    search.monitoring_enabled = False  # Disable verbose logging
    
    # Simplified test positions
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("White advantage", "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1"),
        ("Black advantage", "rnbqkbnr/pppppppp/8/8/8/8/PPP1PPPP/RNB1KBNR w KQkq - 0 1")
    ]
    
    print("\nTesting search functions with various positions:")
    print("================================================")
    
    for desc, fen in test_positions:
        board = chess.Board(fen)
        print(f"\nPosition: {desc} (FEN: {fen})")
        print(board)
        
        # Test direct evaluation from both perspectives
        white_eval = scoring_calculator.evaluate_position(board)
        print(f"White's standard evaluation: {white_eval:+.2f}")
        
        white_perspective = scoring_calculator.evaluate_position_from_perspective(board, chess.WHITE)
        print(f"White's perspective evaluation: {white_perspective:+.2f}")
        
        black_perspective = scoring_calculator.evaluate_position_from_perspective(board, chess.BLACK)
        print(f"Black's perspective evaluation: {black_perspective:+.2f}")
        
        # Test minimax and negamax from White's perspective
        search.current_perspective = chess.WHITE
        search.color_name = "White"
        white_minimax = search._minimax_search(board, 0, -float('inf'), float('inf'), True)
        white_negamax = search._negamax_search(board, 0, -float('inf'), float('inf'))
        print(f"White minimax: {white_minimax:+.2f}, White negamax: {white_negamax:+.2f}")
        
        # Test minimax and negamax from Black's perspective
        search.current_perspective = chess.BLACK
        search.color_name = "Black"
        black_minimax = search._minimax_search(board, 0, -float('inf'), float('inf'), True)
        black_negamax = search._negamax_search(board, 0, -float('inf'), float('inf'))
        print(f"Black minimax: {black_minimax:+.2f}, Black negamax: {black_negamax:+.2f}")
        
        print("-" * 50)
    
    print("\nSearch perspective test completed!")
    return True

if __name__ == "__main__":
    test_simplified_search_perspectives()
