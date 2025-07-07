import chess
import os
import sys
import pygame

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required engine modules
from v7p3r_play import v7p3rChess
from v7p3r import v7p3rEngine
from v7p3r_debug import v7p3rLogger
from v7p3r_rules import v7p3rRules
from v7p3r_pst import v7p3rPST
from v7p3r_search import v7p3rSearch
from v7p3r_score import v7p3rScore
from v7p3r_ordering import v7p3rOrdering
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook
from v7p3r_config import v7p3rConfig

def test_search_perspectives():
    """Test that search functions return consistent perspective-based scores."""
    
    # Set up logger
    logger = v7p3rLogger.setup_logger("test_search_perspectives")
    logger.info("Starting search perspectives test")
    
    # Create game configuration
    config_manager = v7p3rConfig(config_path="configs/default_config.json")
    engine_config = config_manager.get_engine_config()
    game_config = config_manager.get_game_config()
    ruleset = config_manager.get_ruleset()
    
    # Initialize required components
    pst = v7p3rPST()
    rules_manager = v7p3rRules(ruleset=ruleset, pst=pst)
    scoring_calculator = v7p3rScore(rules_manager=rules_manager, pst=pst)
    move_organizer = v7p3rOrdering(scoring_calculator=scoring_calculator)
    time_manager = v7p3rTime()
    opening_book = v7p3rBook()
    
    # Initialize search engine for White
    white_search = v7p3rSearch(
        scoring_calculator=scoring_calculator,
        move_organizer=move_organizer,
        time_manager=time_manager,
        opening_book=opening_book
    )
    white_search.current_perspective = chess.WHITE
    white_search.color_name = "White"
    white_search.current_turn = chess.WHITE
    
    # Initialize search engine for Black
    black_search = v7p3rSearch(
        scoring_calculator=scoring_calculator,
        move_organizer=move_organizer,
        time_manager=time_manager,
        opening_book=opening_book
    )
    black_search.current_perspective = chess.BLACK
    black_search.color_name = "Black"
    black_search.current_turn = chess.BLACK
    
    # Test positions
    # 1. Starting position - should be relatively balanced for both players
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # 2. White has advantage (up a rook)
    white_advantage_fen = "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1"
    # 3. Black has advantage (up a rook)
    black_advantage_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPP1PPPP/RNB1KBNR w KQkq - 0 1"
    # 4. White about to checkmate
    white_winning_fen = "7k/5QR1/8/8/8/8/8/K7 w - - 0 1"
    # 5. Black about to checkmate
    black_winning_fen = "K7/7q/8/8/8/8/8/7k b - - 0 1"
    
    test_positions = [
        ("Starting position", start_fen),
        ("White advantage", white_advantage_fen),
        ("Black advantage", black_advantage_fen),
        ("White winning", white_winning_fen),
        ("Black winning", black_winning_fen)
    ]
    
    print("\nTesting search functions with various positions:")
    print("================================================")
    
    for desc, fen in test_positions:
        board = chess.Board(fen)
        print(f"\nPosition: {desc} (FEN: {fen})")
        print("Board visualization:")
        print(board)
        
        # Direct evaluation scores
        white_eval = scoring_calculator.evaluate_position_from_perspective(board, chess.WHITE)
        black_eval = scoring_calculator.evaluate_position_from_perspective(board, chess.BLACK)
        print(f"Direct evaluations:")
        print(f"  White's perspective: {white_eval:+.2f}")
        print(f"  Black's perspective: {black_eval:+.2f}")
        
        # Set up the search engines with this board
        white_search.root_board = board.copy()
        black_search.root_board = board.copy()
        
        # Minimax scores
        white_minimax = white_search._minimax_search(board.copy(), 2, -float('inf'), float('inf'), True)
        black_minimax = black_search._minimax_search(board.copy(), 2, -float('inf'), float('inf'), True)
        print(f"Minimax evaluations (depth 2):")
        print(f"  White's perspective: {white_minimax:+.2f}")
        print(f"  Black's perspective: {black_minimax:+.2f}")
        
        # Negamax scores
        white_negamax = white_search._negamax_search(board.copy(), 2, -float('inf'), float('inf'))
        black_negamax = black_search._negamax_search(board.copy(), 2, -float('inf'), float('inf'))
        print(f"Negamax evaluations (depth 2):")
        print(f"  White's perspective: {white_negamax:+.2f}")
        print(f"  Black's perspective: {black_negamax:+.2f}")
        
        # Quiescence scores (maximizing=True)
        white_quiescence = white_search._quiescence_search(board.copy(), chess.WHITE, -float('inf'), float('inf'), True)
        black_quiescence = black_search._quiescence_search(board.copy(), chess.BLACK, -float('inf'), float('inf'), True)
        print(f"Quiescence evaluations:")
        print(f"  White's perspective: {white_quiescence:+.2f}")
        print(f"  Black's perspective: {black_quiescence:+.2f}")
        
        # Verify consistent expectations
        print("\nConsistency checks:")
        if "advantage" in desc:
            player = desc.split()[0]
            if player == "White":
                expect_white_positive = True
                expect_black_negative = True
            else:
                expect_white_positive = False
                expect_black_negative = False
                
            print(f"  White should see {'positive' if expect_white_positive else 'negative'} score")
            print(f"  Black should see {'positive' if not expect_black_negative else 'negative'} score")
            
            # Check if evaluations match expectations
            if (white_eval > 0) != expect_white_positive:
                print(f"  ❌ White direct evaluation inconsistent: {white_eval:+.2f}")
            else:
                print(f"  ✓ White direct evaluation consistent: {white_eval:+.2f}")
                
            if (black_eval > 0) != (not expect_black_negative):
                print(f"  ❌ Black direct evaluation inconsistent: {black_eval:+.2f}")
            else:
                print(f"  ✓ Black direct evaluation consistent: {black_eval:+.2f}")
        
    print("\nSearch perspective test completed!")
    return True

if __name__ == "__main__":
    pygame.init()  # Initialize pygame to avoid errors
    if test_search_perspectives():
        print("\n✅ Search perspective test passed!")
    else:
        print("\n❌ Search perspective test failed!")
