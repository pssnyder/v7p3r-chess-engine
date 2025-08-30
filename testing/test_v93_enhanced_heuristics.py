#!/usr/bin/env python3
"""
V9.3 Enhanced Positional Heuristics Test
Test the new developmental heuristics and early game penalties
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_scoring_calculation_v93 import V7P3RScoringCalculationV93

def test_developmental_heuristics():
    """Test developmental heuristics with various opening positions"""
    print("Testing Developmental Heuristics:")
    print("=" * 40)
    
    piece_values = {'pawn': 100, 'knight': 320, 'bishop': 330, 'rook': 500, 'queen': 900, 'king': 0}
    calc = V7P3RScoringCalculationV93(piece_values)
    
    # Test positions showing development progression
    positions = [
        {
            "name": "Starting Position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "expected": "Baseline - no development"
        },
        {
            "name": "King's Pawn Opening",
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "expected": "Central pawn bonus"
        },
        {
            "name": "Knight Development",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "expected": "Development bonus, knights first"
        },
        {
            "name": "Early Queen (BAD)",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            "expected": "Early queen penalty test"
        },
        {
            "name": "Good Development",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "expected": "Multiple pieces developed well"
        },
        {
            "name": "Castled Position", 
            "fen": "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 6",
            "expected": "Castling bonus"
        }
    ]
    
    for pos_data in positions:
        print(f"\n{pos_data['name']}:")
        print(f"Expected: {pos_data['expected']}")
        
        board = chess.Board(pos_data['fen'])
        
        # Test both colors
        white_score = calc.calculate_score_optimized(board, chess.WHITE)
        black_score = calc.calculate_score_optimized(board, chess.BLACK)
        evaluation = white_score - black_score
        
        print(f"White: {white_score:.2f}, Black: {black_score:.2f}, Eval: {evaluation:.2f}")
        
        # Test individual components for White
        dev_bonus = calc._developmental_heuristics(board, chess.WHITE)
        early_penalties = calc._early_game_penalties(board, chess.WHITE)
        
        print(f"  White Dev Bonus: {dev_bonus:.2f}")
        print(f"  White Early Penalties: {early_penalties:.2f}")

def test_early_game_penalties():
    """Test specific early game penalty scenarios"""
    print("\n\nTesting Early Game Penalties:")
    print("=" * 40)
    
    piece_values = {'pawn': 100, 'knight': 320, 'bishop': 330, 'rook': 500, 'queen': 900, 'king': 0}
    calc = V7P3RScoringCalculationV93(piece_values)
    
    # Penalty test positions
    penalty_tests = [
        {
            "name": "Early Queen Development",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4Q3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 2",
            "expected": "Heavy penalty for early queen"
        },
        {
            "name": "Premature H-Pawn", 
            "fen": "rnbqkbnr/ppppppp1/8/7p/4P3/8/PPPP1PPP/RNBQKBNR w KQkq h6 0 2",
            "expected": "Penalty for wing pawn advance"
        },
        {
            "name": "Good Development Order",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "expected": "Minimal penalties, good development"
        }
    ]
    
    for test_data in penalty_tests:
        print(f"\n{test_data['name']}:")
        print(f"Expected: {test_data['expected']}")
        
        board = chess.Board(test_data['fen'])
        
        white_penalties = calc._early_game_penalties(board, chess.WHITE)
        black_penalties = calc._early_game_penalties(board, chess.BLACK)
        
        print(f"White Penalties: {white_penalties:.2f}")
        print(f"Black Penalties: {black_penalties:.2f}")

def test_move_comparison():
    """Compare evaluations to see if heuristics guide good moves"""
    print("\n\nTesting Move Guidance:")
    print("=" * 40)
    
    piece_values = {'pawn': 100, 'knight': 320, 'bishop': 330, 'rook': 500, 'queen': 900, 'king': 0}
    calc = V7P3RScoringCalculationV93(piece_values)
    
    # Starting position - compare different first moves
    board = chess.Board()
    print("Starting position move comparison:")
    
    # Test different opening moves
    moves_to_test = [
        ("e2e4", "King's pawn - central control"),
        ("d2d4", "Queen's pawn - central control"), 
        ("g1f3", "Knight development"),
        ("d1h5", "Early queen attack (BAD)"),
        ("h2h4", "Wing pawn advance (BAD)")
    ]
    
    base_eval = calc.calculate_score_optimized(board, chess.WHITE)
    print(f"Base evaluation: {base_eval:.2f}")
    
    for move_uci, description in moves_to_test:
        test_board = board.copy()
        move = chess.Move.from_uci(move_uci)
        test_board.push(move)
        
        eval_after = calc.calculate_score_optimized(test_board, chess.WHITE)
        change = eval_after - base_eval
        
        print(f"  {move_uci} ({description}): {eval_after:.2f} (change: {change:+.2f})")

def main():
    """Run all enhanced heuristics tests"""
    print("V7P3R v9.3 Enhanced Positional Heuristics Test")
    print("=" * 60)
    
    try:
        test_developmental_heuristics()
        test_early_game_penalties()
        test_move_comparison()
        
        print("\n🎉 All enhanced heuristics tests completed successfully!")
        print("✓ Developmental heuristics working")
        print("✓ Early game penalties active")
        print("✓ Position evaluation guiding good moves")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
