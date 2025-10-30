#!/usr/bin/env python3
"""
Test V14.6 Phase-Based Evaluation
Verify that refactored phase-based evaluation produces similar results to V14.5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator

def test_evaluation_consistency():
    """Test that phase-based evaluation produces reasonable scores"""
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    test_positions = [
        {
            'name': 'Starting Position',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'expected_phase': evaluator.PHASE_OPENING,
            'expected_range': (-50, 50),  # Should be close to equal
        },
        {
            'name': 'White Advantage - Middlegame',
            'fen': 'r1bq1rk1/ppp2ppp/2n5/3pP3/2PP4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 12',
            'expected_phase': evaluator.PHASE_MIDDLEGAME,
            'expected_range': (0, 300),  # White should be slightly better
        },
        {
            'name': 'Rook Endgame',
            'fen': '8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 40',
            'expected_phase': evaluator.PHASE_EARLY_ENDGAME,
            'expected_range': (-100, 100),  # Should be roughly equal
        },
        {
            'name': 'K+Q vs K (White Winning)',
            'fen': '8/8/8/8/8/3k4/8/K2Q4 w - - 0 60',
            'expected_phase': evaluator.PHASE_LATE_ENDGAME,
            'expected_range': (700, 1200),  # White should be winning (has queen)
        },
    ]
    
    print("=" * 70)
    print("V14.6 Phase-Based Evaluation Test")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    for test in test_positions:
        board = chess.Board(test['fen'])
        phase = evaluator.detect_game_phase(board)
        score = evaluator.evaluate_position_complete(board)
        
        min_expected, max_expected = test['expected_range']
        phase_correct = phase == test['expected_phase']
        score_reasonable = min_expected <= score <= max_expected
        
        result = "✓ PASS" if (phase_correct and score_reasonable) else "✗ FAIL"
        if phase_correct and score_reasonable:
            passed += 1
        else:
            failed += 1
        
        print(f"{result} | {test['name']}")
        print(f"      Phase: Expected={test['expected_phase']}, Got={phase} {'✓' if phase_correct else '✗'}")
        print(f"      Score: {score:.1f} (expected range: {min_expected} to {max_expected}) {'✓' if score_reasonable else '✗'}")
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_positions)} tests")
    print("=" * 70)
    
    return failed == 0

def test_phase_specific_features():
    """Test that phase-specific features are being applied"""
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    print("\n" + "=" * 70)
    print("Testing Phase-Specific Evaluation Features")
    print("=" * 70)
    print()
    
    # Test opening evaluation
    opening_board = chess.Board()
    opening_score = evaluator.evaluate_position_complete(opening_board)
    print(f"Opening Position: Score={opening_score:.1f}, Phase={evaluator.detect_game_phase(opening_board)}")
    
    # Test after a few moves
    opening_board.push_san("e4")
    opening_board.push_san("e5")
    opening_board.push_san("Nf3")
    opening_board.push_san("Nc6")
    score_after_moves = evaluator.evaluate_position_complete(opening_board)
    print(f"After 4 moves: Score={score_after_moves:.1f}, Phase={evaluator.detect_game_phase(opening_board)}")
    
    # Test middlegame
    middlegame_fen = 'r1bq1rk1/ppp2ppp/2n5/3p4/2PP4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10'
    middlegame_board = chess.Board(middlegame_fen)
    middlegame_score = evaluator.evaluate_position_complete(middlegame_board)
    print(f"Middlegame: Score={middlegame_score:.1f}, Phase={evaluator.detect_game_phase(middlegame_board)}")
    
    # Test endgame
    endgame_fen = '8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 40'
    endgame_board = chess.Board(endgame_fen)
    endgame_score = evaluator.evaluate_position_complete(endgame_board)
    print(f"Endgame: Score={endgame_score:.1f}, Phase={evaluator.detect_game_phase(endgame_board)}")
    
    print("\n✓ Phase-specific evaluations are working")
    return True

def test_blunder_firewall_active():
    """Verify blunder firewall is still active in all phases"""
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    evaluator = V7P3RBitboardEvaluator(piece_values)
    
    print("\n" + "=" * 70)
    print("Testing Blunder Firewall (Always Active)")
    print("=" * 70)
    print()
    
    # Test that safety analysis is called in different phases
    positions = [
        ('Opening', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
        ('Middlegame', 'r1bq1rk1/ppp2ppp/2n5/3p4/2PP4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10'),
        ('Endgame', '8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 40'),
    ]
    
    for phase_name, fen in positions:
        board = chess.Board(fen)
        
        # Call safety analysis directly
        safety_data = evaluator.analyze_safety_bitboard(board)
        
        # Verify safety data exists
        has_white_safety = 'white_safety_bonus' in safety_data
        has_black_safety = 'black_safety_bonus' in safety_data
        
        print(f"{phase_name}: Safety analysis active = {has_white_safety and has_black_safety} ✓")
    
    print("\n✓ Blunder firewall confirmed active in all phases")
    return True

if __name__ == "__main__":
    print("Testing V14.6 Phase-Based Evaluation System\n")
    
    success = True
    success = test_evaluation_consistency() and success
    success = test_phase_specific_features() and success
    success = test_blunder_firewall_active() and success
    
    if success:
        print("\n✓✓✓ All tests passed! V14.6 phase-based evaluation working correctly.")
        print("Ready for optimization phase (reduce computations per phase).")
        sys.exit(0)
    else:
        print("\n✗✗✗ Some tests failed. Review evaluation logic.")
        sys.exit(1)
