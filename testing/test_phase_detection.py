#!/usr/bin/env python3
"""
Test Phase Detection for V14.6
Verify that game phase detection is accurate and deterministic
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_bitboard_evaluator import V7P3RBitboardEvaluator

def test_phase_detection():
    """Test phase detection across various positions"""
    
    # Standard piece values
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
            'expected': evaluator.PHASE_OPENING,
            'description': 'Move 1, full material'
        },
        {
            'name': 'Italian Game (Move 5)',
            'fen': 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 5',
            'expected': evaluator.PHASE_OPENING,
            'description': 'Move 5, high material, development phase'
        },
        {
            'name': 'Middlegame Position',
            'fen': 'r1bq1rk1/ppp2ppp/2n5/3p4/2PP4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10',
            'expected': evaluator.PHASE_MIDDLEGAME,
            'description': 'Move 10+, moderate material, both castled'
        },
        {
            'name': 'Complex Middlegame',
            'fen': '1r3rk1/p4ppp/2p5/q1n1p3/2B1P3/2N2Q1P/PP3PP1/3R1RK1 w - - 0 20',
            'expected': evaluator.PHASE_MIDDLEGAME,
            'description': 'Move 20, queens on board, tactical position'
        },
        {
            'name': 'Early Endgame (Rook Endgame)',
            'fen': '8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 40',
            'expected': evaluator.PHASE_EARLY_ENDGAME,
            'description': 'Rook + pawns, material ~1100'
        },
        {
            'name': 'Late Endgame (King + Pawns)',
            'fen': '8/8/4k3/8/3K4/8/3P4/8 w - - 0 50',
            'expected': evaluator.PHASE_LATE_ENDGAME,
            'description': 'King and pawn endgame, minimal material'
        },
        {
            'name': 'Late Endgame (K+Q vs K)',
            'fen': '8/8/8/8/8/3k4/8/K2Q4 w - - 0 60',
            'expected': evaluator.PHASE_LATE_ENDGAME,
            'description': 'KQ vs K, only 4 pieces total'
        },
        {
            'name': 'Early Endgame (R+B vs R+N)',
            'fen': '8/5pk1/6p1/8/8/3N2P1/5PK1/3R4 b - - 0 45',
            'expected': evaluator.PHASE_EARLY_ENDGAME,
            'description': 'Minor pieces + rooks, material ~1500'
        },
        {
            'name': 'Forced Opening (High Material)',
            'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 20',
            'expected': evaluator.PHASE_MIDDLEGAME,
            'description': 'Move 20 with full material - middlegame (past opening moves)'
        }
    ]
    
    print("=" * 70)
    print("V14.6 Game Phase Detection Test")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    
    for test in test_positions:
        board = chess.Board(test['fen'])
        detected_phase = evaluator.detect_game_phase(board)
        
        # Calculate stats for verification
        moves = len(board.move_stack)
        material = sum([
            len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        ])
        total_material = sum([
            (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))) * 
            [100, 320, 330, 500, 900][pt-1]
            for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        ])
        pieces = len(board.piece_map())
        
        result = "✓ PASS" if detected_phase == test['expected'] else "✗ FAIL"
        if detected_phase == test['expected']:
            passed += 1
        else:
            failed += 1
        
        print(f"{result} | {test['name']}")
        print(f"      Description: {test['description']}")
        print(f"      Expected: {test['expected']}, Detected: {detected_phase}")
        print(f"      Stats: Moves={moves}, Material={total_material}, Pieces={pieces}")
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_positions)} tests")
    print("=" * 70)
    
    if failed == 0:
        print("✓ All phase detection tests passed!")
        return True
    else:
        print(f"✗ {failed} test(s) failed")
        return False

def test_phase_symmetry():
    """Test that phase detection is symmetrical between white and black"""
    
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
    print("Testing Phase Symmetry (White vs Black)")
    print("=" * 70)
    print()
    
    # Test that white and black see same phase
    test_fen = 'r1bq1rk1/ppp2ppp/2n5/3p4/2PP4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10'
    
    board_white_turn = chess.Board(test_fen)
    board_black_turn = chess.Board(test_fen.replace(' w ', ' b '))
    
    phase_white = evaluator.detect_game_phase(board_white_turn)
    phase_black = evaluator.detect_game_phase(board_black_turn)
    
    print(f"Position: {test_fen[:40]}...")
    print(f"White to move: {phase_white}")
    print(f"Black to move: {phase_black}")
    
    if phase_white == phase_black:
        print("✓ Phase detection is symmetrical")
        return True
    else:
        print("✗ Phase detection differs between white and black!")
        return False

if __name__ == "__main__":
    print("Testing V14.6 Phase Detection\n")
    
    success = test_phase_detection()
    success = test_phase_symmetry() and success
    
    if success:
        print("\n✓✓✓ All tests passed! Phase detection ready for integration.")
        sys.exit(0)
    else:
        print("\n✗✗✗ Some tests failed. Review phase detection logic.")
        sys.exit(1)
