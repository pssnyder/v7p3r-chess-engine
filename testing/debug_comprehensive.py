#!/usr/bin/env python3
"""
Comprehensive debug script to trace evaluation components
"""
import sys
import os
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def debug_comprehensive_evaluation():
    """Test comprehensive evaluation component breakdown"""
    
    # Test position: just kings on board (symmetrical endgame)
    fen = "8/8/8/8/3k4/8/8/3K4 w - - 0 1"
    board = chess.Board(fen)
    
    engine = V7P3REngine()
    
    print("=== COMPREHENSIVE EVALUATION DEBUG ===")
    print(f"Position: {fen}")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print()
    
    # Test with White to move
    print("--- WHITE TO MOVE ---")
    board.turn = chess.WHITE
    
    # Get component scores separately
    try:
        # Material evaluation
        material_white = engine._count_material(board, chess.WHITE)
        material_black = engine._count_material(board, chess.BLACK)
        material_diff = material_white - material_black
        print(f"Material: White={material_white}, Black={material_black}, Diff={material_diff}")
        
        # Bitboard evaluator
        if hasattr(engine, 'bitboard_evaluator'):
            bb_score = engine.bitboard_evaluator.calculate_score_optimized(board, chess.WHITE)
            print(f"Bitboard evaluator (White perspective): {bb_score}")
        
        # Advanced pawn evaluator
        if hasattr(engine, 'advanced_pawn_evaluator'):
            pawn_white = engine.advanced_pawn_evaluator.evaluate_pawn_structure(board, chess.WHITE)
            pawn_black = engine.advanced_pawn_evaluator.evaluate_pawn_structure(board, chess.BLACK)
            print(f"Pawn eval: White={pawn_white}, Black={pawn_black}")
        
        # King safety evaluator
        if hasattr(engine, 'king_safety_evaluator'):
            king_white = engine.king_safety_evaluator.evaluate_king_safety(board, chess.WHITE)
            king_black = engine.king_safety_evaluator.evaluate_king_safety(board, chess.BLACK)
            print(f"King safety: White={king_white}, Black={king_black}")
        
        # Tactical pattern detector
        if hasattr(engine, 'tactical_pattern_detector'):
            tactical_white = engine.tactical_pattern_detector.detect_tactical_patterns(board, chess.WHITE)
            tactical_black = engine.tactical_pattern_detector.detect_tactical_patterns(board, chess.BLACK)
            print(f"Tactical: White={tactical_white}, Black={tactical_black}")
        
        # Full evaluation
        full_eval = engine._evaluate_position(board)
        print(f"Full evaluation (White perspective): {full_eval}")
        
    except Exception as e:
        print(f"Error during White evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test with Black to move
    print("--- BLACK TO MOVE ---")
    board.turn = chess.BLACK
    
    try:
        # Material evaluation
        material_white = engine._count_material(board, chess.WHITE)
        material_black = engine._count_material(board, chess.BLACK)
        material_diff = material_white - material_black
        print(f"Material: White={material_white}, Black={material_black}, Diff={material_diff}")
        
        # Bitboard evaluator
        if hasattr(engine, 'bitboard_evaluator'):
            bb_score = engine.bitboard_evaluator.calculate_score_optimized(board, chess.BLACK)
            print(f"Bitboard evaluator (Black perspective): {bb_score}")
        
        # Advanced pawn evaluator
        if hasattr(engine, 'advanced_pawn_evaluator'):
            pawn_white = engine.advanced_pawn_evaluator.evaluate_pawn_structure(board, chess.WHITE)
            pawn_black = engine.advanced_pawn_evaluator.evaluate_pawn_structure(board, chess.BLACK)
            print(f"Pawn eval: White={pawn_white}, Black={pawn_black}")
        
        # King safety evaluator
        if hasattr(engine, 'king_safety_evaluator'):
            king_white = engine.king_safety_evaluator.evaluate_king_safety(board, chess.WHITE)
            king_black = engine.king_safety_evaluator.evaluate_king_safety(board, chess.BLACK)
            print(f"King safety: White={king_white}, Black={king_black}")
        
        # Tactical pattern detector
        if hasattr(engine, 'tactical_pattern_detector'):
            tactical_white = engine.tactical_pattern_detector.detect_tactical_patterns(board, chess.WHITE)
            tactical_black = engine.tactical_pattern_detector.detect_tactical_patterns(board, chess.BLACK)
            print(f"Tactical: White={tactical_white}, Black={tactical_black}")
        
        # Full evaluation
        full_eval = engine._evaluate_position(board)
        print(f"Full evaluation (Black perspective): {full_eval}")
        
    except Exception as e:
        print(f"Error during Black evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=== EXPECTED RESULTS ===")
    print("For symmetrical position, all component scores should be:")
    print("- Material: Same for both sides")
    print("- Bitboard: Should be opposite signs for White/Black perspective")
    print("- Pawn: Should be 0 (no pawns)")
    print("- King safety: Should be opposite signs for White/Black perspective")
    print("- Tactical: Should be 0 (no tactics)")
    print("- Full eval: Should be opposite signs for White/Black to move")

if __name__ == "__main__":
    debug_comprehensive_evaluation()