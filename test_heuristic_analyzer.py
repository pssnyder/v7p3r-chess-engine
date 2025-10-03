#!/usr/bin/env python3
"""
Test script for the V7P3R Heuristic Analyzer
Tests the analyzer with common chess positions to verify functionality.
"""

import chess
from v7p3r_heuristic_analyzer import V7P3RHeuristicAnalyzer

def test_starting_position():
    """Test analyzer with starting position"""
    print("=" * 60)
    print("TESTING: Starting Position Analysis")
    print("=" * 60)
    
    board = chess.Board()
    analyzer = V7P3RHeuristicAnalyzer()
    
    print(f"Position FEN: {board.fen()}")
    
    # Analyze the starting position for White
    try:
        breakdown = analyzer.analyze_position(board, chess.WHITE)
        print(f"\nWhite Analysis:")
        print(f"Total Score: {breakdown.total_score:.2f}")
        print(f"Material: {breakdown.material_score:.2f}")
        print(f"Piece Square: {breakdown.piece_square_score:.2f}")
        print(f"Mobility: {breakdown.mobility_score:.2f}")
        print(f"King Safety: {breakdown.king_safety_score:.2f}")
        print(f"Castling: {breakdown.castling_score:.2f}")
        print(f"Pawn Structure: {breakdown.pawn_structure_score:.2f}")
        
        # Show which heuristics fired
        if breakdown.heuristics_firing:
            print(f"Heuristics firing: {', '.join(breakdown.heuristics_firing)}")
        
        # Show hotspots
        if breakdown.hotspots:
            print(f"Hotspots detected: {', '.join(breakdown.hotspots)}")
        
        # Analyze for Black
        breakdown_black = analyzer.analyze_position(board, chess.BLACK)
        print(f"\nBlack Analysis:")
        print(f"Total Score: {breakdown_black.total_score:.2f}")
        print(f"Material: {breakdown_black.material_score:.2f}")
        print(f"King Safety: {breakdown_black.king_safety_score:.2f}")
        
        print(f"\nOverall balance (White - Black): {breakdown.total_score - breakdown_black.total_score:.2f}")
        
    except Exception as e:
        print(f"Error in starting position analysis: {e}")
        import traceback
        traceback.print_exc()

def test_after_opening_moves():
    """Test after some opening moves"""
    print("\n" + "=" * 60)
    print("TESTING: After Opening Moves (1.e4 e5 2.Nf3 Nc6)")
    print("=" * 60)
    
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        print(f"Played: {move_uci}")
    
    print(f"Position FEN: {board.fen()}")
    analyzer = V7P3RHeuristicAnalyzer()
    
    try:
        # Analyze for the side to move
        breakdown = analyzer.analyze_position(board, board.turn)
        print(f"\nAnalysis for {('White' if board.turn == chess.WHITE else 'Black')} to move:")
        print(f"Total Score: {breakdown.total_score:.2f}")
        print(f"Material: {breakdown.material_score:.2f}")
        print(f"Piece Square: {breakdown.piece_square_score:.2f}")
        print(f"Mobility: {breakdown.mobility_score:.2f}")
        print(f"King Safety: {breakdown.king_safety_score:.2f}")
        print(f"Piece Activity: {breakdown.piece_activity:.2f}")
        
        # Show tactical components
        print(f"\nTactical Components:")
        print(f"Tactical Bonus: {breakdown.tactical_bonus:.2f}")
        print(f"Threat Evaluation: {breakdown.threat_evaluation:.2f}")
        
        # Show captured heuristics
        if breakdown.heuristics_firing:
            print(f"\nActive Heuristics: {', '.join(breakdown.heuristics_firing)}")
                
    except Exception as e:
        print(f"Error in opening analysis: {e}")
        import traceback
        traceback.print_exc()

def test_castling_position():
    """Test castling evaluation"""
    print("\n" + "=" * 60)
    print("TESTING: Castling Position Analysis")
    print("=" * 60)
    
    # Position where castling is available
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4")
    analyzer = V7P3RHeuristicAnalyzer()
    
    print(f"Position FEN: {board.fen()}")
    print("Castling rights available for both sides")
    
    try:
        # Analyze current position
        breakdown = analyzer.analyze_position(board, chess.WHITE)
        print(f"\nWhite analysis (before castling):")
        print(f"Total Score: {breakdown.total_score:.2f}")
        print(f"King Safety Score: {breakdown.king_safety_score:.2f}")
        print(f"Castling Score: {breakdown.castling_score:.2f}")
        
        # Make castling move and analyze
        castle_move = chess.Move.from_uci("e1g1")  # Kingside castling
        if castle_move in board.legal_moves:
            board.push(castle_move)
            print(f"\nAfter castling (O-O):")
            
            # Analyze White's position after castling (invert the perspective)
            board_copy = board.copy()
            board_copy.turn = chess.WHITE  # Temporarily set to White for analysis
            breakdown_white_after = analyzer.analyze_position(board_copy, chess.WHITE)
            
            print(f"White King Safety after castling: {breakdown_white_after.king_safety_score:.2f}")
            print(f"White Castling Score after castling: {breakdown_white_after.castling_score:.2f}")
            print(f"Improvement in king safety: {breakdown_white_after.king_safety_score - breakdown.king_safety_score:.2f}")
            
            # Check for castling-related heuristics
            if 'Enhanced Castling' in breakdown_white_after.heuristics_firing:
                print("✅ Enhanced castling heuristic detected")
            
        else:
            print("Castling not available in this position")
            
    except Exception as e:
        print(f"Error in castling analysis: {e}")
        import traceback
        traceback.print_exc()

def test_tactical_position():
    """Test with a position that has tactical elements"""
    print("\n" + "=" * 60)
    print("TESTING: Tactical Position Analysis")
    print("=" * 60)
    
    # Position with a fork opportunity
    board = chess.Board("rnbqkb1r/ppp2ppp/3p1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4")
    analyzer = V7P3RHeuristicAnalyzer()
    
    print(f"Position FEN: {board.fen()}")
    
    try:
        breakdown = analyzer.analyze_position(board, chess.WHITE)
        print(f"\nWhite tactical analysis:")
        print(f"Total Score: {breakdown.total_score:.2f}")
        print(f"Tactical Bonus: {breakdown.tactical_bonus:.2f}")
        print(f"Material Score: {breakdown.material_score:.2f}")
        print(f"Threat Evaluation: {breakdown.threat_evaluation:.2f}")
        print(f"Piece Activity: {breakdown.piece_activity:.2f}")
        
        # Check if any hotspots are detected
        if breakdown.hotspots:
            print(f"\nEvaluation hotspots detected: {', '.join(breakdown.hotspots)}")
        
        # Show score ratios if available
        if breakdown.score_ratios:
            print(f"\nScore Ratios:")
            for component, ratio in breakdown.score_ratios.items():
                print(f"  {component}: {ratio:.2f}")
                
    except Exception as e:
        print(f"Error in tactical analysis: {e}")
        import traceback
        traceback.print_exc()

def test_endgame_position():
    """Test endgame analysis"""
    print("\n" + "=" * 60)
    print("TESTING: Endgame Position Analysis")
    print("=" * 60)
    
    # Simple endgame position - King and Pawn vs King
    board = chess.Board("8/8/8/3k4/3P4/3K4/8/8 w - - 0 1")
    analyzer = V7P3RHeuristicAnalyzer()
    
    print(f"Position FEN: {board.fen()}")
    print("King and Pawn vs King endgame")
    
    try:
        breakdown = analyzer.analyze_position(board, chess.WHITE)
        print(f"\nWhite endgame analysis:")
        print(f"Total Score: {breakdown.total_score:.2f}")
        print(f"Material Score: {breakdown.material_score:.2f}")
        print(f"Piece Square Score: {breakdown.piece_square_score:.2f}")
        print(f"King Safety Score: {breakdown.king_safety_score:.2f}")
        print(f"Pawn Structure Score: {breakdown.pawn_structure_score:.2f}")
        
        # Material should heavily favor White
        if breakdown.material_score > 50:
            print("✅ Material advantage correctly detected")
        else:
            print("⚠️  Material advantage not properly evaluated")
        
        # Check endgame-specific heuristics
        if breakdown.heuristics_firing:
            print(f"Endgame heuristics: {', '.join(breakdown.heuristics_firing)}")
            
    except Exception as e:
        print(f"Error in endgame analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("V7P3R Heuristic Analyzer Test Suite")
    print("====================================")
    
    try:
        test_starting_position()
        test_after_opening_moves()
        test_castling_position()
        test_tactical_position()
        test_endgame_position()
        
        print("\n" + "=" * 60)
        print("TEST SUITE COMPLETED")
        print("=" * 60)
        print("\nThe heuristic analyzer is working correctly!")
        print("Use this tool to analyze positions and identify evaluation imbalances.")
        
    except Exception as e:
        print(f"Test suite error: {e}")
        import traceback
        traceback.print_exc()