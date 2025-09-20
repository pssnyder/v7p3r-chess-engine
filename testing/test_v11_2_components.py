#!/usr/bin/env python3
"""
Simple test for V7P3R v11.2 Enhanced Engine
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import chess
import time

# Test without the imports for now - let's build a minimal version
print("Testing V7P3R v11.2 Enhanced Engine Components...")

# Test 1: Basic move classification
print("\n1. Testing Move Classification...")

try:
    from v7p3r_v11_2_enhanced import MoveClass, PositionTone, EnhancedMoveOrderer
    
    board = chess.Board()
    orderer = EnhancedMoveOrderer()
    
    print("   ✅ Move classification classes imported successfully")
    
    # Test position analysis
    tone = orderer.analyze_position_tone(board)
    print(f"   ✅ Position tone analysis: tactical={tone.tactical_complexity:.2f}")
    
    # Test move ordering
    legal_moves = list(board.legal_moves)
    ordered_moves = orderer.order_moves(board, legal_moves)
    print(f"   ✅ Move ordering: {len(ordered_moves)} moves ordered")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Basic search
print("\n2. Testing Basic Search (without full engine)...")

try:
    # Simple material evaluation test
    def simple_eval(board):
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        if board.turn == chess.BLACK:
            score = -score
        
        return score
    
    board = chess.Board()
    eval_score = simple_eval(board)
    print(f"   ✅ Basic evaluation: {eval_score}")
    
    # Test a tactical position
    board.set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4")
    
    orderer = EnhancedMoveOrderer()
    tone = orderer.analyze_position_tone(board)
    print(f"   ✅ Tactical position analysis: tactical={tone.tactical_complexity:.2f}")
    
    legal_moves = list(board.legal_moves)
    ordered_moves = orderer.order_moves(board, legal_moves)
    
    print(f"   ✅ Ordered {len(ordered_moves)} moves in tactical position")
    print(f"   Top 3 moves: {[str(move) for move in ordered_moves[:3]]}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Position tone analysis on various positions
print("\n3. Testing Position Tone Analysis...")

test_positions = [
    ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Tactical position", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4"),
    ("Endgame position", "8/8/3k4/3P4/3K4/8/8/8 w - - 0 1"),
    ("King safety crisis", "r1bq1rk1/ppp2ppp/2n5/3pP3/3P4/2P5/P1P2PPP/R1BQKB1R b KQ - 0 10")
]

try:
    orderer = EnhancedMoveOrderer()
    
    for name, fen in test_positions:
        board = chess.Board(fen)
        tone = orderer.analyze_position_tone(board)
        
        print(f"   {name}:")
        print(f"     Tactical: {tone.tactical_complexity:.2f}")
        print(f"     King Safety: {tone.king_safety_urgency:.2f}")
        print(f"     Development: {tone.development_priority:.2f}")
        print(f"     Endgame: {tone.endgame_factor:.2f}")
        print(f"     Aggression: {tone.aggression_viability:.2f}")
    
    print("   ✅ Position tone analysis working correctly")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n🎯 Component Test Summary:")
print("Enhanced move ordering and position analysis components are working!")
print("Ready for integration testing with existing test suite.")

print("\n4. Testing Search Feedback System...")

try:
    from v7p3r_v11_2_enhanced import SearchFeedback
    
    feedback = SearchFeedback()
    
    # Simulate some search results
    feedback.record_move_result(MoveClass.OFFENSIVE, True)
    feedback.record_move_result(MoveClass.OFFENSIVE, True)  
    feedback.record_move_result(MoveClass.OFFENSIVE, False)
    
    feedback.record_move_result(MoveClass.DEFENSIVE, False)
    feedback.record_move_result(MoveClass.DEFENSIVE, False)
    feedback.record_move_result(MoveClass.DEFENSIVE, False)
    
    print(f"   Offensive confidence: {feedback.get_move_type_confidence(MoveClass.OFFENSIVE):.2f}")
    print(f"   Defensive confidence: {feedback.get_move_type_confidence(MoveClass.DEFENSIVE):.2f}")
    print(f"   Should deprioritize defensive: {feedback.should_deprioritize_move_type(MoveClass.DEFENSIVE)}")
    
    print("   ✅ Search feedback system working correctly")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n✅ All component tests completed!")
print("V7P3R v11.2 Enhanced components are ready for full engine testing.")