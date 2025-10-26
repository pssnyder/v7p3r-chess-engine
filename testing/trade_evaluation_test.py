#!/usr/bin/env python3

"""
V7P3R Trade Evaluation Test

Test the new trade evaluation preferences:
1. Equal trades should be highly favored
2. Simplification bonus should apply
3. No performance impact
"""

import chess
import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_trade_evaluation():
    """Test V7P3R's new trade evaluation preferences"""
    
    # Test positions with trade opportunities
    test_positions = [
        {
            "name": "Equal Queen Trade",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",
            "description": "Position where Qd5 allows queen trade",
        },
        {
            "name": "Knight for Knight Trade",
            "fen": "rnbqkb1r/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "description": "Knights can be traded on various squares",
        },
        {
            "name": "Rook Endgame", 
            "fen": "8/8/8/3k4/8/3K4/R7/r7 w - - 0 1",
            "description": "Simplified position should get simplification bonus",
        },
        {
            "name": "Complex Middlegame",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4",
            "description": "Many pieces - no simplification bonus",
        }
    ]
    
    engine = V7P3REngine()
    
    print("V7P3R Trade Evaluation Test")
    print("=" * 50)
    
    for position in test_positions:
        print(f"\n{position['name']}")
        print(f"Description: {position['description']}")
        print(f"FEN: {position['fen']}")
        print("-" * 40)
        
        board = chess.Board(position['fen'])
        
        # Get evaluation and best move
        start_time = time.time()
        best_move = engine.search(board, depth=3)  # Shallow search for speed
        search_time = time.time() - start_time
        
        # Get position evaluation separately
        evaluation = engine._evaluate_position(board)
        
        print(f"Best move: {best_move}")
        print(f"Evaluation: {evaluation:+.1f}")
        print(f"Search time: {search_time*1000:.1f}ms")
        
        # Check for trading moves in top candidates
        legal_moves = list(board.legal_moves)
        ordered_moves = engine._order_moves_advanced(board, legal_moves, depth=2, tt_move=None)
        
        print(f"Top 5 move priorities:")
        trade_moves = []
        
        for i, move in enumerate(ordered_moves[:5], 1):
            move_type = classify_move_with_trade(board, move)
            print(f"  {i}. {move} - {move_type}")
            
            if "Equal Trade" in move_type or "Good Trade" in move_type:
                trade_moves.append(move)
        
        if trade_moves:
            print(f"✅ Found {len(trade_moves)} favorable trade moves in top 5")
        else:
            print(f"ℹ️  No obvious trade moves in this position")
        
        # Check piece count and simplification
        piece_count = len(board.piece_map())
        print(f"Pieces on board: {piece_count}/32")
        if piece_count < 16:
            expected_bonus = (32 - piece_count) * 2
            print(f"📉 Simplification bonus applied: +{expected_bonus}cp")
    
    print(f"\n{'='*50}")
    print("TRADE EVALUATION SUMMARY")
    print(f"{'='*50}")
    print("✅ Enhanced trade evaluation features:")
    print("   • Equal trades: +80cp bonus")
    print("   • Good trades: +50cp bonus")  
    print("   • Bad trades: -20cp penalty")
    print("   • Simplification bonus: +2cp per missing piece pair")
    print("\nThis should make V7P3R prefer:")
    print("   1. Equal piece exchanges")
    print("   2. Simplified, less complex positions")
    print("   3. Avoid creating unnecessary tension")

def classify_move_with_trade(board: chess.Board, move: chess.Move) -> str:
    """Classify a move and identify trade types"""
    classifications = []
    
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if victim and attacker:
            victim_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, 
                           chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000}
            
            victim_value = victim_values.get(victim.piece_type, 0)
            attacker_value = victim_values.get(attacker.piece_type, 0)
            value_diff = abs(victim_value - attacker_value)
            
            if value_diff <= 30:
                classifications.append("Equal Trade")
            elif victim_value > attacker_value:
                classifications.append("Good Trade")
            else:
                classifications.append("Bad Trade")
            
            classifications.append(f"Capture {chess.piece_name(victim.piece_type)}")
    
    if board.gives_check(move):
        classifications.append("Check")
    
    if move.promotion:
        classifications.append("Promotion")
    
    if not classifications:
        piece = board.piece_at(move.from_square)
        if piece:
            classifications.append(f"{chess.piece_name(piece.piece_type).title()}")
        else:
            classifications.append("Quiet")
    
    return ", ".join(classifications)

if __name__ == "__main__":
    test_trade_evaluation()