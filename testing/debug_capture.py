#!/usr/bin/env python3

"""
Debug the high-value capture f5e4 to understand why it's not being prioritized
"""

import sys
import chess

# Add v7p3r engine path
v7p3r_path = r"s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src"
sys.path.append(v7p3r_path)

def debug_capture():
    try:
        from v7p3r import V7P3REngine
        engine = V7P3REngine()
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return
    
    # High-value capture position
    board = chess.Board('3r1rk1/pp1q1p1R/2n2p1Q/4pb2/4B3/2P2NP1/PP3PP1/R3K3 b Q - 8 20')
    
    print("HIGH-VALUE CAPTURE DEBUG: f5e4")
    print("=" * 40)
    print(f"Position: {board.fen()}")
    
    # Check the top 10 moves and their categories
    legal_moves = list(board.legal_moves)
    tactical_analysis = engine._analyze_position_for_tactics(board)
    ordered_moves = engine._order_moves_advanced(board, legal_moves, 4)
    
    print(f"\nTop 10 moves and their likely categories:")
    target_move = chess.Move.from_uci('f5e4')
    
    for i, move in enumerate(ordered_moves[:10], 1):
        move_str = str(move)
        
        # Determine category
        category = "unknown"
        tactical_score = engine._calculate_tactical_move_score(board, move, tactical_analysis)
        
        if board.gives_check(move):
            category = "check"
        elif board.is_capture(move):
            victim = board.piece_at(move.to_square)
            if victim:
                victim_value = engine._get_dynamic_piece_value(board, victim.piece_type, not board.turn)
                if victim_value >= 500:
                    category = f"high_capture({victim_value})"
                else:
                    category = f"capture({victim_value})"
        elif tactical_score.get('attacks_multiple', False):
            category = "multi_attack"
        elif tactical_score.get('creates_threat', False):
            threat_value = tactical_score.get('threat_value', 0)
            category = f"threat({threat_value})"
        
        marker = "👈 TARGET" if move == target_move else ""
        print(f"  {i:2}. {move_str:6} - {category:20} {marker}")
    
    # Detailed analysis of f5e4
    print(f"\nDETAILED ANALYSIS of f5e4:")
    tactical_score = engine._calculate_tactical_move_score(board, target_move, tactical_analysis)
    print(f"Tactical score: {tactical_score}")
    
    if board.is_capture(target_move):
        victim = board.piece_at(target_move.to_square)
        if victim:
            victim_value = engine._get_dynamic_piece_value(board, victim.piece_type, not board.turn)
            print(f"Captured piece: {victim.symbol()} worth {victim_value}")
            
            # Check capture scoring
            attacker = board.piece_at(target_move.from_square)
            attacker_value = engine._get_dynamic_piece_value(board, attacker.piece_type, board.turn) if attacker else 0
            mvv_lva = victim_value * 100 - attacker_value
            
            high_value_bonus = 300.0 if victim_value >= 500 else 150.0 if victim_value >= 300 else 0
            safe_capture_bonus = 50.0 if tactical_score.get('safe_capture', False) else 0
            total_capture_score = mvv_lva + high_value_bonus + safe_capture_bonus + tactical_score['base_score']
            
            print(f"MVV-LVA: {mvv_lva} (victim: {victim_value}, attacker: {attacker_value})")
            print(f"High-value bonus: {high_value_bonus}")
            print(f"Safe capture bonus: {safe_capture_bonus}")
            print(f"Total capture score: {total_capture_score}")
            
            if victim_value >= 500:
                print("Should be in HIGH-VALUE CAPTURES category")
            elif victim_value >= 300:
                print("Should get high-value bonus but in regular CAPTURES category")
            else:
                print("Regular capture")

if __name__ == "__main__":
    debug_capture()