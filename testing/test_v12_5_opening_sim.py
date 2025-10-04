#!/usr/bin/env python3
"""
V7P3R v12.5 Quick Opening Test
Test the enhanced opening play in a simulated game
"""

import chess
import sys
import os
import time

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def simulate_opening_game():
    """Simulate first few moves of a game to test opening improvements"""
    print("🎮 V7P3R v12.5 Opening Simulation")
    print("=" * 45)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    moves_played = []
    
    for move_num in range(1, 5):  # Play first 4 moves
        print(f"\nMove {move_num}: {board.fen()}")
        print(f"{'White' if board.turn == chess.WHITE else 'Black'} to move")
        
        # Get engine's move
        start_time = time.time()
        best_move = engine.search(board, depth=4)
        search_time = time.time() - start_time
        
        if best_move:
            moves_played.append(best_move.uci())
            
            print(f"Engine plays: {best_move.uci()} (time: {search_time:.2f}s)")
            
            # Show move analysis
            bonus = engine._get_nudge_bonus(board, best_move)
            if bonus > 0:
                print(f"  🧠 Nudge bonus: +{bonus:.1f}")
            
            board.push(best_move)
        else:
            print("No move found!")
            break
    
    print(f"\nGame moves: {' '.join(moves_played)}")
    print(f"Final position: {board.fen()}")
    
    # Analyze the opening
    print("\n📊 Opening Analysis:")
    
    # Check for center control
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    white_center_control = 0
    black_center_control = 0
    
    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                white_center_control += 1
            else:
                black_center_control += 1
        
        # Count attackers
        white_attackers = len(board.attackers(chess.WHITE, square))
        black_attackers = len(board.attackers(chess.BLACK, square))
        white_center_control += white_attackers * 0.5
        black_center_control += black_attackers * 0.5
    
    print(f"Center control - White: {white_center_control:.1f}, Black: {black_center_control:.1f}")
    
    # Check piece development
    starting_pieces = {
        chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],  # Knights and bishops
        chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8]
    }
    
    for color in [chess.WHITE, chess.BLACK]:
        developed = 0
        total = len(starting_pieces[color])
        
        for square in starting_pieces[color]:
            piece = board.piece_at(square)
            if not piece or piece.piece_type not in [chess.KNIGHT, chess.BISHOP]:
                developed += 1
        
        color_name = "White" if color == chess.WHITE else "Black"
        print(f"{color_name} development: {developed}/{total} pieces")
    
    # Check for typical opening principles
    opening_score = 0
    if any(move.startswith('e2e4') or move.startswith('d2d4') for move in moves_played[:2]):
        opening_score += 2
        print("✅ Central pawn advance")
    
    if any(move.startswith('g1f3') or move.startswith('b1c3') for move in moves_played[:4]):
        opening_score += 1
        print("✅ Knight development")
    
    if white_center_control > black_center_control:
        opening_score += 1
        print("✅ Good center control")
    
    print(f"\nOpening Score: {opening_score}/4")
    if opening_score >= 3:
        print("🎉 Excellent opening play!")
    elif opening_score >= 2:
        print("✅ Good opening principles")
    else:
        print("⚠️  Opening could be improved")

if __name__ == "__main__":
    simulate_opening_game()