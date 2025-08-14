# v7p3r_utils.py

"""Utility Functions for V7P3R Chess Engine
Provides common functions and constants used across multiple modules.
"""

import chess

# Standardized piece values (in centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Critical evaluation scores
CHECKMATE_SCORE = 999999
STALEMATE_PENALTY = -999999
DRAW_PENALTY = -5000
REPETITION_PENALTY = -5000

def get_material_balance(board, color):
    """Get material balance from the perspective of the specified color"""
    our_material = 0
    their_material = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            value = PIECE_VALUES[piece.piece_type]
            if piece.color == color:
                our_material += value
            else:
                their_material += value
    
    return our_material - their_material

def get_game_phase(board):
    """Determine the current game phase based on material and development"""
    # Count total pieces (excluding kings)
    piece_count = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            piece_count += 1
    
    # Simple phase detection
    if piece_count >= 24:  # Most pieces on board
        return "opening"
    elif piece_count >= 12:  # Medium number of pieces
        return "middlegame"
    else:  # Few pieces left
        return "endgame"

def is_draw_position(board):
    """Check if position is a draw"""
    return (board.is_stalemate() or 
            board.is_insufficient_material() or
            board.is_seventyfive_moves() or
            board.is_fivefold_repetition() or
            board.can_claim_threefold_repetition() or
            board.can_claim_fifty_moves())

def is_capture_that_escapes_check(board, move):
    """Check if this move captures a piece while escaping check"""
    if not board.is_check() or not board.is_capture(move):
        return False
    
    # Get the attacking pieces
    king_square = board.king(board.turn)
    attackers = board.attackers(not board.turn, king_square)
    
    # Check if the capture target is one of the checking pieces
    target_square = move.to_square
    return target_square in attackers

def evaluate_exchange(board, move):
    """Evaluate a capture sequence to determine material gain/loss"""
    if not board.is_capture(move):
        return 0
    
    # Get the capturing piece
    capturing_piece = board.piece_at(move.from_square)
    if not capturing_piece:
        return 0
    
    # Get the captured piece
    captured_piece = board.piece_at(move.to_square)
    if not captured_piece:
        # En passant capture
        if board.is_en_passant(move):
            return PIECE_VALUES[chess.PAWN]
        return 0
    
    # Initial material gain (what we capture)
    material_gain = PIECE_VALUES[captured_piece.piece_type]
    
    # Make a copy of the board to simulate the capture
    test_board = board.copy()
    test_board.push(move)
    
    # Check if the opponent can recapture
    target_square = move.to_square
    opponent_attackers = test_board.attackers(not test_board.turn, target_square)
    
    if not opponent_attackers:
        # NO RECAPTURE POSSIBLE - FREE MATERIAL!
        return material_gain
    
    # Find the least valuable attacker that can recapture
    min_attacker_value = float('inf')
    for attacker_square in opponent_attackers:
        attacker_piece = test_board.piece_at(attacker_square)
        if attacker_piece:
            attacker_value = PIECE_VALUES[attacker_piece.piece_type]
            if attacker_value < min_attacker_value:
                min_attacker_value = attacker_value
    
    # Calculate net material exchange
    our_loss = PIECE_VALUES[capturing_piece.piece_type]
    their_loss = PIECE_VALUES[captured_piece.piece_type]
    
    return their_loss - our_loss

def find_hanging_pieces(board, color):
    """Find undefended pieces that can be captured for free"""
    hanging_pieces = []
    
    # Look at all opponent pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color != color:  # Opponent's piece
            # Check if this piece is attacked by us
            our_attackers = board.attackers(color, square)
            if our_attackers:
                # Check if it's defended by opponent
                their_defenders = board.attackers(piece.color, square)
                
                if not their_defenders:
                    # HANGING PIECE! No defenders
                    hanging_pieces.append((square, piece, PIECE_VALUES[piece.piece_type]))
                else:
                    # Check if we can win the exchange
                    min_attacker_value = min(
                        PIECE_VALUES[board.piece_at(att_sq).piece_type] 
                        for att_sq in our_attackers 
                        if board.piece_at(att_sq)
                    )
                    min_defender_value = min(
                        PIECE_VALUES[board.piece_at(def_sq).piece_type] 
                        for def_sq in their_defenders 
                        if board.piece_at(def_sq)
                    )
                    
                    # If we can capture with a less valuable piece than their defender
                    if min_attacker_value < min_defender_value:
                        net_gain = PIECE_VALUES[piece.piece_type] - min_attacker_value
                        if net_gain > 0:
                            hanging_pieces.append((square, piece, net_gain))
    
    # Sort by value (highest first)
    hanging_pieces.sort(key=lambda x: x[2], reverse=True)
    return hanging_pieces
