# v7p3r_tempo.py

"""Tempo Calculation for V7P3R Chess Engine
Handles critical move detection including checkmate threats, stalemate avoidance, and draw prevention.
"""

import chess

class TempoCalculation:
    def __init__(self):
        self.checkmate_score = 999999
        self.stalemate_penalty = -999999
        self.mate_threat_bonus = 50000
    
    def evaluate_tempo(self, board, move, depth):
        """Evaluate tempo factors for a move"""
        # Make the move on a copy to evaluate the resulting position
        board_copy = board.copy()
        board_copy.push(move)
        
        tempo_score = 0
        critical_move = False
        
        # Check for immediate checkmate
        if board_copy.is_checkmate():
            return self.checkmate_score, True
        
        # Check for stalemate (avoid this)
        if board_copy.is_stalemate():
            return self.stalemate_penalty, True
        
        # Check for draw conditions (avoid when ahead)
        if self._is_draw_position(board_copy):
            material_balance = self._get_material_balance(board_copy, board.turn)
            if material_balance > 0:  # We're ahead, avoid draw
                tempo_score -= 10000
        
        # Check for checkmate threats within mate horizon
        mate_threat = self._find_mate_threat(board_copy, depth)
        if mate_threat:
            if mate_threat > 0:  # We have mate threat
                tempo_score += self.mate_threat_bonus
                critical_move = True
            else:  # Opponent has mate threat
                tempo_score += mate_threat
        
        # Check if move gives check (small bonus)
        if board_copy.is_check():
            tempo_score += 100
        
        return tempo_score, critical_move
    
    def _is_draw_position(self, board):
        """Check if position is a draw"""
        return (board.is_stalemate() or 
                board.is_insufficient_material() or
                board.is_seventyfive_moves() or
                board.is_fivefold_repetition())
    
    def _get_material_balance(self, board, our_color):
        """Get material balance from our perspective"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        our_material = 0
        their_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]
                if piece.color == our_color:
                    our_material += value
                else:
                    their_material += value
        
        return our_material - their_material
    
    def _find_mate_threat(self, board, max_depth):
        """Look for mate threats within specified depth"""
        if max_depth <= 0:
            return None
        
        # Simple mate threat detection - look for forced sequences
        if board.is_check():
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 1:
                # Only one legal move - might be forced
                board.push(legal_moves[0])
                if board.is_checkmate():
                    board.pop()
                    return -self.checkmate_score + max_depth  # Opponent mates us
                
                # Look deeper
                deeper_threat = self._find_mate_threat(board, max_depth - 1)
                board.pop()
                if deeper_threat:
                    return deeper_threat
        
        # Look for our mate threats
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return self.checkmate_score - max_depth  # We mate opponent
            board.pop()
        
        return None
    
    def should_short_circuit(self, score):
        """Determine if we should short circuit based on tempo score"""
        return abs(score) >= self.mate_threat_bonus
