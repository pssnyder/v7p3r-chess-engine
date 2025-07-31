
# v7p3r_mvv_lva.py
"""
Most Valuable Victim - Least Valuable Attacker (MVV-LVA) for V7P3R Chess Engine
Evaluates capture moves for move ordering and scoring.
"""

import chess
from v7p3r_utils import PIECE_VALUES, evaluate_exchange

class MVVLVA:
    def __init__(self):
        # Use standardized piece values from utils
        self.piece_values = PIECE_VALUES
        
        # MVV-LVA table [victim][attacker]
        self.mvv_lva_table = {}
        self._build_mvv_lva_table()
    
    def _build_mvv_lva_table(self):
        """Build the MVV-LVA scoring table"""
        pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        
        for victim in pieces:
            self.mvv_lva_table[victim] = {}
            for attacker in pieces:
                # Higher score for capturing valuable pieces with less valuable pieces
                victim_value = self.piece_values[victim]
                attacker_value = self.piece_values[attacker]
                
                # MVV-LVA score: victim value * 10 - attacker value
                score = victim_value * 10 - attacker_value
                self.mvv_lva_table[victim][attacker] = score
    
    def get_capture_score(self, board, move):
        """
        Get MVV-LVA score for a capture move, factoring in free material detection.
        Returns a high score for profitable captures, penalizes losing captures.
        """
        if not board.is_capture(move):
            return 0
        capturing_piece = board.piece_at(move.from_square)
        if not capturing_piece:
            return 0
        captured_piece = board.piece_at(move.to_square)
        if not captured_piece:
            if board.is_en_passant(move):
                return self.mvv_lva_table[chess.PAWN][capturing_piece.piece_type]
            return 0
        base_score = self.mvv_lva_table[captured_piece.piece_type][capturing_piece.piece_type]
        material_gain = evaluate_exchange(board, move)
        if material_gain >= 100:
            if material_gain >= 300:
                return 50000 + material_gain
            else:
                return 30000 + material_gain
        if material_gain < -50:
            return base_score - 5000
        return base_score + material_gain
    
    # Using standardized evaluate_exchange function from v7p3r_utils

    
    def get_threat_score(self, board, square, attacking_piece_type):
        """Get threat score for attacking a piece on a square"""
        threatened_piece = board.piece_at(square)
        if not threatened_piece:
            return 0
        
        # Use MVV-LVA logic for threats
        return self.mvv_lva_table[threatened_piece.piece_type][attacking_piece_type]
    
    def sort_captures(self, board, moves):
        """Sort capture moves by MVV-LVA score (highest first)"""
        capture_moves = []
        other_moves = []
        
        for move in moves:
            if board.is_capture(move):
                score = self.get_capture_score(board, move)
                capture_moves.append((move, score))
            else:
                other_moves.append(move)
        
        # Sort captures by score (highest first)
        capture_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted captures followed by other moves
        return [move for move, score in capture_moves] + other_moves
    
    def find_hanging_pieces(self, board, color):
        """Find undefended pieces that can be captured for free
        Uses the standardized utility function from v7p3r_utils"""
        from v7p3r_utils import find_hanging_pieces
        return find_hanging_pieces(board, color)
    
    def is_free_capture(self, board, move):
        """Check if a capture move wins free material"""
        if not board.is_capture(move):
            return False, 0
        
        material_gain = evaluate_exchange(board, move)
        return material_gain >= 100, material_gain
