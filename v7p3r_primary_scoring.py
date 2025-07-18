# v7p3r_primary_scoring.py

"""Primary Scoring for V7P3R Chess Engine
Handles material count, material score, and piece square table evaluation.
"""

import chess
from v7p3r_pst import PieceSquareTables
from v7p3r_mvv_lva import MVVLVA
from v7p3r_utils import (
    get_material_balance, 
    evaluate_exchange,
    find_hanging_pieces,
    PIECE_VALUES
)

class PrimaryScoring:
    def __init__(self):
        self.pst = PieceSquareTables()
        self.mvv_lva = MVVLVA()
    
    def evaluate_primary_score(self, board, our_color):
        """Calculate primary scoring components"""
        material_count = self._get_material_count(board, our_color)
        material_score = self._get_material_score(board, our_color)
        pst_score = self._get_pst_score(board, our_color)
        capture_score = self._get_capture_potential(board, our_color)
        
        return {
            'material_count': material_count,
            'material_score': material_score,
            'pst_score': pst_score,
            'capture_score': capture_score,
            'total': material_score + pst_score + capture_score
        }
    
    def _get_material_count(self, board, our_color):
        """Get raw piece count difference"""
        our_pieces = 0
        their_pieces = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                if piece.color == our_color:
                    our_pieces += 1
                else:
                    their_pieces += 1
        
        return our_pieces - their_pieces
    
    def _get_material_score(self, board, our_color):
        """Get material value difference using standard utility function"""
        return get_material_balance(board, our_color)
    
    def _get_pst_score(self, board, our_color):
        """Get piece square table evaluation"""
        our_pst = 0
        their_pst = 0
        is_endgame = self.pst.is_endgame(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Mirror squares for Black pieces so they're evaluated from their perspective
                if piece.color == chess.WHITE:
                    pst_square = square
                else:
                    pst_square = chess.square_mirror(square)
                    
                pst_value = self.pst.get_pst_value(piece.piece_type, pst_square, is_endgame)
                
                if piece.color == our_color:
                    our_pst += pst_value
                else:
                    their_pst += pst_value
        
        return our_pst - their_pst
    
    def _get_capture_potential(self, board, our_color):
        """Evaluate immediate capture opportunities using enhanced logic"""
        # Look for immediate favorable exchanges
        exchange_score = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                # Only consider captures by our pieces
                moving_piece = board.piece_at(move.from_square)
                if moving_piece and moving_piece.color == our_color:
                    # Calculate exchange value
                    exchange_value = evaluate_exchange(board, move)
                    if exchange_value > 0:
                        exchange_score += exchange_value
        
        # Find hanging pieces (undefended or underdefended pieces)
        hanging_pieces = find_hanging_pieces(board, our_color)
        hanging_score = sum(value for _, _, value in hanging_pieces)
        
        # MVV-LVA score for additional insight
        mvv_lva_score = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                moving_piece = board.piece_at(move.from_square)
                if moving_piece and moving_piece.color == our_color:
                    mvv_lva_score += self.mvv_lva.get_capture_score(board, move) // 100  # Scale down
        
        # Total capture potential
        return exchange_score + hanging_score + mvv_lva_score
    
    def get_material_balance(self, board, our_color):
        """Get current material balance for external use"""
        return self._get_material_score(board, our_color)
