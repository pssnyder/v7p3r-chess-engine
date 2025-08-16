
# v7p3r_quiescence.py
"""
Quiescence Search for V7P3R Chess Engine
Searches capture sequences to quiet positions for better evaluation.
"""

import chess
from v7p3r_mvv_lva import MVVLVA

class QuiescenceSearch:
    def __init__(self):
        self.mvv_lva = MVVLVA()
        self.max_quiescence_depth = 5
    
    def quiescence_search(self, board, alpha, beta, our_color, primary_scorer, depth=0):
        """
        Quiescence search: recursively search capture sequences to reach a quiet position.
        Only considers captures (and optionally checks/promotions in future).
        """
        if depth >= self.max_quiescence_depth:
            return self._evaluate_quiet_position(board, our_color, primary_scorer)

        stand_pat = self._evaluate_quiet_position(board, our_color, primary_scorer)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        captures = self._get_capture_moves(board)
        if not captures:
            return stand_pat

        captures = self.mvv_lva.sort_captures(board, captures)
        for move in captures:
            # Skip obviously bad captures (SEE - Static Exchange Evaluation would go here)
            if self._is_bad_capture(board, move):
                continue
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, not our_color, primary_scorer, depth + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha
    
    def _get_capture_moves(self, board):
        """Return all legal capture moves from the current position."""
        return [move for move in board.legal_moves if board.is_capture(move)]
    
    def _is_bad_capture(self, board, move):
        """Avoid obviously bad captures (very basic SEE). Returns True if capture is likely bad."""
        if not board.is_capture(move):
            return False
        capturing_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)
        if not capturing_piece:
            return True
        if not captured_piece and board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not capturing_piece.color)
        if not captured_piece:
            return True
        piece_values = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 325, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000}
        capturing_value = piece_values[capturing_piece.piece_type]
        captured_value = piece_values[captured_piece.piece_type]
        if capturing_value > captured_value:
            board.push(move)
            is_defended = board.is_attacked_by(not capturing_piece.color, move.to_square)
            board.pop()
            if is_defended:
                return True
        return False
    
    def _evaluate_quiet_position(self, board, our_color, primary_scorer):
        """Evaluate a quiet position using the primary scorer."""
        if board.is_checkmate():
            return -999999 if board.turn == our_color else 999999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        return primary_scorer.evaluate_primary_score(board, our_color)['total']
    
    def is_quiet_position(self, board):
        """Return True if the position is quiet (no checks, captures, or promotions available)."""
        if board.is_check():
            return False
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
                return False
        return True
