# v7p3r_quiescence.py

"""Quiescence Search for V7P3R Chess Engine
Searches capture sequences to quiet positions for better evaluation.
"""

import chess
from v7p3r_mvv_lva import MVVLVA

class QuiescenceSearch:
    def __init__(self):
        self.mvv_lva = MVVLVA()
        self.max_quiescence_depth = 5
    
    def quiescence_search(self, board, alpha, beta, our_color, primary_scorer, depth=0):
        """Search captures and checks to reach a quiet position"""
        if depth >= self.max_quiescence_depth:
            return self._evaluate_quiet_position(board, our_color, primary_scorer)
        
        # Stand-pat evaluation
        stand_pat = self._evaluate_quiet_position(board, our_color, primary_scorer)
        
        # Beta cutoff
        if stand_pat >= beta:
            return beta
        
        # Alpha update
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Generate and order capture moves
        captures = self._get_capture_moves(board)
        if not captures:
            return stand_pat
        
        # Order captures by MVV-LVA
        captures = self.mvv_lva.sort_captures(board, captures)
        
        # Search captures
        for move in captures:
            # Skip obviously bad captures (SEE - Static Exchange Evaluation would go here)
            if self._is_bad_capture(board, move):
                continue
            
            board.push(move)
            
            # Recursively search this capture
            score = -self.quiescence_search(board, -beta, -alpha, not our_color, primary_scorer, depth + 1)
            
            board.pop()
            
            if score >= beta:
                return beta  # Beta cutoff
            
            if score > alpha:
                alpha = score
        
        return alpha
    
    def _get_capture_moves(self, board):
        """Get all capture moves from current position"""
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(move)
        return captures
    
    def _is_bad_capture(self, board, move):
        """Simple check to avoid obviously bad captures"""
        # This is a simplified version - full SEE would be better
        if not board.is_capture(move):
            return False
        
        # Get pieces involved
        capturing_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)
        
        if not capturing_piece:
            return True
        
        # If no piece captured, check for en passant
        if not captured_piece and board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not capturing_piece.color)
        
        if not captured_piece:
            return True
        
        # Simple material exchange check
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 325,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        capturing_value = piece_values[capturing_piece.piece_type]
        captured_value = piece_values[captured_piece.piece_type]
        
        # Don't capture less valuable pieces with more valuable pieces
        # unless the captured piece is undefended
        if capturing_value > captured_value:
            # Check if the target square is defended
            board.push(move)
            is_defended = board.is_attacked_by(not capturing_piece.color, move.to_square)
            board.pop()
            
            if is_defended:
                return True  # Bad capture - we'll likely lose material
        
        return False
    
    def _evaluate_quiet_position(self, board, our_color, primary_scorer):
        """Evaluate position when it's quiet (no more tactical sequences)"""
        if board.is_checkmate():
            return -999999 if board.turn == our_color else 999999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Use primary scoring for quiet evaluation
        primary_eval = primary_scorer.evaluate_primary_score(board, our_color)
        return primary_eval['total']
    
    def is_quiet_position(self, board):
        """Check if position is quiet (no immediate tactical threats)"""
        # Position is not quiet if:
        # 1. In check
        if board.is_check():
            return False
        
        # 2. Has captures available
        for move in board.legal_moves:
            if board.is_capture(move):
                return False
        
        # 3. Has promotion moves available
        for move in board.legal_moves:
            if move.promotion:
                return False
        
        return True
