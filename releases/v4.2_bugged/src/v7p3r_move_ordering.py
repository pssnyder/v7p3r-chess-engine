# v7p3r_move_ordering.py

"""Move Ordering for V7P3R Chess Engine
Prioritizes moves for better alpha-beta pruning and search efficiency.
"""

import chess
from v7p3r_mvv_lva import MVVLVA
from v7p3r_utils import is_capture_that_escapes_check, evaluate_exchange

class MoveOrdering:
    def __init__(self):
        self.mvv_lva = MVVLVA()
    
    def order_moves(self, board, moves, max_moves=None):
        """Order moves for optimal search (best moves first)"""
        if not moves:
            return []
        
        scored_moves = []
        
        for move in moves:
            score = self._score_move(board, move)
            scored_moves.append((move, score))
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Limit number of moves if specified
        if max_moves:
            scored_moves = scored_moves[:max_moves]
        
        return [move for move, score in scored_moves]
    
    # Using is_capture_that_escapes_check from v7p3r_utils
    
    def _score_move(self, board, move):
        """Score a move for ordering purposes"""
        score = 0
        
        # 1. Checkmate gets highest priority
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_checkmate():
            return 1000000
        
        # 2. Avoid repetition (heavy penalty)
        if board_copy.is_repetition(2):  # Check for threefold repetition
            return -500000
        
        # 3. FREE MATERIAL CAPTURES - Extremely high priority!
        if board.is_capture(move):
            is_free, material_gain = self.mvv_lva.is_free_capture(board, move)
            if is_free:
                # SHORT CIRCUIT for free material - highest priority after checkmate
                if material_gain >= 500:  # Rook or higher
                    return 900000 + material_gain
                elif material_gain >= 300:  # Knight/Bishop
                    return 800000 + material_gain
                else:  # Pawn or small material
                    return 700000 + material_gain
        
        # 4. If in check, prioritize safe captures of checking piece
        if board.is_check():
            if board.is_capture(move):
                # Check if this capture targets the checking piece
                if is_capture_that_escapes_check(board, move):
                    # Verify it's a safe capture using evaluate_exchange
                    exchange_value = evaluate_exchange(board, move)
                    if exchange_value >= 0:  # Safe or profitable
                        return 100000 + exchange_value  # Very high priority
        
        # 5. All other captures (MVV-LVA enhanced evaluation)
        if board.is_capture(move):
            mvv_lva_score = self.mvv_lva.get_capture_score(board, move)
            score += 10000 + mvv_lva_score
        
        # 6. Checks get medium priority
        if board_copy.is_check():
            score += 5000
        
        # 4. Promotions get high priority
        if move.promotion:
            promotion_values = {
                chess.QUEEN: 900,
                chess.ROOK: 500,
                chess.KNIGHT: 300,
                chess.BISHOP: 325
            }
            score += 8000 + promotion_values.get(move.promotion, 0)
        
        # 5. Castling gets medium priority
        if board.is_castling(move):
            score += 3000
        
        # 6. Center control
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        
        # Bonus for center squares (e4, e5, d4, d5)
        if 3 <= to_file <= 4 and 3 <= to_rank <= 4:
            score += 100
        # Smaller bonus for extended center
        elif 2 <= to_file <= 5 and 2 <= to_rank <= 5:
            score += 50
        
        # 7. Piece development (early game)
        moving_piece = board.piece_at(move.from_square)
        if moving_piece:
            if moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Bonus for developing pieces from back rank
                from_rank = chess.square_rank(move.from_square)
                if (moving_piece.color == chess.WHITE and from_rank == 0) or \
                   (moving_piece.color == chess.BLACK and from_rank == 7):
                    score += 200
        
        # 8. Avoid moving same piece twice in opening
        # This would require move history - simplified version
        
        return score
    

    # Killer moves and history heuristic are not implemented in this version.
    # If needed, implement them in a dedicated module or as part of a future enhancement.
    
    def get_hanging_piece_captures(self, board):
        """Return moves that capture hanging (undefended) pieces, sorted by value descending."""
        hanging_captures = []
        hanging_pieces = self.mvv_lva.find_hanging_pieces(board, board.turn)
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            if board.is_capture(move):
                for hanging_square, _, value in hanging_pieces:
                    if move.to_square == hanging_square:
                        hanging_captures.append((move, value))
                        break
        hanging_captures.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in hanging_captures]
    
    def order_moves_with_material_priority(self, board, moves):
        """Order moves with hanging piece captures first, then by score."""
        if not moves:
            return []
        
        # Get hanging captures efficiently
        hanging_captures = self.get_hanging_piece_captures(board)
        hanging_capture_set = set(hanging_captures)
        
        # Only score non-hanging moves
        other_moves = [move for move in moves if move not in hanging_capture_set]
        scored_moves = [(move, self._score_move(board, move)) for move in other_moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        return hanging_captures + [move for move, _ in scored_moves]
    
    def get_good_captures(self, board):
        """Get captures that are likely to be good (win material or equal trades)"""
        good_captures = []
        legal_moves = list(board.legal_moves)
        
        for move in legal_moves:
            if board.is_capture(move):
                is_free, material_gain = self.mvv_lva.is_free_capture(board, move)
                if is_free or material_gain >= 0:  # Free captures or at least equal trades
                    good_captures.append((move, material_gain))
        
        # Sort by material gain (highest first)
        good_captures.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in good_captures]
