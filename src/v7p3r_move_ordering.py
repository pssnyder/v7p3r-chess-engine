# v7p3r_move_ordering.py

"""Move Ordering for V7P3R Chess Engine - Simplified Version
Prioritizes moves for better alpha-beta pruning and search efficiency.
Removes redundant hanging piece detection - let quiescence handle material evaluation.
"""

import chess
from v7p3r_mvv_lva import MVVLVA
from v7p3r_utils import evaluate_exchange

class MoveOrdering:
    def __init__(self):
        self.mvv_lva = MVVLVA()
    
    def order_moves(self, board, moves, max_moves=None, beta=None):
        """Order moves for optimal search (best moves first)"""
        if not moves:
            return []
        
        scored_moves = []
        
        for move in moves:
            score = self._score_move(board, move)
            scored_moves.append((move, score))
            
            # Early termination if we found a really good move and have beta cutoff info
            if beta is not None and score >= 900000:  # Checkmate or extremely good move
                scored_moves.sort(key=lambda x: x[1], reverse=True)
                return [move for move, score in scored_moves]
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Limit number of moves if specified
        if max_moves:
            scored_moves = scored_moves[:max_moves]
        
        return [move for move, score in scored_moves]
    
    def _score_move(self, board, move):
        """Simplified move scoring - focus on essential heuristics only"""
        score = 0
        
        # 1. Checkmate gets highest priority
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_checkmate():
            return 1000000
        
        # 2. Captures - use MVV-LVA for basic scoring
        if board.is_capture(move):
            mvv_lva_score = self.mvv_lva.get_capture_score(board, move)
            # Simple exchange evaluation to avoid bad captures
            exchange_value = evaluate_exchange(board, move)
            if exchange_value >= 0:  # Good or neutral capture
                score += 50000 + mvv_lva_score + exchange_value
            else:  # Bad capture, but still might be worth considering
                score += 10000 + mvv_lva_score + exchange_value
        
        # 3. Checks get medium priority
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
            score += 20000 + promotion_values.get(move.promotion, 0)
        
        # 5. Simple positional bonus for center control
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        
        # Bonus for center squares
        if 3 <= to_file <= 4 and 3 <= to_rank <= 4:
            score += 100
        elif 2 <= to_file <= 5 and 2 <= to_rank <= 5:
            score += 50
        
        return score
    def get_good_captures(self, board):
        """Get captures that are likely to be good (simple exchange evaluation)"""
        good_captures = []
        legal_moves = list(board.legal_moves)
        
        for move in legal_moves:
            if board.is_capture(move):
                exchange_value = evaluate_exchange(board, move)
                if exchange_value >= 0:  # At least equal or better trades
                    good_captures.append((move, exchange_value))
        
        # Sort by exchange value (highest first)
        good_captures.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in good_captures]
