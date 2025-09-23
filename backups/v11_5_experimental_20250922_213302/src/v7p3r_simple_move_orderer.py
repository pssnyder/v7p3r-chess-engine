#!/usr/bin/env python3
"""
V7P3R v11.1 Simplified Move Ordering
Basic, reliable move ordering focused on tactical priorities
Author: Pat Snyder
"""

import chess
from typing import List, Tuple, Optional


class V7P3RSimpleMoveOrderer:
    """Simplified move ordering for reliable tactical performance"""
    
    def __init__(self):
        # Basic piece values for MVV-LVA
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        # Simple killer moves (2 per depth)
        self.killer_moves = {}
        
    def order_moves(self, board: chess.Board, moves: List[chess.Move], depth: int = 0, 
                   tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
        """
        Order moves using simple, reliable tactical priorities:
        1. TT move (if provided)
        2. Winning captures (MVV-LVA)
        3. Checks
        4. Equal captures
        5. Killer moves
        6. Other moves
        """
        if not moves:
            return moves
            
        scored_moves = []
        
        for move in moves:
            score = self._score_move(board, move, depth, tt_move)
            scored_moves.append((move, score))
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        return [move for move, score in scored_moves]
    
    def _score_move(self, board: chess.Board, move: chess.Move, depth: int, tt_move: Optional[chess.Move]) -> int:
        """Score a move for ordering purposes"""
        score = 0
        
        # 1. Transposition table move gets highest priority
        if tt_move and move == tt_move:
            return 10000
        
        # 2. Captures (MVV-LVA)
        if board.is_capture(move):
            victim_value = 0
            attacker_value = 0
            
            captured_piece = board.piece_at(move.to_square)
            moving_piece = board.piece_at(move.from_square)
            
            if captured_piece:
                victim_value = self.piece_values[captured_piece.piece_type]
            if moving_piece:
                attacker_value = self.piece_values[moving_piece.piece_type]
            
            # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
            mvv_lva_score = victim_value - (attacker_value // 10)
            
            if mvv_lva_score > 0:
                score = 8000 + mvv_lva_score  # Winning captures
            else:
                score = 6000 + mvv_lva_score  # Equal/losing captures
        
        # 3. Checks
        elif board.gives_check(move):
            score = 7000
        
        # 4. Killer moves
        elif self._is_killer_move(move, depth):
            score = 5000
        
        # 5. Promotions
        elif move.promotion:
            if move.promotion == chess.QUEEN:
                score = 4500
            else:
                score = 4000
        
        # 6. Castling
        elif board.is_castling(move):
            score = 3500
        
        # 7. Basic piece development/centralization bonus
        else:
            score = self._get_positional_bonus(board, move)
        
        return score
    
    def _is_killer_move(self, move: chess.Move, depth: int) -> bool:
        """Check if move is a killer move at this depth"""
        if depth not in self.killer_moves:
            return False
        return move in self.killer_moves[depth]
    
    def store_killer_move(self, move: chess.Move, depth: int):
        """Store a killer move"""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        
        # Keep only 2 killer moves per depth
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > 2:
                self.killer_moves[depth].pop()
    
    def _get_positional_bonus(self, board: chess.Board, move: chess.Move) -> int:
        """Simple positional bonus for move ordering"""
        bonus = 100  # Base score for quiet moves
        
        piece = board.piece_at(move.from_square)
        if not piece:
            return bonus
        
        # Simple center control bonus
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        if move.to_square in center_squares:
            bonus += 50
        
        # Knight development bonus
        if piece.piece_type == chess.KNIGHT:
            if move.to_square in {chess.C3, chess.F3, chess.C6, chess.F6}:
                bonus += 30
        
        # Bishop development bonus  
        elif piece.piece_type == chess.BISHOP:
            if move.to_square in {chess.C4, chess.F4, chess.C5, chess.F5}:
                bonus += 25
        
        return bonus
    
    def clear_killers(self):
        """Clear all killer moves (for new game)"""
        self.killer_moves.clear()