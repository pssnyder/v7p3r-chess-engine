# v7p3r_ordering.py
""" Move ordering for the V7P3R chess engine.
This module provides functionality to order moves based on their potential effectiveness"""
import os
import sys
import chess
from typing import Optional, List, Tuple, Dict
from v7p3r_config import v7p3rConfig
from v7p3r_mvv_lva import v7p3rMVVLVA

class v7p3rOrdering:
    def __init__(self, mvv_lva: v7p3rMVVLVA | None = None):
        self.mvv_lva = mvv_lva or v7p3rMVVLVA()
        self.history_moves = {}
        self.killer_moves = [[] for _ in range(100)]  # Dynamic lists for killer moves
        self._counter_moves = {}  # Private counter moves dictionary
        
    def sort_moves(self, moves: list[chess.Move], board: chess.Board) -> list[chess.Move]:
        """Sort moves based on likelihood of being best"""
        if not moves:
            return []
            
        move_scores = [(move, self._score_move(move, board)) for move in moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]
        
    def _score_move(self, move: chess.Move, board: chess.Board) -> int:
        """Score a move based on various heuristics"""
        score = 0
        
        # MVV-LVA scoring for captures
        if board.is_capture(move):
            score += self.mvv_lva.score_move(move, board) * 10000
            
        # History heuristic
        history_score = self.get_history_score(move)
        score += history_score * 100
            
        # Killer moves bonus
        if self.is_killer_move(move, board.ply()):
            score += 9000
            
        # Counter moves bonus
        if self.is_counter_move(move, board):
            score += 8000
            
        # Promotions
        if move.promotion:
            score += 15000 + (move.promotion - chess.KNIGHT) * 1000
            
        # Mobility evaluation
        score += self._evaluate_mobility(move, board)
        
        # Center control
        score += self._evaluate_center_control(move)
        
        # Development in opening
        if self._improves_development(move, board):
            score += 500
            
        return score
        
    def get_history_score(self, move: chess.Move) -> int:
        """Get the history heuristic score for a move"""
        move_key = (move.from_square, move.to_square)
        return self.history_moves.get(move_key, 0)
        
    def is_killer_move(self, move: chess.Move, ply: int) -> bool:
        """Check if move is a killer move at given ply"""
        if ply >= len(self.killer_moves):
            return False
        return move in self.killer_moves[ply]
        
    def is_counter_move(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move is a counter move to opponent's last move"""
        if not board.move_stack:
            return False
        last_move = board.peek()
        last_piece = board.piece_at(last_move.to_square)
        if not last_piece:
            return False
        key = (last_move.to_square, last_piece.piece_type)
        return self._counter_moves.get(key) == move
        
    def _evaluate_mobility(self, move: chess.Move, board: chess.Board) -> int:
        """Evaluate how much the move improves piece mobility"""
        score = 0
        piece = board.piece_at(move.from_square)
        
        if not piece:
            return 0
            
        # Make move
        board.push(move)
        
        # Count attacked squares
        attacked = len(list(board.attacks(move.to_square)))
        score = attacked * 10
        
        # Unmake move
        board.pop()
        
        return score
        
    def _evaluate_center_control(self, move: chess.Move) -> int:
        """Evaluate how well the move controls the center"""
        center_squares = {27, 28, 35, 36}  # e4, d4, e5, d5
        extended_center = {18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45}
        
        score = 0
        if move.to_square in center_squares:
            score += 30
        elif move.to_square in extended_center:
            score += 15
            
        return score
        
    def _improves_development(self, move: chess.Move, board: chess.Board) -> bool:
        """Check if move improves piece development"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
            
        # Development is only relevant in the opening
        if board.fullmove_number > 10:
            return False
            
        # Moving the same piece twice in the opening is generally bad
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if move.from_square in self._get_starting_squares(piece.color):
                return True
                
        return False
        
    def _get_starting_squares(self, color: chess.Color) -> set:
        """Get starting squares for pieces of a given color"""
        if color == chess.WHITE:
            return {chess.B1, chess.G1, chess.C1, chess.F1}
        return {chess.B8, chess.G8, chess.C8, chess.F8}
        
    def update_history(self, move: chess.Move, depth: int) -> None:
        """Update history score for a move"""
        move_key = (move.from_square, move.to_square)
        self.history_moves[move_key] = self.history_moves.get(move_key, 0) + depth * depth
        
    def add_killer_move(self, move: chess.Move, ply: int) -> None:
        """Add a killer move at the given ply"""
        if ply >= len(self.killer_moves):
            self.killer_moves.extend([[] for _ in range(ply - len(self.killer_moves) + 1)])
        
        if move not in self.killer_moves[ply]:
            self.killer_moves[ply].insert(0, move)
            if len(self.killer_moves[ply]) > 2:
                self.killer_moves[ply].pop()
                
    def add_counter_move(self, prev_move: chess.Move, counter_move: chess.Move, board: chess.Board) -> None:
        """Add a counter move for the given previous move"""
        if not prev_move:
            return
            
        piece = board.piece_at(prev_move.to_square)
        if not piece:
            return
            
        key = (prev_move.to_square, piece.piece_type)
        self._counter_moves[key] = counter_move
            
    def clear_history(self) -> None:
        """Clear history tables"""
        self.history_moves.clear()
        self.killer_moves = [[] for _ in range(100)]
        self._counter_moves.clear()
        
    def order_moves(self, board: chess.Board, max_moves: Optional[int] = None, tempo_bonus: float = 0.0) -> list[chess.Move]:
        """
        Order legal moves for a given board position.
        
        Args:
            board: The chess board position
            max_moves: Optional maximum number of moves to return (None for all)
            tempo_bonus: Optional tempo bonus to influence move ordering
            
        Returns:
            Ordered list of chess moves
        """
        moves = list(board.legal_moves)
        ordered_moves = self.sort_moves(moves, board)
        
        # Apply tempo bonus if specified
        if tempo_bonus > 0:
            # Re-score moves with tempo consideration
            move_scores = []
            for move in ordered_moves:
                score = self._score_move(move, board)
                # Add tempo bonus for central moves and developing pieces
                if self._is_central_move(move) or self._improves_development(move, board):
                    score += int(tempo_bonus * 500)  # Scale tempo bonus
                move_scores.append((move, score))
                
            # Re-sort with tempo consideration
            move_scores.sort(key=lambda x: x[1], reverse=True)
            ordered_moves = [move for move, _ in move_scores]
        
        # Limit number of moves if specified
        if max_moves and max_moves < len(ordered_moves):
            return ordered_moves[:max_moves]
            
        return ordered_moves
        
    def _is_central_move(self, move: chess.Move) -> bool:
        """Check if a move targets the center of the board."""
        central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        return move.to_square in central_squares
        
    # Constants for move scoring
    QUEEN_PROMOTION_SCORE = 15000
    OTHER_PROMOTION_SCORE = 14000
    WINNING_CAPTURE_SCORE = 10000
    EQUAL_CAPTURE_SCORE = 9000
    LOSING_CAPTURE_SCORE = 8000
