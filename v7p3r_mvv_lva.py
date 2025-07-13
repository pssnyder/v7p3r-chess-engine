# v7p3r_mvv_lva.py

"""V7P3R MVV-LVA (Most Valuable Victim - Least Valuable Attacker) Module.
This module provides centralized MVV-LVA calculations and move evaluation for the V7P3R chess engine.
"""

import os
import sys
import chess
from v7p3r_config import v7p3rConfig
from v7p3r_utilities import get_timestamp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rMVVLVA:
    """Class for MVV-LVA calculations in the V7P3R chess engine."""
    
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    def __init__(self, rules=None):
        """Initialize MVV-LVA calculator.
        
        Args:
            rules: Optional v7p3rRules instance for additional context
        """
        self.rules = rules
        # Initialize MVV-LVA table
        self.mvv_lva_table = [[0] * 7 for _ in range(7)]  # 7x7 table (including 'none' piece type)
        
        # Fill MVV-LVA table
        for victim in range(1, 7):  # 1=PAWN to 6=KING
            for attacker in range(1, 7):
                self.mvv_lva_table[victim][attacker] = self._mvv_lva_score(victim, attacker)
    
    def _mvv_lva_score(self, victim: int, attacker: int) -> int:
        """Calculate MVV-LVA score for a victim-attacker pair"""
        if victim == 0 or attacker == 0:
            return 0
        victim_value = self.PIECE_VALUES[victim]
        attacker_value = self.PIECE_VALUES[attacker]
        return victim_value * 10 - attacker_value
    
    def get_capture_score(self, move: chess.Move, board: chess.Board) -> int:
        """Get the MVV-LVA score for a capturing move"""
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if not victim or not attacker:
            return 0
            
        return self.mvv_lva_table[victim.piece_type][attacker.piece_type]
    
    def score_move(self, move: chess.Move, board: chess.Board) -> int:
        """Score a move using MVV-LVA if it's a capture"""
        if board.is_capture(move):
            return self.get_capture_score(move, board)
        return 0
    
    def calculate_mvv_lva_score(self, move: chess.Move, board: chess.Board) -> float:
        """Enhanced MVV-LVA with safety evaluation and position context"""
        if not board.is_capture(move):
            return 0.0
            
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if victim is None or attacker is None:
            return 0.0
        
        # Base MVV-LVA score
        score = self.mvv_lva_table[victim.piece_type][attacker.piece_type]
        
        # Position context
        center_squares = {chess.D4, chess.E4, chess.D5, chess.E5}
        if move.to_square in center_squares:
            score += 10
        
        # Safety evaluation - check if the attacker is protected
        if len(list(board.attackers(attacker.color, move.from_square))) > 0:
            score += 20  # Bonus for protected attacker
            
        # Evaluate exchange - what happens after the capture
        board.push(move)
        if len(list(board.attackers(not attacker.color, move.to_square))) == 0:
            score += 30  # Safe capture
        board.pop()
        
        return float(score)
    
    def evaluate_tactical_pattern(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate tactical patterns like forks, pins, discovered attacks"""
        score = 0.0
        
        # Make the move to analyze resulting position
        board.push(move)
        
        # Check for discovered attacks
        from_square = move.from_square
        attacker_color = board.piece_at(move.to_square).color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == attacker_color:
                # Count number of pieces attacked after the move
                attacked = len(list(board.attackers(attacker_color, square)))
                score += attacked * 5
                
        # Check for forks (attacking multiple pieces)
        fork_targets = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != attacker_color:
                if board.is_attacked_by(attacker_color, square):
                    fork_targets.append((square, piece))
        if len(fork_targets) >= 2:
            # Calculate fork value based on least valuable pieces
            fork_values = sorted([self.PIECE_VALUES[p[1].piece_type] for p in fork_targets])
            score += (fork_values[0] + fork_values[1]) * 0.1
            
        # Check for pins against king
        king_square = board.king(not attacker_color)
        if king_square:
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != attacker_color and board.is_pinned(not attacker_color, square):
                    score += self.PIECE_VALUES[piece.piece_type] * 0.2
                    
        board.pop()
        return score
    
    def sort_captures(self, moves: list[chess.Move], board: chess.Board) -> list[chess.Move]:
        """Sort capture moves by MVV-LVA score"""
        capture_scores = []
        for move in moves:
            if board.is_capture(move):
                score = self.get_capture_score(move, board)
                capture_scores.append((move, score))
        
        capture_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in capture_scores]


