# v7p3r_mvv_lva.py

"""V7P3R MVV-LVA (Most Valuable Victim - Least Valuable Attacker) Module.
This module is the single source of truth for piece values and capture evaluation
in the V7P3R chess engine.
"""

import chess

class v7p3rMVVLVA:
    """Class for MVV-LVA calculations in the V7P3R chess engine."""
    
    # Centralized piece values used throughout the engine
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    def __init__(self):
        """Initialize MVV-LVA calculator with pre-computed score matrix."""
        # Initialize MVV-LVA table (6x6 for piece types 1-6)
        self.mvv_lva_table = [[0] * 7 for _ in range(7)]
        
        # Pre-compute MVV-LVA scores
        for victim in range(1, 7):  # 1=PAWN to 6=KING
            for attacker in range(1, 7):
                # Basic MVV-LVA score: victim value - attacker value
                victim_value = self.PIECE_VALUES[victim]
                attacker_value = self.PIECE_VALUES[attacker]
                self.mvv_lva_table[victim][attacker] = victim_value - attacker_value

    def get_piece_value(self, piece_type: int) -> int:
        """Get the value of a piece type. Used by scoring module."""
        return self.PIECE_VALUES.get(piece_type, 0)

    def score_capture(self, move: chess.Move, board: chess.Board) -> int:
        """Score a capturing move using MVV-LVA.
        
        Args:
            move: The move to score
            board: Current board position
            
        Returns:
            int: MVV-LVA score for the capture, 0 for non-captures
        """
        if not board.is_capture(move):
            return 0
            
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if not victim or not attacker:
            return 0
            
        return self.mvv_lva_table[victim.piece_type][attacker.piece_type]

    def sort_captures(self, moves: list[chess.Move], board: chess.Board) -> list[chess.Move]:
        """Sort capture moves by MVV-LVA score.
        
        Args:
            moves: List of moves to sort
            board: Current board position
            
        Returns:
            list[chess.Move]: Sorted list of capture moves
        """
        # Only score and sort actual captures
        capture_scores = [
            (move, self.score_capture(move, board))
            for move in moves
            if board.is_capture(move)
        ]
        
        # Sort by score, highest first
        capture_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in capture_scores]


