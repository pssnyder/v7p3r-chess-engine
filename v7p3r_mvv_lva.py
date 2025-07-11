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
    
    def __init__(self, rules_manager=None):
        """Initialize the MVV-LVA calculator with configuration settings."""
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()
        
        # Required components
        self.rules_manager = rules_manager
        
        # MVV-LVA settings
        self.use_mvv_lva = True
        self.use_safety_checks = True
        self.use_position_context = True
        self.safety_margin = 200
        self.position_bonus = 50

        # Initialize piece values
        self.piece_values = self.engine_config.get('piece_values', {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        })

    def calculate_mvv_lva_score(self, move: chess.Move, board: chess.Board) -> float:
        """Calculate MVV-LVA score for a move."""
        if not self.use_mvv_lva or not board.is_capture(move):
            return 0.0

        # Get victim and attacker pieces
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if not victim or not attacker:
            return 0.0
            
        # Basic MVV-LVA score
        victim_value = self.piece_values.get(victim.piece_type, 100)  # Default to 100 if not found
        attacker_value = self.piece_values.get(attacker.piece_type, 100)  # Default to 100 if not found
        base_score = float(victim_value * 100 - attacker_value)  # Ensure we're working with floats
        
        # Safety evaluation - only if rules_manager is available and properly initialized
        safety_score = 0.0
        if self.use_safety_checks and self.rules_manager is not None:
            try:
                is_protected = self._is_square_protected(board, move.to_square, not attacker.color)
                temp_board = board.copy()
                temp_board.push(move)
                attacker_threatened = self._is_square_attacked(temp_board, move.to_square, attacker.color)
                
                if is_protected and attacker_threatened:
                    safety_score -= self.safety_margin
                elif not is_protected and not attacker_threatened:
                    safety_score += self.safety_margin
            except:
                pass  # Ignore safety calculations if they fail
        
        # Position context bonus - only if rules_manager is available and has the right method
        position_bonus = 0.0
        if self.use_position_context and self.rules_manager is not None:
            try:
                if hasattr(self.rules_manager, 'evaluate_piece_mobility'):
                    mobility_before = self.rules_manager.evaluate_piece_mobility(board, move.from_square)
                    temp_board = board.copy()
                    temp_board.push(move)
                    mobility_after = self.rules_manager.evaluate_piece_mobility(temp_board, move.to_square)
                    position_bonus = float((mobility_after - mobility_before) * self.position_bonus)
            except:
                pass  # Ignore position bonus calculations if they fail
        
        final_score = base_score + safety_score + position_bonus
        return float(final_score)  # Ensure we return a float
    
    def _is_square_protected(self, board: chess.Board, square: chess.Square, color: bool) -> bool:
        """Check if a square is protected by any piece of the given color."""
        attackers = board.attackers(color, square)
        return bool(attackers)
    
    def _is_square_attacked(self, board: chess.Board, square: chess.Square, color: bool) -> bool:
        """Check if a square is attacked by any piece of the given color."""
        return bool(board.attackers(not color, square))

    def sort_moves_by_mvv_lva(self, moves: list, board: chess.Board) -> list:
        """Sort a list of moves by their MVV-LVA scores."""
        scored_moves = [(move, self.calculate_mvv_lva_score(move, board)) for move in moves]
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in scored_moves]


