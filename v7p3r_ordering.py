# v7p3r_ordering.py
""" Move ordering for the V7P3R chess engine.
This module provides functionality to order moves based on their potential effectiveness"""
import os
import sys
import chess
from v7p3r_config import v7p3rConfig
from v7p3r_mvv_lva import v7p3rMVVLVA

# Ensure the parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rOrdering:
    """Class for move ordering in the V7P3R chess engine."""
    def __init__(self, scoring_calculator):
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()  # Ensure it's always a dictionary
        
        # Required Engine Modules
        self.scoring_calculator = scoring_calculator
        self.mvv_lva = v7p3rMVVLVA(scoring_calculator.rules_manager)
        
        # Move Ordering Settings
        self.move_ordering_enabled = self.engine_config.get('move_ordering_enabled', True)
        self.max_ordered_moves = self.engine_config.get('max_ordered_moves', 10)  # Default to 10 moves if not set
        
        # Enhanced move ordering with history and killer moves
        self.history_table = {}  # History heuristic table
        self.killer_moves = [[] for _ in range(10)]  # Killer moves for different depths (max depth 10)

    def order_moves(self, board: chess.Board, moves, depth: int = 0, cutoff: int = 0) -> list:
        """Order moves for better alpha-beta pruning efficiency
        Following the enhanced hierarchy: Hash move -> Winning captures -> Killer moves -> History moves -> Quiet moves"""
        # Safety check to prevent empty move list issues
        if not moves:
            return []
            
        ordered_moves = []
        capture_moves = []
        killer_moves = []
        history_moves = []
        quiet_moves = []
        
        # Categorize moves
        for move in moves:
            if not board.is_legal(move):
                continue
                
            # Check for immediate checkmate (hash move equivalent)
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_checkmate():
                temp_board.pop()
                return [move]  # Return immediately for checkmate
            temp_board.pop()
            
            # Categorize based on move type
            if board.is_capture(move):
                mvv_lva_score = self.scoring_calculator._calculate_mvv_lva_score(move, board)
                capture_moves.append((move, 900.0 + mvv_lva_score))
            elif self._is_killer_move(move, depth):
                killer_moves.append((move, 100.0))
            elif self._get_history_score(move) > 0:
                history_score = self._get_history_score(move)
                history_moves.append((move, 50.0 + history_score))
            else:
                quiet_score = self._order_move_score(board, move)
                quiet_moves.append((move, quiet_score))
        
        # Sort each category
        capture_moves.sort(key=lambda x: x[1], reverse=True)
        killer_moves.sort(key=lambda x: x[1], reverse=True)
        history_moves.sort(key=lambda x: x[1], reverse=True)
        quiet_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Combine in order: Captures -> Killers -> History -> Quiet
        ordered_moves = [move for move, _ in capture_moves + killer_moves + history_moves + quiet_moves]
        
        # Apply cutoff if specified
        max_ordered_moves = cutoff if cutoff > 0 else self.max_ordered_moves
        if max_ordered_moves > 0 and len(ordered_moves) > max_ordered_moves:
            ordered_moves = ordered_moves[:max_ordered_moves]
        
        return ordered_moves

    def _order_move_score(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate a score for a move for ordering purposes.
        Enhanced move ordering with proper MVV-LVA integration."""
        score = 0.0

        # 1. Checkmate moves get highest priority
        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            temp_board.pop()
            return 99999.0  # Score over 10K for checkmate moves
        
        # 2. Check moves get very high priority
        if temp_board.is_check():
            score += 9999.0

        temp_board.pop()
        
        # 3. Winning captures (by MVV-LVA from scoring calculator)
        if board.is_capture(move):
            # Use scoring calculator's enhanced MVV-LVA
            mvv_lva_score = self.scoring_calculator._calculate_mvv_lva_score(move, board)
            score += 900.0 + mvv_lva_score
            
        # 4. Promotions 
        if move.promotion:
            score += 90.0
            if move.promotion == chess.QUEEN:
                score += 9.0
                
        # 5. Center control moves
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        if move.to_square in center_squares:
            score += 5.0
            
        # 6. Piece development (knights and bishops)
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # Moving from back rank is good development
            if piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0:
                score += 3.0
            elif piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7:
                score += 3.0
        
        return score
    
    def _is_killer_move(self, move: chess.Move, depth: int) -> bool:
        """Check if a move is a killer move at the given depth"""
        if depth < len(self.killer_moves):
            return move in self.killer_moves[depth]
        return False
    
    def _get_history_score(self, move: chess.Move) -> float:
        """Get the history heuristic score for a move"""
        move_key = f"{move.from_square}-{move.to_square}"
        return self.history_table.get(move_key, 0.0)
    
    def add_killer_move(self, move: chess.Move, depth: int):
        """Add a move to the killer moves table"""
        if depth < len(self.killer_moves):
            if move not in self.killer_moves[depth]:
                self.killer_moves[depth].append(move)
                # Keep only the 2 best killer moves per depth
                if len(self.killer_moves[depth]) > 2:
                    self.killer_moves[depth].pop(0)
    
    def update_history(self, move: chess.Move, depth: int):
        """Update the history heuristic table"""
        move_key = f"{move.from_square}-{move.to_square}"
        if move_key not in self.history_table:
            self.history_table[move_key] = 0.0
        self.history_table[move_key] += depth * depth  # Bonus increases with depth
