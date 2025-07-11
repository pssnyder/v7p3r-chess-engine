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
        # Convert moves to list if it's not already and filter invalid moves
        try:
            moves = list(moves) if not isinstance(moves, list) else moves
            moves = [move for move in moves if board.is_legal(move)]
        except:
            return list(board.legal_moves)  # Fallback to unordered legal moves
        
        # Safety check to prevent empty move list issues
        if not moves:
            return []

        try:
            # Initialize move categories with scores as floats
            capture_moves = []
            killer_moves = []
            history_moves = []
            quiet_moves = []
            
            # Categorize moves
            for move in moves:
                # Check for immediate checkmate
                temp_board = board.copy()
                temp_board.push(move)
                if temp_board.is_checkmate():
                    temp_board.pop()
                    return [move]  # Return immediately for checkmate
                temp_board.pop()
                
                try:
                    # Evaluate and categorize each move
                    if board.is_capture(move):
                        score = float(900.0)  # Base capture score
                        mvv_lva_score = 0.0
                        try:
                            mvv_lva_score = float(self.scoring_calculator._calculate_mvv_lva_score(move, board))
                        except:
                            pass  # Ignore MVV-LVA if it fails
                        capture_moves.append((move, score + mvv_lva_score))
                    elif self._is_killer_move(move, depth):
                        killer_moves.append((move, float(100.0)))
                    elif self._get_history_score(move) > 0:
                        hist_score = float(self._get_history_score(move))
                        history_moves.append((move, float(50.0) + hist_score))
                    else:
                        quiet_score = float(self._order_move_score(board, move))
                        quiet_moves.append((move, quiet_score))
                except Exception as e:
                    # If scoring fails, add to quiet moves with neutral score
                    quiet_moves.append((move, float(0.0)))

            # Sort each category
            try:
                capture_moves.sort(key=lambda x: float(x[1]), reverse=True)
                killer_moves.sort(key=lambda x: float(x[1]), reverse=True)
                history_moves.sort(key=lambda x: float(x[1]), reverse=True)
                quiet_moves.sort(key=lambda x: float(x[1]), reverse=True)
            except:
                # If sorting fails, just keep original order
                pass

            # Combine moves in priority order
            ordered_moves = []
            for move_list in [capture_moves, killer_moves, history_moves, quiet_moves]:
                ordered_moves.extend([move for move, _ in move_list])

            # Apply cutoff if specified and valid
            max_ordered_moves = cutoff if cutoff > 0 else self.max_ordered_moves
            if max_ordered_moves > 0 and len(ordered_moves) > max_ordered_moves:
                ordered_moves = ordered_moves[:max_ordered_moves]

            return ordered_moves

        except Exception as e:
            # If anything fails, return unordered legal moves
            return list(board.legal_moves)

    def _order_move_score(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate a score for a move for ordering purposes.
        Enhanced move ordering with proper MVV-LVA integration."""
        try:
            score = 0.0

            # 1. Checkmate moves get highest priority
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_checkmate():
                temp_board.pop()
                return float(99999.0)  # Score over 10K for checkmate moves
            
            # 2. Check moves get very high priority
            if temp_board.is_check():
                score += float(9999.0)

            temp_board.pop()
            
            # 3. Winning captures (by MVV-LVA from scoring calculator)
            try:
                if board.is_capture(move):
                    mvv_lva_score = float(self.scoring_calculator._calculate_mvv_lva_score(move, board))
                    score += float(900.0 + mvv_lva_score)
            except:
                # If MVV-LVA fails, use a basic capture score
                if board.is_capture(move):
                    score += float(900.0)
                
            # 4. Promotions 
            if move.promotion:
                score += float(90.0)
                if move.promotion == chess.QUEEN:
                    score += float(9.0)
                    
            # 5. Center control moves
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            if move.to_square in center_squares:
                score += float(5.0)
                
            # 6. Piece development (knights and bishops)
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Moving from back rank is good development
                if piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0:
                    score += float(3.0)
                elif piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7:
                    score += float(3.0)
            
            return float(score)
        except Exception as e:
            return float(0.0)  # Return neutral score if anything fails
    
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
