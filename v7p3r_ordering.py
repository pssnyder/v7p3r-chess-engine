# v7p3r_ordering.py
""" Move ordering for the V7P3R chess engine.
This module provides functionality to order moves based on their potential effectiveness"""
import os
import sys
import chess
from v7p3r_config import v7p3rConfig
from v7p3r_mvv_lva import v7p3rMVVLVA

# Import required types
from typing import List, Optional, Tuple

# Ensure the parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rOrdering:
    """Class for move ordering in the V7P3R chess engine."""
    def __init__(self, scoring_calculator):
        # Engine Configuration
        self.config_manager = v7p3rConfig()
        self.engine_config = self.config_manager.get_engine_config()
        
        # Required Engine Modules
        self.scoring_calculator = scoring_calculator
        self.mvv_lva = v7p3rMVVLVA(scoring_calculator.rules_manager)
        
        # Move Ordering Settings
        self.move_ordering_enabled = self.engine_config.get('move_ordering_enabled', True)
        self.max_ordered_moves = self.engine_config.get('max_ordered_moves', 10)
        
        # History and killer move tracking
        self.history_moves = {}  # {(piece_type, from_square, to_square): count}
        self.killer_moves = [[] for _ in range(10)]  # Killer moves for different depths
        self.max_killer_moves = 2
        self.max_history_score = 1000.0

        # Move score constants
        self.QUEEN_PROMOTION_SCORE = 10000000.0  # Highest priority
        self.OTHER_PROMOTION_SCORE = 5000000.0
        self.WINNING_CAPTURE_SCORE = 1000000.0
        self.EQUAL_CAPTURE_SCORE = 500000.0
        self.LOSING_CAPTURE_SCORE = 100000.0

    def order_moves_with_debug(self, board: chess.Board, max_moves: Optional[int] = None, tempo_bonus: float = 0.0) -> List[Tuple[chess.Move, float]]:
        """Debug version that returns moves with their scores."""
        if not max_moves:
            max_moves = self.max_ordered_moves
            
        # Get all legal moves
        all_moves = list(board.legal_moves)
        if not all_moves:
            return []
            
        # Score all moves
        scored_moves = []
        for move in all_moves:
            score = 0.0
            move_type = "Normal"
            
            # 1. Queen promotions (highest priority)
            if move.promotion == chess.QUEEN:
                score += self.QUEEN_PROMOTION_SCORE
                move_type = "Queen promotion"
                if board.is_capture(move):
                    score += self.WINNING_CAPTURE_SCORE
                    move_type += " with capture"
                    
            # 2. Other promotions
            elif move.promotion:
                score += self.OTHER_PROMOTION_SCORE
                move_type = f"{chess.piece_name(move.promotion).capitalize()} promotion"
                if board.is_capture(move):
                    score += self.EQUAL_CAPTURE_SCORE
                    move_type += " with capture"
                    
            # 3. Captures with MVV-LVA scoring
            elif board.is_capture(move):
                captured = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if captured and attacker:
                    cap_value = self.scoring_calculator.pst.piece_values.get(captured.piece_type, 0)
                    att_value = self.scoring_calculator.pst.piece_values.get(attacker.piece_type, 0)
                    mvv_lva_score = self.mvv_lva.calculate_mvv_lva_score(move, board)
                    
                    if cap_value > att_value:
                        score += self.WINNING_CAPTURE_SCORE + mvv_lva_score * 1000.0
                        move_type = f"Winning capture ({chess.piece_name(attacker.piece_type)} takes {chess.piece_name(captured.piece_type)})"
                    elif cap_value == att_value:
                        score += self.EQUAL_CAPTURE_SCORE + mvv_lva_score * 500.0
                        move_type = f"Equal capture ({chess.piece_name(attacker.piece_type)} takes {chess.piece_name(captured.piece_type)})"
                    else:
                        score += self.LOSING_CAPTURE_SCORE + mvv_lva_score * 100.0
                        move_type = f"Losing capture ({chess.piece_name(attacker.piece_type)} takes {chess.piece_name(captured.piece_type)})"
                    
                    if tempo_bonus > 0:
                        score *= (1.0 + tempo_bonus)
                        
            # 4. Additional scoring
            score += self._calculate_move_score(board, move, tempo_bonus)
            scored_moves.append((move, score))
            
            # Debug output
            if hasattr(self, 'debug_output'):
                print(f"{move_type} {move.uci()}: {score}")
            
        # Sort by score descending
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Debug top moves
        if hasattr(self, 'debug_output'):
            print("\nTop 5 scored moves:")
            for move, score in scored_moves[:5]:
                print(f"{move.uci()}: {score}")
            
        return scored_moves

    def order_moves(self, board: chess.Board, max_moves: Optional[int] = None, tempo_bonus: float = 0.0) -> List[chess.Move]:
        """Order moves based on likely quality."""
        scored_moves = self.order_moves_with_debug(board, max_moves, tempo_bonus)
        return [move for move, _ in scored_moves[:max_moves if max_moves else self.max_ordered_moves]]

    def _calculate_move_score(self, board: chess.Board, move: chess.Move, tempo_bonus: float = 0.0) -> float:
        """Calculate a comprehensive score for move ordering with tempo awareness."""
        score = 0.0
        
        # Get game phase for context
        _, endgame_factor = self.scoring_calculator.tempo.calculate_game_phase(board)
        
        # 1. Immediate promotions (highest priority)
        if move.promotion:
            promotion_value = 100000.0  # Base promotion value
            if move.promotion == chess.QUEEN:
                promotion_value += 10000.0  # Extra for queen
            
            # Add piece value bonus
            promotion_value += self.scoring_calculator.pst.piece_values.get(move.promotion, 0) * 10.0
            
            # Extra for capturing promotions
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    promotion_value += self.scoring_calculator.pst.piece_values.get(captured.piece_type, 0) * 5.0
            
            score += promotion_value
            
        # 2. Captures and tactics (next highest priority)
        if board.is_capture(move):
            capture_value = 50000.0  # Base capture value
            
            # Add MVV-LVA score with phase-based weight
            mvv_lva_score = self.mvv_lva.calculate_mvv_lva_score(move, board)
            capture_value += mvv_lva_score * (100.0 + 50.0 * endgame_factor)  # Higher weight in endgame
            
            # Extra bonus for winning captures
            captured = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if captured and attacker:
                cap_value = self.scoring_calculator.pst.piece_values.get(captured.piece_type, 0)
                att_value = self.scoring_calculator.pst.piece_values.get(attacker.piece_type, 0)
                if cap_value > att_value:
                    capture_value += (cap_value - att_value) * (10.0 + 5.0 * endgame_factor)
            
            score += capture_value
            
        # 3. Tempo and positional scoring
        board_copy = board.copy()
        board_copy.push(move)
        position_assessment = self.scoring_calculator.tempo.assess_position(board_copy, board.turn)
        
        # Apply tempo bonuses with phase consideration
        tempo_score = position_assessment['tempo_score']
        if tempo_score > 0:
            score += tempo_score * (1000.0 + 500.0 * endgame_factor)
        
        # Consider zugzwang risk
        if position_assessment['zugzwang_risk'] < -0.5 and endgame_factor > 0.6:
            score *= 0.8  # Penalize moves that may lead to zugzwang
        
        # 4. Development and central control in opening/middlegame
        if endgame_factor < 0.7:  # Not deep in endgame
            piece = board.piece_at(move.from_square)
            if piece:
                # Development bonus for minor pieces
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    back_rank = 0 if board.turn == chess.WHITE else 7
                    if chess.square_rank(move.from_square) == back_rank:
                        score += 300.0  # Significant bonus for developing moves
                
                # Central control bonus
                to_square_file = chess.square_file(move.to_square)
                to_square_rank = chess.square_rank(move.to_square)
                if 2 <= to_square_file <= 5 and 2 <= to_square_rank <= 5:
                    score += 200.0  # Bonus for controlling central squares
        
        # 5. Apply tempo bonus from parameter
        if tempo_bonus > 0:
            score *= (1.0 + tempo_bonus)
        
        return score

    def _get_history_score(self, board: chess.Board, move: chess.Move) -> float:
        """Get the score from the history table."""
        piece = board.piece_at(move.from_square)
        if not piece:
            return 0.0
            
        key = (piece.piece_type, move.from_square, move.to_square)
        return self.history_moves.get(key, 0.0)

    def _is_killer_move(self, move: chess.Move, depth: int) -> bool:
        """Check if move is a killer move at the current depth."""
        if depth >= len(self.killer_moves):
            return False
        return move in self.killer_moves[depth]

    def add_killer_move(self, move: chess.Move, depth: int) -> None:
        """Add a killer move at the given depth."""
        if depth >= len(self.killer_moves):
            return
            
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > self.max_killer_moves:
                self.killer_moves[depth].pop()

    def add_history_move(self, board: chess.Board, move: chess.Move) -> None:
        """Add a move to the history table."""
        piece = board.piece_at(move.from_square)
        if not piece:
            return
            
        key = (piece.piece_type, move.from_square, move.to_square)
        self.history_moves[key] = min(
            self.history_moves.get(key, 0) + 1,
            self.max_history_score
        )
