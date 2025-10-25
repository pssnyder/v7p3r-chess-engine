#!/usr/bin/env python3
"""
V7P3R v13 "Capablanca" - Positional Simplification Engine

The Capablanca Evolution: From Tactical Complexity to Positional Clarity
Codename: "Capablanca" (internal only) - emphasizing clear, simple, positional play

CORE PHILOSOPHY:
- Acknowledge "low rating engine syndrome" - our best move assumptions are flawed
- Dual-brained evaluation: complex for our moves, simplified for opponent moves
- Seek simplification through equal trades and pawn removal
- Early search exit with "good enough" moves that don't lose material
- Asymmetric pruning - less aggressive on opponent moves due to evaluation limitations

CAPABLANCA PRINCIPLES:
1. Simplify positions when possible (remove pawns, equal trades)
2. Don't overthink - rarely is there only one good move
3. Clear recaptures should be executed quickly
4. Prefer moves leading to familiar, lower-complexity positions
5. Use complexity metrics to guide search allocation

Author: Pat Snyder
October 2025
"""

import chess
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class PlayerPerspective(Enum):
    """Player perspective for dual-brained evaluation"""
    OUR_MOVE = "our_move"      # Full complex evaluation
    OPPONENT_MOVE = "opponent_move"  # Simplified evaluation


@dataclass
class PositionComplexity:
    """Capablanca complexity metrics for position assessment"""
    optionality: float = 0.0        # Number of plausible candidate moves
    volatility: float = 0.0         # Evaluation stability across depths
    tactical_density: float = 0.0   # Number of forcing moves (checks/captures)
    novelty: float = 0.0            # Rarity/unfamiliarity of position
    total_score: float = 0.0        # Combined complexity score


@dataclass
class CapablancaMetrics:
    """Performance tracking for Capablanca engine"""
    simplifications_made: int = 0
    early_exits: int = 0
    dual_brain_evals: int = 0
    complexity_based_pruning: int = 0
    recaptures_executed: int = 0


class CapablancaComplexityAnalyzer:
    """
    Analyzes position complexity using Capablanca metrics
    Guides search allocation and move selection based on complexity
    """
    
    def __init__(self):
        self.position_cache: Dict[str, PositionComplexity] = {}
        self.evaluation_history: Dict[str, List[float]] = {}
        
    def analyze_position_complexity(self, board: chess.Board, 
                                  candidate_moves: List[chess.Move],
                                  current_eval: float = 0.0,
                                  depth: int = 0) -> PositionComplexity:
        """
        Calculate comprehensive complexity score for current position
        Based on optionality, volatility, tactical density, and novelty
        """
        fen = board.fen()
        if fen in self.position_cache:
            return self.position_cache[fen]
        
        complexity = PositionComplexity()
        
        # 1. Optionality/Variability - width of decision tree
        complexity.optionality = self._calculate_optionality(board, candidate_moves, current_eval)
        
        # 2. Evaluation Volatility - stability across depths
        complexity.volatility = self._calculate_volatility(fen, current_eval, depth)
        
        # 3. Tactical Density - forcing moves
        complexity.tactical_density = self._calculate_tactical_density(board)
        
        # 4. Novelty/Rarity - position unfamiliarity
        complexity.novelty = self._calculate_novelty(board)
        
        # Combined complexity score (weighted)
        complexity.total_score = (
            0.3 * complexity.optionality +
            0.4 * complexity.volatility +
            0.2 * complexity.tactical_density +
            0.1 * complexity.novelty
        )
        
        self.position_cache[fen] = complexity
        return complexity
    
    def _calculate_optionality(self, board: chess.Board, 
                             candidate_moves: List[chess.Move],
                             current_eval: float,
                             epsilon: float = 0.5) -> float:
        """
        Calculate optionality: number of moves within epsilon of best eval
        High optionality = many good moves = decision complexity
        """
        if not candidate_moves:
            return 0.0
        
        # For now, use simplified heuristic based on move types
        # In full implementation, would evaluate each move
        good_moves = 0
        for move in candidate_moves:
            # Quick heuristic: captures, checks, and development are "good"
            if (board.is_capture(move) or 
                board.gives_check(move) or
                self._is_development_move(board, move)):
                good_moves += 1
        
        # Normalize by total moves
        return min(good_moves / len(candidate_moves), 1.0)
    
    def _calculate_volatility(self, fen: str, current_eval: float, depth: int) -> float:
        """
        Calculate evaluation volatility across search depths
        High volatility = unstable evaluation = complex position
        """
        if fen not in self.evaluation_history:
            self.evaluation_history[fen] = []
        
        self.evaluation_history[fen].append(current_eval)
        evals = self.evaluation_history[fen]
        
        if len(evals) < 2:
            return 0.0
        
        # Calculate variance in recent evaluations
        recent_evals = evals[-3:]  # Last 3 depth evaluations
        if len(recent_evals) < 2:
            return 0.0
        
        mean_eval = sum(recent_evals) / len(recent_evals)
        variance = sum((e - mean_eval) ** 2 for e in recent_evals) / len(recent_evals)
        
        # Normalize volatility (high variance = high complexity)
        return min(math.sqrt(variance) / 100.0, 1.0)  # Centipawn normalization
    
    def _calculate_tactical_density(self, board: chess.Board) -> float:
        """
        Calculate tactical density: checks + captures + attacks
        High density = many forcing moves = complex calculation required
        """
        checks = sum(1 for move in board.legal_moves if board.gives_check(move))
        captures = sum(1 for move in board.legal_moves if board.is_capture(move))
        
        # Approximate attacks (pieces under attack)
        attacks = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and board.is_attacked_by(not piece.color, square):
                attacks += 1
        
        total_forcing = checks + captures + attacks
        
        # Normalize (typical positions have 0-10 forcing elements)
        return min(total_forcing / 10.0, 1.0)
    
    def _calculate_novelty(self, board: chess.Board) -> float:
        """
        Calculate position novelty/rarity
        High novelty = unfamiliar position = requires original calculation
        """
        # Simplified heuristic based on piece distribution and structure
        # In full implementation, would use opening book/database frequency
        
        # Count piece types and positions
        piece_count = len(board.piece_map())
        
        # Opening phase (high piece count) = low novelty
        # Endgame/complex middlegame = higher novelty
        if piece_count > 28:  # Opening
            return 0.1
        elif piece_count > 20:  # Early middlegame
            return 0.3
        elif piece_count > 12:  # Complex middlegame
            return 0.6
        else:  # Endgame
            return 0.8
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move develops a piece from starting position"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        # Simple development check for opening
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            back_rank = 0 if piece.color == chess.BLACK else 7
            return chess.square_rank(move.from_square) == back_rank
        
        return False


class CapablancaDualBrainEvaluator:
    """
    Dual-brained evaluation system for Capablanca engine
    Complex evaluation for our moves, simplified for opponent moves
    """
    
    def __init__(self):
        self.metrics = CapablancaMetrics()
        
    def evaluate_position(self, board: chess.Board, 
                         perspective: PlayerPerspective,
                         depth: int = 0) -> float:
        """
        Evaluate position with appropriate complexity based on perspective
        Our moves: full evaluation; Opponent moves: simplified evaluation
        """
        self.metrics.dual_brain_evals += 1
        
        if perspective == PlayerPerspective.OUR_MOVE:
            return self._complex_evaluation(board, depth)
        else:
            return self._simplified_evaluation(board)
    
    def _complex_evaluation(self, board: chess.Board, depth: int) -> float:
        """
        Full complexity evaluation for our move decisions
        Includes all heuristics, tactical analysis, positional factors
        """
        eval_score = 0.0
        
        # Material balance (full piece value calculation)
        eval_score += self._calculate_material_balance(board, complex_mode=True)
        
        # Positional factors (piece mobility, king safety, pawn structure)
        eval_score += self._calculate_positional_factors(board, complex_mode=True)
        
        # Tactical threats and opportunities
        eval_score += self._calculate_tactical_factors(board, complex_mode=True)
        
        # Simplification bonus (Capablanca preference)
        eval_score += self._calculate_simplification_bonus(board)
        
        return eval_score
    
    def _simplified_evaluation(self, board: chess.Board) -> float:
        """
        Simplified evaluation for opponent move assessment
        Focus only on critical factors any player would consider
        """
        eval_score = 0.0
        
        # Basic material balance only
        eval_score += self._calculate_material_balance(board, complex_mode=False)
        
        # Critical threats only (checks, captures, major piece attacks)
        eval_score += self._calculate_critical_threats(board)
        
        # Basic king safety
        eval_score += self._calculate_basic_king_safety(board)
        
        return eval_score
    
    def _calculate_material_balance(self, board: chess.Board, complex_mode: bool) -> float:
        """Calculate material balance with complexity based on mode"""
        white_material = 0
        black_material = 0
        
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                
                if complex_mode:
                    # Add positional adjustments for our evaluation
                    value += self._get_positional_adjustment(board, square, piece)
                
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Return from current player's perspective
        if board.turn == chess.WHITE:
            return white_material - black_material
        else:
            return black_material - white_material
    
    def _get_positional_adjustment(self, board: chess.Board, 
                                 square: int, piece: chess.Piece) -> float:
        """Get positional value adjustment for complex evaluation"""
        adjustment = 0.0
        
        # Center control bonus
        if square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            adjustment += 20
        elif square in [chess.C3, chess.C6, chess.F3, chess.F6]:
            adjustment += 10
        
        # Piece-specific adjustments
        if piece.piece_type == chess.KNIGHT:
            # Knights on rim are dim
            if square in [chess.A1, chess.A8, chess.H1, chess.H8]:
                adjustment -= 30
        elif piece.piece_type == chess.BISHOP:
            # Bishop pair bonus (simplified)
            bishops = [p for p in board.piece_map().values() 
                      if p.piece_type == chess.BISHOP and p.color == piece.color]
            if len(bishops) >= 2:
                adjustment += 25
        
        return adjustment
    
    def _calculate_positional_factors(self, board: chess.Board, complex_mode: bool) -> float:
        """Calculate positional evaluation factors"""
        if not complex_mode:
            return 0.0  # Skip for simplified evaluation
        
        score = 0.0
        
        # Piece mobility
        our_mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())  # Switch perspective
        opp_mobility = len(list(board.legal_moves))
        board.pop()
        
        score += (our_mobility - opp_mobility) * 2
        
        # Pawn structure (simplified)
        score += self._evaluate_pawn_structure(board)
        
        return score
    
    def _calculate_tactical_factors(self, board: chess.Board, complex_mode: bool) -> float:
        """Calculate tactical threats and opportunities"""
        score = 0.0
        
        # Check for immediate tactical threats
        if board.is_check():
            score -= 50 if board.turn else score + 50
        
        # Hanging pieces
        score += self._count_hanging_pieces(board) * -30
        
        if complex_mode:
            # Additional tactical analysis for our moves
            score += self._analyze_tactical_motifs(board)
        
        return score
    
    def _calculate_simplification_bonus(self, board: chess.Board) -> float:
        """
        Capablanca-style simplification bonus
        Prefer positions with fewer pieces and clearer structure
        """
        piece_count = len(board.piece_map())
        
        # Bonus for fewer pieces (encourages simplification)
        simplification_bonus = max(0, (32 - piece_count) * 5)
        
        # Extra bonus for equal material trades
        # (Would track this in move generation)
        
        return simplification_bonus
    
    def _calculate_critical_threats(self, board: chess.Board) -> float:
        """Calculate only critical threats for simplified evaluation"""
        score = 0.0
        
        # Immediate checks
        if board.is_check():
            score += 30
        
        # Immediate captures of valuable pieces
        for move in board.legal_moves:
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured and captured.piece_type in [chess.QUEEN, chess.ROOK]:
                    score += 20
        
        return score
    
    def _calculate_basic_king_safety(self, board: chess.Board) -> float:
        """Basic king safety for simplified evaluation"""
        score = 0.0
        
        # Check if king is castled
        if board.has_castling_rights(board.turn):
            score -= 20  # Penalty for not castling yet
        
        return score
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Simplified pawn structure evaluation"""
        score = 0.0
        
        # Count doubled pawns (penalty)
        files_with_pawns = [0] * 8
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file_idx = chess.square_file(square)
                if piece.color == board.turn:
                    files_with_pawns[file_idx] += 1
        
        doubled_pawns = sum(max(0, count - 1) for count in files_with_pawns)
        score -= doubled_pawns * 15
        
        return score
    
    def _count_hanging_pieces(self, board: chess.Board) -> int:
        """Count pieces that are attacked and undefended"""
        hanging = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                # Check if piece is attacked by opponent
                if board.is_attacked_by(not board.turn, square):
                    # Check if piece is defended
                    if not board.is_attacked_by(board.turn, square):
                        hanging += 1
        
        return hanging
    
    def _analyze_tactical_motifs(self, board: chess.Board) -> float:
        """Analyze tactical patterns for complex evaluation"""
        score = 0.0
        
        # Look for pins, forks, skewers, etc.
        # This would integrate with existing tactical detector
        
        # For now, simple check for discovered attacks
        for move in board.legal_moves:
            # Would implement tactical pattern detection here
            pass
        
        return score


class CapablancaMoveOrderer:
    """
    Capablanca move ordering with asymmetric pruning
    Less aggressive pruning on opponent moves due to evaluation limitations
    """
    
    def __init__(self, complexity_analyzer: CapablancaComplexityAnalyzer):
        self.complexity_analyzer = complexity_analyzer
        self.metrics = CapablancaMetrics()
    
    def order_moves_capablanca(self, board: chess.Board, 
                             perspective: PlayerPerspective,
                             depth: int = 0) -> List[chess.Move]:
        """
        Order moves with Capablanca principles and asymmetric pruning
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []
        
        # Analyze position complexity
        complexity = self.complexity_analyzer.analyze_position_complexity(
            board, legal_moves, depth=depth
        )
        
        # Apply different ordering strategies based on perspective
        if perspective == PlayerPerspective.OUR_MOVE:
            return self._order_our_moves(board, legal_moves, complexity)
        else:
            return self._order_opponent_moves(board, legal_moves, complexity)
    
    def _order_our_moves(self, board: chess.Board, 
                        moves: List[chess.Move],
                        complexity: PositionComplexity) -> List[chess.Move]:
        """
        Order our moves with full Capablanca principles and aggressive pruning
        """
        move_scores = []
        
        for move in moves:
            score = 0.0
            
            # 1. Immediate recaptures (high priority)
            if self._is_immediate_recapture(board, move):
                score += 1000
                self.metrics.recaptures_executed += 1
            
            # 2. Simplification moves (Capablanca preference)
            if self._is_simplification_move(board, move):
                score += 500
                self.metrics.simplifications_made += 1
            
            # 3. Standard tactical priorities
            if board.is_capture(move):
                score += 400
            if board.gives_check(move):
                score += 300
            
            # 4. Complexity reduction bonus
            complexity_reduction = self._estimate_complexity_reduction(board, move, complexity)
            score += complexity_reduction * 200
            
            # 5. Avoid moves that increase complexity (if high complexity already)
            if complexity.total_score > 0.7:
                if self._increases_complexity(board, move):
                    score -= 300
            
            move_scores.append((move, score))
        
        # Sort by score descending
        move_scores.sort(key=lambda x: x[1], reverse=True)
        ordered_moves = [move for move, score in move_scores]
        
        # Apply AGGRESSIVE pruning for our moves (we trust our evaluation more)
        # Progressive pruning based on position complexity and move count
        if len(ordered_moves) > 8:
            if complexity.total_score > 0.5:
                # High complexity: Keep only top 40% of moves
                keep_count = max(4, int(len(ordered_moves) * 0.4))
            elif complexity.total_score > 0.3:
                # Medium complexity: Keep top 60% of moves
                keep_count = max(5, int(len(ordered_moves) * 0.6))
            else:
                # Low complexity: Keep top 70% of moves
                keep_count = max(6, int(len(ordered_moves) * 0.7))
            
            ordered_moves = ordered_moves[:keep_count]
            self.metrics.complexity_based_pruning += len(moves) - keep_count
        
        return ordered_moves
    
    def _order_opponent_moves(self, board: chess.Board,
                            moves: List[chess.Move],
                            complexity: PositionComplexity) -> List[chess.Move]:
        """
        Order opponent moves with conservative pruning
        Less aggressive due to our flawed "best move" assumptions
        """
        move_scores = []
        
        for move in moves:
            score = 0.0
            
            # Only basic priorities for opponent evaluation
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    score += self._get_piece_value(captured.piece_type)
            
            if board.gives_check(move):
                score += 200
            
            # Assume opponent prefers simplification too
            if self._is_simplification_move(board, move):
                score += 100
            
            move_scores.append((move, score))
        
        # Sort by score descending
        move_scores.sort(key=lambda x: x[1], reverse=True)
        ordered_moves = [move for move, score in move_scores]
        
        # Conservative pruning for opponent moves
        # Keep more moves since our evaluation might be wrong
        if len(ordered_moves) > 8:
            # Keep top 80% of moves (more conservative than our moves)
            keep_count = max(6, int(len(ordered_moves) * 0.8))
            ordered_moves = ordered_moves[:keep_count]
        
        return ordered_moves
    
    def _is_immediate_recapture(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move is an immediate recapture"""
        if not board.is_capture(move):
            return False
        
        # Check if the captured piece just moved to this square
        # (Would need move history for full implementation)
        return False  # Simplified for now
    
    def _is_simplification_move(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Check if move leads to simplification (equal trades, pawn removal)
        """
        if not board.is_capture(move):
            return False
        
        captured_piece = board.piece_at(move.to_square)
        moving_piece = board.piece_at(move.from_square)
        
        if not captured_piece or not moving_piece:
            return False
        
        # Equal value trade
        captured_value = self._get_piece_value(captured_piece.piece_type)
        moving_value = self._get_piece_value(moving_piece.piece_type)
        
        # Prefer equal trades, especially involving pawns
        if abs(captured_value - moving_value) <= 30:  # Roughly equal
            if captured_piece.piece_type == chess.PAWN or moving_piece.piece_type == chess.PAWN:
                return True
        
        return False
    
    def _estimate_complexity_reduction(self, board: chess.Board,
                                     move: chess.Move,
                                     current_complexity: PositionComplexity) -> float:
        """
        Estimate how much the move reduces position complexity
        """
        # Simplification generally reduces complexity
        if self._is_simplification_move(board, move):
            return 0.3
        
        # Developing moves in opening reduce complexity
        if self._is_development_move(board, move):
            return 0.2
        
        # Castling reduces complexity
        if board.is_castling(move):
            return 0.4
        
        return 0.0
    
    def _increases_complexity(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move significantly increases position complexity"""
        # Moves that create many new tactical possibilities
        if board.gives_check(move):
            # Check if it's a complex check vs simple check
            return True  # Simplified
        
        # Sacrificial moves increase complexity
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            moving = board.piece_at(move.from_square)
            if captured and moving:
                if self._get_piece_value(moving.piece_type) > self._get_piece_value(captured.piece_type) + 100:
                    return True  # Potential sacrifice
        
        return False
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move develops a piece"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            back_rank = 0 if piece.color == chess.BLACK else 7
            return chess.square_rank(move.from_square) == back_rank
        
        return False
    
    def _get_piece_value(self, piece_type: chess.PieceType) -> int:
        """Get basic piece value"""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        return values.get(piece_type, 0)


class CapablancaSearchController:
    """
    Search controller with early exit logic for "good enough" moves
    Implements Capablanca principle: don't overthink, play good moves quickly
    """
    
    def __init__(self):
        self.metrics = CapablancaMetrics()
        self.move_cache: Dict[str, chess.Move] = {}
    
    def should_exit_early(self, board: chess.Board,
                         current_best_move: chess.Move,
                         current_eval: float,
                         depth: int,
                         time_remaining: float) -> bool:
        """
        Determine if we should exit search early with current best move
        Based on move quality, position simplicity, and time pressure
        """
        # 1. Check if we have a "good enough" move
        if self._is_good_enough_move(board, current_best_move, current_eval):
            
            # 2. Check time pressure (bullet games)
            if time_remaining < 5.0 and depth >= 3:  # Under 5 seconds
                self.metrics.early_exits += 1
                return True
            
            # 3. Check if position is simple enough for quick decision
            if self._is_simple_position(board) and depth >= 4:
                self.metrics.early_exits += 1
                return True
            
            # 4. Check if we're ahead and can play safely
            if current_eval > 100 and depth >= 5:  # Up by a pawn
                self.metrics.early_exits += 1
                return True
        
        return False
    
    def _is_good_enough_move(self, board: chess.Board,
                           move: chess.Move,
                           eval_score: float) -> bool:
        """
        Check if move is "good enough" for early exit
        Criteria: doesn't lose material, improves position or maintains balance
        """
        if not move:
            return False
        
        # 1. Must not lose material (basic safety)
        if eval_score < -50:  # Losing half a pawn or more
            return False
        
        # 2. Check for obvious blunders
        if self._is_obvious_blunder(board, move):
            return False
        
        # 3. Prefer moves that simplify or develop
        if (board.is_capture(move) or 
            board.is_castling(move) or
            self._is_development_move(board, move)):
            return True
        
        # 4. Accept moves that maintain rough equality
        if abs(eval_score) < 30:  # Within 0.3 pawns
            return True
        
        return False
    
    def _is_simple_position(self, board: chess.Board) -> bool:
        """Check if position is simple enough for quick decisions"""
        # Count forcing moves (checks and captures)
        forcing_moves = sum(1 for move in board.legal_moves 
                          if board.is_capture(move) or board.gives_check(move))
        
        # Simple if few forcing moves
        if forcing_moves <= 3:
            return True
        
        # Simple if few pieces left
        piece_count = len(board.piece_map())
        if piece_count <= 16:  # Simplified endgame
            return True
        
        return False
    
    def _is_obvious_blunder(self, board: chess.Board, move: chess.Move) -> bool:
        """Check for obvious blunders that should never be played"""
        # 1. Moving piece to square where it can be captured for free
        moving_piece = board.piece_at(move.from_square)
        if moving_piece:
            # Make the move temporarily
            board.push(move)
            
            # Check if piece is now attacked and undefended
            is_attacked = board.is_attacked_by(board.turn, move.to_square)  # Now opponent's turn
            is_defended = board.is_attacked_by(not board.turn, move.to_square)
            
            board.pop()
            
            if is_attacked and not is_defended:
                # Only a blunder if we're losing significant value
                piece_value = self._get_piece_value(moving_piece.piece_type)
                if piece_value >= 300:  # Knight/Bishop or higher
                    return True
        
        return False
    
    def _is_development_move(self, board: chess.Board, move: chess.Move) -> bool:
        """Check if move develops a piece from starting position"""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            back_rank = 0 if piece.color == chess.BLACK else 7
            return chess.square_rank(move.from_square) == back_rank
        
        return False
    
    def _get_piece_value(self, piece_type: chess.PieceType) -> int:
        """Get basic piece value"""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        return values.get(piece_type, 0)


# Example usage and integration points
def create_capablanca_engine_components():
    """Create the main Capablanca engine components"""
    
    complexity_analyzer = CapablancaComplexityAnalyzer()
    dual_brain_evaluator = CapablancaDualBrainEvaluator()
    move_orderer = CapablancaMoveOrderer(complexity_analyzer)
    search_controller = CapablancaSearchController()
    
    return {
        'complexity_analyzer': complexity_analyzer,
        'dual_brain_evaluator': dual_brain_evaluator,
        'move_orderer': move_orderer,
        'search_controller': search_controller
    }


def print_capablanca_metrics(components: Dict):
    """Print performance metrics for all Capablanca components"""
    
    print("\n=== V13 Capablanca Engine Metrics ===")
    
    # Collect metrics from all components
    total_simplifications = 0
    total_early_exits = 0
    total_dual_brain_evals = 0
    total_complexity_pruning = 0
    total_recaptures = 0
    
    for component in components.values():
        if hasattr(component, 'metrics'):
            metrics = component.metrics
            total_simplifications += metrics.simplifications_made
            total_early_exits += metrics.early_exits
            total_dual_brain_evals += metrics.dual_brain_evals
            total_complexity_pruning += metrics.complexity_based_pruning
            total_recaptures += metrics.recaptures_executed
    
    print(f"Simplifications Made: {total_simplifications}")
    print(f"Early Search Exits: {total_early_exits}")
    print(f"Dual-Brain Evaluations: {total_dual_brain_evals}")
    print(f"Complexity-Based Pruning: {total_complexity_pruning}")
    print(f"Quick Recaptures: {total_recaptures}")
    print("=" * 40)


if __name__ == "__main__":
    # Test Capablanca components
    print("V7P3R v13 'Capablanca' Engine Framework")
    print("Positional Simplification and Dual-Brain Evaluation")
    
    components = create_capablanca_engine_components()
    
    # Example position analysis
    board = chess.Board()
    
    # Test complexity analysis
    complexity = components['complexity_analyzer'].analyze_position_complexity(
        board, list(board.legal_moves)
    )
    
    print(f"\nStarting position complexity: {complexity.total_score:.3f}")
    print(f"  Optionality: {complexity.optionality:.3f}")
    print(f"  Volatility: {complexity.volatility:.3f}")
    print(f"  Tactical Density: {complexity.tactical_density:.3f}")
    print(f"  Novelty: {complexity.novelty:.3f}")
    
    # Test dual-brain evaluation
    our_eval = components['dual_brain_evaluator'].evaluate_position(
        board, PlayerPerspective.OUR_MOVE
    )
    opp_eval = components['dual_brain_evaluator'].evaluate_position(
        board, PlayerPerspective.OPPONENT_MOVE
    )
    
    print(f"\nDual-brain evaluation:")
    print(f"  Our perspective: {our_eval:.1f}")
    print(f"  Opponent perspective: {opp_eval:.1f}")
    
    print("\nCapablanca framework initialized successfully!")