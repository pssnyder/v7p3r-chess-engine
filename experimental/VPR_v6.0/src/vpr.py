#!/usr/bin/env python3
"""
VPR Chess Engine v6.0 - Chaos Capitalization Revolution

Core Philosophy:
"We create chaos, they respond traditionally, we capitalize on the mismatch."

Revolutionary v6.0 Architecture:
- Dual-Brain Search System (Chaos Brain + Opponent Model)
- Position Classification for Algorithm Selection  
- Monte Carlo Tree Search for Ultra-Chaotic Positions
- Asymmetric Evaluation Pipeline
- Chaos-Native Pruning (not alpha-beta)

"Traditional engines assume the opponent thinks like them. 
We assume they don't think like us."

Author: Pat Snyder 
Version: VPR Chaos Capitalization v6.0
"""

import chess
import time
import random
import math
from typing import Optional, List, Dict, Tuple, Union
from enum import Enum
from dataclasses import dataclass

class PositionType(Enum):
    """Position classification for algorithm selection"""
    ULTRA_CHAOTIC = "ultra_chaotic"      # MCTS + Full chaos evaluation  
    TACTICAL_SHARP = "tactical_sharp"     # Deep SEE + Chaos search
    POSITIONAL = "positional"             # Hybrid traditional/chaos
    ENDGAME = "endgame"                   # Traditional with chaos bonuses
    OPENING = "opening"                   # Book moves + chaos preparation

@dataclass
class PositionMetrics:
    """Metrics for position classification"""
    legal_moves: int
    tactical_density: float
    material_balance: float  
    piece_activity: float
    king_safety: float
    complexity_score: float

class VPRChaosEngine:
    """
    VPR v6.0 - The Chaos Capitalization Revolution
    
    First chess engine designed to exploit asymmetric thinking:
    - We play chaotically, they respond traditionally
    - We create complexity, they seek simplification
    - We value potential, they value material
    """
    
    def __init__(self):
        self.nodes_searched = 0
        self.search_start_time = 0.0
        self.board = chess.Board()
        
        # Revolutionary v6.0 Components
        self.chaos_brain = VPRChaosBrain()
        self.opponent_brain = TraditionalOpponentModel()
        self.position_classifier = PositionClassifier()
        self.mcts_engine = VPRMonteCarloSearch()
        
        # Performance caches
        self.position_cache = {}
        self.evaluation_cache = {}
        
        # UCI configurable options
        self.chaos_threshold = 80          # When to use full chaos mode
        self.mcts_time_threshold = 1.0     # Minimum time for MCTS
        self.complexity_bonus = 25         # Bonus for chaotic positions
        
        # Initialize attack pattern lookups
        self._init_attack_patterns()
    
    def _init_attack_patterns(self):
        """Initialize pre-computed attack patterns for speed"""
        # Knight attack patterns for each square (pre-computed bitboards)
        self.knight_attacks = {}
        for square in chess.SQUARES:
            attacks = 0
            row, col = divmod(square, 8)
            
            # All 8 possible knight moves
            knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target_square = new_row * 8 + new_col
                    attacks |= (1 << target_square)
            
            self.knight_attacks[square] = attacks
        
        # King attack patterns 
        self.king_attacks = {}
        for square in chess.SQUARES:
            attacks = 0
            row, col = divmod(square, 8)
            
            # All 8 possible king moves
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target_square = new_row * 8 + new_col
                        attacks |= (1 << target_square)
            
            self.king_attacks[square] = attacks
    
    def search(self, board: chess.Board, time_limit: float = 3.0, 
               depth: Optional[int] = None) -> chess.Move:
        """
        VPR v6.0 Revolutionary Search Entry Point
        
        Determines position type and selects appropriate search algorithm:
        - Ultra-chaotic: Monte Carlo Tree Search
        - Tactical: Deep chaos search with SEE
        - Positional: Hybrid approach
        - Endgame: Traditional with chaos bonuses
        """
        self.nodes_searched = 0
        self.search_start_time = time.time()
        self.position_cache.clear()
        self.evaluation_cache.clear()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Step 1: Classify the position
        position_type = self.position_classifier.classify(board)
        
        # Step 2: Select search algorithm based on position type
        search_algorithm = self._select_search_algorithm(position_type, time_limit)
        
        # Step 3: Execute search with selected algorithm
        if search_algorithm == "MCTS":
            return self.mcts_engine.search(board, time_limit)
        elif search_algorithm == "CHAOS_DEEP":
            return self._chaos_search_deep(board, time_limit, depth)
        elif search_algorithm == "HYBRID":
            return self._hybrid_search(board, time_limit, depth)
        else:  # CHAOS_STANDARD
            return self._chaos_search_standard(board, time_limit, depth)
    
    def _select_search_algorithm(self, position_type: PositionType, time_available: float) -> str:
        """Select the best search algorithm for this position type"""
        if position_type == PositionType.ULTRA_CHAOTIC and time_available > self.mcts_time_threshold:
            return "MCTS"
        elif position_type == PositionType.TACTICAL_SHARP:
            return "CHAOS_DEEP"  
        elif position_type == PositionType.POSITIONAL:
            return "HYBRID"
        else:
            return "CHAOS_STANDARD"
    
    def _chaos_search_deep(self, board: chess.Board, time_limit: float, depth: Optional[int]) -> chess.Move:
        """Deep chaos search for tactical positions"""
        target_depth = depth if depth else 6
        best_move = list(board.legal_moves)[0]
        best_score = -999999
        
        # Use chaos brain for move ordering
        moves = self.chaos_brain.order_moves(board, list(board.legal_moves))
        
        for move in moves[:10]:  # Limit for now
            board.push(move)
            score = -self.chaos_brain.evaluate_position(board)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _hybrid_search(self, board: chess.Board, time_limit: float, depth: Optional[int]) -> chess.Move:
        """Hybrid search combining chaos and traditional elements"""
        return self._chaos_search_standard(board, time_limit, depth)
    
    def _chaos_search_standard(self, board: chess.Board, time_limit: float, depth: Optional[int]) -> chess.Move:
        """Standard chaos search - our bread and butter"""
        moves = list(board.legal_moves)
        if not moves:
            return chess.Move.null()
        
        # For now, return the move that creates the most chaos
        best_move = moves[0]
        best_chaos_score = 0
        
        for move in moves:
            board.push(move)
            chaos_score = len(list(board.legal_moves))  # Simple chaos metric
            board.pop()
            
            if chaos_score > best_chaos_score:
                best_chaos_score = chaos_score
                best_move = move
        
        return best_move
    
    def new_game(self):
        """Reset engine state for a new game"""
        self.nodes_searched = 0
        self.position_cache.clear()
        self.evaluation_cache.clear()
    
    def get_engine_info(self) -> dict:
        """Return engine information"""
        return {
            'name': 'VPR Chaos Capitalization',
            'version': '6.0',
            'author': 'V7P3R Project', 
            'description': 'Revolutionary dual-brain asymmetric chess engine',
            'nodes_searched': self.nodes_searched
        }


class VPRChaosBrain:
    """
    The Chaos Brain - Generates chaotic, potential-maximizing moves
    
    Core responsibilities:
    - Dynamic piece potential calculation
    - Chaos-native move ordering
    - Complexity-preserving search
    """
    
    def __init__(self):
        # Traditional piece values for comparison
        self.traditional_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate position from chaos perspective"""
        score = 0.0
        
        # Base evaluation: piece potential instead of material
        our_potential = 0
        their_potential = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                potential = self._calculate_piece_potential(board, square)
                if piece.color == board.turn:
                    our_potential += potential
                else:
                    their_potential += potential
        
        score = our_potential - their_potential
        
        # Chaos bonus - reward complexity
        legal_moves = len(list(board.legal_moves))
        if legal_moves > 40:
            score += 25  # Chaos bonus
        
        # Randomness factor (opponent imperfection)
        score += random.randint(-5, 5)
        
        return score
    
    def _calculate_piece_potential(self, board: chess.Board, square: chess.Square) -> int:
        """Calculate dynamic piece potential"""
        piece = board.piece_at(square)
        if not piece:
            return 0
        
        potential = 0
        piece_type = piece.piece_type
        
        # Count legal moves from this square
        legal_moves = [move for move in board.legal_moves if move.from_square == square]
        potential += len(legal_moves) * 2
        
        # Count attacks (even if not legal moves)
        attacks = board.attacks(square)
        potential += len(attacks)
        
        # Piece-specific bonuses
        if piece_type == chess.QUEEN:
            potential *= 2  # Queens have high potential
        elif piece_type == chess.KNIGHT:
            potential += 5  # Knights create chaos
        elif piece_type == chess.PAWN:
            # Pawns get bonus for advancement
            row = chess.square_rank(square)
            if piece.color == chess.WHITE:
                potential += row * 2
            else:
                potential += (7 - row) * 2
        
        return potential
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """Order moves to prioritize chaos and complexity"""
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Captures get priority
            if board.is_capture(move):
                score += 100
            
            # Checks create chaos
            board.push(move)
            if board.is_check():
                score += 50
            
            # Moves that increase opponent's options (more chaos for them)
            opponent_moves = len(list(board.legal_moves))
            score += opponent_moves // 2
            
            board.pop()
            
            move_scores.append((score, move))
        
        # Sort by score (highest first)
        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in move_scores]


class TraditionalOpponentModel:
    """
    Traditional Opponent Brain - Models how traditional engines think
    
    Core responsibilities:
    - Material-based evaluation
    - Traditional move ordering  
    - Opponent response prediction
    """
    
    def __init__(self):
        # Traditional piece values for opponent modeling
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def predict_response(self, board: chess.Board, depth: int = 2) -> chess.Move:
        """Predict how a traditional engine would respond"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        best_move = legal_moves[0]
        best_score = -999999
        
        for move in legal_moves:
            board.push(move)
            score = self.evaluate_traditionally(board)
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def evaluate_traditionally(self, board: chess.Board) -> float:
        """Evaluate position using traditional material + position"""
        score = 0.0
        
        # Material count
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == board.turn:
                    score += value
                else:
                    score -= value
        
        # Simple positional factors
        # Center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            if board.is_attacked_by(board.turn, square):
                score += 10
        
        return score


class PositionClassifier:
    """
    Position Classification System
    
    Determines position type to select appropriate search algorithm
    """
    
    def classify(self, board: chess.Board) -> PositionType:
        """Classify position type for algorithm selection"""
        metrics = self._calculate_metrics(board)
        
        # Ultra-chaotic: Many moves, high tactical density
        if metrics.legal_moves > 80 and metrics.tactical_density > 7:
            return PositionType.ULTRA_CHAOTIC
        
        # Tactical sharp: High tactical density  
        elif metrics.tactical_density > 5:
            return PositionType.TACTICAL_SHARP
        
        # Endgame: Few pieces remaining
        elif self._count_total_pieces(board) <= 8:
            return PositionType.ENDGAME
        
        # Opening: First 10 moves
        elif len(board.move_stack) < 10:
            return PositionType.OPENING
        
        # Default to positional
        else:
            return PositionType.POSITIONAL
    
    def _calculate_metrics(self, board: chess.Board) -> PositionMetrics:
        """Calculate position metrics for classification"""
        return PositionMetrics(
            legal_moves=len(list(board.legal_moves)),
            tactical_density=self._calculate_tactical_density(board),
            material_balance=self._calculate_material_balance(board),
            piece_activity=self._calculate_piece_activity(board),
            king_safety=self._calculate_king_safety(board),
            complexity_score=self._calculate_complexity(board)
        )
    
    def _calculate_tactical_density(self, board: chess.Board) -> float:
        """Calculate tactical motif density"""
        tactical_score = 0.0
        
        # Count captures
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        tactical_score += len(captures)
        
        # Count checks
        checks = []
        for move in board.legal_moves:
            board.push(move)
            if board.is_check():
                checks.append(move)
            board.pop()
        tactical_score += len(checks) * 2
        
        # Count attacks on pieces
        attacks = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    attacks += 1
        tactical_score += attacks
        
        return tactical_score
    
    def _calculate_material_balance(self, board: chess.Board) -> float:
        """Calculate material balance"""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    balance += value
                else:
                    balance -= value
        
        return balance
    
    def _calculate_piece_activity(self, board: chess.Board) -> float:
        """Calculate total piece activity"""
        activity = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                moves = [move for move in board.legal_moves if move.from_square == square]
                activity += len(moves)
        
        return activity
    
    def _calculate_king_safety(self, board: chess.Board) -> float:
        """Calculate king safety metrics"""
        king_square = board.king(board.turn)
        if king_square is None:
            return -100  # King missing!
        
        safety = 0
        
        # Check if king is under attack
        if board.is_attacked_by(not board.turn, king_square):
            safety -= 50
        
        # Count attackers around king
        king_area = [sq for sq in chess.SQUARES 
                    if chess.square_distance(king_square, sq) <= 2]
        
        for square in king_area:
            if board.is_attacked_by(not board.turn, square):
                safety -= 5
        
        return safety
    
    def _calculate_complexity(self, board: chess.Board) -> float:
        """Calculate overall position complexity"""
        complexity = 0
        
        # Number of legal moves
        complexity += len(list(board.legal_moves))
        
        # Number of pieces
        complexity += self._count_total_pieces(board)
        
        # Tactical density
        complexity += self._calculate_tactical_density(board)
        
        return complexity
    
    def _count_total_pieces(self, board: chess.Board) -> int:
        """Count total pieces on board"""
        return len([sq for sq in chess.SQUARES if board.piece_at(sq) is not None])


class VPRMonteCarloSearch:
    """
    Monte Carlo Tree Search Engine for Ultra-Chaotic Positions
    
    Used when position complexity exceeds traditional search capabilities
    """
    
    def __init__(self):
        self.exploration_constant = 1.4
        self.node_budget = 1000  # Start small for testing
    
    def search(self, board: chess.Board, time_limit: float) -> chess.Move:
        """Execute MCTS for ultra-chaotic positions"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        
        # Simple MCTS placeholder - just random simulation for now
        move_scores = {}
        simulations_per_move = max(1, self.node_budget // len(legal_moves))
        
        for move in legal_moves:
            total_score = 0
            for _ in range(simulations_per_move):
                board.push(move)
                score = self._random_simulation(board, 10)  # 10 random moves
                board.pop()
                total_score += score
            
            move_scores[move] = total_score / simulations_per_move
        
        # Return move with highest average score
        best_move = max(move_scores.items(), key=lambda x: x[1])[0]
        return best_move
    
    def _random_simulation(self, board: chess.Board, depth: int) -> float:
        """Random simulation to evaluate position"""
        if depth <= 0 or board.is_game_over():
            # Simple evaluation: material + random factor
            score = 0
            piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                           chess.ROOK: 5, chess.QUEEN: 9}
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type != chess.KING:
                    value = piece_values[piece.piece_type]
                    if piece.color == board.turn:
                        score += value
                    else:
                        score -= value
            
            return score + random.randint(-2, 2)
        
        # Random move
        moves = list(board.legal_moves)
        if not moves:
            return 0
        
        move = random.choice(moves)
        board.push(move)
        score = -self._random_simulation(board, depth - 1)
        board.pop()
        
        return score


# Main engine class alias for UCI interface compatibility
VPREngine = VPRChaosEngine