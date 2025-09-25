#!/usr/bin/env python3
"""
V7P3R v11 Phase 1: Advanced Time Management System
Dynamic time allocation based on position complexity and adaptive depth targeting
Author: Pat Snyder
"""

import chess
import time
import math
from typing import Dict, Tuple, Optional


class V7P3RTimeManager:
    """Advanced time management for tournament play with complexity-based allocation"""
    
    def __init__(self, base_time: float, increment: float = 0.0):
        """
        Initialize time manager
        
        Args:
            base_time: Total time available (seconds)
            increment: Time increment per move (seconds)
        """
        self.base_time = base_time
        self.increment = increment
        self.time_used = 0.0
        self.moves_played = 0
        
        # Position complexity cache for faster repeated analysis
        self.position_complexity_cache: Dict[str, float] = {}
        
        # Time allocation settings
        self.emergency_time_threshold = 0.1  # 10% of total time
        self.complex_position_multiplier = 2.0
        self.simple_position_multiplier = 0.5
        self.endgame_time_bonus = 1.5
        
        # Adaptive depth settings
        self.base_depth = 5
        self.max_depth = 12
        self.time_depth_ratio = 2.0  # seconds per additional depth level
        
        # Statistics tracking
        self.allocation_stats = {
            'positions_analyzed': 0,
            'complex_positions': 0,
            'simple_positions': 0,
            'emergency_allocations': 0,
            'average_complexity': 0.0
        }
    
    def update_time_info(self, time_remaining: float, moves_played: int):
        """Update current time information"""
        self.time_used = self.base_time - time_remaining
        self.moves_played = moves_played
    
    def calculate_time_allocation(self, board: chess.Board, time_remaining: float) -> Tuple[float, int]:
        """
        Calculate optimal time allocation and target depth for current position
        
        Args:
            board: Current chess position
            time_remaining: Time remaining (seconds)
            
        Returns:
            Tuple of (time_to_use, target_depth)
        """
        # Analyze position complexity
        complexity_score = self._analyze_position_complexity(board)
        
        # Calculate base time allocation
        base_allocation = self._calculate_base_allocation(time_remaining)
        
        # Apply complexity modifier
        complexity_modifier = self._get_complexity_modifier(complexity_score)
        allocated_time = base_allocation * complexity_modifier
        
        # Apply emergency time management
        if self._is_emergency_time(time_remaining):
            allocated_time = self._apply_emergency_allocation(allocated_time, time_remaining)
        
        # Calculate target search depth
        target_depth = self._calculate_target_depth(allocated_time, complexity_score)
        
        # Ensure minimum and maximum bounds
        allocated_time = max(0.1, min(allocated_time, time_remaining * 0.8))
        target_depth = max(3, min(target_depth, self.max_depth))
        
        # Update statistics
        self._update_statistics(complexity_score, allocated_time)
        
        return allocated_time, target_depth
    
    def _analyze_position_complexity(self, board: chess.Board) -> float:
        """
        Analyze position complexity for time allocation decisions
        
        Returns:
            Complexity score (0.0 = simple, 1.0 = very complex)
        """
        # Check cache first
        position_fen = board.fen()
        if position_fen in self.position_complexity_cache:
            return self.position_complexity_cache[position_fen]
        
        complexity_factors = []
        
        # 1. Material complexity
        material_count = self._count_total_material(board)
        material_complexity = min(1.0, material_count / 7800)  # Normalize to opening material
        complexity_factors.append(material_complexity)
        
        # 2. Mobility complexity (number of legal moves)
        mobility = len(list(board.legal_moves))
        mobility_complexity = min(1.0, mobility / 40)  # Normalize to typical complex position
        complexity_factors.append(mobility_complexity)
        
        # 3. Tactical complexity (checks, captures, threats)
        tactical_complexity = self._analyze_tactical_complexity(board)
        complexity_factors.append(tactical_complexity)
        
        # 4. Pawn structure complexity
        pawn_complexity = self._analyze_pawn_complexity(board)
        complexity_factors.append(pawn_complexity)
        
        # 5. King safety complexity
        king_safety_complexity = self._analyze_king_safety_complexity(board)
        complexity_factors.append(king_safety_complexity)
        
        # Calculate weighted average
        weights = [0.2, 0.3, 0.3, 0.1, 0.1]  # Emphasize mobility and tactics
        complexity_score = sum(factor * weight for factor, weight in zip(complexity_factors, weights))
        
        # Cache the result
        self.position_complexity_cache[position_fen] = complexity_score
        
        return complexity_score
    
    def _analyze_tactical_complexity(self, board: chess.Board) -> float:
        """Analyze tactical complexity of position"""
        tactical_score = 0.0
        
        # Count checks available
        checks = sum(1 for move in board.legal_moves if board.gives_check(move))
        tactical_score += min(0.3, checks / 5)
        
        # Count captures available
        captures = sum(1 for move in board.legal_moves if board.is_capture(move))
        tactical_score += min(0.4, captures / 10)
        
        # Check if currently in check
        if board.is_check():
            tactical_score += 0.3
        
        return min(1.0, tactical_score)
    
    def _analyze_pawn_complexity(self, board: chess.Board) -> float:
        """Analyze pawn structure complexity"""
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Count pawn islands and passed pawns
        total_pawns = len(white_pawns) + len(black_pawns)
        if total_pawns == 0:
            return 0.0
        
        # Simple heuristic: fewer pawns = more complex endgame
        pawn_complexity = 1.0 - (total_pawns / 16)
        
        # Add complexity for advanced pawns
        advanced_pawns = 0
        for pawn in white_pawns:
            if chess.square_rank(pawn) >= 5:  # Advanced white pawn
                advanced_pawns += 1
        for pawn in black_pawns:
            if chess.square_rank(pawn) <= 2:  # Advanced black pawn
                advanced_pawns += 1
        
        pawn_complexity += min(0.3, advanced_pawns / 4)
        
        return min(1.0, pawn_complexity)
    
    def _analyze_king_safety_complexity(self, board: chess.Board) -> float:
        """Analyze king safety complexity"""
        complexity = 0.0
        
        # Check if kings are exposed (not castled)
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king and chess.square_file(white_king) in [4, 5]:  # King on e/f file
            complexity += 0.2
        if black_king and chess.square_file(black_king) in [4, 5]:
            complexity += 0.2
        
        # Check for potential king attacks
        if board.is_check():
            complexity += 0.6
        
        return min(1.0, complexity)
    
    def _count_total_material(self, board: chess.Board) -> int:
        """Count total material on board"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        total = 0
        for piece_type, value in piece_values.items():
            total += len(board.pieces(piece_type, chess.WHITE)) * value
            total += len(board.pieces(piece_type, chess.BLACK)) * value
        
        return total
    
    def _calculate_base_allocation(self, time_remaining: float) -> float:
        """Calculate base time allocation before modifiers"""
        # Estimate remaining moves based on game phase
        estimated_moves_left = max(10, 40 - self.moves_played)
        
        # Basic allocation: time_remaining / estimated_moves + increment
        base_allocation = (time_remaining / estimated_moves_left) + self.increment
        
        return base_allocation
    
    def _get_complexity_modifier(self, complexity_score: float) -> float:
        """Get time modifier based on position complexity"""
        if complexity_score < 0.3:
            return self.simple_position_multiplier
        elif complexity_score > 0.7:
            return self.complex_position_multiplier
        else:
            # Linear interpolation between simple and complex
            t = (complexity_score - 0.3) / 0.4
            return self.simple_position_multiplier + t * (self.complex_position_multiplier - self.simple_position_multiplier)
    
    def _is_emergency_time(self, time_remaining: float) -> bool:
        """Check if we're in emergency time situation"""
        return time_remaining < (self.base_time * self.emergency_time_threshold)
    
    def _apply_emergency_allocation(self, allocated_time: float, time_remaining: float) -> float:
        """Apply emergency time management"""
        self.allocation_stats['emergency_allocations'] += 1
        
        # In emergency, use minimum time but ensure we don't time out
        emergency_allocation = min(allocated_time, time_remaining * 0.1)
        return max(0.1, emergency_allocation)
    
    def _calculate_target_depth(self, allocated_time: float, complexity_score: float) -> int:
        """Calculate target search depth based on allocated time and complexity"""
        # Base depth calculation
        depth_from_time = self.base_depth + int(allocated_time / self.time_depth_ratio)
        
        # Complexity adjustment
        if complexity_score > 0.7:
            depth_adjustment = 1  # Search deeper for complex positions
        elif complexity_score < 0.3:
            depth_adjustment = -1  # Search less deep for simple positions
        else:
            depth_adjustment = 0
        
        target_depth = depth_from_time + depth_adjustment
        
        return max(3, min(target_depth, self.max_depth))
    
    def _update_statistics(self, complexity_score: float, allocated_time: float):
        """Update allocation statistics"""
        self.allocation_stats['positions_analyzed'] += 1
        
        if complexity_score > 0.7:
            self.allocation_stats['complex_positions'] += 1
        elif complexity_score < 0.3:
            self.allocation_stats['simple_positions'] += 1
        
        # Update running average of complexity
        total_positions = self.allocation_stats['positions_analyzed']
        current_avg = self.allocation_stats['average_complexity']
        new_avg = (current_avg * (total_positions - 1) + complexity_score) / total_positions
        self.allocation_stats['average_complexity'] = new_avg
    
    def get_statistics(self) -> Dict:
        """Get current allocation statistics"""
        return self.allocation_stats.copy()
    
    def reset_statistics(self):
        """Reset allocation statistics"""
        self.allocation_stats = {
            'positions_analyzed': 0,
            'complex_positions': 0,
            'simple_positions': 0,
            'emergency_allocations': 0,
            'average_complexity': 0.0
        }
    
    def should_extend_search(self, current_depth: int, time_used: float, allocated_time: float, 
                           best_move_stable: bool = False) -> bool:
        """
        Determine if search should be extended
        
        Args:
            current_depth: Current search depth completed
            time_used: Time already used in search
            allocated_time: Total time allocated for this move
            best_move_stable: Whether best move has been stable across depths
            
        Returns:
            True if search should continue
        """
        # Don't extend if we've used allocated time
        if time_used >= allocated_time:
            return False
        
        # Don't extend beyond maximum depth
        if current_depth >= self.max_depth:
            return False
        
        # If best move is stable and we've searched reasonable depth, consider stopping
        if best_move_stable and current_depth >= self.base_depth and time_used > allocated_time * 0.7:
            return False
        
        # If we have lots of time left, continue searching
        if time_used < allocated_time * 0.5:
            return True
        
        # Default: continue if within time budget
        return time_used < allocated_time * 0.9


# Example usage and testing
if __name__ == "__main__":
    # Test the time manager
    time_manager = V7P3RTimeManager(base_time=300.0, increment=3.0)  # 5+3 time control
    
    # Test different positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),  # Complex
        chess.Board("8/8/8/4k3/4K3/8/8/8 w - - 0 1")  # Simple endgame
    ]
    
    print("V7P3R Time Manager Test")
    print("=" * 40)
    
    for i, board in enumerate(test_positions):
        time_remaining = 250.0  # Simulate time remaining
        allocated_time, target_depth = time_manager.calculate_time_allocation(board, time_remaining)
        
        print(f"\nPosition {i+1}:")
        print(f"  FEN: {board.fen()[:50]}...")
        print(f"  Allocated Time: {allocated_time:.2f}s")
        print(f"  Target Depth: {target_depth}")
    
    print(f"\nStatistics: {time_manager.get_statistics()}")