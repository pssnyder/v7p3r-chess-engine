#!/usr/bin/env python3
"""
V7P3R v11 Phase 3A - Advanced Pawn Structure Evaluator
Enhanced pawn evaluation with structural patterns and strategic assessment
Author: Pat Snyder
"""

import chess
from typing import Dict, List


class V7P3RAdvancedPawnEvaluator:
    """Advanced pawn structure evaluation for strategic play enhancement"""
    
    def __init__(self):
        # Pawn evaluation bonuses/penalties
        self.passed_pawn_bonus = [0, 20, 30, 50, 80, 120, 180, 250]  # By rank
        self.isolated_pawn_penalty = 15
        self.doubled_pawn_penalty = 25
        self.backward_pawn_penalty = 12
        self.connected_pawn_bonus = 8
        self.pawn_chain_bonus = 5
        
        # Advanced pawn structure patterns
        self.pawn_storm_bonus = 10
        self.pawn_shelter_bonus = 15
        self.advanced_passed_bonus = 30
        
    def evaluate_pawn_structure(self, board: chess.Board, color: bool) -> float:
        """
        Comprehensive pawn structure evaluation
        Returns score from the perspective of the given color
        """
        total_score = 0.0
        
        # Get all pawns for this color
        pawns = board.pieces(chess.PAWN, color)
        
        # Evaluate each pawn and overall structure
        total_score += self._evaluate_passed_pawns(board, pawns, color)
        total_score += self._evaluate_isolated_pawns(board, pawns, color)
        total_score += self._evaluate_doubled_pawns(board, pawns, color)
        total_score += self._evaluate_backward_pawns(board, pawns, color)
        total_score += self._evaluate_connected_pawns(board, pawns, color)
        total_score += self._evaluate_pawn_chains(board, pawns, color)
        total_score += self._evaluate_pawn_storms(board, pawns, color)
        
        return total_score
    
    def _evaluate_passed_pawns(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate passed pawns with rank-based bonuses"""
        score = 0.0
        
        for pawn_square in pawns:
            if self._is_passed_pawn(board, pawn_square, color):
                rank = chess.square_rank(pawn_square)
                if not color:  # Black pawns
                    rank = 7 - rank
                
                # Base passed pawn bonus
                bonus = self.passed_pawn_bonus[rank]
                
                # Advanced passed pawn (6th rank or higher)
                if rank >= 5:
                    bonus += self.advanced_passed_bonus
                
                # Connected passed pawns get extra bonus
                if self._has_connected_passed_pawn(board, pawn_square, color):
                    bonus += 20
                
                score += bonus
        
        return score
    
    def _evaluate_isolated_pawns(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate isolated pawns (no friendly pawns on adjacent files)"""
        score = 0.0
        
        for pawn_square in pawns:
            if self._is_isolated_pawn(board, pawn_square, color):
                penalty = self.isolated_pawn_penalty
                
                # Isolated pawns on open files are worse
                if self._is_on_open_file(board, pawn_square):
                    penalty += 10
                
                score -= penalty
        
        return score
    
    def _evaluate_doubled_pawns(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate doubled pawns on the same file"""
        score = 0.0
        file_counts = {}
        
        # Count pawns per file
        for pawn_square in pawns:
            file_idx = chess.square_file(pawn_square)
            file_counts[file_idx] = file_counts.get(file_idx, 0) + 1
        
        # Penalize multiple pawns on same file
        for file_idx, count in file_counts.items():
            if count > 1:
                # Penalty increases with more pawns
                penalty = self.doubled_pawn_penalty * (count - 1)
                score -= penalty
        
        return score
    
    def _evaluate_backward_pawns(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate backward pawns (cannot advance safely)"""
        score = 0.0
        
        for pawn_square in pawns:
            if self._is_backward_pawn(board, pawn_square, color):
                penalty = self.backward_pawn_penalty
                
                # Backward pawns on semi-open files are worse
                if self._is_on_semi_open_file(board, pawn_square, color):
                    penalty += 8
                
                score -= penalty
        
        return score
    
    def _evaluate_connected_pawns(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate connected pawns (pawns protecting each other)"""
        score = 0.0
        
        for pawn_square in pawns:
            if self._has_pawn_support(board, pawn_square, color):
                score += self.connected_pawn_bonus
        
        return score
    
    def _evaluate_pawn_chains(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate pawn chains (connected pawns in diagonal formation)"""
        score = 0.0
        chain_lengths = self._find_pawn_chains(board, pawns, color)
        
        for length in chain_lengths:
            if length >= 2:
                # Bonus increases with chain length
                score += self.pawn_chain_bonus * length
        
        return score
    
    def _evaluate_pawn_storms(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate pawn storms (advancing pawns toward enemy king)"""
        score = 0.0
        
        # Find enemy king
        enemy_king_square = board.king(not color)
        if enemy_king_square is None:
            return 0.0
        
        enemy_king_file = chess.square_file(enemy_king_square)
        
        for pawn_square in pawns:
            pawn_file = chess.square_file(pawn_square)
            pawn_rank = chess.square_rank(pawn_square)
            
            # Check if pawn is advancing toward enemy king
            if abs(pawn_file - enemy_king_file) <= 1:  # Adjacent or same file
                if color and pawn_rank >= 4:  # White pawn advanced
                    score += self.pawn_storm_bonus
                elif not color and pawn_rank <= 3:  # Black pawn advanced
                    score += self.pawn_storm_bonus
        
        return score
    
    # Helper methods for pawn structure analysis
    
    def _is_passed_pawn(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is passed (no enemy pawns blocking its advance)"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Check for enemy pawns that could block or capture
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        for enemy_pawn in enemy_pawns:
            enemy_file = chess.square_file(enemy_pawn)
            enemy_rank = chess.square_rank(enemy_pawn)
            
            # Check if enemy pawn can block or capture
            if abs(enemy_file - pawn_file) <= 1:  # Same or adjacent file
                if color and enemy_rank > pawn_rank:  # White pawn
                    return False
                elif not color and enemy_rank < pawn_rank:  # Black pawn
                    return False
        
        return True
    
    def _is_isolated_pawn(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is isolated (no friendly pawns on adjacent files)"""
        pawn_file = chess.square_file(pawn_square)
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        # Check adjacent files for friendly pawns
        for friendly_pawn in friendly_pawns:
            friendly_file = chess.square_file(friendly_pawn)
            if abs(friendly_file - pawn_file) == 1:
                return False
        
        return True
    
    def _is_backward_pawn(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is backward (cannot advance safely)"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Check if pawn can advance without being captured
        advance_square = pawn_square + (8 if color else -8)
        
        if not (0 <= advance_square <= 63):
            return False
        
        # Check if advance square is attacked by enemy pawns
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        for enemy_pawn in enemy_pawns:
            enemy_file = chess.square_file(enemy_pawn)
            enemy_rank = chess.square_rank(enemy_pawn)
            
            # Check if enemy pawn attacks advance square
            if abs(enemy_file - pawn_file) == 1:
                if color and enemy_rank == pawn_rank + 2:  # White pawn
                    return True
                elif not color and enemy_rank == pawn_rank - 2:  # Black pawn
                    return True
        
        return False
    
    def _has_pawn_support(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn has friendly pawn support"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        # Check for supporting pawns on adjacent files
        for friendly_pawn in friendly_pawns:
            if friendly_pawn == pawn_square:
                continue
                
            friendly_file = chess.square_file(friendly_pawn)
            friendly_rank = chess.square_rank(friendly_pawn)
            
            # Check if friendly pawn provides support
            if abs(friendly_file - pawn_file) == 1:
                if color and friendly_rank == pawn_rank - 1:  # White support
                    return True
                elif not color and friendly_rank == pawn_rank + 1:  # Black support
                    return True
        
        return False
    
    def _has_connected_passed_pawn(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if passed pawn has a connected passed pawn"""
        if not self._is_passed_pawn(board, pawn_square, color):
            return False
        
        pawn_file = chess.square_file(pawn_square)
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        # Check adjacent files for connected passed pawns
        for friendly_pawn in friendly_pawns:
            if friendly_pawn == pawn_square:
                continue
                
            friendly_file = chess.square_file(friendly_pawn)
            
            if abs(friendly_file - pawn_file) == 1:
                if self._is_passed_pawn(board, friendly_pawn, color):
                    return True
        
        return False
    
    def _is_on_open_file(self, board: chess.Board, pawn_square: int) -> bool:
        """Check if pawn is on an open file (no pawns of either color)"""
        pawn_file = chess.square_file(pawn_square)
        
        all_pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
        
        for pawn in all_pawns:
            if pawn != pawn_square and chess.square_file(pawn) == pawn_file:
                return False
        
        return True
    
    def _is_on_semi_open_file(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is on a semi-open file (no enemy pawns on file)"""
        pawn_file = chess.square_file(pawn_square)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        for enemy_pawn in enemy_pawns:
            if chess.square_file(enemy_pawn) == pawn_file:
                return False
        
        return True
    
    def _find_pawn_chains(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> List[int]:
        """Find pawn chains and return their lengths"""
        chains = []
        visited = set()
        
        for pawn_square in pawns:
            if pawn_square in visited:
                continue
            
            chain_length = self._get_chain_length(board, pawn_square, color, visited)
            if chain_length > 1:
                chains.append(chain_length)
        
        return chains
    
    def _get_chain_length(self, board: chess.Board, start_pawn: int, color: bool, visited: set) -> int:
        """Get the length of a pawn chain starting from a given pawn"""
        if start_pawn in visited:
            return 0
        
        visited.add(start_pawn)
        length = 1
        
        # Check for connected pawns
        pawn_file = chess.square_file(start_pawn)
        pawn_rank = chess.square_rank(start_pawn)
        
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        for friendly_pawn in friendly_pawns:
            if friendly_pawn in visited:
                continue
                
            friendly_file = chess.square_file(friendly_pawn)
            friendly_rank = chess.square_rank(friendly_pawn)
            
            # Check if pawns are connected diagonally
            if abs(friendly_file - pawn_file) == 1 and abs(friendly_rank - pawn_rank) == 1:
                length += self._get_chain_length(board, friendly_pawn, color, visited)
        
        return length
