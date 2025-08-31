#!/usr/bin/env python3
"""
V7P3R v9.4-beta Scoring Calculation  
Refined evaluation after v9.4-alpha analysis - focused on beating v7.0
Author: Pat Snyder

IMPROVEMENTS FROM v9.4-alpha ANALYSIS:
- Reduced tactical penalty harshness  
- Better balanced development bonuses
- Maintained strong endgame improvements
- Fine-tuned weights based on head-to-head results
"""

import chess
from typing import Dict


class V7P3RScoringCalculationV94Beta:
    """v9.4-beta: Refined evaluation to consistently beat v7.0"""
    
    def __init__(self, piece_values: Dict):
        self.piece_values = piece_values
        self._init_evaluation_tables()
    
    def _init_evaluation_tables(self):
        """Initialize refined evaluation tables"""
        
        # Core strategic squares
        self.center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        self.extended_center = {chess.C3, chess.C4, chess.C5, chess.C6,
                               chess.D3, chess.D6, chess.E3, chess.E6,
                               chess.F3, chess.F4, chess.F5, chess.F6}
        
        # Optimized development squares (based on analysis)
        self.knight_excellent = {
            chess.WHITE: {chess.F3, chess.C3, chess.E5, chess.D5},
            chess.BLACK: {chess.F6, chess.C6, chess.E4, chess.D4}
        }
        
        self.knight_good = {
            chess.WHITE: {chess.D2, chess.E2, chess.F6, chess.C6, chess.H3, chess.A3},
            chess.BLACK: {chess.D7, chess.E7, chess.F3, chess.C3, chess.H6, chess.A6}
        }
        
        self.bishop_excellent = {
            chess.WHITE: {chess.C4, chess.F4, chess.B5, chess.G5},
            chess.BLACK: {chess.C5, chess.F5, chess.B4, chess.G4}
        }
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Refined evaluation optimized to beat v7.0
        Returns positive values for good positions for the given color.
        """
        score = 0.0
        
        # 1. Material evaluation (unchanged - proven)
        score += self._material_score(board, color)
        
        # 2. King safety (refined weights)
        score += self._king_safety_refined(board, color)
        
        # 3. Development (rebalanced) 
        score += self._development_optimized(board, color)
        
        # 4. Tactical opportunities (reduced penalties, more bonuses)
        score += self._tactical_awareness(board, color)
        
        # 5. Strategic advantages (new component)
        score += self._strategic_bonuses(board, color)
        
        # 6. Endgame improvements (proven strong)
        if self._is_endgame(board):
            score += self._endgame_refined(board, color)
            
        return score
    
    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Material count - identical to v7.0"""
        score = 0.0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            piece_count = len(board.pieces(piece_type, color))
            score += piece_count * self.piece_values.get(piece_type, 0)
        return score
    
    def _king_safety_refined(self, board: chess.Board, color: chess.Color) -> float:
        """Refined king safety - more balanced than alpha"""
        score = 0.0
        king_square = board.king(color)
        if not king_square:
            return -1000.0
            
        # Basic exposure (same as v7.0)
        if self._is_king_exposed(board, color, king_square):
            score -= 30.0  # Reduced from 50.0
            
        # Castling evaluation (refined)
        if board.has_castling_rights(color):
            score += 10.0  # Rights available
        elif self._is_castled(board, color, king_square):
            score += 25.0  # Successfully castled
            
        # Attack detection (reduced impact)
        attack_count = self._count_attacks_near_king(board, color, king_square)
        score -= attack_count * 4.0  # Reduced from 8.0
            
        return score
    
    def _development_optimized(self, board: chess.Board, color: chess.Color) -> float:
        """Optimized development evaluation"""
        score = 0.0
        
        # Knight development (refined bonuses)
        for square in board.pieces(chess.KNIGHT, color):
            if square in self.knight_excellent[color]:
                score += 25.0  # Excellent squares
            elif square in self.knight_good[color]:
                score += 12.0  # Good squares
            elif (color == chess.WHITE and chess.square_rank(square) == 0) or \
                 (color == chess.BLACK and chess.square_rank(square) == 7):
                score -= 5.0   # Back rank penalty (reduced)
                
        # Bishop development (refined)
        for square in board.pieces(chess.BISHOP, color):
            if square in self.bishop_excellent[color]:
                score += 20.0  # Excellent diagonals
            elif (color == chess.WHITE and chess.square_rank(square) > 1) or \
                 (color == chess.BLACK and chess.square_rank(square) < 6):
                score += 8.0   # Any development
            elif (color == chess.WHITE and chess.square_rank(square) == 0) or \
                 (color == chess.BLACK and chess.square_rank(square) == 7):
                score -= 3.0   # Back rank penalty (reduced)
                
        # Center control (enhanced but balanced)
        score += self._center_control_balanced(board, color)
        
        return score
    
    def _tactical_awareness(self, board: chess.Board, color: chess.Color) -> float:
        """Tactical awareness - more bonuses, fewer penalties"""
        score = 0.0
        
        # Fork opportunities (enhanced detection)
        score += self._detect_tactical_motifs(board, color)
        
        # Hanging pieces (reduced penalty)
        hanging_penalty = self._count_hanging_pieces(board, color) * 15.0  # Reduced from 25.0
        score -= hanging_penalty
        
        # Attack coordination (new bonus)
        score += self._attack_coordination(board, color)
        
        return score
    
    def _strategic_bonuses(self, board: chess.Board, color: chess.Color) -> float:
        """Strategic bonuses to exploit v7.0 weaknesses"""
        score = 0.0
        
        # Piece coordination
        score += self._piece_coordination(board, color)
        
        # Pawn structure basics
        score += self._basic_pawn_structure(board, color)
        
        # Control of key squares
        score += self._key_square_control(board, color)
        
        return score
    
    def _center_control_balanced(self, board: chess.Board, color: chess.Color) -> float:
        """Balanced center control evaluation"""
        score = 0.0
        
        # Core center (most important)
        for square in self.center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if piece.piece_type == chess.PAWN:
                    score += 15.0  # Strong central pawn
                else:
                    score += 8.0   # Central piece
            elif board.is_attacked_by(color, square):
                score += 2.0   # Control without occupation
                
        # Extended center support
        extended_control = sum(1 for sq in self.extended_center if board.is_attacked_by(color, sq))
        score += extended_control * 0.5
                
        return score
    
    def _detect_tactical_motifs(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced tactical pattern detection"""
        score = 0.0
        
        # Knight fork detection (improved)
        for square in board.pieces(chess.KNIGHT, color):
            attacks = board.attacks(square)
            high_value_targets = 0
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                if target_piece and target_piece.color != color:
                    if target_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                        high_value_targets += 1
                    elif target_piece.piece_type == chess.KING:
                        score += 5.0  # Check bonus
            
            if high_value_targets >= 2:
                score += 30.0  # Major fork opportunity
            elif high_value_targets == 1:
                score += 10.0  # Potential tactic
                
        # Pin detection (simplified but effective)
        score += self._detect_pins_simple(board, color)
        
        return score
    
    def _attack_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Bonus for coordinated attacks"""
        score = 0.0
        
        # Count squares attacked by multiple pieces
        multi_attacked = 0
        for square in chess.SQUARES:
            attackers = len(board.attackers(color, square))
            if attackers >= 2:
                multi_attacked += 1
                
        score += multi_attacked * 0.5  # Small bonus for coordination
        return score
    
    def _piece_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Piece coordination bonuses"""
        score = 0.0
        
        # Rooks on same file/rank
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) >= 2:
            r1, r2 = rooks[0], rooks[1]
            if chess.square_file(r1) == chess.square_file(r2) or \
               chess.square_rank(r1) == chess.square_rank(r2):
                score += 15.0  # Rook coordination
        
        # Bishops controlling different colored squares
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) >= 2:
            colors = [(chess.square_file(b) + chess.square_rank(b)) % 2 for b in bishops]
            if len(set(colors)) == 2:  # One light, one dark square bishop
                score += 12.0  # Bishop pair bonus
                
        return score
    
    def _basic_pawn_structure(self, board: chess.Board, color: chess.Color) -> float:
        """Basic pawn structure evaluation"""
        score = 0.0
        
        pawn_files = {}
        for square in board.pieces(chess.PAWN, color):
            file_idx = chess.square_file(square)
            if file_idx not in pawn_files:
                pawn_files[file_idx] = []
            pawn_files[file_idx].append(square)
        
        # Doubled pawn penalty
        for file_idx, pawns in pawn_files.items():
            if len(pawns) > 1:
                score -= 8.0 * (len(pawns) - 1)  # Penalty for doubled pawns
        
        # Passed pawn detection
        for square in board.pieces(chess.PAWN, color):
            if self._is_passed_pawn(board, square, color):
                rank = chess.square_rank(square)
                advancement = rank if color == chess.WHITE else (7 - rank)
                score += advancement * 5.0  # Bonus increases as pawn advances
                
        return score
    
    def _key_square_control(self, board: chess.Board, color: chess.Color) -> float:
        """Control of strategically important squares"""
        score = 0.0
        
        # Control of squares in front of enemy king
        enemy_king = board.king(not color)
        if enemy_king:
            king_file = chess.square_file(enemy_king)
            king_rank = chess.square_rank(enemy_king)
            
            # Check control of squares in front of enemy king
            target_ranks = [king_rank + 1, king_rank - 1] if color == chess.WHITE else [king_rank - 1, king_rank + 1]
            for target_rank in target_ranks:
                if 0 <= target_rank <= 7:
                    for file_offset in [-1, 0, 1]:
                        target_file = king_file + file_offset
                        if 0 <= target_file <= 7:
                            target_square = chess.square(target_file, target_rank)
                            if board.is_attacked_by(color, target_square):
                                score += 3.0
        
        return score
    
    def _detect_pins_simple(self, board: chess.Board, color: chess.Color) -> float:
        """Simplified but effective pin detection"""
        score = 0.0
        
        for piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                # Check each attacked square for potential pins
                attacks = board.attacks(square)
                for direction in attacks:
                    piece_at_direction = board.piece_at(direction)
                    if piece_at_direction and piece_at_direction.color != color:
                        # Simple pin bonus
                        score += 3.0
                        
        return score
    
    def _is_passed_pawn(self, board: chess.Board, pawn_square: int, color: chess.Color) -> bool:
        """Check if pawn is passed (simplified)"""
        file_idx = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check for enemy pawns blocking advancement
        for enemy_square in board.pieces(chess.PAWN, not color):
            enemy_file = chess.square_file(enemy_square)
            enemy_rank = chess.square_rank(enemy_square)
            
            # Check if enemy pawn can block this pawn's advancement
            if abs(enemy_file - file_idx) <= 1:  # Adjacent or same file
                if color == chess.WHITE and enemy_rank > rank:
                    return False
                elif color == chess.BLACK and enemy_rank < rank:
                    return False
                    
        return True
    
    def _count_hanging_pieces(self, board: chess.Board, color: chess.Color) -> float:
        """Count undefended pieces (refined)"""
        hanging_count = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                # More sophisticated hanging piece detection
                if board.is_attacked_by(not color, square):
                    defenders = len(board.attackers(color, square))
                    attackers = len(board.attackers(not color, square))
                    
                    if defenders == 0 or attackers > defenders:
                        hanging_count += 1
                    
        return hanging_count
    
    def _endgame_refined(self, board: chess.Board, color: chess.Color) -> float:
        """Refined endgame logic (proven strong in alpha)"""
        score = 0.0
        
        # King activity (enhanced)
        king_square = board.king(color)
        if king_square:
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            center_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
            score += (7 - center_distance) * 6  # Increased weight
            
        # Pawn promotion threats (proven very effective)
        for square in board.pieces(chess.PAWN, color):
            if color == chess.WHITE:
                distance_to_promotion = 7 - chess.square_rank(square)
            else:
                distance_to_promotion = chess.square_rank(square)
                
            if distance_to_promotion <= 3:  # Extended range
                promotion_bonus = 20.0 * (4 - distance_to_promotion)
                score += promotion_bonus
                
        return score
    
    # Helper methods (refined from alpha)
    def _is_king_exposed(self, board: chess.Board, color: chess.Color, king_square: int) -> bool:
        """Check if king is dangerously exposed"""
        if color == chess.WHITE:
            return chess.square_rank(king_square) > 2
        else:
            return chess.square_rank(king_square) < 5
    
    def _is_castled(self, board: chess.Board, color: chess.Color, king_square: int) -> bool:
        """Check if king has castled"""
        if color == chess.WHITE:
            return king_square in [chess.G1, chess.C1]
        else:
            return king_square in [chess.G8, chess.C8]
    
    def _count_attacks_near_king(self, board: chess.Board, color: chess.Color, king_square: int) -> int:
        """Count enemy attacks near the king"""
        attack_count = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                if file_offset == 0 and rank_offset == 0:
                    continue
                    
                check_file = king_file + file_offset
                check_rank = king_rank + rank_offset
                
                if 0 <= check_file <= 7 and 0 <= check_rank <= 7:
                    check_square = chess.square(check_file, check_rank)
                    if board.is_attacked_by(not color, check_square):
                        attack_count += 1
                        
        return attack_count
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect endgame phase"""
        major_pieces = 0
        for color in [chess.WHITE, chess.BLACK]:
            major_pieces += len(board.pieces(chess.QUEEN, color)) * 2
            major_pieces += len(board.pieces(chess.ROOK, color))
        return major_pieces <= 6
