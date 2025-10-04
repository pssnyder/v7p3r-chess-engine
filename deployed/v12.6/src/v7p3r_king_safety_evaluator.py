#!/usr/bin/env python3
"""
V7P3R v11 Phase 3A - Enhanced King Safety Evaluator
Advanced king safety evaluation with pawn shelter, escape squares, and attack assessment
Author: Pat Snyder
"""

import chess
from typing import List, Dict


class V7P3RKingSafetyEvaluator:
    """Enhanced king safety evaluation for better positional assessment"""
    
    def __init__(self):
        # King safety bonuses/penalties
        self.pawn_shelter_bonus = [0, 5, 10, 15, 20]  # By number of shelter pawns
        self.castling_rights_bonus = 25
        self.king_exposure_penalty = 30
        self.escape_square_bonus = 8
        self.attack_zone_penalty = 12
        
        # King activity bonuses (for endgame)
        self.king_activity_bonus = 5
        self.king_centralization_bonus = [0, 2, 4, 8, 12, 8, 4, 2]  # By rank/file distance from center
        
        # Pawn storm penalties
        self.enemy_pawn_storm_penalty = 15
        self.advanced_enemy_pawn_penalty = 10
        
    def evaluate_king_safety(self, board: chess.Board, color: bool) -> float:
        """
        Comprehensive king safety evaluation
        Returns score from the perspective of the given color
        """
        total_score = 0.0
        
        king_square = board.king(color)
        if king_square is None:
            return -1000.0  # King missing - critical error
        
        # Determine game phase for king safety vs activity balance
        material_count = self._count_material(board)
        is_endgame = material_count < 2000  # Rough endgame threshold
        
        if is_endgame:
            # Endgame: King activity is important
            total_score += self._evaluate_king_activity(board, king_square, color)
        else:
            # Opening/Middlegame: King safety is paramount
            total_score += self._evaluate_pawn_shelter(board, king_square, color)
            total_score += self._evaluate_castling_rights(board, color)
            total_score += self._evaluate_king_exposure(board, king_square, color)
            total_score += self._evaluate_escape_squares(board, king_square, color)
            total_score += self._evaluate_attack_zone(board, king_square, color)
            total_score += self._evaluate_enemy_pawn_storms(board, king_square, color)
        
        return total_score
    
    def _evaluate_pawn_shelter(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate pawn shelter around the king"""
        score = 0.0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        friendly_pawns = board.pieces(chess.PAWN, color)
        shelter_pawns = 0
        
        # Check files around the king (king file and adjacent files)
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            if 0 <= check_file <= 7:
                # Look for pawn shelter in front of king
                shelter_found = False
                for pawn_square in friendly_pawns:
                    pawn_file = chess.square_file(pawn_square)
                    pawn_rank = chess.square_rank(pawn_square)
                    
                    if pawn_file == check_file:
                        # Check if pawn is providing shelter
                        if color and pawn_rank > king_rank:  # White king
                            shelter_found = True
                            # Closer pawns provide better shelter
                            if pawn_rank - king_rank <= 2:
                                shelter_pawns += 1
                            break
                        elif not color and pawn_rank < king_rank:  # Black king
                            shelter_found = True
                            if king_rank - pawn_rank <= 2:
                                shelter_pawns += 1
                            break
                
                # Penalty for missing pawn shelter
                if not shelter_found:
                    score -= 10
        
        # Bonus for pawn shelter
        if shelter_pawns < len(self.pawn_shelter_bonus):
            score += self.pawn_shelter_bonus[shelter_pawns]
        else:
            score += self.pawn_shelter_bonus[-1]
        
        return score
    
    def _evaluate_castling_rights(self, board: chess.Board, color: bool) -> float:
        """Evaluate castling rights value"""
        score = 0.0
        
        if color:  # White
            if board.has_kingside_castling_rights(chess.WHITE):
                score += self.castling_rights_bonus
            if board.has_queenside_castling_rights(chess.WHITE):
                score += self.castling_rights_bonus * 0.8  # Queenside slightly less valuable
        else:  # Black
            if board.has_kingside_castling_rights(chess.BLACK):
                score += self.castling_rights_bonus
            if board.has_queenside_castling_rights(chess.BLACK):
                score += self.castling_rights_bonus * 0.8
        
        return score
    
    def _evaluate_king_exposure(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate king exposure to enemy attacks"""
        score = 0.0
        
        # Check if king is on an open file or rank
        if self._is_on_open_file(board, king_square):
            score -= self.king_exposure_penalty
        
        if self._is_on_open_rank(board, king_square):
            score -= self.king_exposure_penalty * 0.5  # Less dangerous than open file
        
        # Check for enemy pieces attacking king vicinity
        enemy_attacks = self._count_enemy_attacks_near_king(board, king_square, color)
        score -= enemy_attacks * 5
        
        return score
    
    def _evaluate_escape_squares(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate available escape squares for the king"""
        score = 0.0
        escape_squares = 0
        
        # Check all adjacent squares
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue  # Skip king's current square
                
                target_square = king_square + rank_offset * 8 + file_offset
                
                if 0 <= target_square <= 63:
                    # Check if square is safe and accessible
                    if self._is_safe_escape_square(board, target_square, color):
                        escape_squares += 1
        
        score += escape_squares * self.escape_square_bonus
        
        # Penalty for having very few escape squares
        if escape_squares <= 1:
            score -= 20
        
        return score
    
    def _evaluate_attack_zone(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate enemy control of squares around the king"""
        score = 0.0
        
        # Define attack zone (2x2 squares around king)
        attack_zone_squares = []
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    target_square = target_rank * 8 + target_file
                    attack_zone_squares.append(target_square)
        
        # Count enemy attacks in the zone
        enemy_controlled = 0
        for square in attack_zone_squares:
            if self._is_attacked_by_enemy(board, square, color):
                enemy_controlled += 1
        
        score -= enemy_controlled * self.attack_zone_penalty
        
        return score
    
    def _evaluate_enemy_pawn_storms(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate enemy pawn storms advancing toward the king"""
        score = 0.0
        king_file = chess.square_file(king_square)
        
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        for pawn_square in enemy_pawns:
            pawn_file = chess.square_file(pawn_square)
            pawn_rank = chess.square_rank(pawn_square)
            
            # Check if pawn is advancing toward our king
            if abs(pawn_file - king_file) <= 1:  # Same or adjacent file
                storm_threat = 0
                
                if color:  # White king, black pawns advancing
                    if pawn_rank <= 3:  # Advanced black pawn
                        storm_threat = self.enemy_pawn_storm_penalty
                        if pawn_rank <= 2:  # Very advanced
                            storm_threat += self.advanced_enemy_pawn_penalty
                else:  # Black king, white pawns advancing
                    if pawn_rank >= 4:  # Advanced white pawn
                        storm_threat = self.enemy_pawn_storm_penalty
                        if pawn_rank >= 5:  # Very advanced
                            storm_threat += self.advanced_enemy_pawn_penalty
                
                score -= storm_threat
        
        return score
    
    def _evaluate_king_activity(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate king activity in endgame positions"""
        score = 0.0
        
        # King centralization bonus
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Distance from center (3.5, 3.5)
        center_distance = max(abs(king_file - 3.5), abs(king_rank - 3.5))
        centralization_index = int(center_distance)
        
        if centralization_index < len(self.king_centralization_bonus):
            score += self.king_centralization_bonus[centralization_index]
        
        # King mobility bonus - calculate moves for this color's king
        # We need to temporarily set the turn to calculate legal moves correctly
        original_turn = board.turn
        board.turn = color
        king_moves = len([move for move in board.legal_moves if move.from_square == king_square])
        board.turn = original_turn  # Restore original turn
        
        score += king_moves * self.king_activity_bonus
        
        return score
    
    # Helper methods
    
    def _count_material(self, board: chess.Board) -> int:
        """Count total material on the board"""
        piece_values = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, 
                       chess.ROOK: 500, chess.QUEEN: 900}
        
        total = 0
        for piece_type, value in piece_values.items():
            total += len(board.pieces(piece_type, chess.WHITE)) * value
            total += len(board.pieces(piece_type, chess.BLACK)) * value
        
        return total
    
    def _is_on_open_file(self, board: chess.Board, square: int) -> bool:
        """Check if square is on an open file"""
        file_idx = chess.square_file(square)
        
        all_pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
        
        for pawn in all_pawns:
            if chess.square_file(pawn) == file_idx:
                return False
        
        return True
    
    def _is_on_open_rank(self, board: chess.Board, square: int) -> bool:
        """Check if square is on an open rank"""
        rank_idx = chess.square_rank(square)
        
        # Check if there are any pieces on the same rank
        for file_idx in range(8):
            check_square = rank_idx * 8 + file_idx
            if check_square != square and board.piece_at(check_square) is not None:
                return False
        
        return True
    
    def _count_enemy_attacks_near_king(self, board: chess.Board, king_square: int, color: bool) -> int:
        """Count enemy piece attacks near the king"""
        attacks = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check 3x3 area around king
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    target_square = target_rank * 8 + target_file
                    if self._is_attacked_by_enemy(board, target_square, color):
                        attacks += 1
        
        return attacks
    
    def _is_safe_escape_square(self, board: chess.Board, square: int, color: bool) -> bool:
        """Check if a square is a safe escape square for the king"""
        # Square must be empty or contain enemy piece that can be captured
        piece = board.piece_at(square)
        if piece and piece.color == color:
            return False  # Occupied by friendly piece
        
        # Square must not be attacked by enemy
        if self._is_attacked_by_enemy(board, square, color):
            return False
        
        return True
    
    def _is_attacked_by_enemy(self, board: chess.Board, square: int, our_color: bool) -> bool:
        """Check if a square is attacked by enemy pieces"""
        # This is a simplified version - in a full implementation,
        # you'd want to check all enemy piece attacks efficiently
        enemy_color = not our_color
        
        # Check for enemy pawn attacks
        enemy_pawns = board.pieces(chess.PAWN, enemy_color)
        for pawn in enemy_pawns:
            if self._pawn_attacks_square(pawn, square, enemy_color):
                return True
        
        # Check for enemy piece attacks
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            enemy_pieces = board.pieces(piece_type, enemy_color)
            for piece_square in enemy_pieces:
                if square in board.attacks(piece_square):
                    return True
        
        return False
    
    def _pawn_attacks_square(self, pawn_square: int, target_square: int, pawn_color: bool) -> bool:
        """Check if a pawn attacks a specific square"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        target_file = chess.square_file(target_square)
        target_rank = chess.square_rank(target_square)
        
        # Check diagonal pawn attacks
        if abs(target_file - pawn_file) == 1:
            if pawn_color and target_rank == pawn_rank + 1:  # White pawn
                return True
            elif not pawn_color and target_rank == pawn_rank - 1:  # Black pawn
                return True
        
        return False
