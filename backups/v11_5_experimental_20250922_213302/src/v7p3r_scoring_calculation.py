#!/usr/bin/env python3
"""
V7P3R Scoring Calculation v10.0 - Tactical Enhanced Edition
Clean evaluation module with comprehensive tactical pattern recognition
Author: Pat Snyder

TACTICAL ENHANCEMENTS:
- Pin detection and exploitation
- Fork detection (all pieces, not just knights)
- Skewer detection
- Discovered attack detection  
- Deflection opportunities
- Removing the guard tactics
- Double check detection
- Battery formation (double piece attacks)
- Piece defense heuristics for tie-breaking
- Enhanced endgame patterns (edge-driving + centralization)
"""

import chess
from typing import Dict, List, Set, Tuple, Optional


class V7P3RScoringCalculation:
    """Enhanced scoring with comprehensive tactical pattern recognition"""
    
    def __init__(self, piece_values: Dict):
        self.piece_values = piece_values
        self._init_tactical_tables()
    
    def _init_tactical_tables(self):
        """Initialize tactical pattern recognition tables"""
        
        # Strategic squares
        self.center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        self.extended_center = {chess.C3, chess.C4, chess.C5, chess.C6,
                               chess.D3, chess.D6, chess.E3, chess.E6,
                               chess.F3, chess.F4, chess.F5, chess.F6}
        
        # Edge squares for king-driving in endgame
        self.edge_squares = set()
        for rank in [0, 7]:
            for file in range(8):
                self.edge_squares.add(chess.square(file, rank))
        for file in [0, 7]:
            for rank in range(8):
                self.edge_squares.add(chess.square(file, rank))
                
        # Corner squares (worst for enemy king in endgame)
        self.corner_squares = {chess.A1, chess.A8, chess.H1, chess.H8}
        
        # Piece values for tactical calculations
        self.tactical_piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Enhanced evaluation with comprehensive tactical awareness.
        Returns positive values for good positions for the given color.
        """
        score = 0.0
        
        # 1. Core evaluation (V10 foundation)
        score += self._material_score(board, color)
        score += self._king_safety(board, color)
        score += self._piece_development(board, color)
        score += self._castling_bonus(board, color)
        score += self._rook_coordination(board, color)
        score += self._center_control(board, color)
        
        # NEW: Piece defense heuristics (STUBBED for performance testing)
        # score += self._piece_defense_coordination(board, color)
        
        # Enhanced endgame logic (TESTING: temporarily simplified)
        if self._is_endgame(board):
            score += self._endgame_logic(board, color)  # Use simple version
        else:
            score += self._endgame_logic(board, color)
            
        return score
    
    def _material_score(self, board: chess.Board, color: chess.Color) -> float:
        """Material count for given color - C0BR4 style"""
        score = 0.0
        for piece_type, value in self.piece_values.items():
            if piece_type != chess.KING:  # King safety handled separately
                piece_count = len(board.pieces(piece_type, color))
                score += piece_count * value
        return score
    
    def _king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """Basic king safety - C0BR4 style"""
        score = 0.0
        king_square = board.king(color)
        if not king_square:
            return -1000.0  # No king = very bad
            
        # Penalty for exposed king
        if self._is_king_exposed(board, color, king_square):
            score -= 50.0
            
        return score
    
    def _piece_development(self, board: chess.Board, color: chess.Color) -> float:
        """Piece development bonus - C0BR4 PST equivalent"""
        score = 0.0
        
        # Bonus for developed knights and bishops
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for square in board.pieces(piece_type, color):
                # Bonus for pieces not on back rank
                if color == chess.WHITE and chess.square_rank(square) > 0:
                    score += 5.0
                elif color == chess.BLACK and chess.square_rank(square) < 7:
                    score += 5.0
                    
        return score
    
    def _castling_bonus(self, board: chess.Board, color: chess.Color) -> float:
        """Castling bonus - C0BR4 style"""
        score = 0.0
        king_square = board.king(color)
        
        if king_square:
            # Bonus for castled king
            if color == chess.WHITE and king_square in [chess.G1, chess.C1]:
                score += 20.0
            elif color == chess.BLACK and king_square in [chess.G8, chess.C8]:
                score += 20.0
        
        return score
    
    def _rook_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Rook coordination - C0BR4 style"""
        score = 0.0
        rooks = list(board.pieces(chess.ROOK, color))
        
        if len(rooks) >= 2:
            # Simple bonus for having both rooks
            score += 10.0
            
        return score
    
    def _center_control(self, board: chess.Board, color: chess.Color) -> float:
        """Center control bonus"""
        score = 0.0
        
        for square in self.center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if piece.piece_type == chess.PAWN:
                    score += 10.0  # Pawn in center
                else:
                    score += 5.0   # Other piece in center
                    
        return score
    
    def _endgame_logic(self, board: chess.Board, color: chess.Color) -> float:
        """Basic endgame logic"""
        score = 0.0
        
        # King activity in endgame
        king_square = board.king(color)
        if king_square:
            # Bonus for centralized king
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            center_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
            score += (7 - center_distance) * 2  # Closer to center = better
            
        return score
    
    # ============================================================================
    # TACTICAL PATTERN RECOGNITION (NEW ENHANCEMENTS)
    # ============================================================================
    
    def _tactical_pin_detection(self, board: chess.Board, color: chess.Color) -> float:
        """Comprehensive pin detection and exploitation"""
        score = 0.0
        
        # Check for pins we can create
        for piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for attacker_square in board.pieces(piece_type, color):
                score += self._detect_pin_opportunities(board, attacker_square, color)
        
        # Penalty for being pinned ourselves
        for our_square in chess.SQUARES:
            our_piece = board.piece_at(our_square)
            if our_piece and our_piece.color == color:
                if self._is_piece_pinned(board, our_square, color):
                    pin_penalty = self.tactical_piece_values.get(our_piece.piece_type, 0) * 0.1
                    score -= pin_penalty
        
        return score
    
    def _tactical_fork_detection(self, board: chess.Board, color: chess.Color) -> float:
        """Comprehensive fork detection for all pieces"""
        score = 0.0
        
        # Knight forks (most common and powerful)
        for knight_square in board.pieces(chess.KNIGHT, color):
            fork_value = self._detect_knight_fork(board, knight_square, color)
            score += fork_value
        
        # Queen forks
        for queen_square in board.pieces(chess.QUEEN, color):
            fork_value = self._detect_queen_fork(board, queen_square, color)
            score += fork_value
        
        # Pawn forks
        for pawn_square in board.pieces(chess.PAWN, color):
            fork_value = self._detect_pawn_fork(board, pawn_square, color)
            score += fork_value
        
        # Bishop/Rook forks (less common but still valuable)
        for piece_type in [chess.BISHOP, chess.ROOK]:
            for piece_square in board.pieces(piece_type, color):
                fork_value = self._detect_piece_fork(board, piece_square, color)
                score += fork_value
        
        return score
    
    def _tactical_skewer_detection(self, board: chess.Board, color: chess.Color) -> float:
        """Skewer detection - forcing valuable piece to move, exposing less valuable one"""
        score = 0.0
        
        for piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for attacker_square in board.pieces(piece_type, color):
                score += self._detect_skewer_opportunities(board, attacker_square, color)
        
        return score
    
    def _tactical_discovered_attack(self, board: chess.Board, color: chess.Color) -> float:
        """Discovered attack detection - moving one piece to reveal attack from another"""
        score = 0.0
        
        # Look for pieces that could move to create discovered attacks
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += self._detect_discovered_attack_potential(board, square, color)
        
        return score
    
    def _tactical_deflection_detection(self, board: chess.Board, color: chess.Color) -> float:
        """Deflection - forcing enemy piece away from important duty"""
        score = 0.0
        
        # Look for enemy pieces defending important targets
        for enemy_square in chess.SQUARES:
            enemy_piece = board.piece_at(enemy_square)
            if enemy_piece and enemy_piece.color != color:
                deflection_value = self._detect_deflection_opportunity(board, enemy_square, color)
                score += deflection_value
        
        return score
    
    def _tactical_removing_guard(self, board: chess.Board, color: chess.Color) -> float:
        """Removing the guard - eliminating defender to win material"""
        score = 0.0
        
        # Find enemy pieces that are defending other pieces
        for defender_square in chess.SQUARES:
            defender = board.piece_at(defender_square)
            if defender and defender.color != color:
                guard_value = self._detect_guard_removal(board, defender_square, color)
                score += guard_value
        
        return score
    
    def _tactical_double_check(self, board: chess.Board, color: chess.Color) -> float:
        """Double check detection - checking with two pieces simultaneously"""
        score = 0.0
        
        enemy_king = board.king(not color)
        if enemy_king:
            # Look for potential double check setups
            score += self._detect_double_check_potential(board, enemy_king, color)
        
        return score
    
    def _tactical_battery_formation(self, board: chess.Board, color: chess.Color) -> float:
        """Battery formation - aligning pieces for coordinated attack"""
        score = 0.0
        
        # Queen-Rook batteries
        score += self._detect_queen_rook_battery(board, color)
        
        # Bishop-Queen batteries on diagonals
        score += self._detect_bishop_queen_battery(board, color)
        
        # Rook doubling/tripling
        score += self._detect_rook_battery(board, color)
        
        return score
    
    # ============================================================================
    # PIECE DEFENSE HEURISTICS (NEW)
    # ============================================================================
    
    def _piece_defense_coordination(self, board: chess.Board, color: chess.Color) -> float:
        """Piece defense heuristics for strategic tie-breaking"""
        score = 0.0
        
        # Bonus for pieces defending other pieces
        score += self._calculate_defense_network(board, color)
        
        # Penalty for undefended pieces
        score += self._penalize_undefended_pieces(board, color)
        
        # Bonus for defending key pieces (queen, rooks)
        score += self._defend_key_pieces_bonus(board, color)
        
        return score
    
    # ============================================================================
    # ENHANCED ENDGAME LOGIC (UPGRADED)
    # ============================================================================
    
    def _endgame_enhanced(self, board: chess.Board, color: chess.Color) -> float:
        """Enhanced endgame evaluation with edge-driving and centralization"""
        score = 0.0
        
        # Our king centralization (maintain from V10)
        our_king = board.king(color)
        if our_king:
            king_file = chess.square_file(our_king)
            king_rank = chess.square_rank(our_king)
            center_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
            score += (7 - center_distance) * 4  # Centralize our king
        
        # NEW: Enemy king edge-driving
        enemy_king = board.king(not color)
        if enemy_king:
            score += self._drive_enemy_king_to_edge(board, enemy_king, color)
        
        # NEW: King proximity in endgame (opposition)
        if our_king and enemy_king:
            score += self._calculate_king_proximity_bonus(our_king, enemy_king, color)
        
        # Enhanced pawn promotion
        for pawn_square in board.pieces(chess.PAWN, color):
            score += self._enhanced_pawn_promotion_bonus(pawn_square, color)
        
        return score
    
    def _is_king_exposed(self, board: chess.Board, color: chess.Color, king_square: int) -> bool:
        """Check if king is dangerously exposed"""
        # Simple check: king on starting rank = safer
        if color == chess.WHITE:
            return chess.square_rank(king_square) > 2
        else:
            return chess.square_rank(king_square) < 5
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect endgame phase"""
        # Count major pieces
        major_pieces = 0
        for color in [chess.WHITE, chess.BLACK]:
            major_pieces += len(board.pieces(chess.QUEEN, color)) * 2
            major_pieces += len(board.pieces(chess.ROOK, color))
            
        return major_pieces <= 6  # Arbitrary threshold
    
    # ============================================================================
    # TACTICAL PATTERN IMPLEMENTATION DETAILS
    # ============================================================================
    
    def _detect_pin_opportunities(self, board: chess.Board, attacker_square: int, color: chess.Color) -> float:
        """Detect pin opportunities from a specific attacking piece"""
        score = 0.0
        piece = board.piece_at(attacker_square)
        if not piece:
            return 0.0
        
        # Only sliding pieces can create pins
        if piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            return 0.0
        
        # Check all squares this piece attacks
        attacks = board.attacks(attacker_square)
        
        for attacked_square in attacks:
            attacked_piece = board.piece_at(attacked_square)
            if attacked_piece and attacked_piece.color != color:
                # Look for a more valuable piece behind this one in the same direction
                pin_bonus = self._check_for_pin_behind(board, attacker_square, attacked_square, color)
                score += pin_bonus
        
        return score
    
    def _check_for_pin_behind(self, board: chess.Board, attacker_square: int, target_square: int, color: chess.Color) -> float:
        """Check if there's a valuable piece behind the target that could be pinned"""
        attacker_file = chess.square_file(attacker_square)
        attacker_rank = chess.square_rank(attacker_square)
        target_file = chess.square_file(target_square)
        target_rank = chess.square_rank(target_square)
        
        # Calculate direction vector
        file_diff = target_file - attacker_file
        rank_diff = target_rank - attacker_rank
        
        # Normalize to get direction (1, 0, -1 for each axis)
        if file_diff != 0:
            file_dir = 1 if file_diff > 0 else -1
        else:
            file_dir = 0
            
        if rank_diff != 0:
            rank_dir = 1 if rank_diff > 0 else -1
        else:
            rank_dir = 0
        
        # Continue in the same direction to look for pieces behind
        current_file = target_file + file_dir
        current_rank = target_rank + rank_dir
        
        while 0 <= current_file <= 7 and 0 <= current_rank <= 7:
            behind_square = chess.square(current_file, current_rank)
            behind_piece = board.piece_at(behind_square)
            
            if behind_piece:
                if behind_piece.color != color:  # Enemy piece behind
                    target_piece = board.piece_at(target_square)
                    if target_piece and behind_piece.piece_type in [chess.KING, chess.QUEEN, chess.ROOK]:
                        # Valuable piece is pinned!
                        if behind_piece.piece_type == chess.KING:
                            return 25.0  # Absolute pin
                        else:
                            return 15.0  # Relative pin
                break  # Stop at first piece found
            
            current_file += file_dir
            current_rank += rank_dir
        
        return 0.0
    
    def _detect_knight_fork(self, board: chess.Board, knight_square: int, color: chess.Color) -> float:
        """Detect knight fork opportunities"""
        score = 0.0
        attacks = board.attacks(knight_square)
        
        high_value_targets = []
        check_bonus = 0.0
        
        for target_square in attacks:
            target_piece = board.piece_at(target_square)
            if target_piece and target_piece.color != color:
                if target_piece.piece_type == chess.KING:
                    high_value_targets.append(target_piece)
                    check_bonus = 5.0  # Bonus for giving check
                elif target_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                    high_value_targets.append(target_piece)
                elif target_piece.piece_type in [chess.BISHOP, chess.KNIGHT]:
                    high_value_targets.append(target_piece)
        
        # Scoring based on number and type of targets
        if len(high_value_targets) >= 2:
            if any(p.piece_type == chess.KING for p in high_value_targets):
                score += 50.0  # Royal fork (king + piece)
            else:
                # Multiple high-value pieces
                total_value = sum(self.tactical_piece_values.get(p.piece_type, 0) for p in high_value_targets)
                if total_value > 800:  # Major pieces
                    score += 30.0
                else:
                    score += 15.0
        elif len(high_value_targets) == 1:
            # Single target - smaller bonus for potential setup
            target = high_value_targets[0]
            if target.piece_type == chess.KING:
                score += check_bonus
            elif target.piece_type in [chess.QUEEN, chess.ROOK]:
                score += 5.0  # Attacking major piece
        
        return score
    
    def _detect_queen_fork(self, board: chess.Board, queen_square: int, color: chess.Color) -> float:
        """Detect queen fork opportunities"""
        score = 0.0
        attacks = board.attacks(queen_square)
        
        targets = []
        for target_square in attacks:
            target_piece = board.piece_at(target_square)
            if target_piece and target_piece.color != color:
                if target_piece.piece_type in [chess.KING, chess.ROOK, chess.KNIGHT, chess.BISHOP]:
                    targets.append(target_piece)
        
        if len(targets) >= 2:
            score += 30.0  # Queen fork
        
        return score
    
    def _detect_pawn_fork(self, board: chess.Board, pawn_square: int, color: chess.Color) -> float:
        """Detect pawn fork opportunities"""
        score = 0.0
        pawn_attacks = []
        
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Pawn attack squares
        if color == chess.WHITE:
            if rank < 7:
                if file > 0:
                    pawn_attacks.append(chess.square(file - 1, rank + 1))
                if file < 7:
                    pawn_attacks.append(chess.square(file + 1, rank + 1))
        else:
            if rank > 0:
                if file > 0:
                    pawn_attacks.append(chess.square(file - 1, rank - 1))
                if file < 7:
                    pawn_attacks.append(chess.square(file + 1, rank - 1))
        
        targets = []
        for attack_square in pawn_attacks:
            piece = board.piece_at(attack_square)
            if piece and piece.color != color:
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    targets.append(piece)
        
        if len(targets) >= 2:
            score += 25.0  # Pawn fork
        
        return score
    
    def _detect_piece_fork(self, board: chess.Board, piece_square: int, color: chess.Color) -> float:
        """Detect fork opportunities for bishops/rooks"""
        score = 0.0
        attacks = board.attacks(piece_square)
        
        targets = []
        for target_square in attacks:
            target_piece = board.piece_at(target_square)
            if target_piece and target_piece.color != color:
                if target_piece.piece_type in [chess.KING, chess.QUEEN, chess.ROOK]:
                    targets.append(target_piece)
        
        if len(targets) >= 2:
            score += 20.0  # Bishop/Rook fork
        
        return score
    
    def _detect_skewer_opportunities(self, board: chess.Board, attacker_square: int, color: chess.Color) -> float:
        """Detect skewer opportunities"""
        score = 0.0
        piece = board.piece_at(attacker_square)
        
        if piece and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            attacks = board.attacks(attacker_square)
            
            for attacked_square in attacks:
                attacked_piece = board.piece_at(attacked_square)
                if attacked_piece and attacked_piece.color != color:
                    # Check if there's a less valuable piece behind on the same line
                    skewer_value = self._check_for_skewer_behind(board, attacker_square, attacked_square, color)
                    score += skewer_value
        
        return score
    
    def _check_for_skewer_behind(self, board: chess.Board, attacker_square: int, front_square: int, color: chess.Color) -> float:
        """Check for skewer opportunity along a line"""
        attacker_file = chess.square_file(attacker_square)
        attacker_rank = chess.square_rank(attacker_square)
        front_file = chess.square_file(front_square)
        front_rank = chess.square_rank(front_square)
        
        # Calculate direction vector
        file_diff = front_file - attacker_file
        rank_diff = front_rank - attacker_rank
        
        # Normalize to get direction
        if file_diff != 0:
            file_dir = 1 if file_diff > 0 else -1
        else:
            file_dir = 0
            
        if rank_diff != 0:
            rank_dir = 1 if rank_diff > 0 else -1
        else:
            rank_dir = 0
        
        # Continue in the same direction to look for pieces behind
        current_file = front_file + file_dir
        current_rank = front_rank + rank_dir
        
        while 0 <= current_file <= 7 and 0 <= current_rank <= 7:
            behind_square = chess.square(current_file, current_rank)
            behind_piece = board.piece_at(behind_square)
            
            if behind_piece:
                if behind_piece.color != color:  # Enemy piece behind
                    front_piece = board.piece_at(front_square)
                    if front_piece:
                        front_value = self.tactical_piece_values.get(front_piece.piece_type, 0)
                        behind_value = self.tactical_piece_values.get(behind_piece.piece_type, 0)
                        
                        # Skewer if front piece is more valuable than back piece
                        if front_value > behind_value and front_value > 100:  # Don't skewer pawns
                            return 10.0 + (front_value - behind_value) / 100
                break  # Stop at first piece found
            
            current_file += file_dir
            current_rank += rank_dir
        
        return 0.0
    
    def _is_piece_pinned(self, board: chess.Board, piece_square: int, color: chess.Color) -> bool:
        """Check if a piece is pinned"""
        # Simple pin detection - try moving the piece and see if king is in check
        piece = board.piece_at(piece_square)
        if not piece or piece.color != color:
            return False
        
        # Create a temporary board without this piece
        temp_board = board.copy()
        temp_board.remove_piece_at(piece_square)
        
        # Check if our king would be in check
        return temp_board.is_check()
    
    def _calculate_defense_network(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate bonus for pieces defending other pieces"""
        score = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                # Count how many of our pieces defend this piece
                defenders = len(board.attackers(color, square))
                if defenders > 0:
                    piece_value = self.tactical_piece_values.get(piece.piece_type, 0)
                    defense_bonus = min(defenders * 2, 10)  # Cap the bonus
                    score += defense_bonus
        
        return score
    
    def _penalize_undefended_pieces(self, board: chess.Board, color: chess.Color) -> float:
        """Penalty for undefended pieces"""
        score = 0.0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_attacked_by(not color, square):
                    defenders = len(board.attackers(color, square))
                    if defenders == 0:
                        piece_value = self.tactical_piece_values.get(piece.piece_type, 0)
                        penalty = piece_value * 0.1  # 10% of piece value
                        score -= penalty
        
        return score
    
    def _defend_key_pieces_bonus(self, board: chess.Board, color: chess.Color) -> float:
        """Bonus for defending key pieces"""
        score = 0.0
        
        # Bonus for defending queen
        queens = list(board.pieces(chess.QUEEN, color))
        for queen_square in queens:
            defenders = len(board.attackers(color, queen_square))
            score += defenders * 5.0
        
        # Bonus for defending rooks
        rooks = list(board.pieces(chess.ROOK, color))
        for rook_square in rooks:
            defenders = len(board.attackers(color, rook_square))
            score += defenders * 3.0
        
        return score
    
    def _drive_enemy_king_to_edge(self, board: chess.Board, enemy_king: int, color: chess.Color) -> float:
        """Drive enemy king to edge of board in endgame"""
        score = 0.0
        
        king_file = chess.square_file(enemy_king)
        king_rank = chess.square_rank(enemy_king)
        
        # Distance from center (4.5, 4.5)
        center_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
        
        # Bonus for enemy king being far from center
        score += center_distance * 5
        
        # Extra bonus if enemy king is on edge
        if enemy_king in self.edge_squares:
            score += 20
        
        # Maximum bonus if enemy king is in corner
        if enemy_king in self.corner_squares:
            score += 50
        
        return score
    
    def _calculate_king_proximity_bonus(self, our_king: int, enemy_king: int, color: chess.Color) -> float:
        """Bonus for king proximity in endgame (opposition)"""
        our_file = chess.square_file(our_king)
        our_rank = chess.square_rank(our_king)
        enemy_file = chess.square_file(enemy_king)
        enemy_rank = chess.square_rank(enemy_king)
        
        distance = abs(our_file - enemy_file) + abs(our_rank - enemy_rank)
        
        # Closer is better in endgame, but not too close
        if distance == 2:
            return 10.0  # Perfect opposition distance
        elif distance == 3:
            return 5.0   # Good proximity
        elif distance > 5:
            return -5.0  # Too far away
        
        return 0.0
    
    def _enhanced_pawn_promotion_bonus(self, pawn_square: int, color: chess.Color) -> float:
        """Enhanced pawn promotion bonus"""
        rank = chess.square_rank(pawn_square)
        
        if color == chess.WHITE:
            distance_to_promotion = 7 - rank
        else:
            distance_to_promotion = rank
        
        if distance_to_promotion <= 3:
            promotion_bonus = 30.0 * (4 - distance_to_promotion)
            return promotion_bonus
        
        return 0.0
    
    def _detect_queen_rook_battery(self, board: chess.Board, color: chess.Color) -> float:
        """Detect queen-rook battery formations"""
        score = 0.0
        queens = list(board.pieces(chess.QUEEN, color))
        rooks = list(board.pieces(chess.ROOK, color))
        
        for queen_square in queens:
            for rook_square in rooks:
                if self._pieces_on_same_line(queen_square, rook_square):
                    score += 15.0  # Battery bonus
        
        return score
    
    def _detect_bishop_queen_battery(self, board: chess.Board, color: chess.Color) -> float:
        """Detect bishop-queen battery formations"""
        score = 0.0
        queens = list(board.pieces(chess.QUEEN, color))
        bishops = list(board.pieces(chess.BISHOP, color))
        
        for queen_square in queens:
            for bishop_square in bishops:
                if self._pieces_on_same_diagonal(queen_square, bishop_square):
                    score += 12.0  # Diagonal battery bonus
        
        return score
    
    def _detect_rook_battery(self, board: chess.Board, color: chess.Color) -> float:
        """Detect rook doubling/tripling"""
        score = 0.0
        rooks = list(board.pieces(chess.ROOK, color))
        
        if len(rooks) >= 2:
            for i, rook1 in enumerate(rooks):
                for rook2 in rooks[i+1:]:
                    if self._pieces_on_same_line(rook1, rook2):
                        score += 10.0  # Rook doubling
        
        return score
    
    def _pieces_on_same_line(self, square1: int, square2: int) -> bool:
        """Check if two pieces are on the same rank or file"""
        return (chess.square_file(square1) == chess.square_file(square2) or
                chess.square_rank(square1) == chess.square_rank(square2))
    
    def _pieces_on_same_diagonal(self, square1: int, square2: int) -> bool:
        """Check if two pieces are on the same diagonal"""
        file_diff = abs(chess.square_file(square1) - chess.square_file(square2))
        rank_diff = abs(chess.square_rank(square1) - chess.square_rank(square2))
        return file_diff == rank_diff and file_diff > 0
    
    # ============================================================================
    # ENHANCED TACTICAL METHODS (Improved implementations)
    # ============================================================================
    
    def _detect_discovered_attack_potential(self, board: chess.Board, piece_square: int, color: chess.Color) -> float:
        """Detect discovered attack potential when piece moves"""
        # This is complex to implement fully without move generation
        # For now, give small bonus for pieces that could potentially discover attacks
        piece = board.piece_at(piece_square)
        if piece and piece.color == color:
            # Small bonus for pieces that could move to discover attacks
            return 1.0
        return 0.0
    
    def _detect_deflection_opportunity(self, board: chess.Board, enemy_square: int, color: chess.Color) -> float:
        """Detect deflection opportunities"""
        enemy_piece = board.piece_at(enemy_square)
        if not enemy_piece or enemy_piece.color == color:
            return 0.0
        
        # Count how many important squares/pieces this enemy piece defends
        defends_count = 0
        attacks = board.attacks(enemy_square)
        
        for defended_square in attacks:
            defended_piece = board.piece_at(defended_square)
            if defended_piece and defended_piece.color == enemy_piece.color:
                # This piece defends another piece
                if defended_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                    defends_count += 1
        
        if defends_count > 0:
            return defends_count * 2.0  # Bonus for deflection opportunity
        
        return 0.0
    
    def _detect_guard_removal(self, board: chess.Board, defender_square: int, color: chess.Color) -> float:
        """Detect guard removal opportunities"""
        defender = board.piece_at(defender_square)
        if not defender or defender.color == color:
            return 0.0
        
        # Check if this defender is protecting valuable pieces
        attacks = board.attacks(defender_square)
        guard_value = 0.0
        
        for defended_square in attacks:
            defended_piece = board.piece_at(defended_square)
            if defended_piece and defended_piece.color == defender.color:
                # Check if this defended piece is also attacked by us
                if board.is_attacked_by(color, defended_square):
                    piece_value = self.tactical_piece_values.get(defended_piece.piece_type, 0)
                    guard_value += piece_value / 200  # Small bonus for removing guard
        
        return guard_value
    
    def _detect_double_check_potential(self, board: chess.Board, enemy_king: int, color: chess.Color) -> float:
        """Detect double check potential"""
        if not enemy_king:
            return 0.0
        
        # Count how many of our pieces can give check
        checking_pieces = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                if board.is_attacked_by(color, enemy_king):
                    attacks = board.attacks(square)
                    if enemy_king in attacks:
                        checking_pieces += 1
        
        if checking_pieces >= 2:
            return 8.0  # Potential double check setup
        
        return 0.0
