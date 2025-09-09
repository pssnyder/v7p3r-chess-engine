#!/usr/bin/env python3
"""
V7P3R v11 Phase 3B - Tactical Pattern Detector
Advanced tactical pattern recognition for enhanced position evaluation
Author: Pat Snyder
"""

import chess
from typing import List, Dict, Set, Tuple


class V7P3RTacticalPatternDetector:
    """Advanced tactical pattern detection and evaluation"""
    
    def __init__(self):
        # Tactical pattern bonuses/penalties
        self.pin_bonus = 25
        self.fork_bonus = 40
        self.skewer_bonus = 35
        self.discovered_attack_bonus = 30
        self.double_attack_bonus = 20
        self.x_ray_bonus = 15
        
        # Piece values for tactical calculations
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 10000
        }
        
        # Tactical pattern weights by game phase
        self.tactical_multipliers = {
            'opening': 0.8,     # Less important in opening
            'middlegame': 1.2,  # Most important in middlegame
            'endgame': 1.0      # Important but balanced in endgame
        }
    
    def evaluate_tactical_patterns(self, board: chess.Board, color: bool) -> float:
        """
        Comprehensive tactical pattern evaluation
        Returns score from the perspective of the given color
        """
        total_score = 0.0
        
        # Determine game phase for tactical weighting
        game_phase = self._get_game_phase(board)
        multiplier = self.tactical_multipliers[game_phase]
        
        # Evaluate different tactical patterns
        total_score += self._evaluate_pins(board, color) * multiplier
        total_score += self._evaluate_forks(board, color) * multiplier
        total_score += self._evaluate_skewers(board, color) * multiplier
        total_score += self._evaluate_discovered_attacks(board, color) * multiplier
        total_score += self._evaluate_double_attacks(board, color) * multiplier
        total_score += self._evaluate_x_ray_attacks(board, color) * multiplier
        
        return total_score
    
    def _evaluate_pins(self, board: chess.Board, color: bool) -> float:
        """Detect and evaluate pin patterns"""
        score = 0.0
        
        # Find our pieces that can create pins (bishops, rooks, queens)
        pinning_pieces = []
        for piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pinning_pieces.extend(board.pieces(piece_type, color))
        
        for piece_square in pinning_pieces:
            piece = board.piece_at(piece_square)
            if piece is None:
                continue
            
            # Check for pins created by this piece
            pins_found = self._find_pins_by_piece(board, piece_square, piece.piece_type, color)
            
            for pin_data in pins_found:
                pin_value = self._calculate_pin_value(pin_data)
                score += pin_value
        
        return score
    
    def _evaluate_forks(self, board: chess.Board, color: bool) -> float:
        """Detect and evaluate fork patterns"""
        score = 0.0
        
        # Check for potential forks by our pieces
        our_pieces = []
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.PAWN]:
            our_pieces.extend([(sq, piece_type) for sq in board.pieces(piece_type, color)])
        
        for piece_square, piece_type in our_pieces:
            fork_targets = self._find_fork_targets(board, piece_square, piece_type, color)
            
            if len(fork_targets) >= 2:
                fork_value = self._calculate_fork_value(fork_targets)
                score += fork_value
        
        return score
    
    def _evaluate_skewers(self, board: chess.Board, color: bool) -> float:
        """Detect and evaluate skewer patterns"""
        score = 0.0
        
        # Find our pieces that can create skewers (bishops, rooks, queens)
        skewering_pieces = []
        for piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            skewering_pieces.extend(board.pieces(piece_type, color))
        
        for piece_square in skewering_pieces:
            piece = board.piece_at(piece_square)
            if piece is None:
                continue
            
            skewers_found = self._find_skewers_by_piece(board, piece_square, piece.piece_type, color)
            
            for skewer_data in skewers_found:
                skewer_value = self._calculate_skewer_value(skewer_data)
                score += skewer_value
        
        return score
    
    def _evaluate_discovered_attacks(self, board: chess.Board, color: bool) -> float:
        """Detect and evaluate discovered attack patterns"""
        score = 0.0
        
        # Look for pieces that can move to create discovered attacks
        our_pieces = []
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            our_pieces.extend(board.pieces(piece_type, color))
        
        for piece_square in our_pieces:
            discovered_attacks = self._find_discovered_attacks(board, piece_square, color)
            
            for attack_data in discovered_attacks:
                attack_value = self._calculate_discovered_attack_value(attack_data)
                score += attack_value
        
        return score
    
    def _evaluate_double_attacks(self, board: chess.Board, color: bool) -> float:
        """Detect and evaluate double attack patterns"""
        score = 0.0
        
        # Find squares that are attacked by multiple pieces
        our_attacks = self._get_all_attacks(board, color)
        
        for target_square, attacking_pieces in our_attacks.items():
            if len(attacking_pieces) >= 2:
                target_piece = board.piece_at(target_square)
                if target_piece and target_piece.color != color:
                    double_attack_value = self._calculate_double_attack_value(target_piece, attacking_pieces)
                    score += double_attack_value
        
        return score
    
    def _evaluate_x_ray_attacks(self, board: chess.Board, color: bool) -> float:
        """Detect and evaluate X-ray attack patterns"""
        score = 0.0
        
        # Find our pieces that can create X-ray attacks
        x_ray_pieces = []
        for piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            x_ray_pieces.extend(board.pieces(piece_type, color))
        
        for piece_square in x_ray_pieces:
            piece = board.piece_at(piece_square)
            if piece is None:
                continue
            
            x_ray_attacks = self._find_x_ray_attacks(board, piece_square, piece.piece_type, color)
            
            for x_ray_data in x_ray_attacks:
                x_ray_value = self._calculate_x_ray_value(x_ray_data)
                score += x_ray_value
        
        return score
    
    # Helper methods for tactical pattern detection
    
    def _get_game_phase(self, board: chess.Board) -> str:
        """Determine current game phase for tactical weighting"""
        # Count total material
        total_material = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                pieces = board.pieces(piece_type, color)
                total_material += len(pieces) * self.piece_values[piece_type]
        
        # Determine phase based on material
        if total_material > 6000:  # Most pieces on board
            return 'opening'
        elif total_material > 3000:  # Some pieces traded
            return 'middlegame'
        else:  # Few pieces left
            return 'endgame'
    
    def _find_pins_by_piece(self, board: chess.Board, piece_square: int, piece_type: int, color: bool) -> List[Dict]:
        """Find pins created by a specific piece"""
        pins = []
        
        # Get the attack rays for this piece type
        attack_directions = self._get_attack_directions(piece_type)
        
        for direction in attack_directions:
            ray_squares = self._get_ray_squares(piece_square, direction)
            
            # Look for pin pattern: attacker -> piece1 -> piece2 (where piece2 is more valuable)
            pieces_on_ray = []
            for square in ray_squares:
                piece = board.piece_at(square)
                if piece:
                    pieces_on_ray.append((square, piece))
                    if len(pieces_on_ray) >= 2:
                        break
            
            # Check if we have a pin pattern
            if len(pieces_on_ray) == 2:
                piece1_square, piece1 = pieces_on_ray[0]
                piece2_square, piece2 = pieces_on_ray[1]
                
                # Valid pin: piece1 is enemy, piece2 is enemy and more valuable
                if (piece1.color != color and piece2.color != color and
                    self.piece_values[piece2.piece_type] > self.piece_values[piece1.piece_type]):
                    
                    pins.append({
                        'attacker': piece_square,
                        'pinned_piece': piece1_square,
                        'protected_piece': piece2_square,
                        'pinned_value': self.piece_values[piece1.piece_type],
                        'protected_value': self.piece_values[piece2.piece_type]
                    })
        
        return pins
    
    def _find_fork_targets(self, board: chess.Board, piece_square: int, piece_type: int, color: bool) -> List[int]:
        """Find potential fork targets for a piece"""
        targets = []
        
        # Get all squares this piece can attack
        attack_squares = self._get_piece_attacks(board, piece_square, piece_type)
        
        # Find enemy pieces that could be attacked
        for square in attack_squares:
            piece = board.piece_at(square)
            if piece and piece.color != color:
                targets.append(square)
        
        return targets
    
    def _find_skewers_by_piece(self, board: chess.Board, piece_square: int, piece_type: int, color: bool) -> List[Dict]:
        """Find skewers created by a specific piece"""
        skewers = []
        
        # Get the attack rays for this piece type
        attack_directions = self._get_attack_directions(piece_type)
        
        for direction in attack_directions:
            ray_squares = self._get_ray_squares(piece_square, direction)
            
            # Look for skewer pattern: attacker -> valuable_piece -> less_valuable_piece
            pieces_on_ray = []
            for square in ray_squares:
                piece = board.piece_at(square)
                if piece:
                    pieces_on_ray.append((square, piece))
                    if len(pieces_on_ray) >= 2:
                        break
            
            # Check if we have a skewer pattern
            if len(pieces_on_ray) == 2:
                piece1_square, piece1 = pieces_on_ray[0]
                piece2_square, piece2 = pieces_on_ray[1]
                
                # Valid skewer: both enemy pieces, piece1 more valuable than piece2
                if (piece1.color != color and piece2.color != color and
                    self.piece_values[piece1.piece_type] > self.piece_values[piece2.piece_type]):
                    
                    skewers.append({
                        'attacker': piece_square,
                        'front_piece': piece1_square,
                        'back_piece': piece2_square,
                        'front_value': self.piece_values[piece1.piece_type],
                        'back_value': self.piece_values[piece2.piece_type]
                    })
        
        return skewers
    
    def _find_discovered_attacks(self, board: chess.Board, piece_square: int, color: bool) -> List[Dict]:
        """Find discovered attacks when a piece moves"""
        discovered_attacks = []
        
        # Look behind this piece for friendly long-range pieces
        for direction in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            # Check if there's a friendly piece behind that could create discovered attack
            behind_square = self._get_square_in_direction(piece_square, direction, -1)
            if behind_square is not None:
                behind_piece = board.piece_at(behind_square)
                if behind_piece and behind_piece.color == color:
                    # Check if moving our piece would create an attack
                    attack_ray = self._get_ray_squares(behind_square, direction)
                    for target_square in attack_ray:
                        if target_square == piece_square:
                            continue  # Skip our own piece
                        target_piece = board.piece_at(target_square)
                        if target_piece and target_piece.color != color:
                            discovered_attacks.append({
                                'moving_piece': piece_square,
                                'attacking_piece': behind_square,
                                'target': target_square,
                                'target_value': self.piece_values[target_piece.piece_type]
                            })
                            break  # First enemy piece on ray
        
        return discovered_attacks
    
    def _find_x_ray_attacks(self, board: chess.Board, piece_square: int, piece_type: int, color: bool) -> List[Dict]:
        """Find X-ray attacks through enemy pieces"""
        x_ray_attacks = []
        
        attack_directions = self._get_attack_directions(piece_type)
        
        for direction in attack_directions:
            ray_squares = self._get_ray_squares(piece_square, direction)
            
            # Look for X-ray pattern: attacker -> enemy_piece -> valuable_target
            pieces_on_ray = []
            for square in ray_squares:
                piece = board.piece_at(square)
                if piece:
                    pieces_on_ray.append((square, piece))
                    if len(pieces_on_ray) >= 2:
                        break
            
            if len(pieces_on_ray) == 2:
                piece1_square, piece1 = pieces_on_ray[0]
                piece2_square, piece2 = pieces_on_ray[1]
                
                # Valid X-ray: piece1 is enemy, piece2 is enemy and valuable
                if (piece1.color != color and piece2.color != color and
                    self.piece_values[piece2.piece_type] >= self.piece_values[piece1.piece_type]):
                    
                    x_ray_attacks.append({
                        'attacker': piece_square,
                        'blocker': piece1_square,
                        'target': piece2_square,
                        'blocker_value': self.piece_values[piece1.piece_type],
                        'target_value': self.piece_values[piece2.piece_type]
                    })
        
        return x_ray_attacks
    
    def _get_all_attacks(self, board: chess.Board, color: bool) -> Dict[int, List[int]]:
        """Get all squares attacked by pieces of given color"""
        attacks = {}
        
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            pieces = board.pieces(piece_type, color)
            for piece_square in pieces:
                attack_squares = self._get_piece_attacks(board, piece_square, piece_type)
                for attack_square in attack_squares:
                    if attack_square not in attacks:
                        attacks[attack_square] = []
                    attacks[attack_square].append(piece_square)
        
        return attacks
    
    # Utility methods for pattern calculations
    
    def _calculate_pin_value(self, pin_data: Dict) -> float:
        """Calculate the value of a pin"""
        base_value = self.pin_bonus
        
        # Absolute pins (protecting king) are more valuable
        if pin_data['protected_value'] >= 10000:  # King
            base_value *= 2
        
        # Pins of more valuable pieces are worth more
        value_ratio = pin_data['pinned_value'] / 100  # Normalize to pawn value
        return base_value * (1 + value_ratio * 0.1)
    
    def _calculate_fork_value(self, fork_targets: List[int]) -> float:
        """Calculate the value of a fork"""
        if len(fork_targets) < 2:
            return 0.0
        
        # Base fork value increases with number of targets
        base_value = self.fork_bonus * (1 + (len(fork_targets) - 2) * 0.3)
        
        return base_value
    
    def _calculate_skewer_value(self, skewer_data: Dict) -> float:
        """Calculate the value of a skewer"""
        base_value = self.skewer_bonus
        
        # More valuable back piece makes skewer more effective
        value_ratio = skewer_data['back_value'] / 100
        return base_value * (1 + value_ratio * 0.15)
    
    def _calculate_discovered_attack_value(self, attack_data: Dict) -> float:
        """Calculate the value of a discovered attack"""
        base_value = self.discovered_attack_bonus
        
        # Higher value targets make discovered attacks more valuable
        value_ratio = attack_data['target_value'] / 100
        return base_value * (1 + value_ratio * 0.1)
    
    def _calculate_double_attack_value(self, target_piece: chess.Piece, attacking_pieces: List[int]) -> float:
        """Calculate the value of a double attack"""
        base_value = self.double_attack_bonus
        
        # More attackers and higher value targets increase value
        attacker_bonus = (len(attacking_pieces) - 1) * 0.2
        value_ratio = self.piece_values[target_piece.piece_type] / 100
        
        return base_value * (1 + attacker_bonus + value_ratio * 0.1)
    
    def _calculate_x_ray_value(self, x_ray_data: Dict) -> float:
        """Calculate the value of an X-ray attack"""
        base_value = self.x_ray_bonus
        
        # Higher value targets make X-ray attacks more valuable
        value_ratio = x_ray_data['target_value'] / 100
        return base_value * (1 + value_ratio * 0.1)
    
    # Low-level utility methods
    
    def _get_attack_directions(self, piece_type: int) -> List[Tuple[int, int]]:
        """Get attack directions for a piece type"""
        if piece_type == chess.BISHOP:
            return [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif piece_type == chess.ROOK:
            return [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif piece_type == chess.QUEEN:
            return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            return []
    
    def _get_ray_squares(self, start_square: int, direction: Tuple[int, int]) -> List[int]:
        """Get squares along a ray from start square in given direction"""
        squares = []
        current_file = chess.square_file(start_square)
        current_rank = chess.square_rank(start_square)
        
        file_delta, rank_delta = direction
        
        while True:
            current_file += file_delta
            current_rank += rank_delta
            
            if not (0 <= current_file <= 7 and 0 <= current_rank <= 7):
                break
            
            square = chess.square(current_file, current_rank)
            squares.append(square)
        
        return squares
    
    def _get_square_in_direction(self, start_square: int, direction: Tuple[int, int], steps: int) -> int | None:
        """Get square at specified steps in direction from start square"""
        current_file = chess.square_file(start_square)
        current_rank = chess.square_rank(start_square)
        
        file_delta, rank_delta = direction
        
        new_file = current_file + file_delta * steps
        new_rank = current_rank + rank_delta * steps
        
        if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
            return chess.square(new_file, new_rank)
        else:
            return None
    
    def _get_piece_attacks(self, board: chess.Board, piece_square: int, piece_type: int) -> Set[int]:
        """Get all squares attacked by a piece (simplified version)"""
        # This is a simplified implementation
        # In a full version, you'd want to use board.attacks(piece_square)
        # but implement custom logic for better tactical detection
        
        try:
            return set(board.attacks(piece_square))
        except:
            return set()
