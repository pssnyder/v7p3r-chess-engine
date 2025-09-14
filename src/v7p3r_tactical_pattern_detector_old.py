#!/usr/bin/env python3
"""
V7P3R Tactical Pattern Detector - Phase 10.9.1 Implementation
Time-scaled tactical pattern recognition for V7P3R v10.9 → v11.0

Key Features:
- Time-budget aware pattern detection
- Format-adaptive tactical intensity 
- Priority-based pattern recognition
- Emergency fallback for time pressure
- Lightweight implementation (avoid v10.7 performance issues)

Author: Pat Snyder
"""

import time
import chess
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TacticalPattern:
    """Represents a detected tactical pattern"""
    pattern_type: str
    piece_square: int
    target_square: int
    confidence: float
    time_to_detect_ms: float


class TimeControlAdaptiveTacticalDetector:
    """
    Time-control adaptive tactical pattern detector for V7P3R v10.9
    
    Implements format-specific tactical budgets:
    - 30-minute: 15-20ms tactical budget
    - 10-minute: 8-12ms tactical budget (PRIMARY TARGET)
    - 5:5: 5-8ms tactical budget
    - 2:1: 2-3ms tactical budget (STRESS TEST)
    - 60s: 1ms tactical budget (MINIMAL)
    """
    
    def __init__(self):
        # Time control detection and budgets (in milliseconds)
        self.time_control_budgets = {
            'bullet_60s': {'tactical_ms': 1, 'emergency_threshold': 0.9},
            'blitz_2+1': {'tactical_ms': 3, 'emergency_threshold': 0.8},
            'rapid_5+5': {'tactical_ms': 8, 'emergency_threshold': 0.7},
            'standard_10min': {'tactical_ms': 12, 'emergency_threshold': 0.6},
            'long_30min': {'tactical_ms': 20, 'emergency_threshold': 0.4}
        }
        
        # Pattern detection priority queue (fastest patterns first)
        self.pattern_detectors = {
            'hanging_pieces': {'priority': 1, 'avg_time_ms': 0.5, 'method': self._detect_hanging_pieces},
            'simple_forks': {'priority': 2, 'avg_time_ms': 1.5, 'method': self._detect_simple_forks},
            'basic_pins': {'priority': 3, 'avg_time_ms': 2.0, 'method': self._detect_basic_pins},
            'skewers': {'priority': 4, 'avg_time_ms': 2.5, 'method': self._detect_skewers},
            'discovered_attacks': {'priority': 5, 'avg_time_ms': 4.0, 'method': self._detect_discovered_attacks}
        }
        
        # Statistics tracking
        self.detection_stats = {
            'total_calls': 0,
            'time_budget_exceeded': 0,
            'emergency_fallbacks': 0,
            'patterns_found': 0,
            'avg_detection_time_ms': 0.0
        }
        
        # Current game state tracking
        self.current_time_control = 'standard_10min'  # Default to primary target
        self.game_time_remaining_ms = 600000  # 10 minutes default
        self.moves_played = 0
        
    def detect_tactical_patterns(self, board: chess.Board, time_remaining_ms: int, 
                                moves_played: int) -> Tuple[List[TacticalPattern], float]:
        """
        Main entry point for tactical pattern detection
        
        Returns:
            Tuple of (patterns_found, tactical_score_bonus)
        """
        start_time = time.time()
        self.detection_stats['total_calls'] += 1
        
        # Update game state
        self.game_time_remaining_ms = time_remaining_ms
        self.moves_played = moves_played
        
        # Determine time control and tactical budget
        self.current_time_control = self._detect_time_control(time_remaining_ms, moves_played)
        tactical_budget_ms = self._get_tactical_budget()
        
        # Emergency fallback: no tactical analysis if budget too low
        if tactical_budget_ms < 0.5:
            self.detection_stats['emergency_fallbacks'] += 1
            return [], 0.0
        
        # Detect patterns within time budget
        patterns_found = []
        budget_start = time.time()
        
        # Get enabled patterns for current budget
        enabled_patterns = self._get_enabled_patterns(tactical_budget_ms)
        
        for pattern_name in enabled_patterns:
            # Check if we have time remaining
            elapsed_ms = (time.time() - budget_start) * 1000
            if elapsed_ms >= tactical_budget_ms:
                self.detection_stats['time_budget_exceeded'] += 1
                break
                
            # Detect this pattern type
            detector = self.pattern_detectors[pattern_name]
            try:
                pattern_results = detector['method'](board)
                patterns_found.extend(pattern_results)
            except Exception as e:
                # Continue with other patterns if one fails
                continue
        
        # Calculate tactical score bonus
        tactical_score = self._calculate_tactical_score(patterns_found)
        
        # Update statistics
        total_time_ms = (time.time() - start_time) * 1000
        self._update_detection_stats(total_time_ms, len(patterns_found))
        
        return patterns_found, tactical_score
    
    def _detect_time_control(self, time_remaining_ms: int, moves_played: int) -> str:
        """Detect time control format from remaining time and moves played"""
        # Estimate initial time based on remaining time and moves
        estimated_initial_ms = time_remaining_ms + (moves_played * (time_remaining_ms / max(30 - moves_played, 1)))
        initial_minutes = estimated_initial_ms / (1000 * 60)
        
        if initial_minutes <= 1.2:    # 60-90 seconds
            return 'bullet_60s'
        elif initial_minutes <= 3:    # 2+1 format
            return 'blitz_2+1'
        elif initial_minutes <= 7:    # 5+5 format  
            return 'rapid_5+5'
        elif initial_minutes <= 12:   # 10 minute format (PRIMARY TARGET)
            return 'standard_10min'
        else:
            return 'long_30min'
    
    def _get_tactical_budget(self) -> float:
        """Get tactical time budget based on current game state"""
        base_budget = self.time_control_budgets[self.current_time_control]['tactical_ms']
        emergency_threshold = self.time_control_budgets[self.current_time_control]['emergency_threshold']
        
        # Calculate time pressure factor
        estimated_moves_remaining = max(40 - self.moves_played, 10)
        time_per_move_ms = self.game_time_remaining_ms / estimated_moves_remaining
        
        # If we're in severe time pressure, reduce tactical budget
        if time_per_move_ms < 3000:  # Less than 3 seconds per move
            time_pressure_factor = min(time_per_move_ms / 3000, 1.0)
            return base_budget * time_pressure_factor * 0.3  # Severe reduction
        elif time_per_move_ms < 8000:  # Less than 8 seconds per move
            time_pressure_factor = min(time_per_move_ms / 8000, 1.0)
            return base_budget * time_pressure_factor * 0.6  # Moderate reduction
        else:
            return base_budget  # Full tactical budget available
    
    def _get_enabled_patterns(self, tactical_budget_ms: float) -> List[str]:
        """Get list of patterns that can be executed within time budget"""
        enabled = []
        cumulative_time = 0.0
        
        # Sort patterns by priority (fastest first)
        sorted_patterns = sorted(self.pattern_detectors.items(), 
                               key=lambda x: x[1]['priority'])
        
        for pattern_name, info in sorted_patterns:
            if cumulative_time + info['avg_time_ms'] <= tactical_budget_ms:
                enabled.append(pattern_name)
                cumulative_time += info['avg_time_ms']
            else:
                break  # No more patterns fit in budget
                
        return enabled
    
    def _detect_hanging_pieces(self, board: chess.Board) -> List[TacticalPattern]:
        """Detect hanging pieces - highest priority, fastest detection"""
        start_time = time.time()
        patterns = []
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == (not board.turn):  # Opponent pieces
                # Check if piece is undefended
                if not self._is_piece_defended(board, square, piece.color):
                    # Check if we can attack it
                    if self._can_attack_square(board, square, board.turn):
                        pattern = TacticalPattern(
                            pattern_type='hanging_piece',
                            piece_square=square,
                            target_square=square,
                            confidence=0.9,
                            time_to_detect_ms=(time.time() - start_time) * 1000
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_simple_forks(self, board: chess.Board) -> List[TacticalPattern]:
        """Detect simple fork patterns"""
        start_time = time.time()
        patterns = []
        
        # Look for knight forks (most common)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn and piece.piece_type == chess.KNIGHT:
                # Check all knight move squares
                knight_attacks = chess.SquareSet(chess.BB_KNIGHT_ATTACKS[square])
                targets = []
                
                for target_square in knight_attacks:
                    target_piece = board.piece_at(target_square)
                    if target_piece and target_piece.color != board.turn:
                        # Check if this would be a valuable target
                        if target_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                            targets.append(target_square)
                
                # If we can attack 2+ valuable pieces, it's a fork
                if len(targets) >= 2:
                    pattern = TacticalPattern(
                        pattern_type='knight_fork',
                        piece_square=square,
                        target_square=targets[0],  # Primary target
                        confidence=0.8,
                        time_to_detect_ms=(time.time() - start_time) * 1000
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_basic_pins(self, board: chess.Board) -> List[TacticalPattern]:
        """Detect basic pin patterns"""
        start_time = time.time()
        patterns = []
        
        # Look for pieces that could create pins
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
                    # Check for pin opportunities
                    pin_patterns = self._find_pin_patterns(board, square, piece)
                    patterns.extend(pin_patterns)
        
        return patterns
    
    def _detect_skewers(self, board: chess.Board) -> List[TacticalPattern]:
        """Detect skewer patterns"""
        start_time = time.time()
        patterns = []
        
        # Similar to pins but with valuable piece in front
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                if piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
                    skewer_patterns = self._find_skewer_patterns(board, square, piece)
                    patterns.extend(skewer_patterns)
        
        return patterns
    
    def _detect_discovered_attacks(self, board: chess.Board) -> List[TacticalPattern]:
        """Detect discovered attack patterns"""
        start_time = time.time()
        patterns = []
        
        # Look for pieces that can move to create discovered attacks
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                discovered_patterns = self._find_discovered_attack_patterns(board, square, piece)
                patterns.extend(discovered_patterns)
        
        return patterns
    
    def _is_piece_defended(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        """Check if a piece is defended by same color pieces"""
        attackers = board.attackers(color, square)
        return len(attackers) > 0
    
    def _can_attack_square(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        """Check if we can attack a square"""
        attackers = board.attackers(color, square)
        return len(attackers) > 0
    
    def _find_pin_patterns(self, board: chess.Board, piece_square: int, piece: chess.Piece) -> List[TacticalPattern]:
        """Find pin patterns for a specific piece"""
        patterns = []
        # Simplified pin detection - would need full implementation
        return patterns
    
    def _find_skewer_patterns(self, board: chess.Board, piece_square: int, piece: chess.Piece) -> List[TacticalPattern]:
        """Find skewer patterns for a specific piece"""
        patterns = []
        # Simplified skewer detection - would need full implementation
        return patterns
    
    def _find_discovered_attack_patterns(self, board: chess.Board, piece_square: int, piece: chess.Piece) -> List[TacticalPattern]:
        """Find discovered attack patterns for a specific piece"""
        patterns = []
        # Simplified discovered attack detection - would need full implementation
        return patterns
    
    def _calculate_tactical_score(self, patterns: List[TacticalPattern]) -> float:
        """Calculate tactical score bonus based on detected patterns"""
        if not patterns:
            return 0.0
        
        score = 0.0
        
        for pattern in patterns:
            # Base scores by pattern type
            pattern_values = {
                'hanging_piece': 50.0,
                'knight_fork': 75.0,
                'pin': 30.0,
                'skewer': 60.0,
                'discovered_attack': 40.0
            }
            
            base_value = pattern_values.get(pattern.pattern_type, 20.0)
            score += base_value * pattern.confidence
        
        return min(score, 200.0)  # Cap tactical bonus to prevent evaluation imbalance
    
    def _update_detection_stats(self, detection_time_ms: float, patterns_found: int):
        """Update detection statistics for monitoring"""
        self.detection_stats['patterns_found'] += patterns_found
        
        # Update average detection time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if self.detection_stats['avg_detection_time_ms'] == 0:
            self.detection_stats['avg_detection_time_ms'] = detection_time_ms
        else:
            self.detection_stats['avg_detection_time_ms'] = (
                alpha * detection_time_ms + 
                (1 - alpha) * self.detection_stats['avg_detection_time_ms']
            )
    
    def get_detection_stats(self) -> Dict:
        """Get current detection statistics for debugging/monitoring"""
        return self.detection_stats.copy()
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total_calls': 0,
            'time_budget_exceeded': 0,
            'emergency_fallbacks': 0,
            'patterns_found': 0,
            'avg_detection_time_ms': 0.0
        }


# Legacy compatibility for existing v10.x imports
class V7P3RTacticalPatternDetector:
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self):
        self.new_detector = TimeControlAdaptiveTacticalDetector()
        
    def evaluate_tactical_patterns(self, board: chess.Board, color: bool) -> float:
        """Legacy interface for v10.x compatibility"""
        # Use default time parameters for legacy calls
        time_remaining_ms = 600000  # 10 minutes
        moves_played = 20  # Assume mid-game
        
        patterns, score = self.new_detector.detect_tactical_patterns(
            board, time_remaining_ms, moves_played
        )
        
        # Return score from perspective of requested color
        if color == board.turn:
            return score
        else:
            return -score  # Negate for opponent
        
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
