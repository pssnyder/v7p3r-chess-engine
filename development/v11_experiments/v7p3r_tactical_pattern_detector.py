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

# V11.3 PERFORMANCE OPTIMIZATION: Emergency time budgets
EMERGENCY_TIME_BUDGET_MS = 2.0  # Maximum 2ms for tactical analysis
FAST_TACTICAL_BUDGET_MS = 0.5   # Ultra-fast mode for bullet games


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
        V11.3 OPTIMIZED: Tactical pattern detection with strict time budgets
        
        Returns:
            Tuple of (patterns_found, tactical_score_bonus)
        """
        # V11.3 CRITICAL OPTIMIZATION: Emergency time budget check
        pattern_start_time = time.time()
        
        # Determine maximum time budget based on game speed
        if time_remaining_ms < 30000:  # < 30 seconds
            max_budget_ms = FAST_TACTICAL_BUDGET_MS
        else:
            max_budget_ms = min(EMERGENCY_TIME_BUDGET_MS, time_remaining_ms / 1000 * 0.001)  # 0.1% of remaining time
        
        # Emergency fallback: skip tactical analysis if time is too tight
        if max_budget_ms < 0.1:
            return [], 0.0
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
            # V11.3 CRITICAL: Check time budget every pattern
            elapsed_ms = (time.time() - pattern_start_time) * 1000
            if elapsed_ms >= max_budget_ms:
                self.detection_stats['time_budget_exceeded'] += 1
                break  # Emergency exit
                
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
        # More conservative estimation: assume games last 60 moves total on average
        moves_remaining = max(60 - moves_played, 10)
        estimated_initial_ms = time_remaining_ms + (moves_played * (time_remaining_ms / moves_remaining))
        initial_minutes = estimated_initial_ms / (1000 * 60)
        
        if initial_minutes <= 1.5:    # 60-90 seconds
            return 'bullet_60s'
        elif initial_minutes <= 3.5:  # 2+1 or 3+0 format
            return 'blitz_2+1'
        elif initial_minutes <= 7:    # 5+5 or 5+3 format  
            return 'rapid_5+5'
        elif initial_minutes <= 15:   # 10 minute format (PRIMARY TARGET)
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