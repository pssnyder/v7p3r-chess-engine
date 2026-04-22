#!/usr/bin/env python3
"""
V7P3R Time Manager - C0BR4-inspired Simple Time Allocation

WHY THIS EXISTS: Time forfeits were 75% of v18.4 losses. Complex time management
was causing bugs. C0BR4 plays 100+ games/day with simple logic - we need the same.

PHILOSOPHY: Simple, proven, conservative time allocation
- Emergency mode for time pressure
- Conservative base allocation
- Game phase awareness (minimal complexity)
- Safety margins to prevent timeouts

Based on C0BR4 v3.4 TimeManager.cs (proven in production)
"""

import chess
from typing import Tuple


class TimeManager:
    """
    Simple, reliable time allocation following C0BR4's proven approach
    """
    
    @staticmethod
    def calculate_time_allocation(
        remaining_time: float,
        increment: float,
        moves_played: int,
        board: chess.Board = None
    ) -> Tuple[float, float]:
        """
        WHY THIS EXISTS: Calculate how much time to spend on this move
        
        WHAT IT DOES: Conservative time allocation based on remaining time and increment
        
        IMPACT: Critical for preventing time forfeits (was 75% of losses in v18.4)
        
        Args:
            remaining_time: Time remaining in seconds
            increment: Time increment per move in seconds
            moves_played: Number of moves played so far
            board: Current board position (optional, for game phase)
        
        Returns:
            (target_time, max_time) tuple in seconds
        """
        
        # EMERGENCY MODE: <3 seconds remaining
        if remaining_time < 3.0:
            # Use 3% of remaining time, minimum 30ms
            emergency_time = max(0.03, remaining_time * 0.03)
            return (emergency_time, emergency_time)
        
        # LOW TIME MODE: <15 seconds remaining
        if remaining_time < 15.0:
            # Use ~5% of time + 1/3 of increment
            low_time = max(0.08, remaining_time / 20.0 + increment / 3.0)
            return (low_time, low_time * 1.1)
        
        # NORMAL MODE: Calculate based on estimated moves remaining
        game_phase = TimeManager._estimate_game_phase(moves_played, board)
        estimated_moves = TimeManager._estimate_moves_remaining(game_phase, remaining_time)
        
        # Base allocation: time / moves_left + 75% of increment
        # (We'll get the increment back, so we can use most of it)
        base_time = remaining_time / estimated_moves + (increment * 0.75)
        
        # Apply game phase multiplier (minimal adjustment)
        phase_multiplier = TimeManager._calculate_phase_multiplier(game_phase)
        base_time *= phase_multiplier
        
        # Apply safety margins
        base_time = TimeManager._apply_safety_margins(base_time, remaining_time)
        
        # Target time and max time (allow 10% overage for critical positions)
        target_time = max(0.05, min(base_time, remaining_time / 5.0))
        max_time = target_time * 1.1
        
        return (target_time, max_time)
    
    @staticmethod
    def _estimate_game_phase(moves_played: int, board: chess.Board = None) -> float:
        """
        WHY THIS EXISTS: Different game phases need different time usage
        
        WHAT IT DOES: Estimates game phase 0.0 (endgame) to 1.0 (opening)
        
        Returns:
            Game phase estimate (0.0 = endgame, 1.0 = opening)
        """
        # Simple move-based phase estimate
        if moves_played < 12:
            return 1.0  # Opening
        elif moves_played < 25:
            return 0.6  # Early middlegame
        elif moves_played < 40:
            return 0.4  # Late middlegame
        else:
            return 0.2  # Endgame
        
        # Note: Could use material count if board provided, but move count is simpler
    
    @staticmethod
    def _estimate_moves_remaining(game_phase: float, remaining_time: float) -> int:
        """
        WHY THIS EXISTS: Need to estimate how to divide remaining time
        
        WHAT IT DOES: Estimates moves left in game based on phase and time control
        
        IMPACT: Critical for time allocation - too low = timeouts, too high = weak play
        
        Returns:
            Estimated moves remaining
        """
        # Base estimate by game phase
        # Opening: ~50 moves, Middlegame: ~40 moves, Endgame: ~25 moves
        phase_estimate = int(25 + game_phase * 25)
        
        # Adjust based on remaining time - longer games = more conservative
        if remaining_time > 1800.0:  # 30+ minutes
            phase_estimate = max(phase_estimate, 50)
        elif remaining_time > 600.0:  # 10+ minutes
            phase_estimate = max(phase_estimate, 40)
        elif remaining_time > 300.0:  # 5+ minutes
            phase_estimate = max(phase_estimate, 30)
        elif remaining_time > 60.0:  # 1+ minute
            phase_estimate = max(phase_estimate, 20)
        
        return phase_estimate
    
    @staticmethod
    def _calculate_phase_multiplier(game_phase: float) -> float:
        """
        WHY THIS EXISTS: Middlegame positions are more complex, need more time
        
        WHAT IT DOES: Returns time multiplier based on game phase
        
        Returns:
            Multiplier for base time allocation
        """
        # Spend slightly more time in middlegame, less in opening/endgame
        # Opening: 0.95x, Middlegame: 1.1x, Endgame: 0.9x
        if game_phase > 0.7:  # Opening
            return 0.95
        elif game_phase > 0.3:  # Middlegame
            return 1.1
        else:  # Endgame
            return 0.9
    
    @staticmethod
    def _apply_safety_margins(base_time: float, remaining_time: float) -> float:
        """
        WHY THIS EXISTS: Prevent time troubles by capping per-move usage
        
        WHAT IT DOES: Ensures we never use too much time on one move
        
        Returns:
            Time allocation with safety margins applied
        """
        # Never use more than 1/3 of remaining time on a single move
        if base_time > remaining_time / 3.0:
            base_time = remaining_time / 3.0
        
        # Reserve 50ms for communication overhead
        base_time = max(0.1, base_time - 0.05)
        
        return base_time


# Quick test function
if __name__ == "__main__":
    print("Time Manager Test Cases:")
    print("=" * 60)
    
    test_cases = [
        # (remaining_time, increment, moves_played, description)
        (2.5, 0, 30, "Emergency: 2.5s left"),
        (10.0, 2.0, 25, "Low time: 10s + 2s inc"),
        (120.0, 4.0, 15, "Blitz: 2min + 4s inc"),
        (300.0, 10.0, 10, "Rapid: 5min + 10s inc"),
        (900.0, 10.0, 8, "Classical: 15min + 10s inc"),
    ]
    
    for remaining, inc, moves, desc in test_cases:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        print(f"{desc:30s} → Target: {target:.2f}s, Max: {max_time:.2f}s")
    
    print("=" * 60)
