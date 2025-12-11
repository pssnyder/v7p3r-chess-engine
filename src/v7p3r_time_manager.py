"""
V7P3R Time Management Module
Simple time allocation strategy for different time controls

Version: 17.8
"""


class TimeManager:
    def __init__(self, base_time_ms, increment_ms, time_control_type='blitz'):
        """
        Args:
            base_time_ms: Starting time in milliseconds
            increment_ms: Increment per move in milliseconds  
            time_control_type: 'bullet', 'blitz', 'rapid', 'classical'
        """
        self.base_time = base_time_ms
        self.increment = increment_ms
        self.time_control = time_control_type
        
    def calculate_move_time(self, time_remaining_ms, moves_played, is_endgame=False):
        """
        Calculate how much time to spend on this move
        
        Strategy:
        - Use increment + small buffer from base time
        - Reduce thinking time in simple positions
        - Increase in complex middle game
        
        Args:
            time_remaining_ms: Time remaining in milliseconds
            moves_played: Number of moves made so far
            is_endgame: Whether position is in endgame
            
        Returns:
            max_think_time_ms: Maximum milliseconds for this move
        """
        # Safety margin - always keep a time cushion
        SAFETY_BUFFER_MS = 2000  # 2 seconds minimum reserve
        
        if time_remaining_ms < SAFETY_BUFFER_MS:
            # Emergency time - move instantly
            return 100  # 0.1 second
        
        # Available time = remaining - safety buffer
        available_time = time_remaining_ms - SAFETY_BUFFER_MS
        
        # Estimate moves remaining
        if moves_played < 20:
            moves_remaining = 40  # Opening/early middle
        elif moves_played < 40:
            moves_remaining = 30  # Middle game
        else:
            moves_remaining = 20  # Endgame
        
        # Base allocation: divide remaining time by moves remaining
        base_allocation = available_time / moves_remaining
        
        # Add increment (we get it back after moving)
        base_allocation += self.increment
        
        # Adjust for game phase
        if is_endgame:
            # Endgames: use less time (50% of base)
            base_allocation *= 0.5
        elif 20 <= moves_played < 40:
            # Critical middle game: use more time (120% of base)
            base_allocation *= 1.2
        
        # Apply time control multipliers
        if self.time_control == 'bullet':
            # Bullet: move faster, rely on increment
            max_time = min(base_allocation, self.increment * 1.5)
        elif self.time_control == 'blitz':
            # Blitz: balanced approach
            max_time = min(base_allocation, self.increment * 2.5)
        elif self.time_control == 'rapid':
            # Rapid: can think longer
            max_time = min(base_allocation, self.increment * 5.0)
        else:
            # Classical: no restriction
            max_time = base_allocation
        
        # Floor and ceiling
        min_time = 100  # At least 0.1 second
        max_time = max(min_time, min(max_time, time_remaining_ms - SAFETY_BUFFER_MS))
        
        return int(max_time)
    
    def get_time_control_type(self, base_time_ms):
        """
        Classify time control based on base time
        
        Args:
            base_time_ms: Base time in milliseconds
            
        Returns:
            Time control type string
        """
        if base_time_ms < 120000:  # < 2 minutes
            return 'bullet'
        elif base_time_ms < 600000:  # < 10 minutes
            return 'blitz'
        elif base_time_ms < 1800000:  # < 30 minutes
            return 'rapid'
        else:
            return 'classical'
