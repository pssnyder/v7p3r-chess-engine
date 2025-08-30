#!/usr/bin/env python3
"""
V7P3R Chess Engine V8.3 - Dynamic Memory Manager
Intelligent memory management with adaptive sizing and cleanup strategies
"""

import time
import gc
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import threading


@dataclass
class MemoryPolicy:
    """Memory management policy configuration"""
    max_cache_size: int = 50000  # Maximum entries in evaluation cache
    max_tt_size: int = 100000    # Maximum transposition table entries
    max_killer_moves: int = 1000  # Maximum killer move entries
    max_history_size: int = 10000 # Maximum history heuristic entries
    
    # Age-based cleanup thresholds (in seconds)
    cache_ttl: float = 30.0      # Time-to-live for cache entries
    tt_ttl: float = 60.0         # Time-to-live for transposition entries
    killer_ttl: float = 10.0     # Time-to-live for killer moves
    history_ttl: float = 20.0    # Time-to-live for history scores
    
    # Memory pressure thresholds
    memory_pressure_mb: float = 100.0  # Start cleanup at this memory usage
    critical_memory_mb: float = 200.0  # Emergency cleanup threshold
    
    # Cleanup frequencies
    cleanup_interval: float = 5.0      # Seconds between routine cleanups
    pressure_cleanup_ratio: float = 0.3 # Fraction to remove under pressure


class LRUCacheWithTTL:
    """LRU cache with time-to-live support for memory-efficient storage"""
    
    def __init__(self, max_size: int, ttl: float):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.access_count = 0
        self.hit_count = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value with LRU update and TTL check"""
        self.access_count += 1
        current_time = time.time()
        
        if key in self.cache:
            # Check if entry has expired
            if current_time - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hit_count += 1
            return self.cache[key]
        
        return None
    
    def put(self, key: Any, value: Any):
        """Store value with automatic size management"""
        current_time = time.time()
        
        if key in self.cache:
            # Update existing entry
            self.cache[key] = value
            self.timestamps[key] = current_time
            self.cache.move_to_end(key)
        else:
            # Add new entry
            self.cache[key] = value
            self.timestamps[key] = current_time
            
            # Remove oldest entries if over size limit
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
        
        return len(expired_keys)
    
    def cleanup_pressure(self, ratio: float) -> int:
        """Remove oldest entries under memory pressure"""
        target_removal = int(len(self.cache) * ratio)
        removed = 0
        
        keys_to_remove = list(self.cache.keys())[:target_removal]
        for key in keys_to_remove:
            del self.cache[key]
            del self.timestamps[key]
            removed += 1
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_ratio = self.hit_count / max(self.access_count, 1)
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_ratio': hit_ratio,
            'access_count': self.access_count,
            'hit_count': self.hit_count
        }
    
    def clear(self):
        """Clear all cache data"""
        self.cache.clear()
        self.timestamps.clear()
        self.access_count = 0
        self.hit_count = 0


class V7P3RMemoryManager:
    """Advanced memory management system for V7P3R chess engine"""
    
    def __init__(self, policy: Optional[MemoryPolicy] = None):
        self.policy = policy or MemoryPolicy()
        
        # Initialize managed caches
        self.evaluation_cache = LRUCacheWithTTL(
            self.policy.max_cache_size, 
            self.policy.cache_ttl
        )
        
        self.transposition_table = LRUCacheWithTTL(
            self.policy.max_tt_size,
            self.policy.tt_ttl
        )
        
        # Killer moves with depth-aware management
        self.killer_moves: Dict[int, List[Tuple[Any, float]]] = {}
        self.killer_timestamps: Dict[int, float] = {}
        
        # History scores with decay
        self.history_scores: Dict[str, Tuple[float, float]] = {}  # (score, timestamp)
        
        # Memory monitoring
        self.last_cleanup = time.time()
        self.cleanup_stats = {
            'total_cleanups': 0,
            'pressure_cleanups': 0,
            'entries_removed': 0
        }
        
        # Performance tracking
        self.memory_usage_history: List[Tuple[float, float]] = []  # (time, memory_mb)
        self.performance_metrics = {
            'cache_efficiency': 0.0,
            'memory_stability': 0.0,
            'cleanup_frequency': 0.0
        }
    
    def get_evaluation(self, position_hash: str) -> Optional[float]:
        """Get cached evaluation with automatic cleanup"""
        result = self.evaluation_cache.get(position_hash)
        self._maybe_cleanup()
        return result
    
    def store_evaluation(self, position_hash: str, evaluation: float):
        """Store evaluation in managed cache"""
        self.evaluation_cache.put(position_hash, evaluation)
    
    def get_transposition(self, position_hash: str) -> Optional[Dict[str, Any]]:
        """Get transposition table entry"""
        result = self.transposition_table.get(position_hash)
        self._maybe_cleanup()
        return result
    
    def store_transposition(self, position_hash: str, data: Dict[str, Any]):
        """Store transposition table entry"""
        self.transposition_table.put(position_hash, data)
    
    def get_killer_moves(self, ply: int) -> List[Any]:
        """Get killer moves for given ply with age filtering"""
        current_time = time.time()
        
        if ply in self.killer_moves:
            # Filter out expired moves
            valid_moves = [
                move for move, timestamp in self.killer_moves[ply]
                if current_time - timestamp <= self.policy.killer_ttl
            ]
            
            if valid_moves:
                # Update timestamp for accessed ply
                self.killer_timestamps[ply] = current_time
                return valid_moves
            else:
                # Remove expired ply entry
                del self.killer_moves[ply]
                if ply in self.killer_timestamps:
                    del self.killer_timestamps[ply]
        
        return []
    
    def store_killer_move(self, ply: int, move: Any):
        """Store killer move with automatic size management"""
        current_time = time.time()
        
        if ply not in self.killer_moves:
            self.killer_moves[ply] = []
        
        # Add move with timestamp
        self.killer_moves[ply].append((move, current_time))
        self.killer_timestamps[ply] = current_time
        
        # Limit moves per ply (keep only 2 most recent)
        if len(self.killer_moves[ply]) > 2:
            self.killer_moves[ply] = self.killer_moves[ply][-2:]
        
        # Clean up old plies if too many entries
        if len(self.killer_moves) > self.policy.max_killer_moves:
            self._cleanup_killer_moves()
    
    def get_history_score(self, move_key: str) -> float:
        """Get history heuristic score with decay"""
        current_time = time.time()
        
        if move_key in self.history_scores:
            score, timestamp = self.history_scores[move_key]
            
            # Apply time-based decay
            age = current_time - timestamp
            if age <= self.policy.history_ttl:
                decay_factor = max(0.1, 1.0 - (age / self.policy.history_ttl))
                return score * decay_factor
            else:
                # Remove expired entry
                del self.history_scores[move_key]
        
        return 0.0
    
    def update_history_score(self, move_key: str, delta: float):
        """Update history heuristic score"""
        current_time = time.time()
        current_score = self.get_history_score(move_key)
        new_score = current_score + delta
        
        self.history_scores[move_key] = (new_score, current_time)
        
        # Cleanup if too many entries
        if len(self.history_scores) > self.policy.max_history_size:
            self._cleanup_history_scores()
    
    def _cleanup_killer_moves(self):
        """Clean up old killer move entries"""
        current_time = time.time()
        
        # Remove expired plies
        expired_plies = [
            ply for ply, timestamp in self.killer_timestamps.items()
            if current_time - timestamp > self.policy.killer_ttl
        ]
        
        for ply in expired_plies:
            if ply in self.killer_moves:
                del self.killer_moves[ply]
            del self.killer_timestamps[ply]
        
        # If still too many, remove oldest
        while len(self.killer_moves) > self.policy.max_killer_moves:
            oldest_ply = min(self.killer_timestamps.keys(), 
                           key=lambda p: self.killer_timestamps[p])
            del self.killer_moves[oldest_ply]
            del self.killer_timestamps[oldest_ply]
    
    def _cleanup_history_scores(self):
        """Clean up old history heuristic entries"""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, (score, timestamp) in self.history_scores.items()
            if current_time - timestamp > self.policy.history_ttl
        ]
        
        for key in expired_keys:
            del self.history_scores[key]
        
        # If still too many, remove lowest scoring entries
        while len(self.history_scores) > self.policy.max_history_size:
            # Find entry with lowest adjusted score (score * recency)
            worst_key = min(
                self.history_scores.keys(),
                key=lambda k: self.history_scores[k][0] * (
                    1.0 - (current_time - self.history_scores[k][1]) / self.policy.history_ttl
                )
            )
            del self.history_scores[worst_key]
    
    def _maybe_cleanup(self):
        """Perform cleanup if interval has elapsed"""
        current_time = time.time()
        
        if current_time - self.last_cleanup >= self.policy.cleanup_interval:
            self.routine_cleanup()
            self.last_cleanup = current_time
    
    def routine_cleanup(self):
        """Perform routine cleanup of all caches"""
        removed_count = 0
        
        # Clean expired entries from all caches
        removed_count += self.evaluation_cache.cleanup_expired()
        removed_count += self.transposition_table.cleanup_expired()
        
        # Clean killer moves and history
        old_killer_count = len(self.killer_moves)
        old_history_count = len(self.history_scores)
        
        self._cleanup_killer_moves()
        self._cleanup_history_scores()
        
        removed_count += (old_killer_count - len(self.killer_moves))
        removed_count += (old_history_count - len(self.history_scores))
        
        # Update stats
        self.cleanup_stats['total_cleanups'] += 1
        self.cleanup_stats['entries_removed'] += removed_count
        
        # Force garbage collection periodically
        if self.cleanup_stats['total_cleanups'] % 10 == 0:
            gc.collect()
    
    def pressure_cleanup(self):
        """Perform aggressive cleanup under memory pressure"""
        removed_count = 0
        
        # Aggressive cache reduction
        removed_count += self.evaluation_cache.cleanup_pressure(self.policy.pressure_cleanup_ratio)
        removed_count += self.transposition_table.cleanup_pressure(self.policy.pressure_cleanup_ratio)
        
        # Clear older killer moves and history
        current_time = time.time()
        
        # Remove killer moves older than half TTL
        aggressive_killer_ttl = self.policy.killer_ttl / 2
        expired_plies = [
            ply for ply, timestamp in self.killer_timestamps.items()
            if current_time - timestamp > aggressive_killer_ttl
        ]
        
        for ply in expired_plies:
            if ply in self.killer_moves:
                del self.killer_moves[ply]
            del self.killer_timestamps[ply]
            removed_count += 1
        
        # Remove history scores older than half TTL
        aggressive_history_ttl = self.policy.history_ttl / 2
        expired_keys = [
            key for key, (score, timestamp) in self.history_scores.items()
            if current_time - timestamp > aggressive_history_ttl
        ]
        
        for key in expired_keys:
            del self.history_scores[key]
            removed_count += 1
        
        # Force garbage collection
        gc.collect()
        
        # Update stats
        self.cleanup_stats['pressure_cleanups'] += 1
        self.cleanup_stats['entries_removed'] += removed_count
        
        return removed_count
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics"""
        eval_stats = self.evaluation_cache.get_stats()
        tt_stats = self.transposition_table.get_stats()
        
        return {
            'evaluation_cache': eval_stats,
            'transposition_table': tt_stats,
            'killer_moves': {
                'plies': len(self.killer_moves),
                'total_moves': sum(len(moves) for moves in self.killer_moves.values())
            },
            'history_scores': {
                'entries': len(self.history_scores)
            },
            'cleanup_stats': self.cleanup_stats,
            'memory_policy': {
                'max_cache_size': self.policy.max_cache_size,
                'max_tt_size': self.policy.max_tt_size,
                'cache_ttl': self.policy.cache_ttl,
                'tt_ttl': self.policy.tt_ttl
            }
        }
    
    def optimize_for_game_phase(self, game_phase: str):
        """Dynamically adjust memory policy based on game phase"""
        if game_phase == "opening":
            # Opening: Larger TT for opening knowledge, smaller evaluation cache
            self.policy.max_tt_size = 150000
            self.policy.max_cache_size = 30000
            self.policy.tt_ttl = 120.0  # Keep opening knowledge longer
            
        elif game_phase == "middlegame":
            # Middlegame: Balanced approach with frequent cleanup
            self.policy.max_tt_size = 100000
            self.policy.max_cache_size = 50000
            self.policy.cleanup_interval = 3.0  # More frequent cleanup
            
        elif game_phase == "endgame":
            # Endgame: Larger evaluation cache for precise calculations
            self.policy.max_tt_size = 50000
            self.policy.max_cache_size = 80000
            self.policy.cache_ttl = 60.0  # Keep evaluations longer
    
    def clear_all(self):
        """Clear all managed memory structures"""
        self.evaluation_cache.clear()
        self.transposition_table.clear()
        self.killer_moves.clear()
        self.killer_timestamps.clear()
        self.history_scores.clear()
        
        # Reset stats
        self.cleanup_stats = {
            'total_cleanups': 0,
            'pressure_cleanups': 0,
            'entries_removed': 0
        }
        
        gc.collect()


# Factory function for easy integration
def create_memory_manager(
    max_memory_mb: float = 100.0,
    game_phase: str = "middlegame"
) -> V7P3RMemoryManager:
    """Create optimized memory manager for specific constraints"""
    
    # Scale cache sizes based on available memory
    base_cache_size = int(max_memory_mb * 500)  # ~500 entries per MB
    base_tt_size = int(max_memory_mb * 1000)    # ~1000 entries per MB
    
    policy = MemoryPolicy(
        max_cache_size=min(base_cache_size, 100000),
        max_tt_size=min(base_tt_size, 200000),
        memory_pressure_mb=max_memory_mb * 0.8,
        critical_memory_mb=max_memory_mb
    )
    
    manager = V7P3RMemoryManager(policy)
    manager.optimize_for_game_phase(game_phase)
    
    return manager
