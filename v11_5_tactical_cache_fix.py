#!/usr/bin/env python3
"""
V11.5 Tactical Cache Performance Fix
====================================

PROBLEM: Tactical pattern detector called 5-6 times per move during search:
- 3x in _order_moves_advanced (lines 584, 595, 611)
- 2x in position evaluation (lines 746, 747)
- 1x in _detect_bitboard_tactics (line 896)

Each call does expensive bitboard analysis for 8+ tactical patterns.
At depth 3 with ~30 moves, that's HUNDREDS of redundant tactical calls!

SOLUTION: Add position-based tactical cache with FEN key lookup.
Cache tactical results for each unique position during search.

Expected improvement: 5-10x search speed boost by eliminating redundant tactical analysis.
"""

import chess
import time
from typing import Dict, Tuple, List, Optional

class TacticalCache:
    """High-speed tactical result cache for eliminating redundant pattern detection"""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, Tuple[float, List, float]] = {}  # FEN -> (score, patterns, timestamp)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.cache_enabled = True
        
    def get_cached_result(self, board: chess.Board, color: chess.Color) -> Optional[float]:
        """Get cached tactical score for position"""
        if not self.cache_enabled:
            return None
            
        fen = board.fen()
        if fen in self.cache:
            self.hits += 1
            score, patterns, timestamp = self.cache[fen]
            # Return score from perspective of requested color
            if color == board.turn:
                return score
            else:
                return -score
        
        self.misses += 1
        return None
    
    def cache_result(self, board: chess.Board, score: float, patterns: List, color: chess.Color):
        """Cache tactical result for position"""
        if not self.cache_enabled:
            return
            
        fen = board.fen()
        
        # Adjust score to always be from white's perspective for consistent caching
        if color != chess.WHITE:
            score = -score
            
        self.cache[fen] = (score, patterns, time.time())
        
        # Prune cache if too large
        if len(self.cache) > self.max_size:
            self._prune_cache()
    
    def _prune_cache(self):
        """Remove oldest 25% of cache entries"""
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1][2])
        prune_count = len(sorted_entries) // 4
        
        for i in range(prune_count):
            del self.cache[sorted_entries[i][0]]
    
    def clear(self):
        """Clear cache between games"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """Get cache performance statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self.cache),
            'enabled': self.cache_enabled
        }


def patch_v7p3r_with_tactical_cache():
    """
    Instructions for patching v7p3r.py with tactical cache:
    
    1. Add to imports:
       from v11_5_tactical_cache_fix import TacticalCache
    
    2. Add to V7P3REngine.__init__():
       self.tactical_cache = TacticalCache(max_size=5000)
    
    3. Modify _get_tactical_evaluation():
       def _get_tactical_evaluation(self, board: chess.Board, color: chess.Color) -> float:
           # Check cache first
           cached_result = self.tactical_cache.get_cached_result(board, color)
           if cached_result is not None:
               return cached_result
           
           # Original tactical detection code
           time_remaining_ms = 600000
           moves_played = 20
           patterns, score = self.new_detector.detect_tactical_patterns(
               board, time_remaining_ms, moves_played
           )
           
           # Cache the result
           self.tactical_cache.cache_result(board, score, patterns, color)
           
           # Return score from perspective of requested color
           if color == board.turn:
               return score
           else:
               return -score
    
    4. Add to _order_moves_advanced() before tactical calls:
       # Use cached tactical evaluation instead of calling detector directly
       tactical_bonus = self._get_tactical_evaluation(board, board.turn)
    
    5. Add cache clearing to uci_newgame():
       self.tactical_cache.clear()
    
    Expected Results:
    - 80-90% cache hit rate during search
    - 5-10x speed improvement in NPS
    - Reduced from 300-600 NPS to 2000-5000+ NPS
    """
    pass

if __name__ == "__main__":
    print("V11.5 Tactical Cache Performance Fix")
    print("====================================")
    print()
    print("This module provides a high-speed tactical cache to eliminate")
    print("redundant tactical pattern detection during search.")
    print()
    print("Current Problem:")
    print("- Tactical detector called 5-6 times per move") 
    print("- Each call analyzes 8+ complex tactical patterns")
    print("- Hundreds of redundant calls at depth 3")
    print("- Results in 300-600 NPS (extremely slow)")
    print()
    print("Solution:")
    print("- Cache tactical results by position (FEN)")
    print("- 80-90% expected cache hit rate")
    print("- 5-10x search speed improvement")
    print("- Target: 2000-5000+ NPS")
    print()
    print("See patch_v7p3r_with_tactical_cache() for implementation details.")