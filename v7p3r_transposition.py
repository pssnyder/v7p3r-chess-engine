"""
V7P3R Chess Engine - Simple Transposition Table
A lightweight FEN-based transposition table for position caching.
"""

import time
from typing import Optional, Dict, Tuple, Any

class V7P3RTranspositionEntry:
    """Simple entry for storing position information."""
    
    def __init__(self, depth: int, score: int, best_move=None, node_type: str = "exact"):
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.node_type = node_type  # "exact", "alpha", "beta"
        self.timestamp = time.time()

class V7P3RTranspositionTable:
    """
    Simple FEN-based transposition table.
    Keeps things straightforward and compatible with existing engine style.
    """
    
    def __init__(self, max_size: int = 2000):
        self.table: Dict[str, V7P3RTranspositionEntry] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get_position_key(self, board) -> str:
        """Get a simplified FEN key for position lookup."""
        # Remove move counters for transposition detection
        fen_parts = board.fen().split()
        # Keep position, active color, castling, en passant
        # Remove halfmove and fullmove counters
        return " ".join(fen_parts[:4])
    
    def store(self, board, depth: int, score: int, best_move=None, node_type: str = "exact"):
        """Store a position in the transposition table."""
        key = self.get_position_key(board)
        
        # If table is full, make room by removing oldest entry
        if len(self.table) >= self.max_size and key not in self.table:
            self._make_room()
        
        # Store or update entry
        self.table[key] = V7P3RTranspositionEntry(depth, score, best_move, node_type)
    
    def lookup(self, board, depth: int, alpha: int, beta: int) -> Optional[Tuple[Optional[int], Any]]:
        """
        Look up a position in the transposition table.
        Returns (score, best_move) if usable, None otherwise.
        """
        key = self.get_position_key(board)
        
        if key not in self.table:
            self.misses += 1
            return None
        
        entry = self.table[key]
        
        # Only use if we searched to at least the same depth
        if entry.depth < depth:
            self.misses += 1
            return None
        
        self.hits += 1
        
        # Check if we can use this score
        if entry.node_type == "exact":
            return (entry.score, entry.best_move)
        elif entry.node_type == "alpha" and entry.score <= alpha:
            return (entry.score, entry.best_move)
        elif entry.node_type == "beta" and entry.score >= beta:
            return (entry.score, entry.best_move)
        
        # Can't use score, but return best move if available
        return (None, entry.best_move)
    
    def _make_room(self):
        """Remove oldest entry to make room for new one."""
        if not self.table:
            return
        
        oldest_key = min(self.table.keys(), 
                        key=lambda k: self.table[k].timestamp)
        del self.table[oldest_key]
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transposition table statistics."""
        total_lookups = self.hits + self.misses
        hit_rate = (self.hits / total_lookups * 100) if total_lookups > 0 else 0
        
        return {
            "size": len(self.table),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }

# Global transposition table instance
tt = V7P3RTranspositionTable()

def get_transposition_table():
    """Get the global transposition table instance."""
    return tt

def clear_transposition_table():
    """Clear the global transposition table."""
    tt.clear()
