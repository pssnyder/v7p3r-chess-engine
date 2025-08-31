# V10.0 Development Log - PERFORMANCE ISSUE FIXED

## Move Ordering Performance Issue - RESOLVED ✅

**Problem**: Enhanced move ordering caused 13% NPS slowdown (21-23k vs 24k baseline)
**Root Cause**: Over-engineered MVV-LVA implementation with too much overhead per move
**Solution**: Reverted to V7.0's actual design philosophy

### What Was Wrong:
```python
# SLOW - Too much work per move
for move in moves:
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)        # Board lookup
        attacker = board.piece_at(move.from_square)    # Board lookup  
        victim_value = self.piece_values.get(...)      # Dict lookup
        attacker_value = self.piece_values.get(...)    # Dict lookup
        # + comparisons and list management
```

### What V7.0 Actually Intended:
**"Simple Move Ordering: Captures first, then other moves"** - No complex evaluation!

### Fixed Implementation:
```python
# FAST - Minimal overhead
def _order_moves_enhanced(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    """V7.0 simple move ordering: captures first, then other moves"""
    if len(moves) <= 2:
        return moves
    
    captures = []
    other_moves = []
    
    for move in moves:
        if board.is_capture(move):  # Only check - no evaluation
            captures.append(move)
        else:
            other_moves.append(move)
    
    return captures + other_moves  # Simple grouping
```

### Performance Results:
- **V7.0 baseline**: ~24,000 NPS
- **Complex MVV-LVA**: ~21,000-23,000 NPS (❌ 13% slower)
- **V7.0-style simple**: ~23,000-24,000 NPS (✅ Performance restored!)

### Key Learning:
**Your original V7.0 design was right**: Move ordering overhead must be minimal to provide net benefit. The cost of ordering should never exceed the benefit of better cutoffs.

## V10.0 Status - Back on Track:
- **Foundation**: ✅ V7.0-style engine (24k NPS)
- **Enhancement #1**: ✅ V7.0-style move ordering (24k NPS maintained)
- **Performance**: ✅ Baseline NPS restored
- **Alignment**: ✅ Now matches your original V7.0 design intent

**Ready for next incremental improvement**: Simple killer moves or basic transposition table
