# V13.x Move Ordering Implementation Plan

## Executive Summary
Based on weakness analysis of V12.6 revealing 75% bad move ordering and 70% tactical misses, this plan implements aggressive move pruning in V13.x to achieve **83% search tree reduction** and **5.9x speedup**.

## Problem Analysis
### Current V13.0 Issues
- `_order_moves_advanced()` includes ALL legal moves in search
- No pruning of weak quiet moves
- Search tree too large in complex positions (38+ moves in middlegame)
- Categories: TT→Nudge→Captures→Checks→Tactical→Killers→**ALL Quiet Moves**

### V12.6 Weakness Statistics
- 75% of bad moves were not in top 5 engine choices
- 70% tactical opportunities missed
- 38% overall weakness rate in analyzed games
- Engine considers too many irrelevant quiet moves

## V13.x Solution: Focused Move Ordering

### Core Philosophy
**"Most positions have 3-8 critical moves that matter"**
- Search only high-impact moves in main tree
- Defer quiet moves to separate "waiting moves" list
- Use quiescence search for quiet move evaluation
- Dramatically reduce branching factor

### Priority System
```
1. CHECKS (1000+ points) - V12.6 missed these
2. KING SAFETY (800+ points) - Save king from attacks  
3. PIECE SAFETY (700+ points) - Save hanging pieces
4. GOOD CAPTURES (600+ points) - MVV-LVA with safety
5. TACTICAL THREATS (500+ points) - Pins, forks, skewers
6. DEVELOPMENT (300+ points) - Opening piece development
7. CASTLING (350+ points) - King safety
8. PROMOTIONS (300+ points) - Pawn promotion
9. CENTER CONTROL (80+ points) - Early game only
10. QUIET MOVES (5+ points) - Heavily penalized
```

### Pruning Thresholds
- **Critical Moves (400+ points)**: Must be searched
- **Important Moves (200+ points)**: Include up to 4 total  
- **Quiet Moves (100+ points)**: Only if <2 critical moves
- **Maximum**: 6 critical moves per position
- **Minimum**: 2 moves per position (unless fewer legal)

## Implementation Strategy

### Phase 1: Replace Move Ordering Function
**File**: `src/v7p3r.py`
**Function**: `_order_moves_advanced()` (lines 539-639)

**Current Code Pattern**:
```python
def _order_moves_advanced(self, board, legal_moves, depth=0):
    # ... existing categorization ...
    return tt_moves + nudge_moves + captures + checks + tactical_moves + killers + quiet_moves
```

**New V13.x Code Pattern**:
```python
def _order_moves_v13x(self, board, legal_moves, depth=0, tt_move=None):
    orderer = V13FocusedMoveOrderer()
    critical_moves, waiting_moves = orderer.order_moves_v13_focused(board, legal_moves, depth, tt_move)
    
    # Store waiting moves for later use
    self.waiting_moves = waiting_moves
    
    return critical_moves  # Only return critical moves for main search
```

### Phase 2: Add Waiting Move Support
**New Methods**:
```python
def get_waiting_moves(self):
    """Return quiet moves for zugzwang/time situations"""
    return getattr(self, 'waiting_moves', [])

def should_use_waiting_moves(self, position_score, time_remaining):
    """Determine if waiting moves should be considered"""
    # Use in specific situations:
    # 1. Zugzwang positions (no good moves)
    # 2. Time pressure (need fast move)
    # 3. Quiescence search extension
    return position_score < -50 or time_remaining < 10.0
```

### Phase 3: Integration Points
**Search Function Integration**:
- Modify `search()` function to use only critical moves
- Add waiting move fallback for special cases
- Update quiescence search to handle quiet moves

**Time Management Integration**:
- Use waiting moves when time is short
- Prioritize critical moves in time pressure

## Expected Performance Improvements

### Search Tree Reduction
- **Opening**: 80% reduction (20 → 4 moves)
- **Middlegame**: 84% reduction (38 → 6 moves)  
- **Tactical**: 84% reduction (37 → 6 moves)
- **Overall**: 83% reduction with 5.9x speedup

### Tactical Accuracy Improvement
- Focus on checks, captures, attacks, threats
- Eliminate weak quiet moves causing confusion
- Better handling of hanging pieces (27.7% miss rate in V12.6)
- Improved king safety evaluation

### NPS Performance
- Current V13.0: ~1762 NPS
- Expected V13.x: ~10,400 NPS (5.9x improvement)
- Maintains tactical accuracy while dramatically increasing speed

## Risk Assessment

### Low Risk Areas
- Algorithm is additive (waiting moves preserved)
- Maintains all existing move categories
- Can be rolled back easily
- No changes to evaluation functions

### Medium Risk Areas
- Integration with existing search algorithms
- Transposition table compatibility
- Time management coordination

### Mitigation Strategies
1. **Gradual Implementation**: Replace move ordering first, then add features
2. **A/B Testing**: Compare V13.0 vs V13.x on test positions
3. **Fallback Option**: Keep original `_order_moves_advanced` as backup
4. **Validation**: Test against known tactical positions

## Testing Plan

### Phase 1 Testing
1. **Tactical Test Suite**: Run on 100 tactical positions
2. **Game Replay**: Test on historical V12.6 weakness positions
3. **Perft Testing**: Verify move generation accuracy
4. **NPS Measurement**: Confirm expected speedup

### Phase 2 Testing  
1. **Engine vs Engine**: V13.0 vs V13.x matches
2. **Tournament Testing**: Multi-engine tournament
3. **Weakness Analysis**: Re-run weakness analyzer on V13.x
4. **Long Games**: Test endgame handling

### Phase 3 Validation
1. **Stockfish Comparison**: Move quality vs Stockfish
2. **Opening Book**: Test against known opening theory
3. **Blitz Performance**: High-speed game testing
4. **Memory Usage**: Verify no memory leaks

## Implementation Timeline

### Immediate (Next Session)
- [ ] Copy V13FocusedMoveOrderer class to v7p3r.py
- [ ] Replace _order_moves_advanced function
- [ ] Basic integration testing

### Short Term (1-2 Sessions)  
- [ ] Add waiting move support
- [ ] Integrate with search function
- [ ] Time management updates
- [ ] Comprehensive testing

### Medium Term (3-5 Sessions)
- [ ] Performance optimization
- [ ] Edge case handling  
- [ ] Tournament validation
- [ ] Documentation updates

## Success Metrics

### Performance Targets
- **NPS Improvement**: >5x current speed (target: 8,800+ NPS)
- **Tactical Accuracy**: <30% tactical miss rate (vs 70% in V12.6)  
- **Move Quality**: >50% good moves in top 3 (vs 25% in V12.6)
- **Search Efficiency**: <20% of moves searched on average

### Validation Criteria
- Passes all existing tactical tests
- Defeats V13.0 in head-to-head matches
- Shows improvement in weakness analyzer
- Maintains or improves Elo rating

## User Approval Required

**Before proceeding with implementation, please confirm**:

1. **Approach**: Do you approve the aggressive pruning strategy?
2. **Risk Level**: Are you comfortable with modifying the core move ordering?
3. **Testing**: Should we implement full system or start with limited testing?
4. **Rollback**: Do you want engine_freeze backup before changes?
5. **Timeline**: Proceed immediately or wait for additional analysis?

**Key Decision Points**:
- Replace `_order_moves_advanced` function entirely
- Implement waiting moves system  
- Target 83% search reduction
- Accept risk of temporary instability during integration

---

*This plan addresses V12.6's critical weakness: "engine is leaving them grouped by piece" and including too many quiet moves. The V13.x system focuses search on your priority: "checks, king safety, pieces under attack, captures, development, tactics"*