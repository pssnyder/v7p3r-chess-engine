# V19.1 Emergency Performance Fix

**Date**: April 22, 2026  
**Branch**: v19.1-emergency-perf-fix  
**Critical**: YES - Production engine timing out on 30% of games

---

## Problem Statement

v19.0 showed promise (48.3% vs v18.4) but has **catastrophic performance regression**:

- **Expected**: 100,000+ nodes/sec, depth 8-10 in 5 seconds
- **Actual**: 6,800 nodes/sec, depth 3-6 in 5 seconds
- **Impact**: 9/30 games (30%) timed out, engine hangs on complex positions
- **Cause**: Expensive move safety checks (0.083ms/move) called for ALL moves at depth ≥3

---

## Root Cause Analysis

### Performance Profiling Results

```
Component Benchmarks:
├─ Move Generation:      0.038ms ✓ OK
├─ Evaluation:           0.001ms ✓ EXCELLENT  
├─ Quiescence Search:    0.91ms  ✓ OK
└─ Move Safety Check:    0.083ms ✗ TOO EXPENSIVE

Impact per position (35 moves):
- Move safety overhead: 35 × 0.083ms = 2.9ms
- This is 2,400x more expensive than evaluation!
- At 6,800 NPS, move ordering happens ~6,800 times/sec
- Total overhead: 2.9ms × 6,800 = 19.7 seconds of wasted time per second!
```

### Code Location

**src/v7p3r.py** lines 640-700 (`_order_moves` function):

```python
for move in moves:
    # V18.0: Apply safety check to all non-TT moves
    if depth >= 3:
        safety_score = self.move_safety.evaluate_move_safety(board, move)
    # ... rest of ordering logic
```

This is called **for every single move** at **every node** at depth ≥3, creating exponential overhead.

---

## Fix Strategy

### Phase 1: Remove Expensive Features (NOW)

**1. Disable Move Safety Checks in Ordering**
   - Remove `evaluate_move_safety()` calls from `_order_moves()`
   - Safety checks were defensive feature (v18.0) but cost is too high
   - Can re-add later with lazy evaluation or only for shallow depths

**2. Simplify Tactical Pattern Detection**
   - `detect_bitboard_tactics()` is also called per move
   - Simplify or remove from hot path
   - Keep for root move selection only if needed

**3. Remove Depth ≥3 Branching**
   - Current code does expensive work at depth ≥3
   - This is the opposite of what we want (do MORE work at shallow depths)
   - Simplify to consistent move ordering

### Phase 2: Expected Performance Gains

**After removing move safety overhead:**
- Per-position time: 2.9ms → ~0.1ms (29x faster)
- Expected NPS: 6,800 → 100,000+ (15x improvement)
- Expected depth in 5s: depth 3-6 → depth 8-10
- Timeout rate: 30% → 0%

---

## Implementation Plan

### Step 1: Simplify Move Ordering (15 min)

```python
def _order_moves(self, board, moves, depth, tt_move):
    """V19.1: Simplified high-performance move ordering"""
    if len(moves) <= 2:
        return moves
    
    # 1. TT move first
    if tt_move and tt_move in moves:
        moves.remove(tt_move)
        return [tt_move] + self._order_remaining_moves(board, moves, depth)
    
    return self._order_remaining_moves(board, moves, depth)

def _order_remaining_moves(self, board, moves, depth):
    """Fast ordering without expensive safety checks"""
    scored_moves = []
    
    for move in moves:
        score = 0.0
        
        # 1. Captures (MVV-LVA only, no tactics)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
            attacker = board.piece_at(move.from_square)
            attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
            score = (victim_value * 100 - attacker_value) + 10000  # Captures first
        
        # 2. Checks (simple bonus)
        elif board.gives_check(move):
            score = 5000
        
        # 3. Killer moves
        elif move in self.killer_moves.get_killers(depth):
            score = 3000
        
        # 4. History heuristic
        else:
            score = self.history_heuristic.get_history_score(move)
        
        scored_moves.append((score, move))
    
    # Sort by score descending
    scored_moves.sort(key=lambda x: x[0], reverse=True)
    return [move for _, move in scored_moves]
```

### Step 2: Test Performance (5 min)

```bash
# Quick benchmark
python testing/profile_v19_performance.py

# Expected results:
# - NPS: 80,000-120,000 (up from 6,800)
# - Depth in 5s: 8-10 (up from 3-6)
# - Component benchmarks: Move ordering <0.2ms per position
```

### Step 3: Validation Tournament (30 min)

```bash
# v19.1 vs v18.4: Should maintain 45%+ win rate with 0 timeouts
python testing/test_v19_vs_v18_4.py

# Success criteria:
# ✓ Score ≥45% (maintain strength)
# ✓ 0 timeouts (fix the hangs)
# ✓ 0 crashes (stability)
```

---

## Risk Assessment

### Low Risk Changes
✅ Removing move safety checks - feature is defensive, not critical for strength  
✅ Simplifying tactical detection - can re-add later with better implementation  
✅ Well-tested code path - just removing overhead from existing logic

### Expected Outcomes
- **Best case**: 48-52% vs v18.4, 0 timeouts, ready for deployment
- **Likely case**: 45-50% vs v18.4, 0 timeouts, slight tactical regression
- **Worst case**: 40-45% vs v18.4, still much better than v18.4's declining performance

### Rollback Plan
If v19.1 performs worse than v19.0:
1. Keep simplified ordering but add back tactical detection
2. Add move safety only at depth 1-2 (root nodes)
3. Profile again to find remaining bottlenecks

---

## Success Criteria

v19.1 is ready for production deployment when:

1. ✅ **Performance**: NPS ≥80,000 (vs current 6,800)
2. ✅ **Depth**: Reaches depth 8-10 in 5s blitz (vs current 3-6)
3. ✅ **Stability**: 0 timeouts in 30-game tournament (vs current 9/30)
4. ✅ **Strength**: Scores ≥45% vs v18.4 (maintain or improve)

---

## Timeline

- **Now**: Implement simplified move ordering (15 min)
- **+15min**: Run performance benchmark (5 min)
- **+20min**: If NPS >80k, run validation tournament (30 min)
- **+50min**: If tournament passes, tag v19.1 and deploy

Total: **~1 hour from now to deployment-ready**

---

## Next Steps After v19.1

Once performance is stable:

1. **Profile remaining bottlenecks** (quiescence search is 0.91ms - could be faster)
2. **Optimize evaluation caching** (currently good at 0.001ms but check hit rate)
3. **Add back tactical detection** (but only at root or depth 1-2)
4. **Consider search extensions** (check extension, recapture extension)
5. **Test at various time controls** (bullet, blitz, rapid, classical)

But first: **Make it fast. Make it stable. Then make it smart.**
