# V17.1 Profiling Analysis & V18 Action Plan

**Date**: 2025-12-29  
**Objective**: Data-driven optimization for v18 to achieve 1800+ rating (100+ ELO gain from v17.1's ~1700)

---

## Critical Findings from Profiling

### 1. Evaluation Component Breakdown
```
Material:   34-35% of eval time (0.0163ms)
PST:        55-56% of eval time (0.0256ms)  ← BOTTLENECK
Strategic:   9-10% of eval time (0.0044ms)
Total:      0.046ms per evaluation
```

**Key Insight**: **PST is 56% of evaluation cost**, not strategic bonuses (9.5%). Our optimization targeted the wrong component!

### 2. Search Depth Scaling
```
Depth 3: 8,815 nodes   (1.0s)   - Baseline
Depth 4: 42,601 nodes  (4.9s)   - 4.8x branching
Depth 5: 92,094 nodes  (11.7s)  - 2.2x branching  
Depth 6: 92,094 nodes  (11.4s)  - 1.0x branching  ← ANOMALY
Depth 7: 129,828 nodes (15.3s)  - 1.4x branching
```

**Key Insights**:
- Effective branching factor varies 1.0x - 4.8x
- **Depth 5→6 shows NO node increase** (pruning working too well or bug?)
- Average branching ~2.5x (not 35x theoretical)
- **Good pruning already in place**

### 3. Cache Hit Rates
```
Transposition Table:  100% hit rate (when used)
Evaluation Cache:     15-20% hit rate  ← LOW
```

**Key Insight**: Eval cache is ineffective (80% misses). Either positions are unique or cache is being cleared too often.

### 4. NPS (Nodes Per Second)
```
Average: 8,000-8,500 NPS
Range:   7,200-10,200 NPS
```

**Key Insight**: Consistent NPS indicates evaluation isn't the bottleneck (would see variation if eval was slow).

---

## Root Cause Analysis

### Why Can't We Reach Depth 8?

**Not Because**:
- ✗ Evaluation is slow (0.046ms is fast)
- ✗ Strategic bonuses are expensive (only 9.5%)
- ✗ Bad caching (TT working well)

**Actually Because**:
1. **Time budget exhausted**: 10s limit reached at depth 6-7
2. **Node explosion**: Need to search smarter, not faster
3. **PST overhead**: 56% of eval time on piece-square lookups

---

## Optimization Opportunities (Ranked by Impact)

### TIER 1: High Impact (Target for v18.3)

#### 1. PST Optimization (56% of eval time)
**Current**: Array lookups + rank flipping for every piece
**Improvement**: Pre-compute flipped tables, use direct indexing
**Expected gain**: 30-40% faster evaluation
**Impact**: +0.3-0.4 ply depth

```python
# Current (slow):
if piece.color == chess.BLACK:
    rank = 7 - rank  # Flip every time
value = PAWN_PST[rank][file]  # Lookup

# Optimized:
value = PAWN_PST_WHITE[square] if white else PAWN_PST_BLACK[square]  # Direct
```

#### 2. Lazy Evaluation
**Current**: Always compute Material + PST + Strategic
**Improvement**: Skip strategic in quiescence, skip PST in pure material positions
**Expected gain**: 20-30% fewer eval calls
**Impact**: +0.2-0.3 ply depth

```python
def evaluate(board, mode='full'):
    if mode == 'material_only':  # Quiescence, desperate
        return quick_material(board)
    elif mode == 'skip_strategic':  # Time pressure
        return material + pst
    else:  # Full evaluation
        return material + pst + strategic
```

#### 3. Incremental PST Updates
**Current**: Recalculate all pieces every eval
**Improvement**: Update PST score when moves are made/unmade
**Expected gain**: 50-70% faster PST
**Impact**: +0.4-0.5 ply depth

---

### TIER 2: Medium Impact (Consider for v18.4)

#### 4. Null Move Pruning
**Current**: Not implemented
**Improvement**: Try "passing" and see if position still fails high
**Expected gain**: 2-3x fewer nodes in many positions
**Impact**: +0.7-1.0 ply depth
**Risk**: Medium (zugzwang positions, tactical misses)

#### 5. History Heuristic Improvements
**Current**: Simple counter
**Improvement**: Butterfly tables, move-pair heuristic
**Expected gain**: 10-15% better move ordering
**Impact**: +0.2-0.3 ply depth

#### 6. Evaluation Cache Improvements
**Current**: 15-20% hit rate
**Improvement**: Larger cache, better key generation, LRU eviction
**Expected gain**: 40-60% hit rate
**Impact**: +0.1-0.2 ply depth

---

### TIER 3: Code Quality (No depth impact, better maintainability)

#### 7. Eliminate Redundancy
- Multiple calls to `_is_endgame()` in same evaluation
- Repeated `piece_at()` calls for same squares
- Duplicate pawn structure checks

#### 8. Profile-Guided Optimization
- Move frequently-accessed data to class attributes
- Inline small helper functions
- Use array operations where possible

---

## Recommended V18.3 Implementation Plan

### Phase 1: PST Optimization (Week 1)
**Goal**: 30-40% faster PST evaluation

1. **Pre-compute flipped tables**:
   ```python
   PAWN_PST_BLACK = [[PAWN_PST[7-r][f] for f in range(8)] for r in range(8)]
   ```

2. **Direct square indexing**:
   ```python
   # Instead of rank/file -> lookup -> flip
   value = PST_DIRECT[piece_type][color][square]  # One lookup
   ```

3. **Test**: Verify 30%+ speedup in eval benchmark
4. **Validate**: 100% move parity with v17.1

**Expected**: Evaluation 0.046ms → 0.032ms (+40% faster eval, +4% total search)

---

### Phase 2: Lazy Evaluation (Week 2)
**Goal**: Skip expensive components when not needed

1. **Evaluation Modes**:
   - `MATERIAL_ONLY`: Quiescence search, material-only positions
   - `FAST`: Skip strategic bonuses (time pressure, desperate)
   - `FULL`: Everything (normal search)

2. **Dynamic Selection**:
   ```python
   if in_quiescence or abs(material_balance) > 300:
       mode = MATERIAL_ONLY
   elif time_pressure or desperate:
       mode = FAST
   else:
       mode = FULL
   ```

3. **Test**: Measure eval call reduction (expect 20-30%)
4. **Validate**: Win rate vs v17.1 >= 100%

**Expected**: Fewer expensive evals → +0.2-0.3 ply depth

---

### Phase 3: Incremental PST (Week 3-4)
**Goal**: Update PST scores instead of recalculating

1. **Add PST Tracking**:
   ```python
   class SearchState:
       pst_score: int  # Maintained across make/unmake
   ```

2. **Update on Move**:
   ```python
   def make_move(move):
       pst_score -= PST[piece][from_square]
       pst_score += PST[piece][to_square]
       if captured:
           pst_score -= PST[captured][to_square]
   ```

3. **Test**: Verify exact scores match full calculation
4. **Validate**: Search produces identical results

**Expected**: PST calculation nearly free → +0.4-0.5 ply depth

---

### Combined Expected Gains

| Optimization | Eval Speedup | Total Speedup | Depth Gain |
|--------------|--------------|---------------|------------|
| PST Pre-compute | +40% eval | +4% search | +0.1 ply |
| Lazy Eval | -25% eval calls | +5% search | +0.2 ply |
| Incremental PST | +60% PST | +6% search | +0.2 ply |
| **TOTAL** | - | **+15% search** | **+0.5 ply** |

**Projected Depth**: 6.0 → 6.5 (realistic, conservative estimate)

**Not enough for depth 8**, but:
- Clean foundation for Tier 2 optimizations (NMP, better history)
- Tier 1 + Tier 2 combined → depth 7.5-8.0 realistic

---

## Quality Gates (Non-Negotiable)

### Before ANY v18 deployment:

1. **100% Move Parity Test**:
   - v18 must make identical moves to v17.1 on 50 test positions
   - Any divergence requires analysis and justification

2. **50-Game Tournament**:
   - v18 vs v17.1 head-to-head
   - Minimum 48% win rate (within statistical noise)
   - NO critical blunders or time forfeits

3. **Regression Suite**:
   - All historical failure cases must pass
   - Mate-in-3 detection (v17.4 failure)
   - Endgame conversion (R+B vs K)
   - Time management (no forfeits)

4. **Performance Baseline**:
   - Depth >= v17.1 in all profiles
   - NPS within 10% of v17.1
   - Win rate >= v17.1 in 100-game sample

**If ANY gate fails**: Rollback and analyze before proceeding.

---

## Implementation Strategy

### Conservative Approach (Recommended)
1. **One optimization at a time**
2. **Test after each change**
3. **Validate against v17.1 after each change**
4. **Only proceed if no regressions**

### Timeline
- **Week 1**: PST pre-compute + validation
- **Week 2**: Lazy evaluation + validation  
- **Week 3**: Incremental PST + validation
- **Week 4**: Combined testing + 100-game tournament

### Rollback Plan
- Git tag before each optimization
- Keep v17.1 binary for rapid comparison
- Automated tests run on every change

---

## Success Metrics

### v18.3 (After Tier 1 Optimizations)
- **Depth**: 6.5 average (vs 6.0 baseline)
- **NPS**: 9,500-10,000 (vs 8,500 baseline)
- **Win Rate vs v17.1**: 52-55% (slight improvement)
- **Rating**: 1720-1750 (maintaining v17.1's trajectory)

### v18.5 (After Tier 2 Optimizations)
- **Depth**: 7.5-8.0 average
- **NPS**: 10,000-11,000
- **Win Rate vs v17.1**: 60-65% (significant improvement)
- **Rating**: 1800+ (TARGET ACHIEVED)

---

## Next Immediate Actions

1. ✅ **Profile v17.1** - DONE
2. ⏭️ **Implement PST pre-compute** - Start here
3. ⏭️ **Benchmark PST improvement** - Verify 30%+ speedup
4. ⏭️ **Test move parity** - Ensure identical to v17.1
5. ⏭️ **50-game validation** - Confirm no regressions

**Start with PST optimization - it's safe, measurable, and has clear impact.**
