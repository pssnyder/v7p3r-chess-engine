# V7P3R V10 Feature Restoration Plan

## Current Status
- ✅ Clean codebase (399 lines, no duplicates)
- ✅ Performance target achieved (15,000+ NPS)
- ✅ Bitboard evaluation working
- ✅ Basic search functionality

## Phase 1: Core Search Features (High Value, Low Risk)
### Step 1: Transposition Table
- **Goal**: Add TT back for massive search speedup
- **Expected Impact**: 2-5x performance improvement
- **Risk**: Low - well-tested feature

### Step 2: Enhanced Move Ordering
- **Goal**: Add back MVV-LVA, checks, killer moves
- **Expected Impact**: Better move ordering = fewer nodes
- **Risk**: Medium - complex but known working

### Step 3: Null Move Pruning
- **Goal**: Add back null move pruning for search reduction
- **Expected Impact**: 20-30% node reduction
- **Risk**: Medium - needs careful tuning

## Phase 2: Tactical Features (Medium Value, Medium Risk)
### Step 4: Late Move Reduction (LMR)
- **Goal**: Search later moves at reduced depth
- **Expected Impact**: 15-25% speedup
- **Risk**: Medium - can cause tactical blindness

### Step 5: Quiescence Search
- **Goal**: Search captures at leaf nodes for tactical stability
- **Expected Impact**: Much stronger tactical play
- **Risk**: High - can slow down significantly if not tuned

## Phase 3: Advanced Features (High Value, High Risk)
### Step 6: Principal Variation (PV) Extraction & Following
- **Goal**: Show full PV line AND implement PV following optimization
- **Expected Impact**: Better debugging + 50-80% time savings on expected moves
- **Risk**: Low-Medium - PV following needs careful validation

### Step 7: Aspiration Windows
- **Goal**: Narrow search windows for speed
- **Expected Impact**: 10-20% speedup
- **Risk**: Medium - can cause research overhead

### Step 8: Tactical Pattern Detection
- **Goal**: Add back pin, fork, skewer detection as bonuses
- **Expected Impact**: Stronger tactical play
- **Risk**: High - was causing major slowdown before

## Testing Strategy
After each step:
1. Run performance test (target: maintain 12,000+ NPS)
2. Run tactical test positions
3. Compare strength vs previous version
4. If performance drops below 10,000 NPS, rollback and optimize

## Success Criteria
- Maintain 12,000+ NPS minimum
- Pass all tactical test positions
- Stronger play than V9.2
- Stable, no crashes or infinite loops
