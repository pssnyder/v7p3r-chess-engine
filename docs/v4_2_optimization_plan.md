# V7P3R Chess Engine v4.2 Performance Optimization Plan

## Performance Analysis

### Current State
- **Baseline perft(4)**: 4,626,791 nodes in 20.77s (222,779 NPS)
- **5-second capacity**: ~1.1M nodes
- **Required reduction**: 76% to meet 5-second time control
- **Current issue**: Engine finds tactical shots immediately, doesn't stress-test search

### Performance Targets
- **Primary goal**: Reduce search tree from 4.6M to <1.1M nodes (76% reduction)
- **Secondary goal**: Maintain or improve tactical strength
- **Time control**: Games under 300 seconds (5 minutes total)
- **Target depth**: 4-5 at 5 seconds per move

## Targeted Optimizations (Minimal Changes)

### 1. Search Algorithm Optimizations

#### A. Repetition Check Optimization
**Current**: Checks `board.is_repetition(2)` on every node
**Problem**: Expensive operation called millions of times
**Solution**: Check repetition only at depth > 2 or when material is equal

#### B. Alpha-Beta Pruning Enhancement
**Current**: Basic alpha-beta with move ordering
**Problem**: Not achieving expected 90%+ reduction
**Solution**: Ensure moves are properly ordered for maximum cutoffs

#### C. Early Termination
**Current**: No early termination for clearly won/lost positions
**Solution**: Add cutoffs for major material advantages (>1500 centipawns)

### 2. Move Ordering Improvements

#### A. Remove Move Limit Restriction
**Current**: Limited to `max_ordered_moves = 10`
**Problem**: Missing good moves in positions with many captures
**Solution**: Remove or increase limit for positions with hanging pieces

#### B. Better Capture Prioritization
**Current**: MVV-LVA with hanging piece detection
**Enhancement**: Prioritize captures that win material, demote equal trades

#### C. Killer Move Heuristic
**Addition**: Remember moves that caused cutoffs at each depth

### 3. Evaluation Optimizations

#### A. Lazy Evaluation
**Addition**: Quick material count before full evaluation
**Benefit**: Skip expensive evaluation when material difference is huge

#### B. Reduce Evaluation Complexity
**Current**: Complex multi-phase evaluation
**Optimization**: Simplify evaluation at non-critical nodes

## Implementation Priority

### Phase 1: Critical Path (Immediate Impact)
1. **Repetition check optimization** - 10-20% speedup expected
2. **Remove move ordering limits** - Better alpha-beta efficiency
3. **Early termination for mate/material** - Skip hopeless branches

### Phase 2: Search Enhancements
1. **Killer move heuristic** - Improve move ordering
2. **Lazy evaluation** - Reduce evaluation overhead
3. **Better capture ordering** - More alpha-beta cutoffs

### Phase 3: Fine-tuning
1. **Adjust search parameters** based on testing
2. **Profile and optimize** remaining bottlenecks

## Risk Assessment

### Low Risk Changes
- Repetition check optimization
- Early termination additions
- Move ordering improvements

### Medium Risk Changes
- Evaluation simplification
- Search algorithm modifications

### Mitigation Strategy
- Test each change individually
- Preserve current tactical strength
- Benchmark against known positions

## Success Metrics

### Performance
- [ ] Achieve <1.1M nodes searched in test position
- [ ] Maintain >200K NPS performance
- [ ] Enable depth 4-5 search in 5 seconds

### Quality
- [ ] Preserve tactical awareness (still find hanging pieces)
- [ ] Maintain positional understanding
- [ ] No regression in test games vs Stockfish

## Testing Plan

1. **Performance testing** with standard perft positions
2. **Tactical testing** with puzzle positions
3. **Speed testing** with time-controlled games
4. **Regression testing** against previous version

This plan focuses on achieving the 76% search reduction needed for faster time controls while preserving the engine's tactical and positional strength.
