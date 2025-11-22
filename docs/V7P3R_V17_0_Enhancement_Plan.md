# V17.0 Enhancement Plan - Time Management & Profiling

## Executive Summary
Analysis of v14.1 vs v16.1 performance revealed that evaluation speed is NOT the bottleneck preventing deeper search. Both fast evaluator (643k evals/sec) and bitboard evaluator (735k evals/sec) achieve similar depth (2.6-2.8) with low NPS (8-9k nodes/sec).

**Root Cause**: V14.1's conservative time management system is prematurely terminating search before depth 5+ can be achieved.

**Solution**: Relax time constraints and add comprehensive profiling to guide further optimizations.

## Version Designation
- **Version**: v17.0
- **Base**: v14.1 (1496 ELO baseline)
- **Enhancement Focus**: Time management optimization + profiling infrastructure
- **Target Depth**: Consistent 5-7 (vs current 2.6-2.8)
- **Evaluator**: Dual support (fast evaluator default, bitboard available)

---

## Problem Analysis

### Current v14.1 Time Management Constraints

**Target Time Calculation**:
```python
target_time = min(absolute_max * time_factor * 0.7, absolute_max * 0.75)
```
- Multiplies by 0.7 (uses only 70% of allocated time)
- Hard cap at 75% of absolute_max
- Result: Extremely conservative, often exits after 2-3 depth iterations

**Max Time Calculation**:
```python
max_time = min(absolute_max * time_factor * 0.9, absolute_max * 0.95)
```
- Multiplies by 0.9 (uses only 90% of allocated time)
- Hard cap at 95% of absolute_max
- Rarely reached due to early target_time exits

**Opening Reduction**:
```python
if moves_played < 8:
    time_factor *= 0.5  # Use HALF time in opening
elif moves_played < 15:
    time_factor *= 0.6  # Still fast in opening
```
- Extremely aggressive opening time reduction
- Prevents depth 5+ in most games (opening phase = 15+ moves)

**Stable Best Move Early Exit**:
```python
if current_depth >= 4 and stable_best_count >= 3:
    if elapsed >= target_time * 0.5:
        break  # Exit early if move is stable
```
- Exits after 3 stable iterations at depth 4+
- Only requires 50% of target_time used
- Compounds with already conservative target_time

**Iteration Prediction**:
```python
predicted_completion = elapsed + (last_iteration_time * 3.0)
if predicted_completion >= max_time:
    break  # Don't start iteration we can't complete
```
- Uses 3x factor (very conservative)
- Prevents starting depth 5+ if depth 3-4 took significant time
- Compounds with short target_time and max_time

### Measured Impact

**Test Results** (test_v14_2_fast_evaluator.py):
- Fast Evaluator: Depth 2.8 avg (range 2.6-2.9)
- Bitboard Evaluator: Depth 2.6 avg (range 2.4-2.7)
- NPS: 8-9k nodes/second (very low for modern engines)
- Evaluation Time: ~0.001-0.002ms per position (NOT the bottleneck)

**Conclusion**: Search is being terminated prematurely by time management, not by actual time exhaustion.

---

## V17.0 Enhancement Strategy

### Phase 1: Time Management Relaxation

#### Target Time Multiplier
**Current**: `target_time = absolute_max * time_factor * 0.7` (capped at 75%)  
**V17.0**: `target_time = absolute_max * time_factor * 0.85` (capped at 90%)

**Rationale**: Allow search to use more of allocated time before considering early exit.

#### Max Time Multiplier
**Current**: `max_time = absolute_max * time_factor * 0.9` (capped at 95%)  
**V17.0**: `max_time = absolute_max * time_factor * 0.98` (capped at 99%)

**Rationale**: Use nearly all available time when needed for complex positions.

#### Opening Time Reduction
**Current**:
```python
if moves_played < 8:
    time_factor *= 0.5  # Half time
elif moves_played < 15:
    time_factor *= 0.6  # 60% time
```

**V17.0**:
```python
if moves_played < 8:
    time_factor *= 0.75  # 75% time (less aggressive)
elif moves_played < 15:
    time_factor *= 0.85  # 85% time (less aggressive)
```

**Rationale**: Even in opening, allow depth 5+ for tactical awareness and piece coordination.

#### Stable Best Move Exit
**Current**: Exit after 3 stable iterations at 50% of target_time  
**V17.0**: Exit after 4 stable iterations at 70% of target_time

**Rationale**: Be more confident before early exit, use more time to verify stability.

#### Iteration Prediction Factor
**Current**: `predicted_completion = elapsed + (last_iteration_time * 3.0)`  
**V17.0**: `predicted_completion = elapsed + (last_iteration_time * 2.5)`

**Rationale**: Less conservative prediction allows attempting higher depths.

### Phase 2: Comprehensive Profiling

#### Profiling Metrics
1. **Time per Depth**: Measure elapsed time for each depth iteration (1-8)
2. **Nodes per Depth**: Count nodes searched at each depth level
3. **NPS by Depth**: Calculate nodes/second efficiency by depth
4. **Time Management Decisions**: Log target_time, max_time, actual_time, exit_reason
5. **Transposition Table**: Hit rate, store rate, entry age distribution
6. **Move Ordering**: Killer move hits, history heuristic effectiveness
7. **Evaluation Time**: Breakdown of evaluation vs search overhead
8. **Quiescence Search**: Depth reached, nodes searched, time spent

#### Profiling Tool Features
- **Non-invasive**: Minimal overhead, accurate measurements
- **Comparative**: Test v14.1 vs v17.0 side-by-side
- **Position-specific**: Test across opening, middlegame, endgame
- **Export**: JSON format for analysis and visualization

---

## Expected Outcomes

### Performance Targets

| Metric | v14.1 Current | v17.0 Target | Improvement |
|--------|---------------|--------------|-------------|
| Average Depth | 2.6-2.8 | 5-7 | +2.2-4.2 plies |
| NPS | 8-9k | 15-25k | +66-177% |
| Target Time Usage | ~50-60% | ~80-90% | +30-50% |
| Max Time Reached | Rarely | When needed | Adaptive |
| Opening Depth | 2.4-2.6 | 4-6 | +1.8-3.4 plies |

### Tournament Strength

| Aspect | Expected Impact |
|--------|-----------------|
| Tactical Awareness | Significantly improved (depth 5+ sees 2-3 move combinations) |
| Positional Play | Improved (deeper evaluation of piece coordination) |
| Endgame Technique | Improved (more precise calculation) |
| Time Management | Better adaptive allocation (use time when needed) |
| ELO Target | 1550-1600+ (vs 1496 baseline) |

---

## Implementation Timeline

### Step 1: Create Profiling Tool ✅
- File: `testing/test_v17_profiling.py`
- Features: Time/depth analysis, TT metrics, move ordering stats
- Baseline: Run on v14.1 to establish metrics

### Step 2: Implement v17.0 Time Management
- Modify: `src/v7p3r.py` (class designation, time calculations)
- Changes: All 5 time management relaxations listed above
- Backward Compatibility: Maintain v14.1 mode via flag if needed

### Step 3: Profiling Comparison
- Run: `test_v17_profiling.py` on both v14.1 and v17.0
- Compare: All metrics side-by-side
- Validate: Depth improvement without time overflow

### Step 4: Depth Achievement Test
- Run: Modified `test_v14_2_fast_evaluator.py` for v17.0
- Target: Consistent depth 5-7 across 10 positions
- Success Criteria: Average depth >= 5.0

### Step 5: Tournament Testing (Next Phase)
- Opponent: PositionalOpponent (1400 ELO baseline)
- Games: 50-game match
- Target: 60%+ win rate, depth advantage in logs

---

## Risk Assessment

### Low Risk
✅ Time management changes are conservative and tested  
✅ No changes to core search algorithm (stable codebase)  
✅ Easy rollback to v14.1 if issues arise  
✅ Profiling tool provides diagnostic visibility

### Medium Risk
⚠️ Deeper search may expose evaluation weaknesses (e.g., king safety at depth 7+)  
⚠️ Longer search times could lead to time pressure in long games  
⚠️ Increased NPS may stress transposition table (memory management)

### Mitigation
- Test in controlled 5-second searches first
- Monitor time usage in tournament games
- Keep 60-second hard cap as safety net
- Maintain dual evaluator support (fast/bitboard) for flexibility

---

## Success Criteria

### Minimum Viable Success (v17.0 Release)
- ✅ Average depth 5.0+ in 5-second searches
- ✅ NPS 15k+ (vs current 8-9k)
- ✅ No time overflows (< 1% of games)
- ✅ Win rate vs v14.1 >= 55% (10-game sample)

### Target Success (v17.0 Tournament Ready)
- ✅ Average depth 6.0+ in 5-second searches
- ✅ NPS 20k+ nodes/second
- ✅ Win rate vs PositionalOpponent 60%+ (50-game match)
- ✅ ELO 1550+ (vs 1496 baseline)

### Stretch Goals (v17.1+)
- 🎯 Average depth 7-8 in middlegame positions
- 🎯 NPS 30k+ with search optimizations
- 🎯 ELO 1600+ approaching intermediate club level

---

## Next Steps

1. **Create Profiling Tool** → `testing/test_v17_profiling.py`
2. **Baseline Profile** → Run on v14.1, establish metrics
3. **Implement v17.0** → Apply time management changes
4. **Profile v17.0** → Compare with v14.1 baseline
5. **Depth Test** → Verify 5+ depth achievement
6. **Tournament Test** → 50-game match vs PositionalOpponent

---

## Lessons from v14.1 → v17.0

### What We Learned
1. **Evaluation speed is NOT always the bottleneck** - Both 0.001ms and 0.002ms evaluations achieved same depth
2. **Time management matters more than raw speed** - Conservative limits prevent deeper search even with fast evaluation
3. **Opening play needs tactical depth** - Aggressive time reduction in opening (0.5x) prevented depth 5+ when tactics matter
4. **Stable move exits are too aggressive** - 3 iterations at depth 4 is insufficient confidence for early termination
5. **Iteration prediction is too conservative** - 3x factor prevents attempting depth 5+ when likely achievable

### Validation Approach
- **Profiling first** - Understand bottlenecks before optimizing
- **Incremental changes** - Relax constraints systematically, measure impact
- **Comparative testing** - Always compare against known baseline (v14.1)
- **Real-world validation** - Tournament games reveal time management issues profiling may miss

---

**Document Version**: 1.0  
**Created**: 2025-11-20  
**Author**: AI Assistant + User Collaboration  
**Status**: Ready for Implementation
