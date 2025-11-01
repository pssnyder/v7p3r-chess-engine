# V7P3R Performance Regression Analysis
## October 26, 2025 Tournament Data

### Tournament Results Summary (70 games each)

| Rank | Engine | Score | Win% | Key Observations |
|------|--------|-------|------|------------------|
| 1 | Stockfish 1% | 58.0/70 | 82.9% | Baseline comparison |
| 2 | C0BR4_v2.9 | 48.0/70 | 68.6% | Competitor |
| 3 | **V7P3R_v14.0** | 47.0/70 | **67.1%** | ✅ **BEST V7P3R** |
| 4 | **V7P3R_v14.2** | 42.5/70 | **60.7%** | ⚠️ Regression from 14.0 |
| 5 | **V7P3R_v12.6** | 40.0/70 | **57.1%** | Older baseline |
| 6 | SlowMate_v3.1 | 31.5/70 | 45.0% | |
| 7 | **V7P3R_v14.3** | 12.0/70 | **17.1%** | ❌ **CATASTROPHIC FAILURE** |
| 8 | Random | 1.0/70 | 1.4% | |

### Version-by-Version Analysis

#### V12.6 (Stable Baseline) - 57.1%
- **Status**: Older bitboard-based version
- **Performance**: Solid 57.1% (40/70 points)
- **Search depth**: Consistently reaches depth 3-4
- **Time management**: 1-12 seconds per move, reasonable allocation
- **Evaluation**: Basic bitboard evaluation, no phase detection

#### V14.0 (Peak Performance) - 67.1%
- **Status**: ✅ **BEST PERFORMING VERSION**
- **Performance**: 67.1% (47/70 points) - **10 percentage points better than V12.6**
- **Key features**: 
  - Consolidated performance optimizations
  - Tactical move ordering working well
  - Reasonable time management
  - Good search depth achievement
- **Conclusion**: This is the version to use as base for future work

#### V14.1 (Minor Regression) - 53.3%
- **Status**: ⚠️ Starting to degrade
- **Performance**: 53.3% (16/30 in limited tournament)
- **Issues**: Unknown changes that reduced effectiveness vs V14.0

#### V14.2 (Continued Regression) - 60.7%
- **Status**: ⚠️ Further degradation
- **Performance**: 60.7% (42.5/70 points) - **6.4 percentage points worse than V14.0**
- **Issues**: Performance optimizations that backfired

#### V14.3 (Catastrophic Failure) - 17.1%
- **Status**: ❌ **CRITICAL FAILURE**
- **Performance**: 17.1% (12/70 points) - **MASSIVE 50 percentage point drop from V14.0**
- **Root cause identified**: "Emergency time management" changes
- **Problem analysis**:
  - Moves played at **depth 1-3 only** (should be 4-6)
  - Emergency time limits TOO AGGRESSIVE (60-70% thresholds)
  - Search terminated prematurely on almost every move
  - Example from PGN: `(Ng8-f6 f2-f3 Nb8-c6) -0.08/3 1` - Only depth 3, 1 second
  - V12.6 same position: `(e2-e3 e7-e6 Bf1-b5 Bf8-b4) 0.00/4 2` - Depth 4, 2 seconds

**V14.3 Time Management Issue:**
```python
# V14.3 changes (from v7p3r.py):
if elapsed > time_limit * 0.6:  # ULTRA-AGGRESSIVE 60% limit
    self.emergency_stop_flag = True
    return self._evaluate_position(board), None
```
This causes search to stop at 60% of allocated time, preventing depth achievement.

#### V14.4 (Known Regression) - Not in tournament
- **Status**: ❌ Documented regression
- **Issues**: 
  - UCI output broken
  - Safety prioritization destroying move ordering
  - Excessive blundering

#### V14.5 (UCI Fixes) - Not in tournament
- **Status**: ⚠️ Fixes but incomplete
- **Changes**: 
  - Fixed UCI output
  - Disabled safety prioritization
  - Relaxed emergency limits to 85%
- **Concerns**: May still have time management issues

#### V14.6 (Phase-Based Evaluation) - Not in tournament
- **Status**: ⚠️ Unproven
- **Changes**: 
  - Phase-based dynamic evaluation
  - Different heuristics per game phase
- **Test results**: 137% overall NPS target BUT user reports blunders
- **Issue**: Phase-based changes may have affected evaluation accuracy

#### V14.7 (Blunder Prevention) - Just implemented
- **Status**: ⚠️ Too aggressive filtering
- **Changes**: 
  - Pre-filtering unsafe moves before ordering
  - Four-part safety check (king, queen, pieces, captures)
- **Test results**: 
  - Starting position: 20/20 moves safe (good)
  - But positions only having 1-2 safe moves out of 20-30 legal
  - Likely rejecting good tactical moves
- **Issue**: Safety filter too strict, may reject winning tactics

### Root Causes of Regression

#### Primary Issue: V14.3 Time Management (50 point drop)
**Symptoms:**
- Depth 1-3 searches instead of 4-6
- 1 second per move instead of 2-12 seconds
- Ultra-aggressive 60% emergency stop threshold

**Fix Required:**
- Revert to V14.0 time management
- Remove ultra-conservative emergency limits
- Allow proper iterative deepening

#### Secondary Issue: Over-Optimization (V14.1, V14.2)
**Symptoms:**
- Gradual performance degradation
- Each "optimization" made things worse

**Pattern:**
- V14.0 → V14.1: -13.8 percentage points
- V14.1 → V14.2: +7.4 percentage points (partial recovery)
- V14.2 → V14.3: -43.6 percentage points (catastrophic)

**Lesson:** More code ≠ better performance. Simpler is better.

#### Tertiary Issue: Blunder Prevention Implementation
**V14.7 Safety Filter:**
- Too aggressive - rejects most moves
- May filter out winning tactical sequences
- Needs to be much more permissive

**Correct Approach:**
- Only filter OBVIOUS blunders (hanging queen/king)
- Allow pieces under attack if defended or winning trade
- Prioritize depth over safety filtering

### Recommendations for V14.8

#### 1. Use V14.0 as Base
- V14.0 achieved 67.1% (best performance)
- Stable time management
- Good search depth
- Working tactical ordering

#### 2. Simplify Safety Checks
- Comment out V14.7's aggressive pre-filtering
- Implement MINIMAL blunder detection:
  - Check 1: King hanging (undefended in check)
  - Check 2: Queen hanging with no compensation
  - Check 3: Obvious losing captures (queen for pawn)
- Run safety checks AT ROOT LEVEL ONLY (not in recursive search)

#### 3. Keep Bitboard Evaluation (Simplified)
- Remove phase-based complexity from V14.6
- Use flat evaluation weights (no dynamic phase adjustments)
- Focus on core terms:
  - Material
  - King safety
  - Piece mobility
  - Pawn structure (simplified)

#### 4. Time Management from V14.0
- Revert all V14.3/V14.5 emergency time changes
- Use V14.0's proven time allocation
- Allow full iterative deepening to target depth

#### 5. Testing Protocol
- Test V14.8 vs V14.0 (should be comparable or better)
- Test V14.8 vs V12.6 (should be significantly better)
- Test V14.8 vs V14.3 (should destroy it)
- Measure depth achievement (target: 4-6 in middlegame)

### Success Criteria for V14.8

✅ **Performance Targets:**
- Win rate vs V14.0: ≥45% (comparable)
- Win rate vs V12.6: ≥55% (better than baseline)
- Win rate vs V14.3: ≥85% (vastly superior)
- Tournament score: ≥65% (better than V14.0's 67%)

✅ **Depth Targets:**
- Opening: Depth 3-4
- Middlegame: Depth 4-6
- Endgame: Depth 5-8

✅ **Blunder Prevention:**
- Zero hanging queen blunders
- Zero hanging king blunders (illegal moves)
- <10% minor piece blunders

### Conclusion

The V14.x series suffered from **over-optimization** and **over-conservative time management**. V14.3's emergency time limits caused a catastrophic 50 percentage point drop by preventing proper search depth.

**V14.8 Strategy:**
1. Return to V14.0 foundation (67.1% performance)
2. Add MINIMAL blunder prevention (only obvious blunders)
3. Remove complex phase-based evaluation
4. Simplify, simplify, simplify

**Philosophy:** "Perfect is the enemy of good" - V14.0 was working. Don't fix what isn't broken.
