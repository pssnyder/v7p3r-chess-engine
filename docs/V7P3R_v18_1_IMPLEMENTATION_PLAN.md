# V7P3R v18.1 Implementation Plan
**Version**: v18.1.0 (Evaluation Tuning - Phase 1 Quick Wins)  
**Date**: December 21, 2025  
**Target**: +30 ELO improvement via evaluation parameter tuning  
**Risk Level**: MEDIUM (affects core evaluation, multiple file changes)

## Rationale

Analysis of 10 recent Lichess games (2W-2D-6L) revealed 41 mistakes with clear thematic patterns:
- **Material Imbalance** (14 occurrences) - Incorrect piece value assessments
- **King Safety** (11 occurrences) - Exposed kings not penalized enough
- **Endgame Technique** (9 occurrences) - Missed passed pawn opportunities
- **Pawn Structure** (8 occurrences) - Weak bonuses/penalties
- **Hanging Pieces** (3 occurrences) - Tactical oversights

V18.0's anti-tactical system addresses hanging pieces, but evaluation weights are 3-10x too small for other themes.

See: `docs/EVALUATION_GAP_ANALYSIS.md` for complete analysis

## Changes Overview

### Phase 1: Quick Wins (v18.1.0)
**Files Modified**:
- `src/v7p3r_bitboard_evaluator.py` - Enhanced king safety, endgame eval, material imbalance
- `src/v7p3r.py` - Version update, evaluation integration

**Target ELO Gain**: +30 ELO  
**Implementation Time**: 3-4 days  
**Testing Required**: 50+ game benchmark

---

## Implementation Details

### 1. King Safety Enhancement (src/v7p3r_bitboard_evaluator.py)

#### 1a. High-Value Attacker Penalty
**Location**: `evaluate_king_safety()` method (~line 868)

**Current Behavior**:
- Attack zone evaluation exists but weights are too weak
- Major pieces (Q, R) near king not heavily penalized

**Change**:
```python
def evaluate_king_safety(self, board: chess.Board, color: bool) -> float:
    # ... existing code ...
    
    if not is_endgame:
        # NEW: High-value attacker penalty
        king_square = board.king(color)
        king_zone = self._get_king_zone_squares(king_square)
        
        high_value_attackers = 0
        total_attackers = 0
        
        for square in king_zone:
            attackers = board.attackers(not color, square)
            total_attackers += len(attackers)
            
            for attacker_sq in attackers:
                piece = board.piece_at(attacker_sq)
                if piece and piece.piece_type in [chess.QUEEN, chess.ROOK]:
                    high_value_attackers += 1
        
        # Escalating danger penalties
        if total_attackers > 3:
            total_score -= 50 * (total_attackers - 3)
        
        if high_value_attackers > 0:
            total_score -= 100 * high_value_attackers
    
    return total_score
```

**Expected Impact**: Prevents games like c61zeXG2 (Move 90: Ke4 walking into rook danger)

#### 1b. Center King Penalty (Middlegame)
**Location**: `evaluate_king_safety()` method

**Current Behavior**:
- King centralization only evaluated in endgame
- No penalty for unmoved king in center during middlegame

**Change**:
```python
# Inside evaluate_king_safety(), middlegame section
if not is_endgame:
    # NEW: Center king penalty
    king_file = chess.square_file(king_square)
    
    # Penalty for being in center files (d, e = files 3, 4)
    if king_file in [3, 4]:
        total_score -= 30
    
    # Severe penalty if unmoved (no castling) and in center
    if not self._has_castled(board, color) and king_file in [3, 4]:
        total_score -= 80
```

**Expected Impact**: Forces earlier castling, prevents king exposure in opening

**Helper Method Needed**:
```python
def _has_castled(self, board: chess.Board, color: bool) -> bool:
    """Check if king has moved from starting square"""
    king_square = board.king(color)
    starting_square = chess.E1 if color == chess.WHITE else chess.E8
    return king_square != starting_square
```

---

### 2. Endgame Passed Pawn Boost (src/v7p3r_bitboard_evaluator.py)

#### Location: `evaluate_pawn_structure()` method (~line 530)

**Current Behavior**:
```python
# Existing: Linear passed pawn bonus (~20cp regardless of rank)
if is_passed_pawn:
    score += 20
```

**Change to Exponential Scaling**:
```python
# NEW: Exponential passed pawn bonus by rank
if is_passed_pawn:
    pawn_rank = chess.square_rank(pawn_square)
    
    # Calculate rank from promotion (0-7)
    if color == chess.WHITE:
        rank_from_promotion = 7 - pawn_rank
    else:
        rank_from_promotion = pawn_rank
    
    # Exponential bonus: 20 * 2^rank
    # 2nd rank: 20cp, 3rd: 40cp, 4th: 80cp, 5th: 160cp, 6th: 320cp
    passed_pawn_bonus = 20 * (2 ** rank_from_promotion)
    score += passed_pawn_bonus
    
    # Extra bonus if king supports the pawn (endgame only)
    if is_endgame:
        king_square = board.king(color)
        king_dist = chess.square_distance(king_square, pawn_square)
        if king_dist <= 2:
            score += 30  # King-pawn coordination
```

**Expected Impact**: 
- Fixes games like ZpfFcXRH (Move 103) where advanced g7 pawn was ignored
- Prevents grabbing pawns when should be pushing passed pawns

---

### 3. Material Imbalance - Bishop Pair (src/v7p3r_bitboard_evaluator.py)

#### Location: New method or inside `calculate_score_optimized()`

**Current Behavior**:
- No bishop pair bonus
- Bishops valued same as knights (330cp) regardless of position

**Change**:
```python
def _evaluate_bishop_pair(self, board: chess.Board, color: bool) -> float:
    """Evaluate bishop pair bonus"""
    bishops = board.pieces(chess.BISHOP, color)
    
    if len(bishops) >= 2:
        # Check if bishops are on different colors
        light_square_bishop = False
        dark_square_bishop = False
        
        for bishop_square in bishops:
            square_color = (chess.square_file(bishop_square) + 
                          chess.square_rank(bishop_square)) % 2
            if square_color == 0:
                dark_square_bishop = True
            else:
                light_square_bishop = True
        
        # Only bonus if on different colored squares
        if light_square_bishop and dark_square_bishop:
            # Bonus stronger in open positions
            piece_count = len(board.piece_map())
            if piece_count < 20:  # Open/endgame
                return 50.0
            else:  # Closed position
                return 30.0
    
    return 0.0
```

**Integration**:
Add to `calculate_score_optimized()` or call from `evaluate_king_safety()` (which returns combined score)

---

### 4. King Centralization in Endgame (src/v7p3r_bitboard_evaluator.py)

#### Location: `_evaluate_king_activity_bitboard()` method (~line 1076)

**Current Behavior**:
```python
# Existing: mobility * 5cp bonus
mobility = count_legal_king_moves()
score += mobility * 5
```

**Enhancement**:
```python
# ENHANCED: Add centralization bonus
king_file = chess.square_file(king_square)
king_rank = chess.square_rank(king_square)

# Distance from center (ideal = d4, e4, d5, e5)
center_file_dist = min(abs(king_file - 3), abs(king_file - 4))
center_rank_dist = min(abs(king_rank - 3), abs(king_rank - 4))
total_center_dist = center_file_dist + center_rank_dist

# Bonus for centralization (max 70cp for perfect center)
centralization_bonus = (4 - total_center_dist) * 10
score += centralization_bonus

# Keep existing mobility bonus
mobility = 0
# ... existing mobility code ...
score += mobility * 5
```

**Expected Impact**: 
- King moves toward center in endgames instead of staying on edge
- Prevents passive king play that misses winning chances

---

## File Changes Summary

### src/v7p3r_bitboard_evaluator.py
**Methods Modified**:
1. `evaluate_king_safety()` - Add high-value attacker penalty, center king penalty
2. `evaluate_pawn_structure()` - Change passed pawn bonus to exponential
3. `_evaluate_king_activity_bitboard()` - Add centralization bonus
4. **NEW**: `_evaluate_bishop_pair()` - Bishop pair detection and bonus
5. **NEW**: `_has_castled()` - Helper to check if king has moved
6. **NEW**: `_get_king_zone_squares()` - Helper to get 3x3 king zone

**Lines Changed**: Approximately 80-100 lines total

### src/v7p3r.py
**Changes**:
1. Update version header to v18.1.0
2. Update VERSION_LINEAGE comment

**Lines Changed**: ~10 lines

### src/v7p3r_uci.py
**Changes**:
1. Update UCI "id name" to "V7P3R v18.1"

**Lines Changed**: 1 line

---

## Testing Plan

### Phase 1: Unit Tests (Create New)
**File**: `testing/test_v18_1_evaluation.py`

**Test Cases**:
1. **King Safety**:
   - Position with queen near king → expect -100cp penalty
   - King on e1 (unmoved) vs Kg1 (castled) → expect -80cp difference
   
2. **Passed Pawns**:
   - White pawn on a6 (2 from promotion) → expect ~320cp bonus
   - White pawn on a3 (4 from promotion) → expect ~80cp bonus
   
3. **Bishop Pair**:
   - 2 bishops (opposite colors) → expect +50cp
   - 2 bishops (same color - impossible but test) → expect 0cp
   
4. **King Centralization**:
   - Endgame: Ke4 vs Kh1 → expect ~40cp difference

### Phase 2: Regression Tests
**File**: `testing/regression_suite.py` (existing)

**Must Pass**:
- ✅ Mate-in-3 detection
- ✅ R+B vs K endgame conversion
- ✅ Tactical positions (pins, forks)
- ✅ No time forfeits in test games

### Phase 3: Performance Benchmark
**File**: `testing/performance_benchmark.py` (create)

**Test Configuration**:
- **Opponent**: v18.0.0 (baseline)
- **Games**: 50 minimum
- **Time Control**: 5min+4s (blitz)
- **Opening Book**: Enabled for both
- **Engine Config**: default_config.json

**Acceptance Criteria**:
- Win Rate: ≥48% vs v18.0.0
- Blunders/Game: ≤6.0
- ACPL: ≤150
- No critical errors in logs
- **Theme Verification**:
  - King safety: No more "king walking into danger" blunders
  - Endgame: Pushes passed pawns on 6th/7th rank
  - Material: Values bishop pair positions correctly

### Phase 4: Lichess Validation (Manual)
**After successful benchmark**:
- Deploy to development bot (if available)
- Play 10-20 games manually observing behavior
- Check for regression in specific themes

---

## Rollback Plan

### If Tests Fail
1. **Regression test failure**: 
   - Identify which change broke the test
   - Revert that specific change
   - Re-test remaining changes
   
2. **Performance benchmark failure** (win rate <45%):
   - Revert all v18.1 changes
   - Deploy v18.0.0 backup
   - Analyze game logs for unexpected behavior

### If Production Issues After Deploy
1. **Immediate rollback**:
   ```bash
   # SSH into production
   sudo docker exec v7p3r-production bash -c \
     'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r_v18.1 && \
      mv /lichess-bot/engines/v7p3r.backup /lichess-bot/engines/v7p3r'
   sudo docker restart v7p3r-production
   ```

2. **Update deployment_log.json**:
   ```json
   {
     "version": "18.1.0",
     "rollback": true,
     "rollback_reason": "Production issue description",
     "rollback_to": "18.0.0"
   }
   ```

---

## Version Management

### CHANGELOG.md Entry
```markdown
## [18.1.0] - 2025-12-21

### Changed
- **King Safety**: Added high-value attacker penalty (Q/R near king: -100cp each)
- **King Safety**: Added center king penalty in middlegame (unmoved king: -80cp)
- **Endgame**: Exponential passed pawn bonus (6th rank: 320cp vs previous 20cp)
- **Endgame**: King centralization bonus (center squares: +40-70cp)
- **Material**: Bishop pair bonus (opposite-color bishops: +50cp in open positions)

### Rationale
Analysis of 10 recent Lichess games revealed evaluation weights 3-10x too small.
V18.1 tunes existing evaluation components to proper competitive values.

### Testing
- ✅ Unit tests: King safety, passed pawns, bishop pair, centralization
- ✅ Regression suite: 100% pass
- ✅ Performance: X% win rate vs v18.0 (X games)
- ✅ Theme verification: [King safety/Endgame/Material improvements confirmed]

### Known Issues
- None identified during testing

### Performance Impact
- Evaluation time: No measurable change (tuning only, no new computation)
- Target: +30 ELO vs v18.0.0
```

### deployment_log.json Entry
```json
{
  "version": "18.1.0",
  "deployed": "2025-12-XX",
  "status": "testing",
  "previous_version": "18.0.0",
  "changes": [
    "King safety: High-value attacker penalty",
    "King safety: Center king penalty (middlegame)",
    "Passed pawn: Exponential scaling by rank",
    "King centralization: Distance-from-center bonus (endgame)",
    "Bishop pair: +50cp bonus"
  ],
  "regression_tests_passed": false,
  "acceptance_criteria": {
    "win_rate": null,
    "blunders_per_game": null,
    "time_forfeit_rate": null,
    "tested": false
  },
  "rollback": false
}
```

---

## Implementation Order

### Day 1: Core Changes
1. ✅ Create `_get_king_zone_squares()` helper
2. ✅ Create `_has_castled()` helper
3. ✅ Modify `evaluate_king_safety()` - high-value attacker penalty
4. ✅ Modify `evaluate_king_safety()` - center king penalty
5. ✅ Create unit tests for king safety changes
6. ✅ Test king safety changes in isolation

### Day 2: Endgame & Material
7. ✅ Modify `evaluate_pawn_structure()` - exponential passed pawn bonus
8. ✅ Modify `_evaluate_king_activity_bitboard()` - centralization
9. ✅ Create `_evaluate_bishop_pair()` method
10. ✅ Integrate bishop pair into scoring
11. ✅ Create unit tests for endgame/material changes
12. ✅ Test all changes together

### Day 3: Integration & Testing
13. ✅ Update version numbers (v7p3r.py, v7p3r_uci.py)
14. ✅ Run full regression suite
15. ✅ Fix any broken tests
16. ✅ Create performance benchmark script
17. ✅ Run 50-game tournament vs v18.0.0

### Day 4: Validation & Documentation
18. ✅ Analyze benchmark results
19. ✅ Update CHANGELOG.md
20. ✅ Update deployment_log.json
21. ✅ Create git tag v18.1.0
22. ✅ Prepare for deployment (if tests pass)

---

## Risk Assessment

**Risk Level**: MEDIUM

**Risks**:
1. **Evaluation Imbalance**: Tuning one component too high may break balance
   - *Mitigation*: Test each change incrementally, revert if needed
   
2. **Performance Regression**: New calculations may slow evaluation
   - *Mitigation*: Profile evaluation time before/after
   
3. **Unexpected Interactions**: Multiple changes may interact negatively
   - *Mitigation*: Unit test each component, benchmark vs baseline
   
4. **Over-Tuning**: Parameters may be overcorrected
   - *Mitigation*: Use conservative values first, can increase in v18.2

**Confidence Level**: HIGH
- Changes are parameter tuning, not architectural
- All components already exist in codebase
- Clear test cases from game analysis
- Rollback plan documented

---

## Success Criteria

### Minimum Requirements (Must Meet ALL)
- ✅ Regression tests: 100% pass
- ✅ Win rate vs v18.0: ≥48%
- ✅ No increase in blunders/game
- ✅ No time management issues

### Stretch Goals
- 🎯 Win rate vs v18.0: ≥52% (+30 ELO)
- 🎯 Reduction in theme-specific mistakes:
  - King safety blunders: <2 per 10 games
  - Missed passed pawn pushes: <1 per 10 games
  - Bishop pair undervaluation: 0 occurrences

---

## Future Work (v18.2+)

**Not included in v18.1**:
- Pawn structure: Backward pawn detection
- Pawn structure: Increased isolated/doubled penalties
- Pawn structure: Pawn chain bonuses
- Endgame: Opposition detection
- Endgame: Zugzwang recognition
- Tactical: Full SEE (Static Exchange Evaluation)

**Reason**: v18.1 focuses on quick wins with minimal risk. More complex changes deferred to v18.2.

---

## Approval Checklist

Before implementation begins:
- [ ] User has reviewed this plan
- [ ] User approves version control approach (branch/commit)
- [ ] User approves backup approach (engine_freeze or manual backup)
- [ ] User confirms testing requirements
- [ ] User confirms deployment timeline
