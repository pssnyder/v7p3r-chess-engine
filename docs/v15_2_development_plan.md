# V7P3R v15.2 Development Plan

## Critical Issues in V15.1

### Problem Analysis from MaterialOpponent Loss

**Game**: V15.1 vs MaterialOpponent (0-1)
**Critical Blunders**:
1. Move 18: `Kd1` - King walks into danger unnecessarily
2. Move 20: `Nd5` - Knight moves to attacked square, immediately captured by queen
3. Move 23: `Bxf7` - Desperate bishop sacrifice for pawn

### Root Causes

1. **Broken Material Floor** (Line 456)
   ```python
   score = max(pst_score, material_balance) if board.turn == chess.WHITE else min(pst_score, material_balance)
   ```
   - Logic is BACKWARDS - creates inflated scores instead of material floor
   - Applied at evaluation time = search tree sees wrong values
   
2. **Insufficient Hanging Piece Detection** (Lines 593-603)
   - Only checks queens/rooks
   - Only checks non-capturing moves
   - Doesn't detect pieces moving TO attacked squares
   - Example: `Nd5` moves knight to square attacked by queen (not detected!)

3. **No Static Exchange Evaluation (SEE)**
   - Can't evaluate if captures are good or bad
   - Can't evaluate if moving to attacked square is safe

## V15.2 Design Goals

### 1. Remove Broken Material Floor
- Delete lines 443-456 material floor code
- Return to pure PST evaluation
- Material awareness should be in SEARCH, not evaluation

### 2. Implement Static Exchange Evaluation (SEE)
```python
def _see(self, board: chess.Board, move: chess.Move) -> int:
    """
    Static Exchange Evaluation - determine if a move loses material
    Returns: Net material gain/loss in centipawns
    """
```

Used for:
- Evaluating captures (is Qxf6 good or bad?)
- Evaluating moves to attacked squares (is Nd5 safe?)
- Move ordering (good captures first)

### 3. Enhanced Move Filtering
```python
def _is_safe_move(self, board: chess.Board, move: chess.Move) -> bool:
    """
    Check if a move is tactically safe (doesn't lose material)
    
    Returns False if:
    - Moving piece to attacked square and SEE < 0
    - Capture with SEE < -200 (allows some compensation)
    - Leaving king in check or moving into check
    """
```

### 4. Multi-Layered Safety Checks

**Layer 1: Move Generation** - Already legal moves only

**Layer 2: Move Ordering** (existing)
- TT moves first
- Good captures (SEE > 0)
- Killers
- History
- Bad captures (SEE < 0) last

**Layer 3: Move Filtering** (NEW)
- Before search, filter out moves that:
  - Lose material without compensation (SEE < -200)
  - Hang major pieces (queen/rook) undefended
  - Are obviously bad tactically

**Layer 4: Search** (existing)
- Alpha-beta pruning
- Quiescence search for captures
- Transposition table

## Implementation Plan

### Phase 1: Remove Broken Code ✓
1. Delete material floor from `_evaluate_position()`
2. Return to pure PST evaluation

### Phase 2: Implement SEE
1. Add piece value constants
2. Implement SEE algorithm
3. Test SEE on known positions

### Phase 3: Integrate SEE into Move Ordering
1. Score captures by SEE
2. Order: good captures > killers > quiet > bad captures

### Phase 4: Add Move Safety Filter
1. Implement `_is_safe_move()`
2. Filter moves in `_search()` before trying them
3. Keep unsafe moves as last resort (in case no safe moves exist)

### Phase 5: Testing
1. Test against MaterialOpponent (should not lose pieces)
2. Test against V14.1 (should maintain performance)
3. Test depth consistency (should still reach 7-8)

## Expected Behavior Changes

### Before (V15.1)
- Qxf6: Captures knight, hangs queen → played anyway
- Nd5: Moves to attacked square → played anyway
- Loses to MaterialOpponent by hanging pieces

### After (V15.2)
- Qxf6: SEE = -400 (lose queen, gain knight) → rejected or ordered last
- Nd5: SEE = -300 (lose knight) → rejected or ordered last  
- Should beat MaterialOpponent by not losing pieces

## Success Criteria

### Minimum (Must Have)
- ✓ Beats MaterialOpponent (doesn't hang pieces)
- ✓ Maintains depth 7-8 capability
- ✓ No regressions in speed (<15s per move)

### Target (Should Have)
- ✓ Win rate vs V14.1 ≥ 50%
- ✓ Win rate vs MaterialOpponent ≥ 80%
- ✓ Fewer than 5 hanging pieces per 100 games

### Stretch (Nice to Have)
- Win rate vs PositionalOpponent ≥ 60%
- Average depth ≥ 7.5
- Tactical strength improvement visible

## Risk Mitigation

### Risk: SEE Slows Down Search
- **Mitigation**: Only call SEE for captures and moves to attacked squares
- **Mitigation**: Cache SEE results in move ordering
- **Fallback**: Simplified SEE (only first exchange)

### Risk: Too Aggressive Filtering Hurts Positional Play
- **Mitigation**: Allow SEE < -200 (compensated losses)
- **Mitigation**: Don't filter if no other moves available
- **Fallback**: Make filtering less aggressive

### Risk: Breaks Existing Strengths
- **Mitigation**: Keep depth 8 and phase-aware time management
- **Mitigation**: Preserve PST evaluation (engine's strength)
- **Rollback**: Can revert to V14.1 if needed

## Timeline

1. **Phase 1-2**: 30 minutes (remove broken code, implement SEE)
2. **Phase 3-4**: 30 minutes (integrate SEE, add filtering)
3. **Phase 5**: 30 minutes (testing against MaterialOpponent)

**Total**: ~90 minutes to working V15.2

## Notes

- Keep depth 8 and phase-aware time management from V15.1 (working well)
- Focus on SEARCH-level material awareness, not evaluation
- SEE is standard in strong engines - this is the right approach
- MaterialOpponent is perfect test opponent (pure material evaluation)

---

**Status**: Ready to implement
**Confidence**: HIGH (SEE is proven technique, V15.1 issues well-understood)
