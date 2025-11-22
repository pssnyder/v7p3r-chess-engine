# V14.1 Enhancement Plan: Selective V16.1 Learnings

**Date:** November 20, 2025  
**Goal:** Apply v16.1's speed and depth lessons to v14.1 without losing its proven tactical strength

## V16.1 Performance Analysis

### What Made V16.1 Fast (Depth 6-8 Consistently):

1. **Simple 60/40 PST+Material Evaluation** (~0.001ms per position)
   - No complex bitboard operations
   - Direct PST table lookups
   - Simple material counting
   - **Speed: 40x faster than v14.1's bitboard evaluator**

2. **Lightweight Middlegame Bonuses** (~0.0005ms overhead)
   - Only 4 checks: rook files, king shield, doubled pawns, passed pawns
   - Applied ONLY in middlegame (not opening or endgame)
   - Minimal branching and loops

3. **Aggressive Move Filtering**
   - Pre-search material delta calculation
   - -50cp threshold for filtering
   - Always allows checks (tactical opportunity)

4. **Simple Opening Book** (52 positions)
   - Lightweight Zobrist lookup
   - Weighted random selection (variety)
   - Exits book early → more search time

### What V14.1 Has That V16.1 Lacks:

1. **PV Following System** (instant moves when opponent follows prediction)
2. **Advanced Time Management** (phase-aware allocation)
3. **Bitboard Evaluator** (more comprehensive but SLOW - 0.04ms per position)
4. **Proven Tactical Understanding** (v14.1's tournament success)

## Proposed Enhancements (Incremental Approach)

### Phase 1: Speed Enhancement (Primary Goal - Target: Depth 6-8)

**Problem:** V14.1's bitboard evaluator is 40x slower than v16.1's simple PST evaluation
- V14.1: ~0.04ms per position
- V16.1: ~0.001ms per position
- **Impact:** At 10,000 positions/search, that's 400ms vs 10ms evaluation time!

**Solution:** Add a "fast mode" simple evaluator alongside existing bitboard evaluator

**Implementation:**
1. Extract v16.1's simple PST evaluation into `v7p3r_fast_evaluator.py`
2. Add v16.1's PST tables (PAWN_PST, KNIGHT_PST, etc.) 
3. Add v16.1's simple middlegame bonuses (rooks, king shield, pawn structure)
4. Add config flag: `use_fast_evaluator = True/False`
5. Keep bitboard evaluator for fallback/comparison

**Benefits:**
- ✅ 40x speed improvement → enables depth 6-8 consistently
- ✅ Maintains v14.1's proven search algorithm
- ✅ Keeps v14.1's PV following and time management
- ✅ Non-destructive (can toggle back to bitboard)

### Phase 2: Move Filtering Enhancement (Secondary - If Needed)

**Only if Phase 1 alone doesn't reach depth 6-8:**

Add v16.1's material delta pre-filtering:
```python
def _calculate_material_delta(self, board, move) -> int:
    """Quick material safety check before adding to search"""
    # Gain from capture
    # Loss if hanging after move
    return delta

def _filter_moves_by_material(self, moves) -> List:
    """Filter -50cp+ losing moves, always keep checks"""
```

**Benefits:**
- Reduces search tree size
- Focuses on safe moves
- Always preserves tactical checks

### Phase 3: Opening Book (Optional Enhancement)

Add v16.1's simple opening book (52 positions):
- Italian Game, Queen's Gambit, King's Indian for White
- Sicilian, King's Indian, French, Caro-Kann for Black
- Weighted random selection (30% variety)
- Early exit → more time for search

## Implementation Strategy

### Step 1: Create Fast Evaluator Module ✅

```python
# New file: v7p3r_fast_evaluator.py
class V7P3RFastEvaluator:
    def __init__(self):
        # PST tables from v16.1
        self.PAWN_PST = [...]
        self.KNIGHT_PST = [...]
        # etc.
    
    def evaluate(self, board) -> int:
        """Simple 60/40 PST+Material evaluation"""
        pst_score = 0
        material_score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                pst_score += self._get_pst_value(piece, square)
                material_score += PIECE_VALUES[piece.piece_type]
        
        # Middlegame bonuses (if applicable)
        if self._is_middlegame(board):
            bonus = self._calc_middlegame_bonus(board)
            combined = int(pst_score * 0.6 + material_score * 0.4 + bonus)
        else:
            combined = int(pst_score * 0.6 + material_score * 0.4)
        
        return combined if board.turn == chess.WHITE else -combined
```

### Step 2: Integrate Into V14.1 Engine

Modify `v7p3r.py`:
```python
from v7p3r_fast_evaluator import V7P3RFastEvaluator

class V7P3REngine:
    def __init__(self, use_fast_evaluator=True):
        if use_fast_evaluator:
            self.evaluator = V7P3RFastEvaluator()
        else:
            self.evaluator = V7P3RScoringCalculationBitboard(...)
    
    def _evaluate_position(self, board):
        """Use selected evaluator"""
        return self.evaluator.evaluate(board)
```

### Step 3: Test Depth Achievement

Run speed test:
```python
# Test positions: 1000 evaluations
# Measure: time per evaluation, depth reached in 5 seconds

V14.1 (bitboard): ~0.04ms → depth 3-4
V14.1 (fast):     ~0.001ms → depth 6-8 (expected)
```

### Step 4: Tournament Validation

Play 20 games vs known opponents:
- MaterialOpponent v2.0 (expect: 70%+)
- PositionalOpponent v2.0 (expect: 40-50%)
- v14.1 original (expect: 50-55%)

If win rate > 60% → SUCCESS, deploy to Lichess

## Risk Mitigation

### Risk 1: Fast evaluator loses tactical understanding
**Mitigation:** 
- Preserve all search logic (alpha-beta, TT, move ordering)
- PV following system intact
- Time management intact
- Only speed up evaluation, not search

### Risk 2: Simple PST too weak positionally
**Mitigation:**
- V16.1 proved 60/40 PST+Material works
- PositionalOpponent won with pure PST (81%)
- Middlegame bonuses add tactical awareness

### Risk 3: Breaking v14.1's proven strength
**Mitigation:**
- Config flag allows instant rollback
- Keep bitboard evaluator code intact
- Incremental testing at each step

## Success Metrics

### Primary Goal: Consistent Depth 6-8
- ✅ Average depth ≥ 6.0 in test games
- ✅ <10% moves at depth ≤ 4
- ✅ Depth std deviation < 1.5 (consistency)

### Secondary Goal: Maintain/Improve Win Rate
- ✅ Win rate ≥ 50% vs v14.1 original
- ✅ Win rate ≥ 60% vs MaterialOpponent
- ✅ No regression in material safety

### Tertiary Goal: Speed to Lichess
- ✅ Deploy in <2 days if successful
- ✅ Monitor first 20 games closely
- ✅ Rollback plan ready (v14.1 backup available)

## Version Naming

**V14.2** - Fast Evaluator Enhancement
- "V14.1 with v16.1's speed lessons"
- Maintains v14.1 architecture + PV following + time management
- Adds v16.1's simple fast evaluator
- Target: Depth 6-8, 60%+ win rate

## Next Steps

1. ✅ Create `v7p3r_fast_evaluator.py` with v16.1's PST tables
2. ✅ Integrate into v14.1 with config toggle
3. ✅ Run depth/speed benchmark
4. ⏳ Play 20 test games vs known opponents
5. ⏳ If successful (>60% win), deploy to Lichess as v14.2
6. ⏳ Monitor Lichess performance for 50 games

---

**Estimated Development Time:** 2-3 hours  
**Estimated Testing Time:** 4-6 hours  
**Total Time to Deployment:** 1-2 days

**Key Insight:** V16.1's speed came from evaluation simplicity, NOT search complexity. We can get v14.1 to depth 6-8 by swapping evaluators while keeping everything else intact.
