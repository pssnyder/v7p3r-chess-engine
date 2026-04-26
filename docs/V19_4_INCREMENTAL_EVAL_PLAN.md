# V7P3R v19.4: Incremental Evaluation Addition Plan

**Strategy**: Strip v7p3r back to MaterialOpponent core (proven 35K NPS, 70%+ win rate), then add evaluations one at a time until performance dips.

**Goal**: Identify the exact evaluation bottleneck and deploy fastest stable version.

---

## Tournament Data: MaterialOpponent Proven Winner

### Performance Record (Nov 2025)
| Tournament | Score | Performance | Key Achievement |
|-----------|-------|-------------|-----------------|
| C0BR4 Regression 20251102 | 21.5/30 | 71.7% | **Beat C0BR4 v3.1** |
| C0BR4 Regression 20251102_2 | 22.5/30 | 75.0% | **Won again** |
| Engine Battle 20251107_2 | 11.5/18 | 63.9% | Beat V7P3R v14.0 & v14.2 |
| C0BR4 Regression Test | 23.0/40 | 57.5% | Strong vs Stockfish 1% |

**Conclusion**: MaterialOpponent's simple material-only evaluation achieves competitive strength through **depth** (10 ply in 5s) rather than evaluation complexity.

---

## Phase 0: Strip to MaterialOpponent Core (Baseline)

### What to Keep
1. **Search Infrastructure** (from v19.3)
   - Alpha-beta with TT ✓
   - Null move pruning ✓
   - Quiescence (optimized depth 1) ✓
   - Killer moves + history ✓
   - Opening book ✓
   - Time manager ✓

2. **Evaluation** (MaterialOpponent style)
   - **Material counting only**
   - **Bishop pair bonus (+50cp)**
   - **NO PSTs**
   - **NO mobility**
   - **NO king safety**
   - **NO pawn structure**

### What to Remove Temporarily
- All PST lookups
- Mobility calculations
- King safety calculations
- Pawn structure analysis
- Piece coordination
- Everything except material + bishop pair

### Expected Performance
- **NPS**: 30,000-40,000 (3x faster than v19.3)
- **Depth in 5s**: 8-10 (2x deeper)
- **Strength**: Competitive via depth (MaterialOpponent proved this)

### Files to Modify
- **src/v7p3r.py**: Replace `_evaluate_position()` with material-only
- **src/v7p3r_fast_evaluator.py**: Strip to pure material (or bypass entirely)
- **Test**: Run quick_test.py to verify NPS improvement

---

## Phase 1: Add Piece-Square Tables (PSTs)

### What to Add
- Basic PST values for piece positioning
- Centralization bonuses
- Pawn advancement bonuses

### Test After Addition
1. **Performance**: `quick_v19_1_test.py`
   - Target NPS: ≥25,000 (acceptable 20% slowdown)
   - If NPS <25K: **STOP, PSTs too expensive**
2. **Strength**: 10-game quick tournament vs Phase 0
   - Target: ≥55% win rate (strength improvement justifies cost)
   - If <50%: **ROLLBACK, no benefit**

### Expected Impact
- NPS: 25,000-30,000 (15-25% slowdown)
- Depth: Still 7-9 ply
- Strength: +30-50 ELO (better piece placement)

### Decision Point
- ✅ **Keep if**: NPS ≥25K AND win rate ≥55%
- ❌ **Remove if**: NPS <25K OR win rate <50%

---

## Phase 2: Add Pawn Structure Basics

### What to Add
- Passed pawn bonus
- Doubled pawn penalty
- Isolated pawn penalty
- Backward pawn detection

### Test After Addition
1. **Performance**: `quick_v19_1_test.py`
   - Target NPS: ≥20,000
   - If NPS <20K: **STOP**
2. **Strength**: 10-game tournament vs Phase 1
   - Target: ≥55% win rate
   - If <50%: **ROLLBACK**

### Expected Impact
- NPS: 20,000-25,000 (15-20% slowdown)
- Depth: 6-8 ply
- Strength: +20-40 ELO (endgame improvement)

### Decision Point
- ✅ **Keep if**: NPS ≥20K AND win rate ≥55%
- ❌ **Remove if**: NPS <20K OR win rate <50%

---

## Phase 3: Add King Safety (Minimal)

### What to Add
- Pawn shield bonus (pawns in front of king)
- King exposure penalty (open files near king)
- **NO complex attack calculations**

### Test After Addition
1. **Performance**: Target NPS ≥18,000
2. **Strength**: 10-game tournament vs Phase 2
   - Target: ≥55% win rate

### Expected Impact
- NPS: 18,000-22,000 (10-15% slowdown)
- Strength: +15-30 ELO (better defensive play)

### Decision Point
- ✅ **Keep if**: NPS ≥18K AND win rate ≥55%
- ❌ **Remove if**: NPS <18K OR win rate <50%

---

## Phase 4: Add Mobility (If still fast enough)

### What to Add
- Legal move count bonus
- Control of center squares

### Test After Addition
1. **Performance**: Target NPS ≥15,000
2. **Strength**: 10-game tournament

### Expected Impact
- NPS: 15,000-20,000 (15-20% slowdown)
- Strength: +20-30 ELO (better piece activity)

### Decision Point
- ✅ **Keep if**: NPS ≥15K AND win rate ≥55%
- ❌ **Remove if**: Performance tanks

---

## Testing Protocol (Repeated Each Phase)

### Step 1: Performance Benchmark
```bash
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine"
python testing/quick_v19_1_test.py
```

**Acceptance Criteria**:
- NPS documented
- Depth reached documented
- Compare to previous phase

### Step 2: Quick Strength Test (10 games)
```bash
python testing/tournament_runner.py \
  --engine1 "python src/v7p3r_uci.py" \
  --engine2 "python src/v7p3r_previous_phase.py" \
  --games 10 \
  --time-control "5+4"
```

**Acceptance Criteria**:
- Win rate ≥55% (statistically significant improvement)
- Zero timeouts
- Zero crashes

### Step 3: Decision
- **Keep addition**: If passes both tests
- **Rollback**: If fails either test
- **Deploy**: When next addition fails OR reach acceptable balance

---

## Deployment Decision Matrix

| Phase | Min NPS | Min Win % | Deploy? |
|-------|---------|-----------|---------|
| Phase 0 (Material only) | 30K-40K | Baseline | If fastest |
| Phase 1 (+ PSTs) | 25K | 55% vs P0 | If balanced |
| Phase 2 (+ Pawn structure) | 20K | 55% vs P1 | If strong |
| Phase 3 (+ King safety) | 18K | 55% vs P2 | If defensive |
| Phase 4 (+ Mobility) | 15K | 55% vs P3 | If complete |

### Deployment Criteria
1. **NPS ≥20,000** (target for blitz stability)
2. **Zero timeouts** in 30-game validation
3. **Win rate ≥48%** vs v18.4
4. **Depth ≥6** in 5 seconds

### Most Likely Outcome
- **Phase 1 or Phase 2** will be deployed
- Material + PSTs + Pawn structure = sweet spot
- ~20-25K NPS, depth 6-8, competitive strength

---

## Implementation Steps

### Step 1: Create Phase 0 (Material Only)
**File**: `src/v7p3r.py`
**Action**: Strip `_evaluate_position()` to material + bishop pair only

**Code Change**:
```python
def _evaluate_position(self, board: chess.Board) -> float:
    """Phase 0: Material-only evaluation (MaterialOpponent style)"""
    if board.is_checkmate():
        return -99999 if board.turn else 99999
    
    # Material counting
    score = 0
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 325,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    # Count material
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                       chess.ROOK, chess.QUEEN]:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    
    # Bishop pair bonus
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 50
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 50
    
    # Return from white's perspective
    return score if board.turn == chess.WHITE else -score
```

**Test**:
```bash
python testing/quick_v19_1_test.py
# Expected: 30K-40K NPS, depth 8-10
```

### Step 2: Version as v19.4.0 (Phase 0)
- Update version in header
- Update UCI name
- Commit as "v19.4.0: MaterialOpponent core baseline"

### Step 3: Add PSTs (Phase 1)
- Add simplified PST lookup
- Test performance
- Keep or rollback based on criteria

### Step 4: Continue Incrementally
- Add one evaluation at a time
- Test after each
- Stop when performance tanks

---

## Success Metrics

### Must Achieve
- **NPS ≥20,000** (blitz-safe)
- **No timeouts** in 30-game validation
- **Win rate ≥48%** vs v18.4
- **Stable** (no crashes, no illegal moves)

### Nice to Have
- **NPS ≥25,000** (1.8x faster than v19.3)
- **Depth ≥7** in 5 seconds (40% deeper)
- **Win rate ≥52%** vs v18.4 (convincing improvement)

---

## Rollback Strategy

If ANY phase fails tests:
1. **Git revert** to previous phase
2. **Deploy previous phase** to production
3. **Document failure** in CHANGELOG
4. **Re-evaluate** failed feature for optimization

---

## Timeline

| Phase | Development | Testing | Total |
|-------|-------------|---------|-------|
| Phase 0 (Material) | 1 hr | 30 min | 1.5 hrs |
| Phase 1 (PSTs) | 1 hr | 30 min | 1.5 hrs |
| Phase 2 (Pawns) | 1 hr | 30 min | 1.5 hrs |
| Phase 3 (King) | 1 hr | 30 min | 1.5 hrs |
| Phase 4 (Mobility) | 1 hr | 30 min | 1.5 hrs |
| **Final validation** | - | 3 hrs | 3 hrs |
| **Total** | ~5 hrs | ~5.5 hrs | **~10-12 hrs** |

Expect to deploy after **Phase 1 or Phase 2** (3-5 hours total).

---

## Relevant Files

### To Modify
- [src/v7p3r.py](../src/v7p3r.py) - Main engine file
  - Lines 780-1100: `_evaluate_position()` method
  - Replace with incremental versions

### Reference Implementation
- [MaterialOpponent/material_opponent.py](e:/Programming Stuff/Chess Engines/Opponent Chess Engines/opponent-chess-engines/src/MaterialOpponent/material_opponent.py)
  - Lines 100-150: Material-only evaluation
  - Proven 35K NPS performance

### Testing Scripts
- [testing/quick_v19_1_test.py](../testing/quick_v19_1_test.py) - Performance benchmark
- [testing/tournament_runner.py](../testing/tournament_runner.py) - Strength validation
- [testing/profile_components.py](../testing/profile_components.py) - Bottleneck identification

---

## Next Immediate Action

**Start Phase 0 implementation**:
1. Strip v7p3r.py evaluation to material-only
2. Run performance test (expect 30K-40K NPS)
3. Save as baseline for comparison
4. Begin Phase 1 (PSTs) if Phase 0 passes

**Command to start**:
```bash
# User approval needed before proceeding
```
