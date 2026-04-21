# V7P3R v19.0.0 - Spring Cleaning Refactoring Plan
**Target Release**: May 2026  
**Philosophy**: Back to basics - match C0BR4's efficiency while preserving v18's 1600+ ELO strength  
**Approach**: Incremental phases with testing between each stage

---

## 🎯 Primary Goals

### Performance Targets
- **Speed**: Match C0BR4's throughput (100+ games/day on e2-micro)
- **Cost**: Reduce from $20/month to <$5/month
- **Time Management**: Eliminate timeouts (currently 75% of losses)
- **ELO**: Maintain or improve 1600 rating achieved in v18.x

### Code Quality Targets
- **Simplicity**: Remove over-engineered abstractions
- **Clarity**: Document "Why this exists" for every function
- **Maintainability**: Remove dead code and redundant calculations
- **Performance**: Profile and optimize hot paths

---

## 📊 Current State Analysis (v18.4)

### Performance Issues
- **Time forfeits**: 3/4 recent games (75%)
- **Games/day**: 5-10 vs C0BR4's 100+
- **Cost**: $20/month on e2-medium vs C0BR4's <$5 on ec2-micro
- **Complexity**: 32+ eval modules with runtime profiling overhead

### Code Complexity Comparison
| Component | C0BR4 | V7P3R v18.4 | Overhead |
|-----------|-------|-------------|----------|
| Core Components | 13 | 13 | None |
| Eval Modules | 7 focused | 32+ modular | 4.5x |
| Profile System | None | 6 profiles | NEW |
| Context Calculator | None | Per-move | NEW |
| Module Registry | None | Cost/criticality | NEW |

### What Works (Keep This!)
- **v18.4 validation**: 90.9% puzzle accuracy (+3.9% improvement)
- **Aspiration windows**: ±50cp, 8-18% node reduction
- **Mate-in-1 fast path**: 100% detection, <10ms
- **1600 ELO achievement**: Something in v18.x works well

---

## 🚀 PHASE 1: Speed & Cleanup (No Eval Changes)
**Goal**: Eliminate performance bottlenecks without changing evaluation logic  
**Timeline**: Week 1-2  
**Testing**: Performance benchmarks only (no ELO testing needed)

### 1.1 Remove Modular Evaluation System ⚡ HIGH IMPACT
**Files to modify/remove:**
- `src/v7p3r_modular_eval.py` - DELETE
- `src/v7p3r_eval_modules.py` - DELETE
- `src/v7p3r_eval_selector.py` - DELETE
- `src/v7p3r_position_context.py` - DELETE
- `src/v7p3r.py` - Remove all modular eval references

**Impact:**
- Remove per-move context calculation overhead
- Remove profile selection logic (6 profiles × decision tree)
- Remove module cost/criticality metadata processing
- Estimated speedup: **30-50% reduction in per-node time**

**Replacement Strategy:**
- Use C0BR4-style simple evaluator pattern
- Always evaluate: material + PST + basic positional
- Conditional evaluation: king safety, pawn structure (based on game phase only)

**Code changes:**
```python
# BEFORE (v18.4):
def search(self, board, time_limit, ...):
    if is_root:
        self.current_context = self.context_calculator.calculate(...)  # EXPENSIVE
        self.current_profile = self.profile_selector.select_profile(...)  # EXPENSIVE
        
# AFTER (v19.0):
def search(self, board, time_limit, ...):
    if is_root:
        # No context calculation - just search!
        pass
```

### 1.2 Simplify Time Management ⚡ HIGH IMPACT
**Problem**: 200+ lines of time allocation logic in UCI, plus adaptive allocation in engine

**Files to modify:**
- `src/v7p3r_uci.py` - Lines 97-200+ (time parsing)
- `src/v7p3r.py` - `_calculate_adaptive_time_allocation()` function

**C0BR4 Strategy (simple and works):**
```csharp
// C0BR4 TimeManager.cs approach:
// 1. Calculate moves remaining (assume 40 moves total)
// 2. Allocate time: (remaining_time + increment * moves_left) / moves_left
// 3. Cap at max_per_move (usually 30-60s)
// 4. Emergency mode: if time < 10s, use max 2s per move
```

**Simplified V7P3R approach:**
```python
def calculate_time_allocation(remaining_time: float, increment: float, moves_played: int) -> float:
    """Simple, proven time allocation - no complexity"""
    # Emergency mode
    if remaining_time < 10.0:
        return min(2.0, remaining_time * 0.15)
    
    # Estimate moves remaining (typical game = 40 moves)
    moves_remaining = max(20, 40 - moves_played)
    
    # Budget: current time + expected increments
    budget = remaining_time + (increment * moves_remaining)
    
    # Allocate per move
    target_time = budget / moves_remaining
    
    # Cap at reasonable limits
    return min(target_time, 30.0)  # Never exceed 30s
```

**Impact:**
- Remove 150+ lines of complex time logic
- Eliminate time calculation bugs
- Reduce per-move overhead
- Estimated speedup: **5-10% reduction in time forfeit risk**

### 1.3 Optimize Move Ordering 🔧 MEDIUM IMPACT
**Current state**: Multiple passes over move list

**Optimization opportunities:**
1. **Pre-allocate score array** instead of creating tuples
2. **Early cutoff on hash move** (don't sort if hash move is available)
3. **Lazy evaluation** of capture scores (only when needed)

**Code change pattern:**
```python
# BEFORE:
scored_moves = []
for move in legal_moves:
    score = self._score_move(move, ...)  # Called for ALL moves
    scored_moves.append((score, move))
scored_moves.sort(reverse=True)

# AFTER:
# If hash move exists and is legal, try it first
if hash_move and hash_move in legal_moves:
    legal_moves.remove(hash_move)
    legal_moves.insert(0, hash_move)
    # Search without sorting - hash move tried first
```

### 1.4 Profile Hot Paths 📊 DIAGNOSTIC
**Action**: Run cProfile on search function during a full game

**Command:**
```bash
python -m cProfile -o v18.4_profile.stats testing/profile_game.py
python -c "import pstats; p = pstats.Stats('v18.4_profile.stats'); p.sort_stats('cumulative').print_stats(50)"
```

**Look for:**
- Functions called >100k times per game
- Any evaluation taking >1% total time
- Unnecessary type conversions or allocations
- Redundant board state checks

### 1.5 Remove UCI Overhead 🔧 LOW IMPACT
**Problem**: UCI interface has duplicate time parsing for White/Black

**Files**: `src/v7p3r_uci.py`

**Change**: Single time parsing function instead of 200-line duplicated if/elif blocks

**Impact:**
- Cleaner code (100+ line reduction)
- Minimal performance impact (UCI overhead is small)
- Easier to maintain

---

## 🧹 PHASE 2: Dead Code & Redundant Eval Removal
**Goal**: Remove calculations that don't impact play quality  
**Timeline**: Week 3-4  
**Testing**: Quick 20-game validation after each removal

### 2.1 Identify Low-Impact Evaluations 🔍 RESEARCH
**Method**: Run games with individual eval components disabled

**Test script:**
```python
# testing/eval_impact_test.py
# For each eval component:
# 1. Disable it
# 2. Play 20 games vs baseline
# 3. Measure: win%, time/move, nodes/move
# 4. If win% drops <2% and speed improves, mark for removal
```

**Candidates for removal (based on C0BR4 comparison):**
1. **Queen activity/mobility** - C0BR4 doesn't have separate queen eval
2. **Complex piece mobility** - Expensive, likely redundant with PST
3. **Space evaluation** - Complex calculation, unclear benefit
4. **Tempo evaluation** - Hard to measure, likely noise
5. **Development tracking** - Opening book handles this
6. **Backward pawns** - Less important than isolated/doubled

### 2.2 Consolidate Pawn Evaluation 📦 MEDIUM IMPACT
**Current state**: 6 separate pawn modules in modular system

**C0BR4 approach**: Single pawn evaluation function

**Proposal**: Keep only high-impact pawn evals
- ✅ **Passed pawns** (major endgame factor)
- ✅ **Doubled pawns** (clear weakness)
- ✅ **Isolated pawns** (positional weakness)
- ❌ **Backward pawns** (rarely decisive)
- ❌ **Pawn chains** (nice to have, not critical)
- ❌ **Pawn structure score** (redundant with above)

### 2.3 Simplify King Safety 👑 LOW IMPACT
**Current state**: `king_safety_basic` + `king_safety_complex`

**Proposal**: Single king safety function
- Pawn shield (3 squares in front of king)
- Castling rights bonus
- Open files near king penalty
- Remove: King tropism (expensive), attack pattern detection

### 2.4 Remove Redundant Position Checks 🔧 LOW IMPACT
**Example pattern to eliminate:**
```python
# BEFORE: Multiple checks for same condition
if len(board.move_stack) < 10:
    # Opening logic
if phase == OPENING:
    # More opening logic
if moves_played < 12:
    # Even more opening logic

# AFTER: Single check
phase = self._calculate_phase(board)
if phase == OPENING:
    # All opening logic here
```

### 2.5 Remove Unused Features/Files 🗑️ CLEANUP
**Files that may be unused:**
- Search for imports of each file
- If only imported in tests or docs, mark for removal
- Review git history to confirm no recent usage

**Candidates:**
- `v7p3r_tactical_cache.py` - Mentioned in comments as "not used"
- `v7p3r_pv_tracker.py` - PV instant moves disabled in v17.1
- Old version files in src/ (if any)

---

## ⚠️ PHASE 3: Careful Eval Modifications (Impact Testing Required)
**Goal**: Optimize remaining evaluations without losing strength  
**Timeline**: Week 5-6  
**Testing**: Full 50-game validation after each change

### 3.1 Optimize PST Lookups 🔧 MEDIUM IMPACT
**Current method**: Dictionary lookups or array indexing

**Optimization**: Pre-compute flipped tables as contiguous arrays
```python
# BEFORE:
pst_score = PST[piece_type][square]
if not is_white:
    pst_score = PST[piece_type][FLIP[square]]

# AFTER: Pre-computed
pst_score = PST_WHITE[piece_type][square] if is_white else PST_BLACK[piece_type][square]
```

### 3.2 Lazy Quiescence Evaluation 🎯 HIGH IMPACT
**Current**: Full evaluation in quiescence search

**Optimization**: Delta pruning in quiescence
```python
# If capturing a pawn won't improve alpha by enough, skip it
if best_score + piece_value[captured] + 200 < alpha:
    continue  # Delta pruning
```

**Testing required**: Ensure tactical accuracy maintained

### 3.3 Simplify Material Counting 🔧 LOW IMPACT
**Current**: Count material in every evaluation

**Optimization**: Incremental material tracking
```python
# Update material count during make/unmake move
def make_move(self, move):
    if move.is_capture:
        captured_piece = board.piece_at(move.to_square)
        self.material_balance -= PIECE_VALUES[captured_piece]
```

### 3.4 Phase-Appropriate Evaluation 📊 MEDIUM IMPACT
**Strategy**: Skip expensive evals when not relevant

```python
def evaluate(self, board):
    score = 0
    phase = self._get_phase_fast(board)  # Cached
    
    # Always evaluate
    score += self.evaluate_material(board)
    score += self.evaluate_pst(board)
    
    # Opening/Middlegame only
    if phase in (OPENING, MIDDLEGAME):
        score += self.evaluate_king_safety(board)
        score += self.evaluate_piece_coordination(board)
    
    # Endgame only
    if phase == ENDGAME:
        score += self.evaluate_passed_pawns(board)  # Critical in endgame
        score += self.evaluate_king_activity(board)
    
    return score
```

---

## 📋 Implementation Workflow

### Week 1: Phase 1 - Remove Modular System
- [ ] Create v19.0 branch
- [ ] Remove modular eval files (4 files)
- [ ] Update v7p3r.py to remove references
- [ ] Run perft tests to ensure move generation intact
- [ ] Profile: measure per-move time vs v18.4
- [ ] Target: 30% speed improvement

### Week 2: Phase 1 - Simplify Time Management
- [ ] Implement C0BR4-style time allocation
- [ ] Test in 20 bullet games (time forfeit test)
- [ ] Test in 20 rapid games (quality test)
- [ ] Verify: 0 time forfeits expected
- [ ] Optimize move ordering
- [ ] Profile again: measure cumulative improvement

### Week 3-4: Phase 2 - Dead Code Removal
- [ ] Run eval impact tests (script in testing/)
- [ ] Remove low-impact evals one at a time
- [ ] 20-game validation after each removal
- [ ] Consolidate pawn evaluation
- [ ] Simplify king safety
- [ ] Profile: ensure no regression

### Week 5-6: Phase 3 - Careful Optimizations
- [ ] Optimize PST lookups
- [ ] Implement delta pruning in quiescence
- [ ] Test incremental material tracking
- [ ] 50-game tournament vs v18.4
- [ ] Target: maintain 1600 ELO, gain 20+ speed

### Week 7: Final Validation
- [ ] 100-game tournament vs v18.4
- [ ] 50-game tournament vs C0BR4
- [ ] Acceptance criteria:
  - Win% vs v18.4: ≥48%
  - Time forfeits: <5%
  - Games/day: ≥50
  - Cost: <$10/month
- [ ] Update CHANGELOG.md
- [ ] Deploy to production

---

## 🎓 Code Documentation Standard (v19)

### Function Documentation Template
```python
def function_name(self, params):
    """
    WHY THIS EXISTS: [Single sentence explaining the purpose]
    
    WHAT IT DOES: [Brief description of functionality]
    
    IMPACT: [Performance/ELO impact if known]
    
    EXAMPLE: [Usage example if complex]
    """
    pass
```

### Example: Material Evaluation
```python
def evaluate_material(self, board: chess.Board) -> int:
    """
    WHY THIS EXISTS: Material is the most important evaluation factor (>80% correlation with winning)
    
    WHAT IT DOES: Counts piece values (P=100, N=320, B=330, R=500, Q=900)
    
    IMPACT: Critical - removing this drops win% to ~10%
    
    PERFORMANCE: O(64) iteration, ~0.01ms/call
    """
    # Implementation
```

---

## 🚨 Rollback Criteria

**Rollback to v18.4 if:**
- Win% vs v18.4 baseline drops below 45% (3+ sigma)
- Time forfeit rate exceeds 15%
- Puzzle validation drops below 85%
- Production errors/crashes occur

**Version preservation:**
- Keep v18.4 deployment package intact
- Tag v19.0-alpha, v19.0-beta, v19.0-rc before production
- Document all changes in CHANGELOG.md

---

## 📈 Success Metrics

### Performance Metrics
| Metric | v18.4 Current | v19.0 Target | C0BR4 Reference |
|--------|---------------|--------------|-----------------|
| Games/day | 5-10 | 50+ | 100+ |
| Time forfeits | 75% | <5% | <1% |
| Cost/month | $20 | <$10 | <$5 |
| Avg move time | 8-15s | 2-4s | 1-2s |

### Quality Metrics
| Metric | v18.4 Current | v19.0 Target |
|--------|---------------|--------------|
| ELO rating | 1600 | 1600+ |
| Puzzle accuracy | 90.9% | ≥88% |
| Win% vs v18.4 | - | ≥48% |
| Blunders/game | ~6 | ≤6 |

### Code Metrics
| Metric | v18.4 Current | v19.0 Target |
|--------|---------------|--------------|
| Source files | ~15 | ~10 |
| Lines of code | ~3500 | ~2500 |
| Functions | ~80 | ~50 |
| Eval modules | 32+ | ~10 |

---

## 💡 Future Enhancements (Post-v19)

**Reserved for v20+:**
- Neural network integration (NNUE)
- Advanced search techniques (null move pruning, LMR)
- Tablebase integration
- Multi-threading support

**Philosophy for v19:**
- Simplicity over complexity
- Speed over features
- Proven techniques over experiments
- C0BR4 parity is the goal

---

## 📚 References

- **C0BR4 Comparison**: `available-evals.md`
- **v18.4 Validation**: `docs/V18_4_0_VALIDATION_REPORT.md` (if exists)
- **Version Management**: `.github/instructions/version_management.instructions.md`
- **Deployment Log**: `deployment_log.json`
- **CHANGELOG**: `CHANGELOG.md`

---

**Version**: Draft 1.0  
**Author**: AI Assistant + User Collaboration  
**Date**: April 21, 2026  
**Status**: READY FOR REVIEW
