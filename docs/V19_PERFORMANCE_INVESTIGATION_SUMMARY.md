# V7P3R Performance Investigation Summary

**Date**: April 22, 2026  
**Investigation**: v19.0-v19.3 performance optimization  
**Result**: Found bottleneck (quiescence), achieved 1.7x speedup, but still 4-7x slower than target

---

## Investigation Timeline

### v19.0: Initial Refactoring
- **Goal**: Remove modular evaluation to match C0BR4's efficiency
- **Expected**: 30-50% speedup from removing overhead
- **Actual**: Same ~7-9k NPS as v18.4 (no improvement)
- **Conclusion**: Modular eval wasn't the bottleneck

### v19.1: Move Safety Removal  
- **Hypothesis**: Move safety checks (0.083ms/move) causing slowdown
- **Action**: Removed evaluate_move_safety() and detect_bitboard_tactics()
- **Result**: Still ~9k NPS (no improvement)
- **Conclusion**: Move safety wasn't the bottleneck

### v19.2: board.is_game_over() Removal
- **Hypothesis**: Expensive game-over checks at every node
- **Action**: Only check when no legal moves
- **Result**: Still ~9k NPS (no improvement)
- **Conclusion**: Not the bottleneck

### v19.3: Quiescence Optimization ✅
- **Discovery**: Component profiling showed quiescence = 78.7% of time!
- **Actions**:
  - Reduced depth 4 → 1
  - Removed check generation (captures only)
  - Added delta pruning
- **Result**: **13,500 NPS** (1.7x improvement) ✓
- **Conclusion**: Quiescence WAS the bottleneck, but still slow overall

---

## Current Performance Profile (v19.3)

```
Component                Time      % Total    Avg Time    Calls
────────────────────────────────────────────────────────────────
Quiescence Search       1.481s     64.9%     0.19ms      7,620
Position Evaluation     0.769s     33.7%     0.04ms     20,802
Move Ordering           0.349s     15.3%     0.19ms      1,880
TT Probe                0.226s      9.9%     0.02ms     10,055
Move Generation         0.084s      3.7%     0.04ms      1,893
TT Store                0.046s      2.0%     0.02ms      1,880
Board Make/Unmake       0.061s      2.7%     0.01ms     19,962
────────────────────────────────────────────────────────────────
TOTAL                   2.28s      NPS: 13,526
```

---

## Key Findings

### 1. **Quiescence is Still the Bottleneck (65%)**
Even optimized to depth 1, it's still the slowest component:
- Called 7,620 times for 30,857 total nodes (25% of nodes!)
- 0.19ms per call (down from 0.44ms, but still expensive)
- Generates moves, filters captures, sorts by value

### 2. **Position Evaluation is Expensive (34%)**
- Called 20,802 times (67% of nodes!)
- 0.04ms per call seems fast, but adds up
- Called in quiescence (stand-pat) AND at leaf nodes

### 3. **Python-chess Overhead**
- `board.push/pop` are surprisingly fast (0.005ms each)
- `legal_moves` generation is reasonable (0.04ms)
- Not the primary bottleneck

### 4. **We're Searching Very Shallow**
- Depth 4 in 2.3 seconds is still too shallow
- Should reach depth 8-10 in 5 seconds for proper play
- Quiescence being called at EVERY depth-0 node

---

## Architecture Analysis

### Why V7P3R is Slow

1. **Quiescence at Every Leaf Node**
   - Every depth-0 node calls quiescence
   - Quiescence generates moves, evaluates captures
   - Even at depth 1, this is expensive

2. **Frequent Evaluation Calls**
   - Stand-pat in quiescence = evaluation
   - Leaf nodes = evaluation  
   - Called 20k+ times per search

3. **Python Overhead**
   - Interpreted language vs C++ engines
   - Function call overhead
   - No SIMD or bitboard optimizations

4. **Complex Move Ordering**
   - Called 1,880 times
   - Loops through all moves scoring each
   - 0.19ms per call adds up

---

## Comparison: V7P3R vs Reference Engines

### V7P3R v19.3 (Python)
- **NPS**: 13,500
- **Depth in 5s**: 4-5
- **Architecture**: Alpha-beta + TT + quiescence + move ordering

### C0BR4 v3.4 (C#)
- **NPS**: Unknown (but 100+ games/day on cheap VM)
- **Cost**: <$5/month
- **Games/day**: 100+

### Stockfish (C++)
- **NPS**: 1,000,000+ (100x faster!)
- **Architecture**: Highly optimized C++, NNUE evaluation
- **Not comparable**: Professional-grade engine

### Typical Python Engines
- **Sunfish**: ~20,000-40,000 NPS (simple Python engine)
- **python-chess examples**: ~10,000-30,000 NPS
- **Conclusion**: We're in the normal Python range!

---

## Root Cause: Python Performance Ceiling

**Hypothesis**: We've hit the fundamental Python performance ceiling.

### Evidence:
1. v18.4 production engine: ~7-8k NPS
2. After all optimizations: ~13-14k NPS
3. Typical Python engines: 10-30k NPS
4. C++ engines: 100k-1M+ NPS

### Why Python is Slow:
- Interpreted bytecode (not compiled machine code)
- Function call overhead (each recursive call has overhead)
- No SIMD bitboard operations
- No inline optimizations
- GIL (Global Interpreter Lock) prevents multi-core search
- Dynamic typing overhead

---

## Realistic Performance Expectations

### What's Achievable in Python:
- **Current**: 13,500 NPS
- **Optimized**: 20,000-30,000 NPS (with further optimizations)
- **Maximum**: 40,000-50,000 NPS (perfect Python code)
- **C++ Equivalent**: 500,000+ NPS (40-50x faster)

### Further Optimizations Possible:
1. **Reduce quiescence depth to 0** (only evaluate captures that recapture)
2. **Simplify evaluation** (faster piece-square tables only)
3. **Reduce move ordering complexity** (no history heuristic)
4. **Disable null move pruning** (saves recursive calls)
5. **Reduce TT size** (faster hash lookups)

**Expected gain**: 1.5-2x → **20,000-30,000 NPS**

---

## Tournament Performance Analysis

### v19.0 vs v18.4 (30 games)
- **Score**: 48.3% (14.5/30)  
- **Timeouts**: 9 games (30%)
- **Depth reached**: 3-5 (should be 8-10)

### Why Timeouts Occurred:
- **Not time management** (v7p3r_time_manager allocates correctly)
- **Fundamental slowness** (8k-14k NPS too slow for blitz)
- **Shallow search** (depth 3-5 = tactically blind)

### Expected with v19.3:
- **NPS**: 13,500 (better but still borderline)
- **Depth**: 5-6 (still too shallow)
- **Timeouts**: Reduced but still possible

---

## Options Moving Forward

### Option A: Accept Python Limitations
- **Goal**: Optimize to 20-30k NPS (1.5-2x more)
- **Actions**:
  - Quiescence depth 0 (or remove entirely)
  - Simpler evaluation
  - Reduce search features
- **Result**: Faster but weaker tactically
- **Deployment**: Might work for slower time controls

### Option B: Port to Faster Language
- **Goal**: 100,000-500,000 NPS
- **Language**: C# (C0BR4 style) or C++
- **Effort**: Complete rewrite
- **Result**: 10-40x faster, proper depth, competitive play

### Option C: Hybrid Approach
- **Core search in C++**: Fast recursive search
- **Python glue**: UCI, time management, opening book
- **Using**: ctypes or Cython for interface
- **Result**: Best of both worlds

### Option D: Focus on Strength, Not Speed
- **Goal**: Smarter search, not faster
- **Actions**:
  - Better move ordering (ML-based?)
  - Better evaluation (piece activity, mobility)
  - Better pruning (SEE, futility pruning)
- **Result**: Higher quality depth 4-5 search
- **Trade-off**: Still slow, but stronger moves

---

## Recommendation

### Immediate (v19.3 → v19.4):
1. Test v19.3 in tournament vs v18.4
2. If timeouts reduced: deploy to Lichess
3. Monitor performance for 48 hours

### Short-term (Next 2 weeks):
1. Try Option A optimizations (target 20-30k NPS)
2. If still timing out: accept Python limitations
3. Consider Option B (port to C#)

### Long-term (Q2 2026):
- **If staying in Python**: Focus on Option D (smarter, not faster)
- **If performance critical**: Port to C# like C0BR4
- **If competitive play**: Full C++ rewrite with NNUE

---

## Conclusion

We successfully identified and optimized the primary bottleneck (quiescence search), achieving **1.7x speedup** (8k → 13.5k NPS). However, we've likely hit the **Python performance ceiling** around 10-30k NPS.

For proper blitz play (depth 8-10), we need **50-100k+ NPS**, which requires:
- Compiled language (C# or C++)
- Bitboard operations
- Optimized evaluation

**V19.3 is the best Python-based V7P3R we can build without fundamental architecture changes.**
