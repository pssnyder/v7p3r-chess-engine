# V17.2 Performance Optimization Plan
**Date**: November 21, 2025  
**Goal**: Reach consistent depth 10 through safe, methodical performance improvements  
**Approach**: NO tactics changes, NO strategy changes, ONLY performance tuning

---

## Executive Summary

V17.1.1 currently achieves depth 4.8-5.2 in tournament play (default depth 6). To reach consistent depth 10, we need approximately **2x node throughput improvement** (~5,000 NPS → ~15,000 NPS).

**v17.2 Strategy**: Focus on **safe, non-invasive optimizations** that improve speed without altering engine behavior:
- Transposition table efficiency improvements
- Eliminate redundant position lookups
- Remove unnecessary list allocations
- Optimize data structures for performance

**Explicitly OUT OF SCOPE**:
- ❌ Removing or modifying `detect_bitboard_tactics()` (preserves tactical strength)
- ❌ Changing move ordering heuristics (preserves strategy)
- ❌ Modifying evaluation logic (preserves playing style)
- ❌ Altering search extensions or reductions (preserves tactical vision)

---

## Current Performance Profile

### V17.1.1 Baseline
```
Default depth: 6
Tournament avg depth: 4.8-5.2
Estimated NPS: 5,000-8,000
Time per move: 3-5 seconds (60+1 control)
TT size: 50,000 entries
TT replacement: Sort-based (expensive)
Evaluation cache: Separate from TT (redundant lookups)
```

### Bottleneck Analysis (Non-Invasive Only)

**High Impact, Low Risk**:
1. **TT Replacement Strategy** (lines 714-717) - Sorts 50,000 entries when full (~50ms per clear, 10x per search = 500ms/search = 18% overhead)
2. **Duplicate TT/Eval Cache Lookups** (lines 264, 270, 622-628) - Every position queries two separate dictionaries
3. **List Allocations in Quiescence** (lines 793-810) - Creates temporary lists for every Q-search node (10-50x more calls than regular search)
4. **Move Ordering List Management** (lines 540-615) - Creates 6 separate lists (tt_moves, captures, checks, killers, tactical_moves, quiet_moves)

**Excluded (Preserve Tactics/Strategy)**:
- `detect_bitboard_tactics()` calls - KEEP (critical for tactical strength)
- `gives_check()` evaluations - KEEP (important for check detection)
- MVV-LVA scoring - KEEP (tactical move ordering)
- History heuristic - KEEP (learning mechanism)

---

## V17.2 Implementation Plan

### Phase 1: TT Replacement Optimization (Highest Impact)
**Effort**: 90 minutes  
**Expected Gain**: +20-25% NPS  
**Risk**: Very Low

#### Current Problem (lines 714-717)
```python
if len(self.transposition_table) >= self.max_tt_entries:
    # Clear 25% of entries when full (simple aging)
    entries = list(self.transposition_table.items())  # Convert dict to list
    entries.sort(key=lambda x: x[1].depth, reverse=True)  # Sort 50,000 entries!
    self.transposition_table = dict(entries[:int(self.max_tt_entries * 0.75)])
```

**Performance Impact**:
- `list()`: ~2ms (copies 50,000 entries)
- `sort()`: ~50ms (O(N log N) comparisons)
- `dict()`: ~3ms (creates new dictionary)
- **Total**: ~55ms per TT clear
- **Frequency**: ~10x per search
- **Cost**: 550ms per 3-second search = **18% overhead**

#### Solution: Two-Tier Bucket System
Replace sort-based aging with O(1) bucket replacement:

```python
def _store_transposition_table(self, board: chess.Board, depth: int, score: int, 
                               best_move: Optional[chess.Move], alpha: int, beta: int):
    """Store position in TT with O(1) two-tier bucket replacement"""
    zobrist_hash = self.zobrist.hash_position(board)
    
    # Two-tier buckets: always-replace + depth-preferred
    primary_bucket = zobrist_hash % self.max_tt_entries
    secondary_bucket = (zobrist_hash % self.max_tt_entries) ^ 1  # Adjacent bucket
    
    # Determine node type
    if score <= alpha:
        node_type = 'upperbound'
    elif score >= beta:
        node_type = 'lowerbound'
    else:
        node_type = 'exact'
    
    entry = TranspositionEntry(depth, score, best_move, node_type, zobrist_hash)
    
    # Check primary bucket
    primary_entry = self.transposition_table.get(primary_bucket)
    if primary_entry is None or primary_entry.zobrist_hash == zobrist_hash:
        # Empty or same position - always replace
        self.transposition_table[primary_bucket] = entry
        self.search_stats['tt_stores'] += 1
        return
    
    # Check secondary bucket
    secondary_entry = self.transposition_table.get(secondary_bucket)
    if secondary_entry is None or secondary_entry.depth < depth:
        # Empty or shallower depth - replace secondary
        self.transposition_table[secondary_bucket] = entry
        self.search_stats['tt_stores'] += 1
        return
    
    # Both buckets occupied by deeper entries - replace primary (always-replace strategy)
    self.transposition_table[primary_bucket] = entry
    self.search_stats['tt_stores'] += 1
```

**Benefits**:
- O(1) replacement (no sorting ever)
- ~2 dictionary lookups vs sorting 50,000 entries
- Preserves deep search results (depth-preferred bucket)
- Fresh position tracking (always-replace bucket)

**Testing Required**:
- Verify TT hit rate doesn't degrade (should improve slightly)
- Confirm no tactical regressions (20 games vs v17.1.1)
- Measure NPS improvement (expect +1,000-1,500 NPS)

---

### Phase 2: Unified TT + Evaluation Cache
**Effort**: 2 hours  
**Expected Gain**: +30-35% NPS  
**Risk**: Low

#### Current Problem (lines 264, 270, 622-628)
```python
# Line 264: Evaluation cache
self.evaluation_cache = {}  # position_hash -> evaluation

# Line 270: Transposition table (separate!)
self.transposition_table: Dict[int, TranspositionEntry] = {}

# Lines 622-628: _evaluate_position() checks both
def _evaluate_position(self, board: chess.Board) -> float:
    cache_key = board._transposition_key()
    
    if cache_key in self.evaluation_cache:  # First lookup
        self.search_stats['cache_hits'] += 1
        return self.evaluation_cache[cache_key]
    
    # ... then later queries TT separately
```

**Performance Impact**:
- Every position: 2 dictionary lookups (eval cache + TT lookup in search)
- Duplicate storage: Same position stored twice (TT entry + eval cache)
- Cache coherency: Eval cache can be stale if TT entry exists

#### Solution: Add Static Eval to TT Entry
Merge evaluation cache into transposition table:

```python
# Update TranspositionEntry class definition (find in code, around line 200)
class TranspositionEntry:
    def __init__(self, depth: int, score: int, best_move: Optional[chess.Move], 
                 node_type: str, zobrist_hash: int, static_eval: Optional[float] = None):
        self.depth = depth
        self.score = score
        self.best_move = best_move
        self.node_type = node_type
        self.zobrist_hash = zobrist_hash
        self.static_eval = static_eval  # NEW: cached evaluation

# Remove separate evaluation cache from __init__ (line 264)
# self.evaluation_cache = {}  ← DELETE THIS

# Update _evaluate_position to use TT cache
def _evaluate_position(self, board: chess.Board) -> float:
    """V17.2: Unified TT + eval cache for single lookup"""
    zobrist_hash = self.zobrist.hash_position(board)
    
    # Check TT for cached evaluation
    if zobrist_hash in self.transposition_table:
        tt_entry = self.transposition_table[zobrist_hash]
        if tt_entry.static_eval is not None:
            self.search_stats['cache_hits'] += 1
            return tt_entry.static_eval
    
    self.search_stats['cache_misses'] += 1
    
    # V14.2: Use selected evaluator (fast or bitboard)
    if self.use_fast_evaluator:
        final_score = float(self.evaluator.evaluate(board))
    else:
        # Bitboard evaluator path
        white_base = self.bitboard_evaluator.calculate_score_optimized(board, True)
        black_base = self.bitboard_evaluator.calculate_score_optimized(board, False)
        # ... rest of evaluation logic
    
    # Store evaluation in TT (create entry if doesn't exist)
    if zobrist_hash in self.transposition_table:
        self.transposition_table[zobrist_hash].static_eval = final_score
    else:
        # Create new entry with just evaluation (depth=0 for eval-only entry)
        entry = TranspositionEntry(0, 0, None, 'eval_only', zobrist_hash, static_eval=final_score)
        # Use same bucket logic from Phase 1
        primary_bucket = zobrist_hash % self.max_tt_entries
        self.transposition_table[primary_bucket] = entry
    
    return final_score
```

**Benefits**:
- Single hash lookup instead of two
- Better memory locality (evaluation stored with TT entry)
- Automatic cache invalidation (when TT entry replaced, eval goes too)
- ~35% reduction in dictionary operations

**Testing Required**:
- Verify cache hit rates (should be similar or better)
- Confirm NPS improvement (expect +2,000-3,000 NPS)
- Tactical validation (20 games vs v17.1.1)

---

### Phase 3: Eliminate List Allocations in Quiescence
**Effort**: 45 minutes  
**Expected Gain**: +5-8% NPS  
**Risk**: Very Low

#### Current Problem (lines 793-810)
```python
# Sort tactical moves by MVV-LVA for better ordering
capture_scores = []  # New list allocation
for move in tactical_moves:
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
        attacker = board.piece_at(move.from_square)
        attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
        mvv_lva = victim_value * 100 - attacker_value
        capture_scores.append((mvv_lva, move))  # Tuple allocation
    else:
        capture_scores.append((0, move))

# Sort by MVV-LVA score
capture_scores.sort(key=lambda x: x[0], reverse=True)
ordered_tactical = [move for _, move in capture_scores]  # Another list allocation
```

**Performance Impact**:
- Q-search called 10-50x more than regular nodes
- Each Q-search node: 3 list allocations (capture_scores, ordered_tactical, tactical_moves earlier)
- At depth 6: ~10,000 Q-nodes × 3 allocations = 30,000 allocations per search

#### Solution: In-Place Sorting with Key Function
```python
# Sort tactical moves by MVV-LVA for better ordering
def mvv_lva_key(move):
    """Calculate MVV-LVA score for sorting (higher is better)"""
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        victim_value = self.piece_values.get(victim.piece_type, 0) if victim else 0
        attacker = board.piece_at(move.from_square)
        attacker_value = self.piece_values.get(attacker.piece_type, 0) if attacker else 0
        return victim_value * 100 - attacker_value
    else:
        return 0  # Check moves get lower priority

# Sort tactical_moves in-place (no new list allocations)
tactical_moves.sort(key=mvv_lva_key, reverse=True)

# Search tactical moves directly (no ordered_tactical list)
best_score = stand_pat
for move in tactical_moves:
    board.push(move)
    score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
    board.pop()
    # ... rest of search logic
```

**Benefits**:
- Eliminates 2 list allocations per Q-node (capture_scores, ordered_tactical)
- In-place sort is faster (no tuple overhead)
- Cleaner code (fewer intermediate variables)

**Testing Required**:
- Verify tactical move ordering unchanged
- Confirm NPS improvement (expect +400-600 NPS)
- Basic tactical test (10 positions)

---

### Phase 4: Optimize Move Ordering List Management
**Effort**: 60 minutes  
**Expected Gain**: +8-12% NPS  
**Risk**: Very Low

#### Current Problem (lines 540-615)
Creates 6 separate lists for move categorization:
```python
captures = []
checks = []
killers = []
quiet_moves = []
tactical_moves = []
tt_moves = []
```

Each list stores tuples `(score, move)`, then sorts, then extracts moves.

#### Solution: Pre-Allocate and Reuse Buffers
```python
def __init__(self, ...):
    # ... existing init code
    
    # V17.2: Pre-allocated move ordering buffers (reused across searches)
    self.move_buffers = {
        'captures': [],
        'checks': [],
        'killers': [],
        'quiet': [],
        'tactical': [],
        'tt': []
    }

def _order_moves_advanced(self, board: chess.Board, moves: List[chess.Move], depth: int, 
                          tt_move: Optional[chess.Move] = None) -> List[chess.Move]:
    """V17.2: Move ordering with buffer reuse"""
    
    # Clear buffers (faster than creating new lists)
    for buffer in self.move_buffers.values():
        buffer.clear()
    
    # Use pre-allocated buffers
    captures = self.move_buffers['captures']
    checks = self.move_buffers['checks']
    killers = self.move_buffers['killers']
    quiet_moves = self.move_buffers['quiet']
    tactical_moves = self.move_buffers['tactical']
    tt_moves = self.move_buffers['tt']
    
    # Rest of move ordering logic stays THE SAME
    # (preserves all tactical detection and scoring)
    killer_set = set(self.killer_moves.get_killers(depth))
    
    for move in moves:
        if tt_move and move == tt_move:
            tt_moves.append(move)
        elif board.is_capture(move):
            # ... same tactical bonus logic (PRESERVED)
            tactical_bonus = self.bitboard_evaluator.detect_bitboard_tactics(board, move)
            captures.append((total_score, move))
        # ... rest of categorization (UNCHANGED)
    
    # ... sorting and combining (UNCHANGED)
    return ordered
```

**Benefits**:
- Eliminates 6 list allocations per move ordering call
- ~1,000 move ordering calls per search = 6,000 fewer allocations
- Memory reuse improves cache performance

**Testing Required**:
- Verify move ordering unchanged (critical for tactics)
- Confirm NPS improvement (expect +600-1,000 NPS)
- Full tactical test suite (preserve v17.1.1 strength)

---

### Phase 5: Zobrist Hash Caching
**Effort**: 30 minutes  
**Expected Gain**: +3-5% NPS  
**Risk**: Very Low

#### Current Problem
Zobrist hash recalculated multiple times per position:
- Line 684: `zobrist_hash = self.zobrist.hash_position(board)` (TT lookup)
- Line 703: `zobrist_hash = self.zobrist.hash_position(board)` (TT store)
- Evaluation cache uses `board._transposition_key()` (different hash!)

#### Solution: Cache Zobrist Hash in Board State
```python
def _recursive_search(self, board: chess.Board, search_depth: int, alpha: float, beta: float, time_limit: float):
    """V17.2: Cache zobrist hash for reuse"""
    
    # Calculate hash once per node
    if not hasattr(board, '_cached_zobrist'):
        board._cached_zobrist = self.zobrist.hash_position(board)
    zobrist_hash = board._cached_zobrist
    
    # ... rest of search uses cached hash
    
    # Clear cache when making/unmaking moves
    board.push(move)
    if hasattr(board, '_cached_zobrist'):
        delattr(board, '_cached_zobrist')
    score = -self._recursive_search(...)
    board.pop()
    if hasattr(board, '_cached_zobrist'):
        delattr(board, '_cached_zobrist')
```

**Alternative**: Use `board._transposition_key()` everywhere (python-chess built-in, already cached)

**Benefits**:
- Eliminates redundant hash calculations
- Minimal code changes
- No behavior changes

---

## Implementation Timeline

### V17.2.0: TT Optimization (Week 1)
- **Day 1-2**: Implement Phase 1 (TT replacement optimization)
- **Day 2**: Test Phase 1 (NPS benchmark, 20 games vs v17.1.1)
- **Day 3**: Deploy v17.2.0 to Lichess if successful

**Expected**: 5,000 NPS → 6,200 NPS (+24%), depth 5.2 → 6.0

### V17.2.1: Cache Unification (Week 2)
- **Day 1-3**: Implement Phase 2 (unified TT + eval cache)
- **Day 3-4**: Test Phase 2 (comprehensive validation)
- **Day 4**: Deploy v17.2.1 to Lichess if successful

**Expected**: 6,200 NPS → 8,400 NPS (+68% total), depth 6.0 → 7.0

### V17.2.2: Polish (Week 3)
- **Day 1**: Implement Phase 3 (quiescence list elimination)
- **Day 1**: Implement Phase 4 (move ordering buffers)
- **Day 2**: Implement Phase 5 (zobrist caching)
- **Day 2-3**: Test all Phase 3-5 changes
- **Day 3**: Deploy v17.2.2 to Lichess if successful

**Expected**: 8,400 NPS → 10,500 NPS (+110% total), depth 7.0 → 8.5

### V17.3 Planning (Future)
If depth 10 not reached by v17.2.2, consider for v17.3:
- Lazy move generation (generator pattern)
- Principal variation caching
- Parallel search (Lazy SMP)

---

## Testing Requirements

### NPS Benchmark (Required for Each Phase)
```python
# Run perft at depth 5 from starting position
board = chess.Board()
start = time.time()
nodes = engine.perft(board, depth=5, divide=False)
elapsed = time.time() - start
nps = int(nodes / elapsed)
print(f"Perft(5) NPS: {nps:,}")
```

**Target**: 200,000+ NPS for perft (pure move generation, no evaluation)

### Depth Test (Required for Each Phase)
```python
# Standard middlegame positions, 5 seconds each
test_positions = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pp2pppp/5n2/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    # ... 8 more positions
]

for fen in test_positions:
    board = chess.Board(fen)
    best_move, score = engine.search(board, time_limit=5.0)
    print(f"Depth reached: {engine.last_depth}")
```

**Target v17.2.0**: Avg depth 6.0  
**Target v17.2.1**: Avg depth 7.0  
**Target v17.2.2**: Avg depth 8.5

### Tactical Validation (Required for Each Phase)
Run existing tactical test suite to ensure no regressions.

### Mini-Tournament (Required for Each Phase)
20 games vs v17.1.1 baseline:
- Time control: 60+1 (same as Lichess)
- Expected: 50-55% win rate (slight improvement from speed)
- Minimum acceptable: 45% (no regression in playing strength)

---

## Risk Assessment

### Phase 1: TT Replacement - **VERY LOW RISK**
- Pure data structure optimization
- No logic changes
- Well-tested algorithm (used in Stockfish, Komodo)
- Easy rollback if issues

### Phase 2: Unified Cache - **LOW RISK**
- Cache consolidation (same data, different structure)
- Preserves all evaluation logic
- Testable in isolation
- Rollback: restore separate eval_cache

### Phase 3: Quiescence Lists - **VERY LOW RISK**
- In-place sorting (same result)
- No ordering changes
- Tactical moves unchanged
- Easy to verify correctness

### Phase 4: Move Buffers - **VERY LOW RISK**
- Buffer reuse (same allocations, just recycled)
- Zero logic changes
- All tactical detection preserved
- Straightforward testing

### Phase 5: Zobrist Caching - **VERY LOW RISK**
- Cache optimization only
- Hash values unchanged
- No search logic affected
- Minimal code changes

---

## Success Metrics

### V17.2.0 Success Criteria
- ✅ NPS improvement: +20-25% (5,000 → 6,200)
- ✅ Depth improvement: +0.5-0.8 (5.2 → 6.0)
- ✅ Win rate vs v17.1.1: ≥45%
- ✅ No tactical regressions (test suite passes)
- ✅ Lichess deployment stable (no crashes, timeouts)

### V17.2.1 Success Criteria
- ✅ NPS improvement: +60-70% cumulative (5,000 → 8,400)
- ✅ Depth improvement: +1.5-1.8 cumulative (5.2 → 7.0)
- ✅ Win rate vs v17.1.1: ≥48%
- ✅ Cache hit rate: ≥60%
- ✅ Tactical strength preserved

### V17.2.2 Success Criteria (Final)
- ✅ NPS improvement: +100-110% cumulative (5,000 → 10,500)
- ✅ Depth improvement: +3.0-3.3 cumulative (5.2 → 8.5)
- ✅ Win rate vs v17.1.1: ≥50%
- ✅ Memory usage stable (<100MB increase)
- ✅ Ready for Lichess long-term deployment

### V17.3 Decision Point
If v17.2.2 reaches depth 8.5 but not depth 10:
- Consider lazy move generation (+15% NPS)
- Consider parallel search (+80% NPS on 4 cores)
- Re-evaluate whether depth 10 is necessary (depth 8.5 may be sufficient for 1650+ ELO)

---

## Conclusion

V17.2 focuses on **safe, methodical performance improvements** that preserve V7P3R's tactical strength and playing style while achieving significant NPS gains through:

1. **Better data structures** (TT two-tier buckets)
2. **Eliminating redundancy** (unified TT/eval cache)
3. **Memory efficiency** (buffer reuse, in-place operations)
4. **Cache optimization** (zobrist hash caching)

**No changes to**:
- Tactical detection (`detect_bitboard_tactics` preserved)
- Move ordering heuristics (same prioritization)
- Evaluation logic (same scoring)
- Search extensions (same tactical vision)

**Expected outcome**: V17.2.2 reaches depth 8-9 consistently, with path to depth 10 in V17.3 if needed.

**Philosophy**: "Make it fast, keep it smart" - performance without compromising the engine's proven tactical strength.
