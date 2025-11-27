# V7P3R v17.3 Development Plan

**Date Created:** November 25, 2025  
**Target Release:** December 2025  
**Current Stable Version:** v17.1.1 (Lichess Cloud: 1609 Rapid, 77.2% puzzle perfect sequences)

---

## Executive Summary

V7P3R v17.3 will focus on **tactical depth improvements** and **strategic heuristic enhancements** while preserving the stability and time management of v17.1.1. We will selectively adopt the **UCI debugging enhancements from v17.2.0** (which were valid) but **reject the performance optimization attempts** that caused the 439% CPU regression and -232 Elo drop.

### Core Objectives

1. **Preserve v17.1.1 Stability** - Maintain proven time management and baseline performance
2. **Adopt v17.2.0 UCI Enhancements** - Pull in debugging/info improvements only
3. **Fix Tactical Blindness** - Address depth limitations and late-depth evaluation shifts
4. **Add Broad-Based Heuristics** - Implement 1-2 new strategic nudges for better positioning

---

## Performance Baseline (v17.1.1)

### Cloud Production (Lichess)
- **Rating:** 1609 Rapid (stabilized from 1558 with v17.2.0)
- **Configuration:** 2 vCPU, 4GB RAM (GCP e2-medium)
- **CPU Usage:** 0.63% idle, <1% during games (88% reduction from v17.2.0's 439%)
- **Time Controls:** Bullet, Blitz, Rapid, Classical (all with ≥1s increment)
- **Status:** ✅ Accepting challenges, performing well

### Puzzle Performance (500-puzzle analysis)
- **Overall Score:** 77.2% perfect sequences (386/500)
- **Weighted Accuracy:** 91.3% across 1,176 positions
- **Estimated Puzzle Rating:** 1400-1450 (strong tactical pattern recognition)
- **Time Efficiency:** 3.2s average per position (vs 15s suggested)
- **Perfect Themes:** discoveredAttack (100%), mateIn2 (99%), backRankMate (98.9%)
- **Weak Themes:** zugzwang (60.5%), complex endgames

### Local Tournament Performance (v17.1)
- **vs Stockfish 1%:** 50% (10/20) - Major Breakthrough ⭐
- **Tournament Standing:** 2nd/9 engines (71.5/80 = 89.4%)
- **Estimated Rating:** ~2600-2650 vs handicapped Stockfish
- **NPS:** 5,845 nodes/second (stable baseline)
- **Strengths:** Tactical middle games (60% win rate), aggressive play, king hunts
- **Weaknesses:** Long endgames (29% win rate), defensive consolidation

---

## Problem Analysis

### Issue #1: Tactical Blindness in Deep Sequences

**Evidence from Puzzle Analysis:**
- **Position 1 Accuracy:** 82.6% (initial pattern recognition)
- **Positions 2-7 Accuracy:** 92-100% (excellent at continuing known sequences)
- **Performance Gap:** -11.6% on first position vs continuation

**Root Causes:**
1. **Depth Limitations:** Search may not reach critical tactical shots consistently
2. **Evaluation Instability:** Late-depth evaluation shifts causing horizon effects
3. **Quiescence Depth:** May terminate too early in complex tactical positions
4. **Move Ordering:** Critical moves might be ordered too low at root

**Symptoms:**
- Missing initial tactical patterns (82.6% vs 92%+ in sequences)
- Zugzwang weakness (60.5% - positional squeeze concepts)
- Occasional tactical oversights in cloud games

### Issue #2: Strategic Positioning Deficiencies

**Evidence from Game Analysis:**
- v17.1 strong in tactics but weaker in strategic positioning
- Long endgame performance: 29% win rate (5 losses in 7 games vs Stockfish 1%)
- Zugzwang puzzle performance: 60.5% (worst theme)
- Puzzle rating ceiling: ~2700 (only 402/500 high accuracy above this)

**Root Causes:**
1. **Lack of Positional Heuristics:** Engine optimized for tactics, not strategy
2. **Piece Coordination:** No explicit heuristics for optimal piece placement
3. **Pawn Structure:** Minimal evaluation of pawn weaknesses/strengths
4. **King Safety:** Could be enhanced beyond basic shelter evaluation

**Symptoms:**
- Overextension in complex positions
- Weak pawn endgame technique
- Suboptimal piece placement in quiet positions
- Draw rate only 10% (too aggressive, not recognizing equal positions)

---

## Rejected v17.2.0 Changes (Performance Regression)

### ❌ DO NOT PORT: Performance "Optimizations" That Failed

**1. Unified TT + Evaluation Cache**
```python
# V17.2: Unified eval cache (CAUSED ISSUES)
self.static_eval = static_eval  # Stored in TT entries
```
- **Result:** 439% CPU usage (4.39 cores on 2-core system)
- **Issue:** Single lookup overhead, potential serialization with Python GIL
- **Decision:** Keep v17.1.1's separate structures

**2. O(1) TT Bucket Replacement**
- **Result:** Potential collision overhead
- **Issue:** Python dictionary overhead may have negated O(1) benefits
- **Decision:** Keep v17.1.1's TT implementation

**3. In-Place Quiescence Sorting**
```python
# V17.2: In-place sorting (CAUSED SLOWDOWN)
captures.sort(key=..., reverse=True)  # Mutation overhead in Python
```
- **Result:** 88% higher CPU usage
- **Issue:** Python list mutation slower than allocation for small lists
- **Decision:** Keep v17.1.1's list allocation approach

**4. Pre-Allocated Move Ordering Buffers**
```python
# V17.2: Pre-allocated buffers (MEMORY MANAGEMENT OVERHEAD)
self._move_scores_buffer = [0] * 256
```
- **Result:** Memory management overhead outweighed benefits
- **Decision:** Keep v17.1.1's per-search allocation

---

## Approved v17.2.0 Changes (UCI Enhancements)

### ✅ PORT TO v17.3: Debugging & Info Improvements

**1. Enhanced UCI Info Output**
```python
# V17.2: Extended UCI info with seldepth and hashfull
print(f"info depth {depth} seldepth {self.selective_depth} "
      f"score cp {score} nodes {self.nodes_searched} "
      f"nps {nps} time {elapsed_ms} hashfull {hashfull} pv {pv_string}")
```
- **Benefit:** Better debugging visibility, GUI compatibility
- **Impact:** Zero performance impact, pure output enhancement
- **Decision:** ✅ Port to v17.3

**2. Selective Depth Tracking**
```python
# V17.2: Track selective depth (maximum depth including extensions)
self.selective_depth = max(self.selective_depth, current_depth)
```
- **Benefit:** Visibility into search extensions and quiescence depth
- **Impact:** Minimal overhead, valuable for analysis
- **Decision:** ✅ Port to v17.3

**3. Hash Table Usage Reporting**
```python
# V17.2: Calculate hashfull (TT usage in per mille 0-1000)
filled_entries = sum(1 for entry in self.transposition_table.values() if entry.zobrist_key != 0)
hashfull = int((filled_entries / max_entries) * 1000)
```
- **Benefit:** Monitor TT efficiency, detect overflow issues
- **Impact:** Calculated once per search, negligible overhead
- **Decision:** ✅ Port to v17.3

**4. Enhanced UCI Comments**
```python
# V17.2: UCI enhancements for debugging
# - Added seldepth tracking
# - Added hashfull percentage
# - Improved PV formatting
```
- **Benefit:** Better documentation and maintainability
- **Decision:** ✅ Port to v17.3

---

## Phase 1: UCI Enhancement Integration

### Objective
Port v17.2.0's UCI debugging improvements to v17.1.1 codebase without touching engine logic.

### Changes to Implement

**File: `src/v7p3r.py`**

1. **Add Selective Depth Tracking**
```python
def __init__(self):
    # Existing v17.1.1 initialization...
    
    # V17.3: Add selective depth tracking (from v17.2.0 UCI enhancements)
    self.selective_depth = 0  # Maximum depth reached including extensions
```

2. **Track Selective Depth in Search**
```python
def search_position(self, board, depth, alpha, beta, ply=0):
    # Existing v17.1.1 search logic...
    
    # V17.3: Track selective depth for UCI reporting
    self.selective_depth = max(self.selective_depth, ply)
    
    # Continue with existing v17.1.1 search...
```

3. **Enhanced UCI Info Output**
```python
def search(self, board, time_limit):
    # Existing v17.1.1 search logic...
    
    # V17.3: Enhanced UCI info (from v17.2.0)
    if current_depth > 0:
        elapsed_ms = int(elapsed_time * 1000)
        nps = int(self.nodes_searched / max(elapsed_time, 0.001))
        
        # Calculate hashfull (TT usage percentage in per mille)
        if hasattr(self, 'transposition_table') and self.transposition_table:
            max_entries = len(self.transposition_table)
            filled_entries = sum(1 for entry in self.transposition_table.values() 
                                if entry and entry.zobrist_key != 0)
            hashfull = int((filled_entries / max(max_entries, 1)) * 1000)
        else:
            hashfull = 0
        
        print(f"info depth {current_depth} seldepth {self.selective_depth} "
              f"score cp {best_score} nodes {self.nodes_searched} "
              f"nps {nps} time {elapsed_ms} hashfull {hashfull} pv {pv_string}")
    
    # Continue with existing v17.1.1 logic...
```

4. **Update Version String**
```python
# File: src/v7p3r_uci.py
print("id name V7P3R v17.3")
```

### Testing Requirements

**Phase 1 Acceptance Criteria:**
1. ✅ All UCI info output includes `seldepth` and `hashfull`
2. ✅ No performance regression vs v17.1.1 (NPS ≥ 5,845)
3. ✅ Cloud deployment maintains <1% CPU usage
4. ✅ 10-puzzle test maintains 100% accuracy on known positions
5. ✅ Arena/CuteChess GUI compatibility verified

---

## Phase 2: Tactical Depth Improvements

### Objective
Address tactical blindness by improving search depth, evaluation stability, and move ordering at root positions.

### Enhancement #1: Adaptive Quiescence Depth

**Problem:** Quiescence search may terminate too early in complex tactical positions.

**Current Implementation (v17.1.1):**
```python
def quiescence_search(self, board, alpha, beta, depth=0):
    MAX_QUIESCE_DEPTH = 10  # Fixed depth limit
    if depth > MAX_QUIESCE_DEPTH:
        return self.evaluate_position(board)
```

**Proposed Enhancement:**
```python
def quiescence_search(self, board, alpha, beta, depth=0):
    # V17.3: Adaptive quiescence depth based on position complexity
    
    # Determine complexity factors
    num_checks = 1 if board.is_check() else 0
    num_captures = len([m for m in board.legal_moves if board.is_capture(m)])
    
    # Extend quiescence depth in tactical positions
    if num_checks > 0:
        max_depth = 15  # Deeper search when in check
    elif num_captures > 5:
        max_depth = 12  # Moderate extension for many captures
    else:
        max_depth = 10  # Standard depth
    
    if depth > max_depth:
        return self.evaluate_position(board)
    
    # Continue with existing quiescence logic...
```

**Expected Impact:**
- Better tactical vision in complex positions (addressing 82.6% → 90%+ first-position target)
- Minimal performance impact (only extends in critical positions)
- Improved puzzle performance on tactical themes

### Enhancement #2: Root Move Ordering Boost

**Problem:** Critical tactical moves may be ordered too low at root, causing missed patterns.

**Proposed Enhancement:**
```python
def search(self, board, time_limit):
    # V17.3: Enhanced root move ordering
    
    # Get all legal moves
    legal_moves = list(board.legal_moves)
    
    # Score moves at root with extra tactical emphasis
    move_scores = []
    for move in legal_moves:
        score = 0
        
        # Existing TT/killer/history scoring...
        
        # V17.3: Boost tactical patterns at root
        if board.gives_check(move):
            score += 5000  # Checks highly valuable for tactics
        
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                # MVV-LVA with bonus for hanging pieces
                if not self._is_defended(board, move.to_square):
                    score += 3000  # Hanging piece capture
                score += self._mvv_lva_score(move, board)
        
        # Bonus for moves that improve piece activity
        board.push(move)
        mobility_after = len(list(board.legal_moves))
        board.pop()
        score += mobility_after * 2  # Mobility bonus
        
        move_scores.append((move, score))
    
    # Sort by score descending
    move_scores.sort(key=lambda x: x[1], reverse=True)
    ordered_moves = [move for move, _ in move_scores]
    
    # Continue with iterative deepening on ordered moves...
```

**Expected Impact:**
- Better first-position pattern recognition (82.6% → 90%+)
- Tactical shots found earlier in search
- Improved move ordering at root without affecting deeper plies

### Enhancement #3: Evaluation Stability Check

**Problem:** Late-depth evaluation shifts causing horizon effects and missed tactics.

**Proposed Enhancement:**
```python
def search_position(self, board, depth, alpha, beta, ply=0):
    # Existing v17.1.1 search logic...
    
    # V17.3: Evaluation stability verification at critical depths
    if depth == 1:  # One ply before quiescence
        # Get static evaluation
        static_eval = self.evaluate_position(board)
        
        # Continue search and get quiescence result
        # ... existing search logic ...
        
        # Check for large evaluation swing (horizon effect indicator)
        eval_delta = abs(best_score - static_eval)
        if eval_delta > 200 and board.is_capture(best_move):
            # Large swing on capture - extend search 1 ply to verify
            extended_score = self.search_position(board, 2, alpha, beta, ply)
            if abs(extended_score - best_score) > 100:
                # Evaluation unstable - use more conservative score
                best_score = (best_score + static_eval) // 2
    
    return best_score
```

**Expected Impact:**
- Reduced horizon effects
- More stable tactical evaluations
- Better handling of deep tactical sequences

---

## Phase 3: Strategic Heuristic Enhancements

### Objective
Add 1-2 broad-based heuristics to nudge engine toward better strategic positions without disrupting tactical strength.

### Heuristic #1: Piece Coordination Bonus

**Rationale:** v17.1.1 excels at tactics but lacks strategic piece placement awareness. Many losses come from overextension.

**Implementation:**
```python
def evaluate_position(self, board):
    # Existing v17.1.1 evaluation...
    
    # V17.3: Piece Coordination Heuristic
    coordination_score = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == board.turn:
            # Count pieces defending this piece
            defenders = len(board.attackers(board.turn, square))
            
            # Count pieces attacking enemy pieces this piece attacks
            attacks = board.attacks(square)
            supported_attacks = 0
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                if target_piece and target_piece.color != board.turn:
                    # How many of our pieces also attack this target?
                    support = len(board.attackers(board.turn, target_square))
                    supported_attacks += support
            
            # Bonus for well-supported pieces
            if defenders >= 2:
                coordination_score += 5  # Piece has backup
            
            # Bonus for coordinated attacks
            if supported_attacks >= 2:
                coordination_score += 8  # Multiple pieces attacking same target
            
            # Penalty for isolated pieces (no defenders, not in home rank)
            if defenders == 0 and chess.square_rank(square) not in [0, 1, 6, 7]:
                coordination_score -= 10  # Overextended piece
    
    # Apply coordination bonus (scaled to not overwhelm tactical evaluation)
    if board.turn == chess.WHITE:
        score += coordination_score
    else:
        score -= coordination_score
    
    return score
```

**Expected Impact:**
- Reduce overextension in complex positions
- Better piece placement in quiet positions
- Improved long-game performance (addressing 29% endgame win rate)
- Estimated: +30-50 Elo

### Heuristic #2: Pawn Structure Awareness

**Rationale:** Weak pawn endgame technique (60.5% zugzwang performance). Add basic pawn structure evaluation.

**Implementation:**
```python
def evaluate_position(self, board):
    # Existing v17.1.1 evaluation...
    
    # V17.3: Pawn Structure Heuristic
    pawn_structure_score = 0
    
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    # Evaluate white pawn structure
    for pawn_square in white_pawns:
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Passed pawn bonus (no enemy pawns ahead or on adjacent files)
        is_passed = True
        for check_rank in range(rank + 1, 8):
            for check_file in [file - 1, file, file + 1]:
                if 0 <= check_file < 8:
                    check_square = chess.square(check_file, check_rank)
                    if check_square in black_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        
        if is_passed:
            # Bonus increases as pawn advances
            passed_bonus = (rank - 1) * 10  # 0 to 60 bonus
            pawn_structure_score += passed_bonus
        
        # Doubled pawn penalty (another white pawn on same file)
        same_file_pawns = [sq for sq in white_pawns if chess.square_file(sq) == file]
        if len(same_file_pawns) > 1:
            pawn_structure_score -= 15
        
        # Isolated pawn penalty (no white pawns on adjacent files)
        has_neighbor = False
        for adj_file in [file - 1, file + 1]:
            if 0 <= adj_file < 8:
                if any(chess.square_file(sq) == adj_file for sq in white_pawns):
                    has_neighbor = True
                    break
        if not has_neighbor and file not in [0, 7]:  # Edge pawns excused
            pawn_structure_score -= 10
    
    # Mirror evaluation for black pawns
    for pawn_square in black_pawns:
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Passed pawn (checking ranks 0 to rank-1)
        is_passed = True
        for check_rank in range(0, rank):
            for check_file in [file - 1, file, file + 1]:
                if 0 <= check_file < 8:
                    check_square = chess.square(check_file, check_rank)
                    if check_square in white_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        
        if is_passed:
            passed_bonus = (6 - rank) * 10  # 0 to 60 bonus
            pawn_structure_score -= passed_bonus
        
        # Doubled/isolated pawn penalties
        same_file_pawns = [sq for sq in black_pawns if chess.square_file(sq) == file]
        if len(same_file_pawns) > 1:
            pawn_structure_score += 15
        
        has_neighbor = False
        for adj_file in [file - 1, file + 1]:
            if 0 <= adj_file < 8:
                if any(chess.square_file(sq) == adj_file for sq in black_pawns):
                    has_neighbor = True
                    break
        if not has_neighbor and file not in [0, 7]:
            pawn_structure_score += 10
    
    # Apply pawn structure score
    score += pawn_structure_score
    
    return score
```

**Expected Impact:**
- Better pawn endgame understanding (addressing 60.5% zugzwang performance)
- Recognition of passed pawns and structural weaknesses
- Improved long-game conversion
- Estimated: +20-40 Elo

---

## Implementation Schedule

### Week 1: UCI Enhancement Integration (Phase 1)
**Days 1-2:** Code integration
- Port v17.2.0 UCI enhancements to v17.1.1 codebase
- Add selective depth tracking
- Enhance info output with hashfull
- Update version strings

**Days 3-4:** Testing
- Verify UCI output compatibility (Arena, CuteChess)
- Confirm no performance regression (NPS test)
- Cloud deployment test (CPU usage check)
- 10-puzzle accuracy verification

**Day 5:** Release v17.3-alpha1
- Deploy to tournament engine folder
- Initial cloud testing

### Week 2: Tactical Depth Improvements (Phase 2)
**Days 6-8:** Code implementation
- Implement adaptive quiescence depth
- Add root move ordering boost
- Implement evaluation stability check

**Days 9-10:** Testing
- 100-puzzle tactical test (compare to v17.1.1's 77.2% perfect)
- Local tournament vs v17.1.1 (20 games)
- NPS performance check

**Day 11:** Release v17.3-alpha2
- Assess tactical improvement impact
- Cloud deployment if promising

### Week 3: Strategic Heuristics (Phase 3)
**Days 12-14:** Code implementation
- Implement piece coordination heuristic
- Implement pawn structure heuristic
- Balance heuristic weights

**Days 15-16:** Testing
- 50-puzzle strategic test (focus on zugzwang, endgames)
- Local tournament vs v17.1.1 (20 games)
- Cloud deployment stress test

**Day 17:** Release v17.3-beta1
- Full feature freeze
- Begin stabilization testing

### Week 4: Integration & Release
**Days 18-20:** Comprehensive testing
- 500-puzzle full analysis (compare to v17.1.1 baseline)
- 100-game local tournament (v17.3 vs v17.1.1 vs Stockfish 1%)
- Cloud production testing (Lichess games)
- Performance profiling

**Days 21-22:** Final adjustments
- Bug fixes only
- Performance tuning if needed
- Documentation updates

**Day 23:** Release v17.3.0
- Deploy to production cloud
- Update tournament engines
- Generate release notes

---

## Success Metrics

### Phase 1 Targets (UCI Enhancements)
- ✅ All UCI info includes seldepth and hashfull
- ✅ NPS ≥ 5,845 (no regression)
- ✅ Cloud CPU usage <1%
- ✅ GUI compatibility verified

### Phase 2 Targets (Tactical Depth)
- 🎯 Puzzle first-position accuracy: 82.6% → **90%+**
- 🎯 Perfect sequence rate: 77.2% → **82%+**
- 🎯 Tactical theme performance: discoveredAttack, mateIn2, backRankMate maintain 100%/99%
- 🎯 No NPS regression (<5% acceptable)

### Phase 3 Targets (Strategic Heuristics)
- 🎯 Zugzwang puzzle performance: 60.5% → **75%+**
- 🎯 Endgame win rate vs opponents: 29% → **40%+**
- 🎯 Cloud rating: 1609 → **1650+** Rapid
- 🎯 Draw rate: 10% → **15-20%** (better equal position recognition)

### Overall v17.3 Targets
- 🎯 **Puzzle Rating:** 1400-1450 → **1500-1550**
- 🎯 **Cloud Rating:** 1609 Rapid → **1650+**
- 🎯 **vs Stockfish 1%:** Maintain 50% (no regression)
- 🎯 **Estimated Elo Gain:** +50-80 Elo over v17.1.1

---

## Risk Mitigation

### Risk #1: Performance Regression
**Mitigation:**
- Maintain v17.1.1 baseline in parallel
- NPS testing after each phase
- Cloud CPU monitoring
- Rollback plan ready

### Risk #2: Tactical Strength Reduction
**Mitigation:**
- Extensive puzzle testing (500+ puzzles)
- Tactical theme focus (discoveredAttack, mateIn2, etc.)
- Regression testing vs v17.1.1
- Incremental changes with validation

### Risk #3: Heuristic Imbalance
**Mitigation:**
- Conservative heuristic weights (start low, tune up)
- A/B testing different weight combinations
- Monitor tactical vs strategic balance
- Keep heuristics optional (feature flags)

### Risk #4: Time Management Disruption
**Mitigation:**
- Do not modify v17.1.1 time management code
- Time control testing before cloud deployment
- Monitor cloud game time usage
- Timeout detection in tests

---

## Rollback Plan

If v17.3 causes regressions:

1. **Immediate:** Revert cloud to v17.1.1 (5 minutes)
2. **Analysis:** Identify problematic phase
3. **Options:**
   - Phase 1 only → Release as v17.2.1 (UCI fixes only)
   - Phase 1+2 → Release as v17.3-tactical
   - Full rollback → Continue with v17.1.1, reassess plan

---

## Post-Release Monitoring

### Week 1 After Release
- Cloud game monitoring (20+ games)
- Rating stability check
- CPU usage verification
- User feedback collection

### Week 2-4 After Release
- 100+ cloud games analysis
- Rating trend analysis
- Performance comparison to v17.1.1
- Identify next optimization opportunities

---

## Future Directions (v17.4+)

Based on v17.3 results, potential next steps:

### If Tactical Focus Successful
- Enhanced endgame tablebase support
- More sophisticated position evaluation
- Opening book expansion

### If Strategic Focus Successful
- Additional positional heuristics
- King safety improvements
- Piece mobility enhancements

### If Both Successful
- Parallel search (NPS boost)
- Neural network evaluation hybrid
- Advanced pruning techniques

---

## Conclusion

V7P3R v17.3 will build on v17.1.1's proven stability while addressing tactical blindness and strategic positioning through targeted enhancements. By selectively adopting v17.2.0's UCI improvements and adding focused tactical/strategic heuristics, we aim for a +50-80 Elo improvement without sacrificing the reliability that made v17.1.1 successful.

**Key Principles:**
- ✅ Preserve what works (v17.1.1 time management, baseline performance)
- ✅ Fix known issues (tactical blindness, strategic weakness)
- ✅ Incremental improvements (test each phase)
- ✅ No risky optimizations (learned from v17.2.0)

**Target:** V7P3R v17.3 - Tactical depth + strategic positioning = **1650+ Lichess Rapid, 1500+ puzzle rating**

---

**Document Status:** Draft  
**Next Review:** After Phase 1 completion  
**Approval Required:** User acceptance before each phase implementation
