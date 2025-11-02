# 🎯 V7P3R V14.9.1 Step-by-Step Workflow

## 📋 Version Summary
**V14.9.1** represents a complete restoration of V12.6's proven simple workflow with enhanced time management tuning. This version removes the complex emergency controls and over-optimization that caused V14.3-V14.8's catastrophic performance regression (17.1%-38.8% accuracy), returning to the fundamentals that made V12.6 successful (85%+ puzzle accuracy, 57.1% tournament performance).

---

## 1. Engine Initialization
```
When V7P3R starts up:
→ Creates main V7P3REngine instance
→ Initializes SIMPLIFIED bitboard evaluator (material + positioning only)
→ Sets up transposition table with Zobrist hashing
→ Configures search parameters (default depth = 6)
→ Initializes killer moves and history heuristic tables
→ Creates PV (Principal Variation) tracker for move following
→ Sets up evaluation cache for position scoring
→ Ready to receive UCI commands
```

**V14.9.1 Key Changes:**
- Removed advanced pawn and king safety evaluators (causing negative baseline)
- Simplified to proven bitboard-only evaluation
- Removed emergency stop flags and complex time management
- Restored simple, predictable architecture

---

## 2. Position Setup
```
When given a chess position:
→ Receives FEN string or move sequence via UCI protocol
→ Creates python-chess Board object
→ Validates position legality
→ Checks PV tracker for instant book moves
→ Ready for move search
```

**V14.9.1 Optimization:**
- PV following checks if position matches known good continuation
- Instant move return if PV match found (0ms thinking time)

---

## 3. Move Search Process (The Core Engine Loop)

### Step 3a: Adaptive Time Allocation (V14.9.1 TUNED)
```
Before starting search, calculate time budget:
→ Detects game phase (opening < 10 moves, middlegame < 40, endgame)
→ Counts tactical complexity (captures available, checks, in-check status)
→ Applies aggressive time factors:

OPENING (moves < 10):
   • Base factor: 30% of time limit
   • Absolute cap: 0.5s target, 1.0s maximum
   • Philosophy: Move quickly, don't waste time
   
EARLY MIDDLEGAME (moves < 15):
   • Base factor: 50% of time limit  
   • Absolute cap: 1.0s target, 2.0s maximum
   • Philosophy: Moderate speed, develop pieces
   
MIDDLEGAME QUIET (moves < 40, not noisy):
   • Base factor: 60% of time limit
   • Philosophy: Find plan and move decisively
   
MIDDLEGAME TACTICAL (moves < 40, noisy):
   • Base factor: 100% of time limit
   • Noisy = captures ≥5 OR checks ≥3 OR in check
   • Philosophy: Calculate deeply, use full time
   
ENDGAME (moves ≥ 40):
   • Base factor: 70% of time limit
   • Philosophy: Precise calculation for technique

→ Additional modifiers:
   • In check: +20% time
   • Many legal moves (≥40): +30% time
   • Few legal moves (≤5): -40% time
   • Behind in material: +10% time
   • Ahead in material: -20% time
```

### Step 3b: Move Generation
```
→ Generate all legal moves for current position
→ Typically 20-40 moves in opening/middlegame
→ 5-15 moves in endgame
→ Each move represents a possible choice
```

### Step 3c: Simple Move Ordering (V14.9.1 RESTORED)
```
→ Calls _order_moves_advanced() function
→ V14.9.1 SIMPLIFIED to V12.6's proven 5-category system:

1. **Transposition Table Move** (if available)
   • Previously best move from TT probe
   • Highest priority - already proven good
   
2. **Captures** (MVV-LVA ordering)
   • Most Valuable Victim - Least Valuable Attacker
   • Queen captures first, pawn captures last
   • Captures ordered by victim value descending
   
3. **Checks** (giving check moves)
   • Forcing moves that put opponent king in check
   • Can lead to tactical opportunities
   
4. **Killer Moves** (non-capture moves that caused cutoffs)
   • Previously successful quiet moves at this depth
   • Position-independent move history
   
5. **Quiet Moves** (remaining moves)
   • History heuristic scoring for move ordering
   • All other legal moves

→ REMOVED V14.x complexity:
   ✗ 12-category over-classification
   ✗ Threat detection scoring
   ✗ Development move prioritization
   ✗ Castling special priority
   ✗ Pawn advance categorization
   ✗ Tactical pattern bonuses

→ Philosophy: Simple, proven ordering examines best moves first
```

### Step 3d: Iterative Deepening Search
```
→ Starts at depth 1, increases to depth 6 (default_depth)
→ For each depth level:

   BEFORE ITERATION:
   → Check if elapsed time > target_time → break
   → Predict next iteration time using previous iteration
   → If predicted_time > max_time → break (FIXED in V14.9.1)
   
   DURING ITERATION:
   → Call _recursive_search() for current depth
   → Track iteration completion time
   → Update best move if valid result returned
   → Extract and display Principal Variation (PV)
   → Store PV for move following optimization
   
   PV STABILITY TRACKING (V14.9.1 NEW):
   → Count consecutive iterations with same best move
   → If PV stable for 2+ iterations AND depth ≥4 AND position quiet:
      • Print "Early exit: PV stable"
      • Break search loop
      • Return best move immediately
   → Philosophy: Don't waste time recalculating obvious moves
   
   AFTER ITERATION:
   → Print UCI info (depth, score, nodes, time, nps, pv)
   → Continue to next depth if time allows

→ Returns best move found at deepest completed depth
```

### Step 3e: Recursive Alpha-Beta Search (V14.9.1 RESTORED)
```
→ _recursive_search() is the core "thinking" algorithm

For each move (starting with highest priority from ordering):
   
   → Make the move on board temporarily
   → Ask: "How would opponent respond to this?"
   
   → If at leaf node (depth = 0):
      • Call _quiescence_search() for tactical stability
      • Return static evaluation
   
   → If game over:
      • Return mate score or draw score
      • Prefer quicker mates (depth bonus)
   
   → NULL MOVE PRUNING (if depth ≥3, not in check):
      • Try passing turn to opponent
      • If we're still winning after null move, prune branch
      • Saves ~30% of search nodes
   
   → For each opponent response:
      • Recursively call _recursive_search() at depth-1
      • Track best score using alpha-beta bounds
      • Prune branches that can't improve position
   
   → Unmake move (board returns to original state)
   → Store result in transposition table
   → Update killer moves if move caused beta cutoff
   → Return best score found

TIME MANAGEMENT (V14.9.1 RESTORED):
→ Check every 1000 nodes (not 50) - 20x less overhead
→ If elapsed > time_limit → return current eval
→ Single abort point - trust the algorithm
→ No emergency stop flags
→ No 85% bailout thresholds
→ Philosophy: Simple, predictable, proven
```

---

## 4. Position Evaluation (The "Judgment" System)

### Step 4a: Simplified Bitboard Evaluation (V14.9.1)
```
For each position reached in search:
→ Check evaluation cache first (fast _transposition_key())
→ If cached, return immediately (cache hit)
→ Otherwise, calculate fresh evaluation
```

### Step 4b: Material Evaluation (SIMPLIFIED)
```
→ Count pieces with STATIC VALUES:
   • Queen = 900 points
   • Rook = 500 points  
   • Bishop = 300 points (constant)
   • Knight = 300 points (constant)
   • Pawn = 100 points
   • King = 0 (safety handled separately)

→ Calculate material balance:
   white_score = bitboard_evaluator.calculate_score_optimized(board, True)
   black_score = bitboard_evaluator.calculate_score_optimized(board, False)
   
→ Return from current player's perspective:
   if white_to_move: score = white_score - black_score
   else: score = black_score - white_score

→ REMOVED V14.x features:
   ✗ Dynamic bishop valuation (325/275)
   ✗ Advanced pawn structure evaluator
   ✗ Advanced king safety evaluator
   ✗ Tactical pattern detection bonuses
   ✗ Threat-aware scoring

→ Philosophy: Simple material + basic positioning is reliable
```

### Step 4c: Positional Evaluation (Bitboard-Based)
```
→ Piece-Square Tables (PST) applied via bitboard evaluator:
   • Knights prefer center squares (+30 bonus)
   • Bishops prefer long diagonals (+20 bonus)
   • Rooks prefer 7th rank and open files (+10 bonus)
   • Pawns prefer advancement (+5 per rank)
   • Kings prefer corners in opening/middlegame
   • Kings prefer center in endgame

→ Applied during calculate_score_optimized():
   for each piece:
      base_value = piece_values[piece_type]
      positional_bonus = piece_square_table[square]
      total += base_value + positional_bonus

→ All positional scoring consolidated in bitboard evaluator
→ No separate evaluator calls (performance optimization)
```

### Step 4d: Quiescence Search (Tactical Stability)
```
→ Called at leaf nodes to prevent horizon effect
→ Continues searching captures until position is quiet
→ Maximum 4 ply extension for tactical sequences

Process:
   → Generate all capture moves
   → Stand-pat evaluation (option to not capture)
   → Try each capture recursively
   → Return best score when no more captures
   
→ Prevents:
   • Hanging pieces after search horizon
   • Missing tactical sequences
   • Incorrect static evaluations in tactical positions

→ V14.9.1: Uses simple evaluation (no complexity)
```

---

## 5. Transposition Table Management

### Step 5a: TT Probe (Before Search)
```
→ Hash position using Zobrist hashing
→ Check if position exists in transposition table
→ If found and depth ≥ current_depth:
   • Return stored score if node_type matches alpha-beta bounds
   • Return stored best_move for move ordering
→ Cache hit rate: ~20-30% in typical positions
```

### Step 5b: TT Store (After Search)
```
→ Determine node type:
   • Exact: score within alpha-beta window
   • Lowerbound: score ≥ beta (fail-high)
   • Upperbound: score ≤ alpha (fail-low)
→ Store: depth, score, best_move, node_type, zobrist_hash
→ Replacement strategy: keep highest depth entries
→ Clear 25% of entries when table full (simple aging)
→ Max entries: 50,000 (reasonable memory usage)
```

---

## 6. Move Selection and Return

### Step 6a: Best Move Selection
```
→ After iterative deepening completes:
   • best_move contains highest-scoring move
   • best_score contains evaluation of resulting position
   
→ V14.9.1 guarantees:
   • Move is legal (from legal_moves list)
   • PV matches move played (fixed root search bug)
   • Sensible opening moves (no more 1.e3!)
```

### Step 6b: UCI Communication
```
→ Returns selected move in UCI format (e.g., "g1f3")
→ During search, prints info strings:
   info depth 4 score cp -172 nodes 5123 time 1440 nps 3557 pv e2e3 e7e5 f1b5 f8b4
   
   Components:
   • depth: Ply depth achieved
   • score cp: Centipawn evaluation (100 = 1 pawn)
   • nodes: Total positions examined
   • time: Milliseconds elapsed
   • nps: Nodes per second (search speed)
   • pv: Principal variation (expected move sequence)

→ V14.9.1: Clean UCI output, no emergency messages
```

---

## 🔧 Key V14.9.1 Architecture Decisions

### ✅ Restored from V12.6 (Proven Components):
```
→ Simple 5-category move ordering
→ 1000-node time checking interval
→ 100% time limit usage (no 60% emergency stops)
→ Single abort point in recursive search
→ Adaptive time allocation (not emergency allocation)
→ Simple iterative deepening loop
→ Basic material + positional evaluation
→ Killer moves and history heuristic
→ Transposition table with Zobrist hashing
```

### ❌ Removed from V14.x (Caused Regressions):
```
→ 12-category move ordering complexity
→ 50-node time checking (excessive overhead)
→ 60% time limit emergency stops
→ Multiple emergency bailout points (85% thresholds)
→ Emergency stop flags
→ Complex minimum/target depth calculations
→ Game phase detection for every search
→ Advanced pawn structure evaluator
→ Advanced king safety evaluator
→ Dynamic bishop valuation (325/275)
→ Threat detection and scoring
→ Development move prioritization
→ Tactical pattern bonuses
```

### 🆕 New in V14.9.1 (Tuning Improvements):
```
→ Aggressive opening time management (30% factor, 0.5s cap)
→ PV stability tracking for early exit
→ Proper iteration time prediction (prevents max_time overflow)
→ Tactical position detection (noisy = captures ≥5 OR checks ≥3)
→ Extended time allocation for noisy positions (100% factor)
→ Quiet position early exit (PV stable 2+ iterations)
→ Fixed root search move selection (PV now matches move played)
→ Simplified evaluation (bitboard-only, no negative baseline)
```

---

## 📊 Performance Characteristics

### Search Speed:
```
→ Nodes per second: 3,000-4,000 nps
→ Typical depth: 4-6 ply (same as V12.6)
→ Opening moves: 0.3-0.5s (FAST - was 3+ seconds in V14.8)
→ Middlegame quiet: 0.9-2.0s (early exit working)
→ Middlegame tactical: 2.0-5.0s (uses full time)
→ Endgame: 0.1-3.0s (depends on complexity)
```

### Time Management:
```
→ Opening speed: ✅ <1s (0.35s measured)
→ Stable PV exit: ✅ ~18% efficiency on quiet positions
→ Tactical depth: ✅ Full time on complex positions
→ Iteration prediction: ✅ Prevents max_time overflow
→ No time flagging: ✅ Reliable time management
```

### Evaluation Factors:
```
→ Material count (6 piece types)
→ Piece-square tables (positioning bonuses)
→ Total: ~8 core evaluation components
→ Simplified from V14.8's 15+ factors
→ Faster evaluation = deeper search
```

---

## 🎯 V14.9.1 Philosophy

**"Simple, Proven, Reliable"**

V14.9.1 represents a return to fundamentals after the V14.3-V14.8 series attempted complex optimizations that backfired:
- V14.3: 17.1% tournament (emergency time management killed search)
- V14.8: 38.8% puzzles (move ordering too complex, time management broken)

V14.9.1 restores V12.6's proven workflow:
- V12.6: 85%+ puzzles, 57.1% tournament (solid baseline)
- V14.9.1: Simple architecture + time tuning = reliable performance

**Key Insight:** Chess engine strength comes from:
1. **Search depth** (seeing further ahead)
2. **Move ordering** (examining best moves first)  
3. **Time management** (using time wisely)
4. **Evaluation accuracy** (judging positions correctly)

V14.9.1 excels at #1-3 with simplified, predictable components. Future improvements (V15+) will enhance #4 with better positional understanding while maintaining the proven simple architecture.

---

## 🔮 Path to V15

V14.9.1 establishes a stable foundation. V15 enhancements should focus on:
1. **Evaluation improvements** (better position judgment)
2. **Opening book** (instant moves in known theory)
3. **Endgame tables** (perfect play in simple endings)
4. **Selective extensions** (search critical positions deeper)

All improvements must maintain V14.9.1's simple, reliable architecture.

---

## 📝 Summary Workflow Diagram

```
UCI Command → Position Setup → Time Allocation → Iterative Deepening Loop
                                                          ↓
                                    ┌─────────────────────────────────┐
                                    │ For each depth 1..6:            │
                                    │  • Check time (target/max)      │
                                    │  • Predict next iteration       │
                                    │  • Call recursive search        │
                                    │  • Track PV stability           │
                                    │  • Early exit if stable         │
                                    └─────────────────────────────────┘
                                                          ↓
                    ┌─────────────────────────────────────────────────┐
                    │ Recursive Alpha-Beta Search:                    │
                    │  • Order moves (5 categories)                   │
                    │  • Try each move recursively                    │
                    │  • Quiescence search at leaves                  │
                    │  • Evaluate positions (bitboard-only)           │
                    │  • Alpha-beta pruning                           │
                    │  • Transposition table cache                    │
                    │  • Time check every 1000 nodes                  │
                    └─────────────────────────────────────────────────┘
                                                          ↓
                                        Return Best Move → UCI Output
```

This workflow represents how V7P3R V14.9.1 "thinks" about chess - it systematically examines possibilities with proven simple ordering, evaluates positions using reliable material + positioning, and selects moves that lead to favorable outcomes with smart time management.
