# 🎯 V7P3R V15 Proposed Step-by-Step Workflow

## 📋 Version Summary
**V15.0 Planned Changes:**
- Heuristic priority review
- Time management verification
- Blunder firewall reinforcement and re-verification
- Opening enhancements and center control

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
→ 5-category system:

1. **Transposition Table Move** (if available)
   • Previously best move from TT probe
   • Highest priority - already proven good
   
2. **Captures** (MVV-LVA + SEE ordering) - V15 ENHANCED
   • Most Valuable Victim - Least Valuable Attacker baseline
   • Queen captures first, pawn captures last (traditional)
   • **NEW: Static Exchange Evaluation (SEE) Enhancement**
     ```python
     # V15 SEE-Enhanced Capture Ordering
     for move in capture_moves:
         mvv_lva_score = VICTIM_VALUES[captured_piece] - ATTACKER_VALUES[moving_piece]
         see_score = self._static_exchange_evaluation(board, move)
         
         if see_score < 0:  # Losing capture
             final_score = mvv_lva_score - 10000  # Heavily deprioritize
         else:  # Winning or equal capture
             final_score = mvv_lva_score + see_score
             
         move_scores.append((move, final_score))
     ```
   • **Benefits:** Prevents examining obviously losing captures first
   • **Philosophy:** Tactical awareness without complexity bloat
   
3. **Checks** (giving check moves) - V15 ENHANCED
   • Forcing moves that put opponent king in check
   • Can lead to tactical opportunities and king safety threats
   • **NEW: Symmetrical Check Awareness (Enhanced King Safety)**
     ```python
     # V15 Enhanced Check Evaluation
     def _evaluate_check_move(self, board, move):
         base_score = 1000  # Standard check bonus
         
         # King safety symmetry - consider our king exposure
         board.push(move)
         opponent_checks = len([m for m in board.legal_moves if board.gives_check(m)])
         king_safety_penalty = opponent_checks * 50  # Penalty for exposing our king
         board.pop()
         
         # Enhanced check types
         if board.is_checkmate():
             return 30000  # Checkmate priority
         elif self._is_discovered_check(board, move):
             return base_score + 200  # Discovered checks powerful
         elif self._is_double_check(board, move):
             return base_score + 300  # Double checks very strong
         else:
             return base_score - king_safety_penalty
     ```
   • **Benefits:** Avoids reckless checks that expose our own king
   • **Philosophy:** Tactical aggression with positional responsibility
   
4. **Killer Moves** (non-capture moves that caused cutoffs)
   • Previously successful quiet moves at this depth
   • Position-independent move history
   
5. **Quiet Moves** (remaining moves)
   • History heuristic scoring for move ordering
   • All other legal moves

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

TIME MANAGEMENT:
→ Check every 1000 nodes (not 50) - 20x less overhead
→ If elapsed > time_limit → return current eval
→ Single abort point - trust the algorithm
→ No emergency stop flags
→ No 85% bailout thresholds
→ Philosophy: Simple, predictable, proven
```

---

## 4. Position Evaluation (The "Judgment" System)

### Step 4a: Simplified Bitboard Evaluation
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

### Step 4c: Positional Evaluation (Bitboard-Based) - V15 ENHANCED
```
→ Piece-Square Tables (PST) applied via bitboard evaluator:
   • Knights prefer center squares (+30 bonus)
   • Bishops prefer long diagonals (+20 bonus)
   • Rooks prefer 7th rank and open files (+10 bonus)
   • Pawns prefer advancement (+5 per rank)
   • Kings prefer corners in opening/middlegame
   • Kings prefer center in endgame

→ **V15 NEW: Enhanced Queen and Pawn Positioning**
   ```python
   # V15 Reduced Queen Early Development Penalty
   QUEEN_OPENING_PST = {
       # Heavily penalize early queen moves in opening
       'd1': 0,    'e1': 0,     # Starting squares neutral
       'd2': -50,  'e2': -50,   # Early development penalty
       'd3': -100, 'e3': -100,  # Further advancement penalty
       'd4': -150, 'e4': -150,  # Center control penalty (too early)
       # ... (encourage queen to stay back until minor pieces developed)
   }
   
   # V15 Enhanced Pawn Center Control
   PAWN_OPENING_PST = {
       # Multi-square advances for center control
       'e2': 0,   'd2': 0,     # Starting squares
       'e3': +10, 'd3': +10,   # One square advance
       'e4': +25, 'd4': +25,   # Two square advance - excellent center control
       'e5': +15, 'd5': +15,   # Advanced pawns (context dependent)
       # Encourage early e4/d4 pawn breaks for center control
   }
   ```

→ **V15 Center Control Philosophy:**
   • Discourage early queen sorties (pieces before queen principle)
   • Reward aggressive pawn center control (e4, d4 advances)
   • Maintain piece activity bonuses for proper development
   
→ Applied during calculate_score_optimized():
   for each piece:
      base_value = piece_values[piece_type]
      
      # V15: Game phase aware PST selection
      if game_phase == "opening" and piece_type == QUEEN:
          positional_bonus = QUEEN_OPENING_PST[square]
      elif game_phase == "opening" and piece_type == PAWN:
          positional_bonus = PAWN_OPENING_PST[square]
      else:
          positional_bonus = piece_square_table[square]
      
      total += base_value + positional_bonus

→ All positional scoring consolidated in bitboard evaluator
→ No separate evaluator calls (performance optimization maintained)
```

### Step 4d: Quiescence Search (Tactical Stability) - V15 ENHANCED
```
→ Called at leaf nodes to prevent horizon effect
→ Continues searching forcing moves until position is quiet
→ Maximum 4 ply extension for tactical sequences

V15 Enhanced Process - Threat-Aware Quiescence:
   ```python
   # V15 Enhanced Quiescence Move Generation
   def _generate_quiescence_moves(self, board):
       forcing_moves = []
       
       # Traditional captures
       for move in board.legal_moves:
           if board.is_capture(move):
               forcing_moves.append(move)
       
       # V15 NEW: Add checks and promotions
       for move in board.legal_moves:
           if board.gives_check(move) and not board.is_capture(move):
               forcing_moves.append(move)  # Non-capture checks
           elif move.promotion:
               forcing_moves.append(move)  # Pawn promotions
       
       # V15 NEW: Add threatened piece escapes (if material behind)
       if self._is_material_behind(board):
           for move in board.legal_moves:
               if self._escapes_threat(board, move):
                   forcing_moves.append(move)
       
       return forcing_moves
   
   # Enhanced quiescence evaluation
   def _quiescence_search(self, board, alpha, beta, depth):
       # Stand-pat evaluation (option to not move)
       stand_pat = self._evaluate_position(board)
       
       if stand_pat >= beta:
           return beta  # Beta cutoff
       if stand_pat > alpha:
           alpha = stand_pat  # Improve alpha
       
       # Generate and try forcing moves
       for move in self._generate_quiescence_moves(board):
           board.push(move)
           score = -self._quiescence_search(board, -beta, -alpha, depth - 1)
           board.pop()
           
           if score >= beta:
               return beta  # Beta cutoff
           if score > alpha:
               alpha = score  # New best
       
       return alpha
   ```
   
→ **V15 Prevents:**
   • Hanging pieces after search horizon ✅
   • Missing tactical sequences (captures, checks, promotions) ✅
   • Missing critical defensive moves when behind in material ✅
   • Incorrect static evaluations in tactical positions ✅

→ **Philosophy:** Comprehensive forcing move detection without explosion
→ **Performance:** Limited to 4 ply max, selective move generation
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

## 🔧 Key V15 Strategic Goals & Technical Implementation

### 1. Enhanced Threat Awareness (SEE Integration)
```python
# Static Exchange Evaluation for capture ordering
def _static_exchange_evaluation(self, board, square):
    """Calculate material gain/loss from captures on given square"""
    attackers = self._get_attackers(board, square, board.turn)
    defenders = self._get_attackers(board, square, not board.turn)
    
    if not attackers:
        return 0
    
    # Simulate capture sequence
    gain = [0] * 32  # Max capture sequence depth
    gain[0] = PIECE_VALUES[board.piece_at(square).piece_type]
    
    # Alternate captures by value
    attacker_values = sorted([PIECE_VALUES[board.piece_at(sq).piece_type] 
                             for sq in attackers])
    defender_values = sorted([PIECE_VALUES[board.piece_at(sq).piece_type] 
                             for sq in defenders])
    
    # Calculate net material after all exchanges
    for i in range(1, min(len(attacker_values) + len(defender_values), 32)):
        if i % 2 == 1:  # Defender captures
            if i // 2 < len(defender_values):
                gain[i] = defender_values[i // 2] - gain[i-1]
        else:  # Attacker captures
            if i // 2 < len(attacker_values):
                gain[i] = gain[i-1] - attacker_values[i // 2 - 1]
    
    # Minimax backwards to find best outcome
    for i in range(len(gain) - 2, -1, -1):
        gain[i] = max(gain[i], gain[i+1])
    
    return gain[0]
```

### 2. King Safety Move Symmetry (Defensive Checks)
```python
# Enhanced check evaluation considering king safety
def _evaluate_check_with_safety(self, board, move):
    """Evaluate check moves while considering our king exposure"""
    if not board.gives_check(move):
        return 0
    
    # Make the check move
    board.push(move)
    
    # Count opponent's check responses
    opponent_check_count = sum(1 for m in board.legal_moves if board.gives_check(m))
    
    # Evaluate check strength
    if board.is_checkmate():
        check_value = 30000
    elif board.is_check():
        if self._is_double_check(board):
            check_value = 1300  # Double check very strong
        elif self._is_discovered_check(board, move):
            check_value = 1200  # Discovered check powerful
        else:
            check_value = 1000  # Standard check
    else:
        check_value = 0
    
    # Apply king safety penalty
    safety_penalty = opponent_check_count * 50
    
    board.pop()
    return max(0, check_value - safety_penalty)
```

### 3. Intelligent Queen Development Control
```python
# Opening-phase queen positioning penalties
OPENING_QUEEN_PENALTIES = {
    'pieces_developed': 0,  # Count of developed pieces (N, B, castled)
    'penalty_per_early_square': {
        'd2': 50,  'd3': 100, 'd4': 150, 'd5': 200,
        'e2': 50,  'e3': 100, 'e4': 150, 'e5': 200,
        'f3': 75,  'c3': 75,  'h5': 200,  # Common early queen squares
    }
}

def _apply_queen_development_penalty(self, board, queen_square):
    """Penalize early queen development before minor pieces"""
    if self._count_developed_pieces(board) < 2:  # Less than 2 minor pieces out
        return OPENING_QUEEN_PENALTIES['penalty_per_early_square'].get(
            chess.square_name(queen_square), 0)
    return 0
```

### 4. Enhanced Center Control (Aggressive Pawn Play)
```python
# Pawn structure evaluation for center control
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [chess.C4, chess.C5, chess.F4, chess.F5]

def _evaluate_pawn_center_control(self, board):
    """Reward aggressive center pawn advances"""
    score = 0
    
    for square in CENTER_SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            if piece.color == board.turn:
                score += 25  # Own pawn in center
            else:
                score -= 15  # Opponent pawn in center
    
    for square in EXTENDED_CENTER:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            if piece.color == board.turn:
                score += 10  # Extended center control
    
    return score
```

---

## 🎯 V15 Implementation Priority & Risk Assessment

### Implementation Order (Recommended):
```
Phase 1 (Low Risk): SEE-Enhanced Capture Ordering
→ Add _static_exchange_evaluation() method
→ Modify capture ordering in _order_moves_advanced()
→ Test: Should improve capture quality immediately
→ Risk: Very low - only affects move ordering, not search logic

Phase 2 (Low Risk): Enhanced Pawn Center Control
→ Modify PAWN_PST values in bitboard evaluator
→ Add center control bonuses for e4/d4 advances
→ Test: Should encourage more aggressive openings
→ Risk: Low - only affects positional evaluation

Phase 3 (Medium Risk): Queen Development Control
→ Add opening phase detection
→ Implement queen early development penalties
→ Modify QUEEN_PST for opening phase
→ Test: Should delay queen development appropriately
→ Risk: Medium - affects opening play significantly

Phase 4 (Medium Risk): Enhanced Quiescence Search
→ Add checks and promotions to forcing moves
→ Add threatened piece escape detection
→ Modify _quiescence_search() generation
→ Test: Should catch more tactical sequences
→ Risk: Medium - affects search tree size

Phase 5 (High Risk): Symmetrical Check Awareness
→ Add king safety evaluation for check moves
→ Implement opponent check counting
→ Modify check move scoring
→ Test: Should reduce reckless check moves
→ Risk: High - complex evaluation, potential search slowdown
```

### Rollback Strategy:
```
→ Each phase implemented as separate commit
→ Performance testing after each phase
→ If any phase degrades performance >10%, rollback immediately
→ Maintain V14.9.1 as stable baseline
→ Each enhancement can be independently disabled
```

### Success Metrics:
```
Phase 1: Improved capture sequences in tactical positions
Phase 2: More e4/d4 openings, better center control
Phase 3: Delayed queen development, better piece coordination  
Phase 4: Improved tactical puzzle accuracy (+5-10%)
Phase 5: Reduced king safety blunders, better defensive play

Overall V15 Target: 90%+ puzzle accuracy, competitive tournament performance
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

V14.9.1 excels at #1-3 with simplified, predictable components. V15 enhances #2 and #4 with tactical awareness and better positional understanding while maintaining the proven simple architecture.

**V15 Enhancement Philosophy:**
- **Tactical Awareness:** SEE prevents wasted nodes on losing captures
- **Positional Improvement:** Better opening principles and center control
- **Defensive Balance:** King safety awareness prevents tactical oversights
- **Forcing Move Coverage:** Enhanced quiescence catches more tactical themes

**Architecture Preservation:**
- Maintain V14.9.1's simple iterative deepening
- Keep proven time management system
- Preserve 5-category move ordering structure
- No complex evaluation subsystems
- All improvements modular and reversible

---

## 🚀 V15 Readiness Checklist

### Before Implementation:
- [ ] Create V15 development branch
- [ ] Run V14.9.1 baseline tests (opening speed, time management, tactical accuracy)
- [ ] Backup current engine state
- [ ] Review each phase implementation details

### During Implementation:
- [ ] Implement phases sequentially (SEE → Pawns → Queen → Quiescence → Checks)
- [ ] Test after each phase with quick validation
- [ ] Monitor performance impact (nodes/second should stay >3000)
- [ ] Validate move selection still sensible

### V15 Validation:
- [ ] Opening speed <1s maintained
- [ ] Time management working (no flags, appropriate allocation)
- [ ] Tactical puzzle accuracy improved (+5-10% target)
- [ ] No regression in tournament play vs V14.9.1
- [ ] Enhanced opening play (more e4/d4, delayed queen development)

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
