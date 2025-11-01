# V14.7 Blunder Prevention Architecture

## Problem Analysis

### V14.6 Failures (Game vs V14.4)
The V14.6 vs V14.4 game showed catastrophic blunders despite having blunder firewall code:

1. **Move 3. b3** - Should capture Ne4, instead played random pawn move allowing Ng5
2. **Move 6. Rg1** - Random rook move after dxc3 was clearly better
3. **Move 12. Bxc6+** - Traded bishop for nothing, giving up material
4. **Move 20. Bxc1** - Blundered bishop for absolutely nothing (major blunder)

### Root Cause
The blunder firewall function `evaluate_move_safety_bitboard()` exists in the bitboard evaluator but **is never called during move selection**. The function `_filter_moves_by_safety()` exists in v7p3r.py but is not integrated into the search pipeline.

**Current Flow:**
1. Generate legal moves
2. Order moves tactically (`_order_moves_advanced`)
3. Search each move with alpha-beta
4. **No safety filtering happens**

**What's Missing:**
- Safety filtering is not called before move ordering
- Unsafe moves are not rejected from consideration
- Hanging piece detection is not preventing material loss
- Capture validation is not preventing bad trades

## V14.7 Architecture

### Priority 1: Move Safety Filtering (PRE-ORDER)

**Integrate safety filtering BEFORE move ordering:**

```python
def _recursive_search(board, depth, alpha, beta, max_time):
    legal_moves = list(board.legal_moves)
    
    # PRIORITY 1: Filter unsafe moves (NEW - V14.7)
    safe_moves = self._filter_unsafe_moves(board, legal_moves)
    
    # PRIORITY 2: Order remaining safe moves tactically
    ordered_moves = self._order_moves_advanced(board, safe_moves, depth, tt_move)
    
    # PRIORITY 3: Search ordered safe moves
    for move in ordered_moves:
        ...
```

### Safety Checks (Three-Part Firewall)

#### 1. **King Safety Check** (CRITICAL)
```python
def _check_king_safety(board, move):
    # After move, is our king under undefended attack?
    board.push(move)
    our_king = board.king(not board.turn)
    enemy_attacks = board.attacks_mask(board.turn)
    our_defense = board.attacks_mask(not board.turn)
    
    if enemy_attacks & chess.BB_SQUARES[our_king]:
        if not (our_defense & chess.BB_SQUARES[our_king]):
            # REJECT: King under undefended attack
            return False
    board.pop()
    return True
```

#### 2. **Queen Safety Check** (HIGH PRIORITY)
```python
def _check_queen_safety(board, move):
    # After move, is our queen hanging (undefended)?
    board.push(move)
    our_queens = board.pieces(chess.QUEEN, not board.turn)
    enemy_attacks = board.attacks_mask(board.turn)
    our_defense = board.attacks_mask(not board.turn)
    
    for queen_sq in our_queens:
        if enemy_attacks & chess.BB_SQUARES[queen_sq]:
            if not (our_defense & chess.BB_SQUARES[queen_sq]):
                # REJECT: Queen hanging
                return False
    board.pop()
    return True
```

#### 3. **Valuable Piece Safety** (MEDIUM PRIORITY)
```python
def _check_valuable_pieces(board, move):
    # After move, are rooks/bishops/knights hanging?
    board.push(move)
    our_color = not board.turn
    enemy_attacks = board.attacks_mask(board.turn)
    our_defense = board.attacks_mask(our_color)
    
    # Check rooks, bishops, knights
    for piece_type in [chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        our_pieces = board.pieces(piece_type, our_color)
        for piece_sq in our_pieces:
            if enemy_attacks & chess.BB_SQUARES[piece_sq]:
                if not (our_defense & chess.BB_SQUARES[piece_sq]):
                    # Hanging piece - reject move
                    return False
    
    board.pop()
    return True
```

#### 4. **Capture Validation** (NEW)
```python
def _validate_capture(board, move):
    # Is this capture safe? Are we trading up or equal?
    if not board.is_capture(move):
        return True  # Not a capture, pass
    
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    
    if not victim:
        return True  # No victim (en passant, etc)
    
    victim_value = get_piece_value(victim.piece_type)
    attacker_value = get_piece_value(attacker.piece_type)
    
    # Check if square is defended after capture
    board.push(move)
    enemy_attacks = board.attacks_mask(board.turn)
    attacker_square = move.to_square
    
    if enemy_attacks & chess.BB_SQUARES[attacker_square]:
        # Capture square is defended - is trade worthwhile?
        if attacker_value > victim_value:
            # Losing trade (e.g., bishop takes pawn but bishop gets captured)
            board.pop()
            return False
    
    board.pop()
    return True
```

### Implementation Strategy

#### Phase 1: Create Unified Safety Filter
Create new method `_filter_unsafe_moves` that combines all safety checks:

```python
def _filter_unsafe_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    """
    V14.7: CRITICAL blunder prevention - filter moves BEFORE ordering
    
    Returns only moves that pass all safety checks:
    1. King safety (CRITICAL)
    2. Queen safety (HIGH)
    3. Valuable piece safety (MEDIUM)
    4. Capture validation (MEDIUM)
    """
    safe_moves = []
    
    for move in moves:
        # Check 1: King safety
        if not self._is_king_safe_after_move(board, move):
            continue  # REJECT
        
        # Check 2: Queen safety
        if not self._is_queen_safe_after_move(board, move):
            continue  # REJECT
        
        # Check 3: Valuable pieces safe
        if not self._are_valuable_pieces_safe(board, move):
            continue  # REJECT
        
        # Check 4: Valid capture
        if not self._is_capture_valid(board, move):
            continue  # REJECT
        
        # Passed all checks - move is safe
        safe_moves.append(move)
    
    # If ALL moves filtered out (very rare), return one legal move to avoid stalemate
    if not safe_moves and moves:
        safe_moves = [moves[0]]
    
    return safe_moves
```

#### Phase 2: Integrate into Search Pipeline
Update `_recursive_search` to call filter before ordering:

```python
def _recursive_search(self, board, depth, alpha, beta, max_time):
    ...
    legal_moves = list(board.legal_moves)
    
    # V14.7: FILTER unsafe moves first
    safe_moves = self._filter_unsafe_moves(board, legal_moves)
    
    # Then order tactically
    ordered_moves = self._order_moves_advanced(board, safe_moves, depth, tt_move)
    
    for move in ordered_moves:
        ...
```

#### Phase 3: Enhanced Evaluation Penalties
Update position evaluation to heavily penalize hanging pieces:

```python
def evaluate_position_complete(board):
    ...
    # V14.7: Heavy penalties for hanging pieces
    hanging_penalty = self._calculate_hanging_pieces_penalty(board)
    
    # Apply to final evaluation
    evaluation -= hanging_penalty  # Subtract from our score
    ...
```

### Performance Considerations

**Concern:** Safety filtering adds computational cost per move.

**Mitigation:**
1. **Bitboard Operations**: All safety checks use fast bitboard masks
2. **Early Rejection**: Most unsafe moves rejected within 1-2 checks
3. **Depth Gating**: Only filter at shallow depths (depth >= 2) where blunders matter most
4. **Caching**: Cache attack masks for reuse across multiple moves

**Expected Cost:**
- ~10-15% NPS reduction at shallow depths (acceptable trade for blunder prevention)
- Negligible cost at deeper depths (fewer moves to filter)

### Testing Validation

#### Test 1: Blunder Position Prevention
Test V14.7 against positions from V14.6 game:

1. Position after 2...Ne4 - Should play 3.Nxe4, not 3.b3
2. Position after 5...Nxc3 - Should play 6.dxc3, not 6.Rg1
3. Position before 12.Bxc6+ - Should not trade bishop for nothing
4. Position before 20.Bxc1 - Should not blunder bishop

#### Test 2: Self-Play Validation
- V14.7 vs V14.6: 50 games, verify <5% blunder rate (V14.6 had ~20%+)
- V14.7 vs V14.4: 50 games, verify no piece-hanging blunders
- V14.7 vs Stockfish 1%: Verify no catastrophic queen/rook blunders

#### Test 3: Performance Regression
- Measure NPS impact: Target <15% reduction
- Verify depth achievement: Should maintain depth 5-6
- Time management: Ensure no flagging

### Success Criteria

✅ **Critical Success:**
- No hanging queen blunders (zero tolerance)
- No hanging rook blunders (zero tolerance)
- <5% minor piece blunders (acceptable due to tactical complexity)

✅ **Performance Success:**
- <15% NPS reduction from V14.6
- Maintain depth 5+ achievement in middlegame
- No time flagging in 180+2 games

✅ **Rating Success:**
- Win rate vs V14.6: >55% (proving blunder prevention improvement)
- Win rate vs V14.4: >60% (proving superiority over broken version)

## Implementation Checklist

- [ ] Create `_filter_unsafe_moves()` master function
- [ ] Implement `_is_king_safe_after_move()`
- [ ] Implement `_is_queen_safe_after_move()`
- [ ] Implement `_are_valuable_pieces_safe()`
- [ ] Implement `_is_capture_valid()`
- [ ] Integrate filter into `_recursive_search()`
- [ ] Update UCI version to V14.7
- [ ] Create blunder position test suite
- [ ] Run self-play validation (V14.7 vs V14.6, V14.4)
- [ ] Measure NPS impact
- [ ] Deploy to Arena for user testing
