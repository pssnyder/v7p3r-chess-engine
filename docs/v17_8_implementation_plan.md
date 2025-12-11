# V17.8 Implementation Plan: Time Management & Mate Detection
**Date**: December 7, 2025
**Target**: Reduce time forfeits and improve mate detection
**Priority**: HIGH (30% of losses are time-related)

## Overview

Based on analysis of 10 recent v17.7 games, two critical issues emerged:
1. **Time forfeit losses** in winning/drawable positions
2. **Missed checkmate threats** leading to sudden losses

## Implementation Approach

### Minimal Impact Philosophy
Following our implementation guardrails:
- Small, incremental changes
- Modular components with low coupling
- Test each feature independently
- No fundamental architecture changes

## Component 1: Basic Time Management

### File: `src/v7p3r_time_manager.py` (NEW)

```python
"""
V7P3R Time Management Module
Simple time allocation strategy for different time controls
"""

class TimeManager:
    def __init__(self, base_time_ms, increment_ms, time_control_type='blitz'):
        """
        Args:
            base_time_ms: Starting time in milliseconds
            increment_ms: Increment per move in milliseconds  
            time_control_type: 'bullet', 'blitz', 'rapid', 'classical'
        """
        self.base_time = base_time_ms
        self.increment = increment_ms
        self.time_control = time_control_type
        
    def calculate_move_time(self, time_remaining_ms, moves_played, is_endgame=False):
        """
        Calculate how much time to spend on this move
        
        Strategy:
        - Use increment + small buffer from base time
        - Reduce thinking time in simple positions
        - Increase in complex middle game
        
        Returns:
            max_think_time_ms: Maximum milliseconds for this move
        """
        # Safety margin - always keep a time cushion
        SAFETY_BUFFER_MS = 2000  # 2 seconds minimum reserve
        
        if time_remaining_ms < SAFETY_BUFFER_MS:
            # Emergency time - move instantly
            return 100  # 0.1 second
        
        # Available time = remaining - safety buffer
        available_time = time_remaining_ms - SAFETY_BUFFER_MS
        
        # Estimate moves remaining
        if moves_played < 20:
            moves_remaining = 40  # Opening/early middle
        elif moves_played < 40:
            moves_remaining = 30  # Middle game
        else:
            moves_remaining = 20  # Endgame
        
        # Base allocation: divide remaining time by moves remaining
        base_allocation = available_time / moves_remaining
        
        # Add increment (we get it back after moving)
        base_allocation += self.increment
        
        # Adjust for game phase
        if is_endgame:
            # Endgames: use less time (50% of base)
            base_allocation *= 0.5
        elif 20 <= moves_played < 40:
            # Critical middle game: use more time (120% of base)
            base_allocation *= 1.2
        
        # Apply time control multipliers
        if self.time_control == 'bullet':
            # Bullet: move faster, rely on increment
            max_time = min(base_allocation, self.increment * 1.5)
        elif self.time_control == 'blitz':
            # Blitz: balanced approach
            max_time = min(base_allocation, self.increment * 2.5)
        elif self.time_control == 'rapid':
            # Rapid: can think longer
            max_time = min(base_allocation, self.increment * 5.0)
        else:
            # Classical: no restriction
            max_time = base_allocation
        
        # Floor and ceiling
        min_time = 100  # At least 0.1 second
        max_time = max(min_time, min(max_time, time_remaining_ms - SAFETY_BUFFER_MS))
        
        return int(max_time)
```

### Integration into `v7p3r_uci.py`

```python
# In _handle_go() method, add time management

def _handle_go(self, tokens):
    """Handle go command with time management"""
    # Parse time control
    wtime = None
    btime = None
    winc = 0
    binc = 0
    
    i = 1
    while i < len(tokens):
        if tokens[i] == 'wtime':
            wtime = int(tokens[i+1])
            i += 2
        elif tokens[i] == 'btime':
            btime = int(tokens[i+1])
            i += 2
        elif tokens[i] == 'winc':
            winc = int(tokens[i+1])
            i += 2
        elif tokens[i] == 'binc':
            binc = int(tokens[i+1])
            i += 2
        else:
            i += 1
    
    # Determine our time
    our_time = wtime if self.engine.board.turn == chess.WHITE else btime
    our_inc = winc if self.engine.board.turn == chess.WHITE else binc
    
    # Calculate move time if time control specified
    max_time_ms = None
    if our_time is not None:
        from v7p3r_time_manager import TimeManager
        
        # Detect time control type
        if our_time < 120000:  # < 2 minutes
            tc_type = 'bullet'
        elif our_time < 600000:  # < 10 minutes
            tc_type = 'blitz'
        else:
            tc_type = 'rapid'
        
        tm = TimeManager(our_time, our_inc, tc_type)
        moves_played = len(self.engine.board.move_stack)
        is_endgame = self._is_endgame(self.engine.board)
        
        max_time_ms = tm.calculate_move_time(our_time, moves_played, is_endgame)
        
        # Convert to seconds for search
        max_time_sec = max_time_ms / 1000.0
        
        # Pass to search with time limit
        best_move = self.engine.search_position(max_time=max_time_sec)
    else:
        # No time control - use default depth
        best_move = self.engine.search_position()
    
    print(f"bestmove {best_move}")
```

## Component 2: Mate-in-1 Detection

### Enhancement to `v7p3r.py`

Add pre-move mate verification:

```python
def _check_immediate_mate(self, board):
    """
    Check if any move delivers immediate checkmate
    Always call this before deeper search
    
    Returns:
        mate_move: Move that delivers mate, or None
    """
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        board.pop()
    
    return None

# In search_position():
def search_position(self, max_time=None):
    """Search with mate-in-1 fast path"""
    
    # ALWAYS check for immediate mate first (fast)
    mate_move = self._check_immediate_mate(self.board)
    if mate_move:
        return mate_move
    
    # Continue with regular search...
    # (existing code)
```

## Component 3: Enhanced King Safety

### Enhancement to `v7p3r_fast_evaluator.py`

```python
def _evaluate_king_safety_v17_8(self, board, color):
    """
    Enhanced king safety evaluation for v17.8
    Focus on back-rank mate patterns
    """
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    
    safety_score = 0
    king_rank = chess.square_rank(king_sq)
    king_file = chess.square_file(king_sq)
    
    # Back-rank weakness detection
    if (color == chess.WHITE and king_rank == 0) or \
       (color == chess.BLACK and king_rank == 7):
        # King on back rank - check for escape squares
        escape_squares = 0
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            if 0 <= check_file <= 7:
                # Check square in front of king
                check_rank = king_rank + (1 if color == chess.WHITE else -1)
                check_sq = chess.square(check_file, check_rank)
                
                # If square is empty or defended, it's an escape square
                if not board.is_attacked_by(not color, check_sq):
                    escape_squares += 1
        
        # Heavy penalty if trapped on back rank
        if escape_squares == 0:
            safety_score -= 80  # BACK_RANK_TRAPPED
        elif escape_squares == 1:
            safety_score -= 40  # Limited escape
        
        # Check for enemy rooks/queens on same file or rank
        enemy_pieces = board.pieces(chess.ROOK, not color) | \
                      board.pieces(chess.QUEEN, not color)
        
        for piece_sq in enemy_pieces:
            piece_rank = chess.square_rank(piece_sq)
            piece_file = chess.square_file(piece_sq)
            
            # Same rank (horizontal attack)
            if piece_rank == king_rank:
                safety_score -= 60  # BACK_RANK_THREAT
            
            # Same file (vertical attack)
            if piece_file == king_file:
                safety_score -= 40  # FILE_PRESSURE
    
    # Exposed king (few defenders nearby)
    defender_count = 0
    for sq in self._get_king_ring(king_sq):
        if board.piece_at(sq) and board.piece_at(sq).color == color:
            defender_count += 1
    
    if defender_count < 2:
        safety_score -= 30  # Exposed king
    
    return safety_score

def _get_king_ring(self, king_sq):
    """Get squares around king (8 squares)"""
    ring = []
    king_rank = chess.square_rank(king_sq)
    king_file = chess.square_file(king_sq)
    
    for rank_offset in [-1, 0, 1]:
        for file_offset in [-1, 0, 1]:
            if rank_offset == 0 and file_offset == 0:
                continue
            new_rank = king_rank + rank_offset
            new_file = king_file + file_offset
            if 0 <= new_rank <= 7 and 0 <= new_file <= 7:
                ring.append(chess.square(new_file, new_rank))
    
    return ring
```

## Component 4: Deeper Mate Extensions

### Enhancement to `v7p3r.py` in `_recursive_search()`

```python
# Current v17.7: +2 plies when mate detected
# V17.8: +3 plies, earlier trigger

if abs(eval_score) > 9000:  # Mate score
    # v17.7 had +2, increase to +3
    extended_depth = depth + 3  
    # Re-search with extended depth
    eval_score = -self._recursive_search(
        board, extended_depth, -beta, -alpha, not maximizing
    )
```

Also add "mate threat" extension:

```python
def _is_in_check_or_mate_threat(self, board):
    """
    Check if position has immediate tactical threats
    """
    if board.is_check():
        return True
    
    # Check if enemy has forcing moves (checks)
    for move in board.legal_moves:
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        if gives_check:
            return True  # Enemy can give check
    
    return False

# In _recursive_search(), before depth cutoff:
if self._is_in_check_or_mate_threat(board):
    depth += 1  # Extend by 1 ply in forcing positions
```

## Testing Plan

### Phase 1: Time Management Testing (Week 1)

```python
# testing/test_v17_8_time_management.py

def test_time_allocation_blitz():
    """Test time allocation for 5min+4s blitz"""
    tm = TimeManager(300000, 4000, 'blitz')
    
    # Move 10, middle game, 4 minutes left
    time_left = 240000
    moves = 10
    allocated = tm.calculate_move_time(time_left, moves, is_endgame=False)
    
    # Should allocate ~8-12 seconds (8000-12000 ms)
    assert 6000 < allocated < 15000, f"Got {allocated}ms"
    
def test_time_pressure():
    """Test behavior with low time"""
    tm = TimeManager(300000, 4000, 'blitz')
    
    # Only 5 seconds left
    time_left = 5000
    allocated = tm.calculate_move_time(time_left, 30, False)
    
    # Should move quickly (use increment only)
    assert allocated < 3000, f"Too slow under time pressure: {allocated}ms"

def test_endgame_speed():
    """Test faster moves in endgame"""
    tm = TimeManager(300000, 4000, 'blitz')
    
    # Endgame with plenty of time
    time_left = 180000
    allocated = tm.calculate_move_time(time_left, 45, is_endgame=True)
    
    # Should use less time in endgame
    normal = tm.calculate_move_time(time_left, 45, is_endgame=False)
    assert allocated < normal * 0.7, "Endgame should be faster"
```

### Phase 2: Mate Detection Testing (Week 1)

```python
# testing/test_v17_8_mate_detection.py

def test_mate_in_1_detection():
    """Test immediate mate detection"""
    from v7p3r import V7P3R
    
    # Back rank mate position
    fen = "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"  # Ra8#
    engine = V7P3R()
    engine.board = chess.Board(fen)
    
    mate_move = engine._check_immediate_mate(engine.board)
    assert mate_move == chess.Move.from_uci("a1a8"), "Missed mate in 1"

def test_back_rank_safety():
    """Test back-rank weakness detection"""
    from v7p3r_fast_evaluator import FastEvaluator
    
    # Trapped king on back rank
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"
    board = chess.Board(fen)
    evaluator = FastEvaluator()
    
    safety = evaluator._evaluate_king_safety_v17_8(board, chess.BLACK)
    assert safety < -60, f"Should detect back-rank weakness: {safety}"
```

### Phase 3: Integration Testing (Week 2)

- Run 50 games with v17.8 vs v17.7
- Monitor time usage per game
- Count time forfeit occurrences
- Verify mate detection accuracy

## Deployment Strategy

### Pre-Deployment Checklist:
1. ✅ Create v17_8_implementation_plan.md (this document)
2. ⏸️ User review and approval
3. ⏸️ Create feature branch `v17.8-time-management`
4. ⏸️ Implement Component 1 (Time Manager)
5. ⏸️ Test Component 1 independently
6. ⏸️ Implement Component 2 (Mate-in-1)
7. ⏸️ Test Component 2 independently
8. ⏸️ Implement Components 3 & 4 (King Safety + Extensions)
9. ⏸️ Full integration testing
10. ⏸️ 50-game validation test
11. ⏸️ User approval for deployment
12. ⏸️ GCP deployment

### Rollback Plan:
- Keep v17.7 backup on GCP
- Can revert in <5 minutes if issues detected
- Monitor first 10 games closely

## Success Criteria

### Must Have:
- ✅ Zero time forfeits in 50-game test
- ✅ 100% mate-in-1 detection (test suite)
- ✅ No performance regression (depth/speed)

### Nice to Have:
- ⭐ +50 ELO from avoiding time losses
- ⭐ 95% mate-in-2 detection
- ⭐ Better king safety evaluation

## Estimated Impact

**Time Forfeit Rate**:
- Current: ~30% of losses
- Target: <5% of losses
- Expected: ~10% of losses (being conservative)

**Rating Improvement**:
- Avoiding time forfeits: +40-60 ELO
- Better mate detection: +10-20 ELO
- Total expected: +50-80 ELO

**Development Time**:
- Component 1: 1-2 hours
- Component 2: 30 minutes
- Component 3: 1 hour
- Component 4: 30 minutes
- Testing: 2-3 hours
- **Total: ~6 hours development + validation time**

---

**Next Steps**: Await user approval to begin implementation.
