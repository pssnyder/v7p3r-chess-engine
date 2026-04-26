# V7P3R Character-Defining Evaluations Analysis

**Phase 0 Baseline Achieved**: 26,000 NPS (1.9x faster than v19.3)

**Goal**: Add back minimal evaluations that give v7p3r its distinctive playing style without killing performance.

---

## What Makes V7P3R Play Like V7P3R?

### Historical Tournament Analysis
From the regression battles and Lichess games, v7p3r is known for:

1. **Aggressive Pawn Pushes** - Advances pawns confidently in middlegame
2. **Bishop Pair Preference** - Keeps bishops, rarely trades them
3. **King Safety Awareness** - Castles early, maintains pawn shield
4. **Passed Pawn Recognition** - Pushes passed pawns aggressively
5. **Center Control** - Fights for central squares early

### Current Phase 0 (Material Only)
✅ Material counting (P=100, N=300, B=325, R=500, Q=900)
✅ Bishop pair bonus (+50cp)
❌ All other positional understanding removed

---

## Candidate Evaluations (Fastest to Slowest)

### Tier 1: Ultra-Fast (Estimated <5% slowdown)

#### 1. **Castling Bonus** ⭐ TOP PRIORITY
**Implementation**: Simple flag check
```python
# Castling bonus (king safety character)
if board.has_castled(chess.WHITE):
    score += 30
if board.has_castled(chess.BLACK):
    score -= 30
```

**Why Add**:
- Virtually free (single board attribute check)
- Gives v7p3r its defensive character
- Encourages early king safety (classic v7p3r style)

**Expected Cost**: <0.001ms per eval
**Character Impact**: HIGH (king safety is core v7p3r trait)

---

#### 2. **Pawn Advancement Bonus** ⭐ TOP PRIORITY
**Implementation**: Count advanced pawns (rank 5+ for white, rank 4- for black)
```python
# Pawn advancement (aggressive character)
for square in board.pieces(chess.PAWN, chess.WHITE):
    rank = chess.square_rank(square)
    if rank >= 4:  # 5th rank or higher
        score += (rank - 3) * 10  # +10/20/30/40/50 for ranks 5/6/7/8
        
for square in board.pieces(chess.PAWN, chess.BLACK):
    rank = chess.square_rank(square)
    if rank <= 3:  # 4th rank or lower (from black's view)
        score -= (4 - rank) * 10
```

**Why Add**:
- Simple rank lookup (O(8) for pawns)
- Gives v7p3r its aggressive pawn play character
- Encourages space advantage (classic v7p3r)

**Expected Cost**: 0.002ms per eval
**Character Impact**: HIGH (pawn aggression is signature v7p3r)

---

#### 3. **Passed Pawn Detection** ⭐ HIGH PRIORITY
**Implementation**: Check if pawn has no opposing pawns ahead
```python
# Passed pawn bonus (endgame awareness)
for square in board.pieces(chess.PAWN, chess.WHITE):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    # Check if passed (no black pawns on file or adjacent files ahead)
    is_passed = True
    for ahead_rank in range(rank + 1, 8):
        for check_file in [file - 1, file, file + 1]:
            if 0 <= check_file < 8:
                check_square = chess.square(check_file, ahead_rank)
                if board.piece_at(check_square) == chess.Piece(chess.PAWN, chess.BLACK):
                    is_passed = False
                    break
    if is_passed:
        score += (rank - 1) * 20  # Exponential: rank 6 = 100cp, rank 7 = 120cp
```

**Why Add**:
- Moderate complexity but high character impact
- Gives v7p3r endgame conversion ability
- Recognizes winning positions (v7p3r strength)

**Expected Cost**: 0.005-0.010ms per eval (worst case 8 pawns)
**Character Impact**: MEDIUM-HIGH (endgame character)

---

### Tier 2: Fast (Estimated 5-15% slowdown)

#### 4. **Center Control (Piece Count)** 
**Implementation**: Count pieces attacking central squares (d4, d5, e4, e5)
```python
# Center control (opening character)
center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
for square in center_squares:
    white_attackers = len(board.attackers(chess.WHITE, square))
    black_attackers = len(board.attackers(chess.BLACK, square))
    score += (white_attackers - black_attackers) * 5
```

**Why Add**:
- Gives v7p3r opening character (fights for center)
- Attackers() is relatively fast (bitboard operation)

**Expected Cost**: 0.003ms per eval
**Character Impact**: MEDIUM (opening phase character)

---

#### 5. **Mobility (Legal Move Count)**
**Implementation**: Count legal moves for each side
```python
# Mobility bonus (active piece play)
white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
# Flip board for black
board.turn = not board.turn
black_moves = len(list(board.legal_moves))
board.turn = not board.turn
score += (white_moves - black_moves) * 2
```

**Why Add**:
- Encourages active piece placement
- Legal move generation already fast in python-chess

**Expected Cost**: 0.010-0.015ms per eval
**Character Impact**: MEDIUM (piece activity)

---

### Tier 3: Moderate Cost (Estimated 15-30% slowdown)

#### 6. **Simplified PSTs (Center Only)**
**Implementation**: Only evaluate pieces on central 16 squares (d3-e6)
```python
# Simplified PST - center squares only
central_squares = [chess.D3, chess.E3, chess.D4, chess.E4, 
                   chess.D5, chess.E5, chess.D6, chess.E6]
for square in central_squares:
    piece = board.piece_at(square)
    if piece:
        if piece.piece_type == chess.KNIGHT:
            score += 30 if piece.color == chess.WHITE else -30
        elif piece.piece_type == chess.BISHOP:
            score += 20 if piece.color == chess.WHITE else -20
```

**Why Add**:
- Encourages centralization (v7p3r style)
- Much cheaper than full PSTs (only 16 squares vs 64)

**Expected Cost**: 0.005ms per eval
**Character Impact**: MEDIUM (piece positioning)

---

### Tier 4: Expensive (DON'T ADD - too slow)

❌ **Full PSTs** - 22.6% of v19.3 time, kills NPS
❌ **King Safety Calculations** - Complex, expensive
❌ **Pawn Structure Analysis** - Too detailed
❌ **Mobility (Full)** - Double move generation expensive

---

## Recommended Incremental Addition Order

### Phase 1: Ultra-Fast Character (Tier 1)
**Add**: Castling bonus + Pawn advancement + Passed pawn detection

**Expected NPS**: 23,000-25,000 (5-10% slowdown from 26K)
**Expected Depth**: 6-7 in 5 seconds
**Character Gain**: HIGH (king safety + aggressive pawns + endgame)

**Test After Phase 1**:
```bash
python testing/quick_v19_1_test.py  # Check NPS ≥ 23K
python testing/tournament_runner.py --games 10  # Check win rate vs Phase 0
```

**Decision**: 
- ✅ Keep if: NPS ≥23K AND win rate ≥55% vs Phase 0
- ❌ Rollback if: NPS <23K OR win rate <50%

---

### Phase 2: Center Control (Tier 2 - first item)
**Add**: Center square control (d4, d5, e4, e5 attacker count)

**Expected NPS**: 21,000-23,000 (8-10% slowdown from Phase 1)
**Expected Depth**: 6 in 5 seconds
**Character Gain**: MEDIUM (opening phase character)

**Test After Phase 2**:
```bash
python testing/quick_v19_1_test.py  # Check NPS ≥ 21K
python testing/tournament_runner.py --games 10  # Check win rate vs Phase 1
```

**Decision**:
- ✅ Keep if: NPS ≥21K AND win rate ≥55% vs Phase 1
- ❌ Rollback if: Fails either test

---

### Phase 3: STOP and Validate
**Don't add more** - instead run full tournament vs v18.4

**Full Validation** (30 games):
```bash
python testing/tournament_runner.py \
  --engine1-cmd "python src/v7p3r_uci.py" \
  --engine2-cmd "python lichess/engines/V7P3R_v18.4_20260417/src/v7p3r_uci.py" \
  --games 30 \
  --time-control "5+4"
```

**Deployment Criteria**:
- ✅ NPS ≥20,000 (blitz-safe)
- ✅ Win rate ≥48% vs v18.4
- ✅ Zero timeouts
- ✅ Depth ≥6 in 5 seconds

If Phase 2 fails, deploy Phase 1.
If Phase 1 fails, deploy Phase 0 (material-only is competitive via depth).

---

## Implementation Code Snippets

### Phase 1 Addition (to _evaluate_position)
```python
# === PHASE 1: CHARACTER-DEFINING EVALS ===

# 1. Castling bonus (king safety character)
if board.has_castled(chess.WHITE):
    score += 30
if board.has_castled(chess.BLACK):
    score -= 30

# 2. Pawn advancement bonus (aggressive character)
for square in board.pieces(chess.PAWN, chess.WHITE):
    rank = chess.square_rank(square)
    if rank >= 4:  # 5th rank or higher
        score += (rank - 3) * 10
        
for square in board.pieces(chess.PAWN, chess.BLACK):
    rank = chess.square_rank(square)
    if rank <= 3:  # 4th rank or lower
        score -= (4 - rank) * 10

# 3. Passed pawn detection (endgame character)
for square in board.pieces(chess.PAWN, chess.WHITE):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    is_passed = True
    for ahead_rank in range(rank + 1, 8):
        for check_file in [max(0, file-1), file, min(7, file+1)]:
            check_square = chess.square(check_file, ahead_rank)
            if board.piece_at(check_square) == chess.Piece(chess.PAWN, chess.BLACK):
                is_passed = False
                break
        if not is_passed:
            break
    if is_passed and rank >= 4:  # Only count advanced passed pawns
        score += (rank - 3) * 20  # 20/40/60/80 for ranks 5/6/7/8
        
# Similar for Black passed pawns (reverse logic)
```

### Phase 2 Addition
```python
# === PHASE 2: CENTER CONTROL ===
center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
for square in center_squares:
    white_attackers = len(board.attackers(chess.WHITE, square))
    black_attackers = len(board.attackers(chess.BLACK, square))
    score += (white_attackers - black_attackers) * 5
```

---

## Expected Final Performance

### Best Case (Phase 2 succeeds)
- **NPS**: 21,000-23,000
- **Depth**: 6-7 in 5 seconds
- **Character**: Aggressive pawns + king safety + center control + endgame
- **Strength**: Competitive via depth + key positional understanding

### Likely Case (Phase 1 succeeds, Phase 2 fails)
- **NPS**: 23,000-25,000
- **Depth**: 6-7 in 5 seconds
- **Character**: Aggressive pawns + king safety + endgame
- **Strength**: Strong via depth + core v7p3r traits

### Worst Case (Phase 0 only)
- **NPS**: 26,000
- **Depth**: 7-8 in 5 seconds
- **Character**: Pure material (like MaterialOpponent)
- **Strength**: Competitive via depth alone (MaterialOpponent proved this)

---

## Next Steps

1. **Implement Phase 1** (castling + pawns + passed pawns)
2. **Test performance** (expect NPS ≥23K)
3. **Quick tournament** (10 games vs Phase 0, expect ≥55% win rate)
4. **If pass**: Implement Phase 2 (center control)
5. **If pass**: Full tournament vs v18.4 (30 games)
6. **Deploy winner** to production

Ready to implement Phase 1?
