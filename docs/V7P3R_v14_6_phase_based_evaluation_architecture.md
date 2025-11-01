# Phase-Based Dynamic Evaluation Architecture
**Version:** V14.6 Proposal  
**Date:** October 29, 2025  
**Status:** Pending User Approval

## Overview
Implement game-phase-aware evaluation where heuristics dynamically opt in/out based on opening/middlegame/endgame phase. This improves both performance (higher NPS) and playing strength (phase-appropriate strategy).

## Core Principles

### 1. Blunder Firewall - ALWAYS ACTIVE
The following safety checks run in ALL phases (non-negotiable):
- Hanging piece detection (pieces attacked but not defended)
- Tactical loss detection (forced material loss)
- King safety threats (check, mate threats, exposed king)
- Move safety analysis (`evaluate_move_safety_bitboard`)

### 2. Phase-Specific Strategy Heuristics
Different evaluation terms activate based on game phase:

#### Opening Phase (Moves 0-12 OR Material >= 30)
**Focus:** Development, Center Control, King Safety
- ✅ Piece development bonuses (knights/bishops off back rank)
- ✅ Center control (e4, d4, e5, d5 squares)
- ✅ Castling incentives (king safety)
- ✅ Rapid development penalties (moving same piece twice)
- ❌ SKIP: Complex pawn structure analysis (save time)
- ❌ SKIP: King centralization (not relevant yet)
- ❌ SKIP: Advanced endgame patterns

#### Middlegame Phase (Moves 13-40 AND Material 16-30)
**Focus:** Tactics, Attacks, Captures, Exchanges
- ✅ Full tactical analysis (pins, forks, skewers)
- ✅ Attack/defense balance
- ✅ Pawn breaks and pawn structure
- ✅ Piece coordination
- ✅ King safety (still important)
- ❌ SKIP: Development bonuses (already developed)
- ❌ SKIP: Endgame king patterns

#### Early Endgame (Material 8-15)
**Focus:** King Activation, Piece Coordination, Passed Pawns
- ✅ King centralization (moving toward center)
- ✅ King near own pieces (protecting them)
- ✅ Passed pawn evaluation
- ✅ Rook activity (7th rank, open files)
- ✅ Pawn structure critical
- ❌ SKIP: King shelter/castling incentives
- ❌ SKIP: Development bonuses
- ⚠️ REDUCED: Tactical complexity (fewer pieces = fewer tactics)

#### Late Endgame (Material < 8 OR Pieces < 10)
**Focus:** Opposition, Edge Restriction, Zugzwang
- ✅ King opposition patterns
- ✅ King activity maximized
- ✅ Edge restriction (forcing opponent king to edges)
- ✅ Passed pawn promotion paths
- ✅ Checkmate patterns (KQ vs K, KR vs K, etc.)
- ❌ SKIP: Most middlegame tactics
- ❌ SKIP: Complex pawn structure

## Technical Implementation

### Phase Detection Function
```python
def detect_game_phase(self, board: chess.Board) -> str:
    """
    Deterministic phase detection - same result for both sides
    Returns: 'opening', 'middlegame', 'early_endgame', 'late_endgame'
    """
    moves_played = len(board.move_stack)
    
    # Calculate total material (excluding kings)
    total_material = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            total_material += PIECE_VALUES.get(piece.piece_type, 0)
    
    # Count total pieces
    total_pieces = len(board.piece_map())
    
    # Phase detection logic
    if moves_played < 12 or total_material >= 30:
        return 'opening'
    elif total_material < 8 or total_pieces <= 10:
        return 'late_endgame'
    elif total_material < 16:
        return 'early_endgame'
    else:
        return 'middlegame'
```

### Modified evaluate_position_complete
```python
def evaluate_position_complete(self, board: chess.Board, evaluation_cache: dict = {}) -> float:
    """
    V14.6: Phase-aware evaluation with dynamic heuristics
    """
    # Detect phase ONCE per position
    phase = self.detect_game_phase(board)
    
    # Cache check...
    
    # ALWAYS: Base material and blunder firewall
    white_base = self.evaluate_bitboard(board, chess.WHITE)
    black_base = self.evaluate_bitboard(board, chess.BLACK)
    
    # ALWAYS: Safety analysis (blunder firewall)
    safety_data = self.analyze_safety_bitboard(board)
    white_safety = safety_data.get('white_safety_bonus', 0)
    black_safety = safety_data.get('black_safety_bonus', 0)
    
    # PHASE-DEPENDENT: Strategic evaluations
    if phase == 'opening':
        white_strategic = self._evaluate_opening_strategy(board, chess.WHITE)
        black_strategic = self._evaluate_opening_strategy(board, chess.BLACK)
        
    elif phase == 'middlegame':
        white_strategic = self._evaluate_middlegame_strategy(board, chess.WHITE)
        black_strategic = self._evaluate_middlegame_strategy(board, chess.BLACK)
        
    elif phase == 'early_endgame':
        white_strategic = self._evaluate_early_endgame_strategy(board, chess.WHITE)
        black_strategic = self._evaluate_early_endgame_strategy(board, chess.BLACK)
        
    else:  # late_endgame
        white_strategic = self._evaluate_late_endgame_strategy(board, chess.WHITE)
        black_strategic = self._evaluate_late_endgame_strategy(board, chess.BLACK)
    
    # Combine: Base + Safety (always) + Strategic (phase-dependent)
    white_total = white_base + white_safety + white_strategic
    black_total = black_base + black_safety + black_strategic
    
    return white_total - black_total
```

### New Phase-Specific Evaluation Methods

#### Opening Strategy
```python
def _evaluate_opening_strategy(self, board: chess.Board, color: chess.Color) -> float:
    """Opening: Development, Center, King Safety"""
    score = 0.0
    
    # Development: Minor pieces off back rank
    score += self._evaluate_piece_development(board, color)
    
    # Center control: e4, d4, e5, d5
    score += self._evaluate_center_control_opening(board, color)
    
    # Castling incentive
    if board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color):
        score += 20.0  # Bonus for maintaining castling rights
    
    # King safety (castled position)
    score += self.evaluate_king_safety(board, color) * 1.2  # Higher weight in opening
    
    return score
```

#### Middlegame Strategy
```python
def _evaluate_middlegame_strategy(self, board: chess.Board, color: chess.Color) -> float:
    """Middlegame: Tactics, Attacks, Structure"""
    score = 0.0
    
    # Full tactical analysis
    tactical_data = self.analyze_position_for_tactics_bitboard(board)
    score += tactical_data.get('white_tactical_bonus' if color == chess.WHITE else 'black_tactical_bonus', 0)
    
    # Pawn structure
    score += self.evaluate_pawn_structure(board, color)
    
    # Pin detection
    pin_data = self.detect_pins_bitboard(board)
    score += pin_data['pin_score_white' if color == chess.WHITE else 'pin_score_black']
    
    # King safety (still important but slightly reduced)
    score += self.evaluate_king_safety(board, color) * 1.0
    
    return score
```

#### Early Endgame Strategy
```python
def _evaluate_early_endgame_strategy(self, board: chess.Board, color: chess.Color) -> float:
    """Early Endgame: King Activation, Passed Pawns"""
    score = 0.0
    
    # King centralization
    king_square = board.king(color)
    if king_square:
        # Distance from center (e4/d4/e5/d5 average)
        center_distance = min(
            chess.square_distance(king_square, chess.E4),
            chess.square_distance(king_square, chess.D4),
            chess.square_distance(king_square, chess.E5),
            chess.square_distance(king_square, chess.D5)
        )
        score += (7 - center_distance) * 5.0  # Closer to center = better
    
    # Simplified pawn structure (focus on passed pawns)
    score += self.evaluate_pawn_structure(board, color) * 1.5  # Higher weight
    
    # Reduced tactical complexity (fewer pieces)
    tactical_data = self.analyze_position_for_tactics_bitboard(board)
    score += tactical_data.get('white_tactical_bonus' if color == chess.WHITE else 'black_tactical_bonus', 0) * 0.5
    
    return score
```

#### Late Endgame Strategy
```python
def _evaluate_late_endgame_strategy(self, board: chess.Board, color: chess.Color) -> float:
    """Late Endgame: Opposition, Edge Restriction, Checkmate"""
    score = 0.0
    
    king_square = board.king(color)
    opponent_king_square = board.king(not color)
    
    if not king_square or not opponent_king_square:
        return score
    
    # King activity (maximum centralization)
    center_distance = min(
        chess.square_distance(king_square, chess.E4),
        chess.square_distance(king_square, chess.D4),
        chess.square_distance(king_square, chess.E5),
        chess.square_distance(king_square, chess.D5)
    )
    score += (7 - center_distance) * 10.0  # Even higher weight than early endgame
    
    # Edge restriction (push opponent king to edges)
    opponent_edge_distance = min(
        chess.square_file(opponent_king_square),
        7 - chess.square_file(opponent_king_square),
        chess.square_rank(opponent_king_square),
        7 - chess.square_rank(opponent_king_square)
    )
    score -= opponent_edge_distance * 15.0  # Penalty if opponent NOT on edge
    
    # King opposition
    file_distance = abs(chess.square_file(king_square) - chess.square_file(opponent_king_square))
    rank_distance = abs(chess.square_rank(king_square) - chess.square_rank(opponent_king_square))
    
    # Reward being 2 squares away (opposition)
    if file_distance == 0 and rank_distance == 2:
        score += 25.0  # Vertical opposition
    elif file_distance == 2 and rank_distance == 0:
        score += 25.0  # Horizontal opposition
    elif file_distance == 2 and rank_distance == 2:
        score += 20.0  # Diagonal opposition
    
    # Passed pawn critical
    score += self.evaluate_pawn_structure(board, color) * 2.0
    
    return score
```

## Performance Benefits

### Expected NPS Improvements
- **Opening:** +15-25% (skip complex pawn analysis, skip endgame patterns)
- **Middlegame:** +5-10% (skip development checks, skip endgame patterns)
- **Endgame:** +20-35% (skip tactical complexity, simpler evaluations)

### Computation Reduction Per Phase
| Evaluation Component | Opening | Middlegame | Early Endgame | Late Endgame |
|---------------------|---------|------------|---------------|--------------|
| Base Material | ✅ Always | ✅ Always | ✅ Always | ✅ Always |
| Blunder Firewall | ✅ Always | ✅ Always | ✅ Always | ✅ Always |
| Development | ✅ Full | ❌ Skip | ❌ Skip | ❌ Skip |
| Center Control | ✅ High Weight | ✅ Standard | ⚠️ Reduced | ❌ Skip |
| Pawn Structure | ⚠️ Basic | ✅ Full | ✅ High Weight | ✅ Critical |
| Tactical Analysis | ⚠️ Light | ✅ Full | ⚠️ Reduced | ⚠️ Minimal |
| King Safety | ✅ High Weight | ✅ Standard | ⚠️ Reduced | ❌ Skip |
| King Centralization | ❌ Skip | ❌ Skip | ✅ Important | ✅ Critical |
| Edge Restriction | ❌ Skip | ❌ Skip | ❌ Skip | ✅ Critical |
| Opposition Patterns | ❌ Skip | ❌ Skip | ⚠️ Basic | ✅ Full |

## Implementation Phases

### Phase 1: Infrastructure (Low Risk)
1. Add `detect_game_phase()` method to bitboard evaluator
2. Add phase logging to verify correct detection
3. Test phase detection across various positions
4. **NO EVALUATION CHANGES YET**

### Phase 2: Refactor Existing Code (Medium Risk)
1. Extract current evaluation logic into phase-specific methods:
   - `_evaluate_opening_strategy()`
   - `_evaluate_middlegame_strategy()`
   - `_evaluate_early_endgame_strategy()`
   - `_evaluate_late_endgame_strategy()`
2. Ensure evaluations produce same results as current (just refactored)
3. Test against V14.5 to confirm no regression

### Phase 3: Optimization (Medium Risk)
1. Remove unnecessary computations per phase
2. Add phase-specific bonuses/penalties
3. Tune weights per phase
4. Performance testing (NPS measurement)

### Phase 4: Validation (High Impact)
1. Self-play: V14.6 vs V14.5 (100 games minimum)
2. Historical opponent test: V14.6 vs V10.8
3. Lichess deployment test (time-controlled games)
4. Rating and blunder-rate monitoring

## Risk Mitigation

### Determinism Guarantee
- Phase detection uses ONLY board state (move count, material, piece count)
- No side-specific logic in phase detection
- Both sides see same phase for same position
- Caching works correctly (phase is position-dependent)

### Blunder Firewall Guarantee
- Safety checks NEVER disabled
- `analyze_safety_bitboard()` always runs
- `evaluate_move_safety_bitboard()` always available for move ordering
- King safety always evaluated (weight varies by phase)

### Rollback Plan
- Create branch: `feature/phase-based-eval-v14.6`
- Create engine_freeze before implementation
- Each phase has its own commit for easy rollback
- Keep V14.5 available for comparison

## Testing Strategy

### Unit Tests
- Phase detection correctness (known positions)
- Symmetry test (white/black see same phase)
- Evaluation consistency (refactored code = same results)

### Performance Tests
- NPS comparison: V14.5 vs V14.6 per phase
- Search depth achievement at fixed time
- Memory usage (should be negligible increase)

### Game Tests
- Quick games (1min+1sec) - 50 games
- Standard games (5min+3sec) - 50 games
- Long games (15min+10sec) - 20 games
- Self-play and vs V10.8

## Questions for User

1. **Phase Boundaries:** Are the proposed thresholds correct?
   - Opening: < 12 moves OR material >= 30
   - Middlegame: 13-40 moves AND material 16-30
   - Early Endgame: Material 8-15
   - Late Endgame: Material < 8 OR pieces < 10

2. **Endgame King Tables:** Do we need to create bitboard tables for:
   - King centralization scores (per square)
   - Edge restriction patterns
   - Opposition detection patterns

3. **Performance Priority:** Which phase is most critical for NPS?
   - Opening (every game starts here)
   - Middlegame (most nodes searched)
   - Endgame (typically faster already)

4. **Branch/Backup Strategy:** Should we:
   - Create feature branch?
   - Run engine_freeze backup?
   - Both?

5. **Implementation Pace:** Prefer:
   - All-at-once implementation with comprehensive testing after?
   - Incremental with testing between each phase?

## Expected Outcomes

### Performance
- Opening NPS: 2500 → 3000+ (+20%)
- Middlegame NPS: 2300 → 2500+ (+8%)
- Endgame NPS: 3000 → 4000+ (+33%)

### Playing Strength
- Better opening play (development-focused)
- More tactical middlegame (full analysis when it matters)
- Cleaner endgames (phase-appropriate technique)
- Maintained blunder protection (firewall always active)

### Code Quality
- Clearer separation of concerns
- Easier to tune phase-specific weights
- Better performance profiling per phase
- More maintainable evaluation logic

---

**Ready for user approval before implementation.**
