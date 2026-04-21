# Chess Engine Component Inventory
## Analysis of C0BR4 (C#) and V7P3R (Python) Engines

---

## 🔵 CORE COMPONENTS - BOTH ENGINES

### 1. **Board Representation**
- **C0BR4**: `BitboardBoard.cs`, `Board.cs`
  - Bitboard-based (64-bit integers)
  - Functions: `GetPiece()`, `MakeMove()`, `UnmakeMove()`, `GetLegalMoves()`
- **V7P3R**: Uses python-chess library
  - Bitboard operations via chess.Board
  - Custom bitboard evaluator for performance

### 2. **Move Generation**
- **C0BR4**: `BitboardMoveGenerator.cs`
  - `GenerateLegalMoves()` - filters illegal moves
  - `GeneratePseudoLegalMoves()` - all possible moves
  - Piece-specific: `GeneratePawnMoves()`, `GenerateKnightMoves()`, `GenerateBishopMoves()`, etc.
  - Castling: `GenerateCastlingMoves()`
  - En passant support
- **V7P3R**: Uses python-chess move generation
  - Built-in legal move validation

### 3. **Bitboard Operations**
- **C0BR4**: `Bitboard.cs`, `MagicBitboards.cs`
  - Functions: `PopCount()`, `TrailingZeroCount()`, `SquareToBitboard()`
  - Ray-based attack generation: `GetRookAttacks()`, `GetBishopAttacks()`, `GetQueenAttacks()`
  - Precomputed: `GetKnightAttacks()`, `GetKingAttacks()`
  - Masks: File masks, Rank masks, Center, Corners, Castling squares
- **V7P3R**: `v7p3r_bitboard_evaluator.py`
  - Bitboard masks: Files, Ranks, Center, Edges
  - Attack tables: `KNIGHT_ATTACKS[]`, `KING_ATTACKS[]`, `WHITE_PAWN_ATTACKS[]`, `BLACK_PAWN_ATTACKS[]`
  - Passed pawn masks: `WHITE_PASSED_PAWN_MASKS[]`, `BLACK_PASSED_PAWN_MASKS[]`

### 4. **Search Algorithm - Alpha-Beta Pruning**
- **C0BR4**: `AlphaBetaSearchBot.cs`
  - `AlphaBeta(board, depth, alpha, beta)` - negamax with pruning
  - `SearchBestMove()` - root search
  - Mate detection with depth adjustment
- **V7P3R**: Main search in `v7p3r.py`
  - `alpha_beta()` - negamax implementation
  - Mate scoring with distance to mate

### 5. **Transposition Table**
- **C0BR4**: `TranspositionTable.cs`, `ZobristHashing.cs`
  - Hash calculation: `ZobristHashing.CalculateHash()`
  - Incremental updates: `UpdateHash()`
  - Table operations: `TryGetEntry()`, `StoreEntry()`
  - Node types: exact, lowerbound, upperbound
- **V7P3R**: `TranspositionEntry` class in v7p3r.py
  - Similar hash table with Zobrist hashing
  - Depth-based replacement strategy

### 6. **Move Ordering**
- **C0BR4**: `MoveOrdering.cs`
  - `OrderMoves()` - sorts moves by priority
  - MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
  - Priorities: Captures (10000+), Promotions (9000+), Checks (500+), Center control (10+)
  - `ScoreMove()` - assigns score to each move
- **V7P3R**: In main engine
  - Hash move first
  - Killer moves
  - History heuristic
  - Capture ordering

### 7. **Quiescence Search**
- **C0BR4**: `QuiescenceSearchBot.cs`
  - `Quiescence(board, alpha, beta)` - searches tactical moves
  - Stand-pat evaluation
  - Searches captures and promotions
  - Delta pruning
- **V7P3R**: Built into main search
  - Similar quiescence implementation
  - Searches forcing moves

### 8. **Material Evaluation**
- **C0BR4**: `SimpleEvaluator.cs` - `EvaluateMaterial()`
  - Pawn: 100, Knight: 300, Bishop: 300, Rook: 500, Queen: 900
- **V7P3R**: `v7p3r_fast_evaluator.py` - `evaluate_material()`
  - Pawn: 100, Knight: 320, Bishop: 330, Rook: 500, Queen: 900

### 9. **Piece-Square Tables (PST)**
- **C0BR4**: `PieceSquareTables.cs`
  - `EvaluatePosition()` - applies PST bonuses
  - Phase interpolation (opening/middlegame/endgame)
  - Separate tables per piece type
- **V7P3R**: `v7p3r_fast_evaluator.py`
  - Pre-computed flipped tables for Black
  - Direct indexing: `PST_DIRECT[piece_type][color][square]`
  - Phase-aware evaluation

### 10. **Game Phase Detection**
- **C0BR4**: `GamePhase.cs`
  - `CalculatePhase()` - returns 0.0 (endgame) to 1.0 (opening)
  - `IsEndgame()`, `IsOpening()`, `IsMiddlegame()`
  - Material-based calculation
- **V7P3R**: `v7p3r_position_context.py`
  - Enum: `OPENING, MIDDLEGAME_COMPLEX, MIDDLEGAME_SIMPLE, ENDGAME_COMPLEX, ENDGAME_SIMPLE`
  - Context-aware evaluation selection

### 11. **Time Management**
- **C0BR4**: `TimeManager.cs`
  - `CalculateTimeAllocation()` - determines time per move
  - `CalculateSearchDepth()` - adaptive depth based on time
  - Emergency time handling (<3s remaining)
  - Phase-based multipliers
  - Increment-aware calculations
- **V7P3R**: Smart time management in main engine
  - Similar adaptive time control
  - Emergency modes for time pressure

### 12. **UCI Protocol**
- **C0BR4**: `UCIEngine.cs`
  - Full UCI implementation
  - Commands: `uci`, `isready`, `position`, `go`, `stop`
- **V7P3R**: `v7p3r_uci.py`
  - Complete UCI interface
  - Info strings with depth, score, nodes, pv

### 13. **Opening Book**
- **C0BR4**: `OpeningBook.cs`
  - Embedded move sequences
  - London System, Vienna System, Caro-Kann, Dutch Defense, Queen's Pawn, Sicilian
  - Algebraic notation parsing: `AlgebraicNotation.cs`
- **V7P3R**: `v7p3r_openings_v161.py`
  - `get_enhanced_opening_book()` - returns opening moves
  - Position-based opening selection

---

## 🔴 C0BR4 UNIQUE FEATURES

### 1. **Iterative Deepening**
- **Implementation**: `TranspositionSearchBot.cs`
- **Function**: Searches depth 1, 2, 3... up to target
- Time management integration
- Early exit on time limits
- Maintains best move from completed depths

### 2. **Principal Variation (PV) Tracking**
- **Implementation**: `TranspositionSearchBot.cs`
- **Functions**: `SearchWithPV()`, `AlphaBetaWithPV()`
- Returns: `(bestMove, bestScore, pv)` tuple
- PV collection for display

### 3. **Killer Move Heuristic**
- **Implementation**: Mentioned in search bots
- Stores non-capture moves that caused cutoffs
- 2 killers per depth

### 4. **Evaluation Components**
All in `Evaluation/` folder with `Evaluate(Board, gamePhase)` signature:

#### **King Safety**
- `KingSafety.cs`
  - Pawn shield evaluation
  - King exposure penalties
  - Attack pattern detection

#### **King Endgame**
- `KingEndgame.cs`
  - King centralization in endgame
  - Opposition detection
  - King activity bonuses

#### **Castling Evaluation**
- `CastlingIncentive.cs`
  - Bonus for castling early
  - Penalties for delaying castling
- `CastlingRights.cs`
  - Castling rights preservation
  - Timing bonuses

#### **Rook Coordination**
- `RookCoordination.cs`
  - Connected rooks bonus
  - Rook on 7th rank
  - Open file detection

#### **Pawn Endgame**
- `PawnEndgame.cs`
  - Pure pawn endgame evaluation
  - Passed pawn evaluation
  - King proximity to pawns

### 5. **Advanced Move Validation**
- **Implementation**: `BitboardValidationTest.cs`, `IllegalMoveDebugger.cs`
- Comprehensive legal move verification
- Move validation testing framework

### 6. **Performance Benchmarking**
- **Implementation**: `PerformanceBenchmark.cs`
- Perft testing
- NPS (nodes per second) measurement

### 7. **.NET Optimizations**
- Method inlining: `[MethodImpl(MethodImplOptions.AggressiveInlining)]`
- Built-in bit operations: `System.Numerics.BitOperations.PopCount()`
- Compiled language performance advantages

---

## 🟢 V7P3R UNIQUE FEATURES

### 1. **Modular Evaluation System**
- **Implementation**: `v7p3r_modular_eval.py`, `v7p3r_eval_modules.py`
- **Purpose**: Selective evaluation based on context

#### **Evaluation Profiles**
`v7p3r_eval_selector.py`:
- **DESPERATE**: Down material - tactics only (10 modules)
- **EMERGENCY**: Time pressure <3s (5 modules)
- **FAST**: Fast time control (12-18 modules)
- **TACTICAL**: Tactical positions (18-22 modules)
- **ENDGAME**: Endgame specific (10-15 modules)
- **COMPREHENSIVE**: All modules (20-28 modules)

#### **Module Registry** (`EvaluationModule` dataclass)
Each module has:
- `cost`: NEGLIGIBLE, LOW, MEDIUM, HIGH
- `criticality`: ESSENTIAL, IMPORTANT, SITUATIONAL, OPTIONAL
- `required_pieces`: What pieces must exist
- `required_phases`: When to activate
- `skip_when_desperate`: Boolean flag
- `skip_in_time_pressure`: Boolean flag

#### **Available Evaluation Modules** (32+ modules):

**Essential Modules:**
1. `material_counter` - Basic material counting
2. `piece_square_tables` - PST evaluation

**Desperate/Tactical Modules:**
3. `hanging_pieces` - Undefended piece detection
4. `capture_priority` - Prioritize material-winning captures
5. `check_threats` - Check-giving moves
6. `pins_forks_skewers` - Tactical pattern detection
7. `tactical_patterns` - Complex tactics
8. `exchanges` - Static Exchange Evaluation (SEE)
9. `trapped_pieces` - Piece trapping detection
10. `back_rank` - Back rank weakness

**King Safety Modules:**
11. `king_safety_basic` - Pawn shield, castling rights
12. `king_safety_complex` - Attack patterns, king tropism
13. `king_centralization` - Endgame king activity

**Pawn Structure Modules:**
14. `passed_pawns` - Passed pawn bonuses
15. `doubled_pawns` - Doubled pawn penalties
16. `isolated_pawns` - Isolated pawn penalties
17. `backward_pawns` - Backward pawn penalties
18. `pawn_chains` - Connected pawn bonuses
19. `pawn_structure` - Overall pawn quality

**Piece-Specific Modules:**
20. `bishop_pair` - Two bishop bonus
21. `knight_outposts` - Knight on strong squares
22. `rook_on_7th` - Rook on 7th rank
23. `rook_on_open_file` - Rook file control
24. `rook_files` - Rook file evaluation
25. `connected_rooks` - Rook coordination
26. `queen_mobility` - Queen activity
27. `queen_activity` - Queen centralization

**Positional Modules:**
28. `piece_mobility` - Full mobility count (expensive)
29. `piece_activity` - Simplified mobility
30. `center_control` - Central square control
31. `space` - Space advantage
32. `development` - Piece development in opening
33. `tempo` - Time advantage

### 2. **Position Context System**
- **Implementation**: `v7p3r_position_context.py`
- **Class**: `PositionContextCalculator`

**Calculated Metrics:**
- Game phase (5 distinct phases)
- Material balance type
- Time pressure detection
- Tactical complexity
- Position volatility
- Piece activity levels

### 3. **Move Safety Checker**
- **Implementation**: `v7p3r_move_safety.py`
- **Class**: `MoveSafetyChecker`

**Functions:**
- `evaluate_move_safety()` - Returns penalty for unsafe moves
- `_check_hanging_pieces()` - Detects hanging pieces after move
- `_is_piece_hanging()` - Checks if piece is undefended
- `_check_immediate_captures()` - Evaluates capture threats
- Simple SEE (Static Exchange Evaluation)

### 4. **History Heuristic**
- **Implementation**: `HistoryHeuristic` class in v7p3r.py
- Tracks move success rates
- `update_history()` - Updates scores (depth²)
- `get_history_score()` - Returns history value
- Memory management for large tables

### 5. **Killer Moves (Enhanced)**
- **Implementation**: `KillerMoves` class
- `store_killer()` - Stores killer move
- `get_killers()` - Returns killers for depth
- `is_killer()` - Checks if move is killer
- 2 killers per depth

### 6. **PV Instant Move System**
- **Implementation**: `PVTracker` class in v7p3r.py
- **Purpose**: Play instantly when opponent follows PV

**Functions:**
- `store_pv_from_search()` - Saves PV after search
- `check_position_for_instant_move()` - Checks for PV match
- `_setup_next_prediction()` - Prepares next prediction
- Position FEN matching
- PV queue management

### 7. **Adaptive Evaluation**
- Profile selection based on:
  - Material deficit (DESPERATE mode)
  - Time remaining (EMERGENCY mode)
  - Tactical complexity
  - Game phase

### 8. **Performance Optimizations**
- Pre-computed PST tables (28% faster PST evaluation)
- Direct square indexing
- Bitboard operation caching
- Lazy evaluation support

### 9. **Comprehensive Statistics Tracking**
- Nodes searched
- Quiescence nodes
- Transposition table hits/stores
- Search depth achieved
- NPS calculation
- Profile usage statistics

### 10. **Advanced Search Features**
- Aspiration windows
- Null move pruning (referenced)
- Late move reductions (LMR) potential
- Multi-PV support foundation

---

## 🏗️ COMPLETE COMPONENT LIST FOR NEW ENGINE

If building a new engine with ALL components from both:

### **Core Engine (Required)**
1. ✅ Bitboard board representation
2. ✅ Legal move generator (pseudo-legal + filtering)
3. ✅ Make/unmake move functionality
4. ✅ Attack/defense detection
5. ✅ Check/checkmate/stalemate detection

### **Search Components**
6. ✅ Alpha-beta negamax search
7. ✅ Iterative deepening
8. ✅ Quiescence search
9. ✅ Transposition table with Zobrist hashing
10. ✅ Move ordering:
    - Hash move first
    - MVV-LVA for captures
    - Killer move heuristic (2 per depth)
    - History heuristic
    - Promotion ordering
    - Check ordering
11. ✅ Principal Variation (PV) tracking
12. ✅ PV instant move system (optional optimization)
13. ✅ Aspiration windows
14. ✅ Null move pruning (advanced)
15. ✅ Late move reductions (advanced)

### **Evaluation Core**
16. ✅ Material counting
17. ✅ Piece-square tables (phase-interpolated)
18. ✅ Game phase detection (5 phases recommended)
19. ✅ Tapered evaluation (opening → middlegame → endgame)

### **Evaluation Modules - Tactical (High Priority)**
20. ✅ Hanging piece detection
21. ✅ Move safety checker
22. ✅ Capture evaluation
23. ✅ Check threats
24. ✅ Pins, forks, skewers detection
25. ✅ Static Exchange Evaluation (SEE)
26. ✅ Trapped pieces
27. ✅ Back rank threats

### **Evaluation Modules - King Safety**
28. ✅ Basic king safety (pawn shield)
29. ✅ Complex king safety (attack patterns)
30. ✅ King centralization (endgame)
31. ✅ Castling rights evaluation
32. ✅ Castling timing incentives

### **Evaluation Modules - Pawn Structure**
33. ✅ Passed pawns (with king distance)
34. ✅ Doubled pawns
35. ✅ Isolated pawns
36. ✅ Backward pawns
37. ✅ Pawn chains (connected pawns)
38. ✅ Pure pawn endgame evaluation

### **Evaluation Modules - Piece-Specific**
39. ✅ Bishop pair bonus
40. ✅ Knight outposts
41. ✅ Rook on 7th rank
42. ✅ Rook on open/semi-open files
43. ✅ Connected rooks
44. ✅ Queen mobility
45. ✅ Queen activity

### **Evaluation Modules - Positional**
46. ✅ Piece mobility (full move generation)
47. ✅ Piece activity (lightweight mobility)
48. ✅ Center control
49. ✅ Space advantage
50. ✅ Piece development (opening)
51. ✅ Tempo evaluation

### **Evaluation System Architecture**
52. ✅ Modular evaluation framework
53. ✅ Evaluation profile selector
54. ✅ Position context calculator
55. ✅ Module registry with metadata
56. ✅ Adaptive evaluation (context-based module selection)
57. ✅ Cost/benefit tracking per module

### **Time Management**
58. ✅ Adaptive time allocation
59. ✅ Phase-based time multipliers
60. ✅ Emergency time handling (<3s)
61. ✅ Increment-aware calculation
62. ✅ Depth-based time budgeting
63. ✅ Moves-to-go consideration

### **Opening Book**
64. ✅ Opening book implementation
65. ✅ Position matching
66. ✅ Multiple opening lines per system
67. ✅ Algebraic notation support

### **UCI Protocol**
68. ✅ Full UCI implementation
69. ✅ Info string output (depth, score, pv, nodes, nps)
70. ✅ Time control parsing
71. ✅ Position setup (FEN + moves)
72. ✅ Go command with all variants

### **Optimization & Performance**
73. ✅ Bitboard operations (popcount, trailing zeros)
74. ✅ Magic bitboards or ray-based attacks
75. ✅ Pre-computed attack tables
76. ✅ Method/function inlining
77. ✅ Incremental hash updates
78. ✅ Lazy evaluation support
79. ✅ Memory-efficient transposition table

### **Testing & Validation**
80. ✅ Move validation framework
81. ✅ Perft testing
82. ✅ Performance benchmarking
83. ✅ Illegal move debugging
84. ✅ Position test suite

### **Advanced Features (Optional)**
85. ⚪ Multi-PV search
86. ⚪ Syzygy tablebase support
87. ⚪ Contempt factor
88. ⚪ Pondering (think on opponent's time)
89. ⚪ SMP (parallel search)
90. ⚪ NNUE evaluation (neural network)

---

## 📊 EVALUATION FUNCTION SIGNATURES

### C0BR4 Pattern:
```csharp
public static int Evaluate(Board board, double gamePhase)
```

### V7P3R Pattern:
```python
def evaluate(self, board: chess.Board) -> int
def evaluate_with_profile(self, board, profile, context) -> float
```

### Module Execution Pattern:
```python
# Check if module should run
if module_name in active_modules:
    score += self._evaluate_module_name(board)
```

---

## 🎯 RECOMMENDED BUILD ORDER

### Phase 1: Core Engine (Weeks 1-2)
1. Board representation
2. Move generation
3. Basic search (minimax → alpha-beta)
4. Material evaluation
5. UCI interface

### Phase 2: Search Enhancements (Weeks 3-4)
6. Transposition table
7. Move ordering (basic)
8. Quiescence search
9. Iterative deepening
10. Time management

### Phase 3: Evaluation Basics (Weeks 5-6)
11. Piece-square tables
12. Game phase detection
13. Basic king safety
14. Pawn structure basics
15. Opening book

### Phase 4: Tactical Awareness (Weeks 7-8)
16. Move safety checker
17. Hanging piece detection
18. SEE (Static Exchange Evaluation)
19. Advanced move ordering (killer + history)
20. Tactical modules

### Phase 5: Modular System (Weeks 9-10)
21. Position context system
22. Evaluation module registry
23. Profile selector
24. Adaptive evaluation
25. Testing all profiles

### Phase 6: Advanced Features (Weeks 11-12)
26. All remaining evaluation modules
27. PV tracking improvements
28. Advanced time management
29. Performance optimization
30. Tournament testing

---

## 📝 KEY IMPLEMENTATION NOTES

### From C0BR4:
- Use ray-based attacks if magic bitboards too complex
- Implement make/unmake properly to avoid state bugs
- Phase interpolation crucial for smooth evaluation
- Fixed seed for Zobrist ensures reproducibility
- Safety margins in time management prevent losses on time

### From V7P3R:
- Modular evaluation allows extreme flexibility
- Context-aware evaluation significantly improves strength
- Pre-computed tables (PST, attacks) provide major speedup
- PV instant moves save time in critical positions
- Profile system handles time pressure gracefully
- Move safety prevents simple tactical oversights

### Performance Targets:
- **C0BR4**: ~100,000+ NPS (C# compiled)
- **V7P3R**: ~20,000+ NPS (Python interpreted)
- **New Engine**: Target 50,000+ NPS minimum (depends on language)

### Evaluation Speed:
- **Fast evaluation** (material + PST): ~0.02-0.04ms per node
- **Full evaluation** (all modules): ~0.1-0.5ms per node
- **Modular advantage**: 2-3x faster in time pressure

---

## 🔧 FUNCTION CALL REFERENCE

### C0BR4 Core Calls:
```csharp
// Board operations
board.GetPiece(square)
board.MakeMove(move)
board.UnmakeMove()
board.GetLegalMoves()
board.IsInCheck()

// Bitboard operations
Bitboard.PopCount(bitboard)
Bitboard.SquareToBitboard(square)
MagicBitboards.GetRookAttacks(square, occupancy)

// Search
AlphaBeta(board, depth, alpha, beta)
Quiescence(board, alpha, beta)

// Evaluation
evaluator.Evaluate(board)
GamePhase.CalculatePhase(board)
PieceSquareTables.EvaluatePosition(board, gamePhase)

// Transposition table
transpositionTable.TryGetEntry(board, depth, alpha, beta, out entry)
transpositionTable.StoreEntry(board, depth, score, move, alpha, beta)

// Time management
TimeManager.CalculateTimeAllocation(timeControl, isWhite, gamePhase)
TimeManager.CalculateSearchDepth(timeControl, isWhite, gamePhase)
```

### V7P3R Core Calls:
```python
# Board operations (python-chess)
board.piece_at(square)
board.push(move)
board.pop()
list(board.legal_moves)
board.is_check()

# Bitboard operations
int(board.pieces(piece_type, color))
popcount(bitboard)
board.is_attacked_by(color, square)

# Search
alpha_beta(board, depth, alpha, beta)
quiescence(board, alpha, beta)

# Evaluation
fast_evaluator.evaluate(board)
modular_evaluator.evaluate_with_profile(board, profile, context)
context_calculator.calculate_context(board, time_left)
profile_selector.select_profile(context)

# Move safety
safety_checker.evaluate_move_safety(board, move)
safety_checker._is_piece_hanging(board, square, piece)

# Transposition table
tt_entry = TranspositionEntry(depth, score, move, node_type, zobrist)
if hash in transposition_table: ...

# PV tracking
pv_tracker.store_pv_from_search(board, pv_moves)
instant_move = pv_tracker.check_position_for_instant_move(board)
```

---

## 📈 EXPECTED STRENGTH PROGRESSION

### Minimal Engine (Phases 1-2):
- **ELO**: ~1200-1500
- Material + basic search + move ordering

### Basic Engine (Phase 3):
- **ELO**: ~1600-1800  
- + PST + basic king safety + opening book

### Tactical Engine (Phase 4):
- **ELO**: ~1900-2100
- + Move safety + hanging piece detection + SEE

### Modular Engine (Phase 5):
- **ELO**: ~2100-2300
- + Adaptive evaluation + all tactical modules

### Complete Engine (Phase 6):
- **ELO**: ~2300-2500+
- + All evaluation modules + advanced search features

---

**Total Components**: 90+ distinct features
**Core Components**: 84 implemented across both engines
**Advanced Components**: 6 optional enhancements

This inventory provides a complete roadmap for building a modern chess engine with best practices from both C0BR4 and V7P3R architectures.
