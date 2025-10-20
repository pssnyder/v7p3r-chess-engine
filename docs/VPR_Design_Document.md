# VPR Chess Engine - Design Document

## Overview

VPR is an experimental barebones rebuild of the V7P3R chess engine, stripped down to minimal heuristics and functions for maximum search depth. The goal is to achieve significantly deeper search by reducing computational overhead per node.

## Design Philosophy

**Core Principle**: Simplicity and speed over sophisticated evaluation

The VPR engine is built on the principle that in chess, **seeing farther is often more valuable than seeing more accurately**. By removing computational overhead from complex heuristics and data structures, VPR can search 10x more nodes in the same amount of time as the full V7P3R engine.

## Performance Achievements

### Benchmark Results (5 positions, 3 second time limit)

| Metric | VPR | V7P3R v12.x | Improvement |
|--------|-----|-------------|-------------|
| **Nodes Searched (avg)** | 45,259 | 10,757 | **10.34x** |
| **NPS (avg)** | 17,677 | 3,970 | **5.24x** |
| **Depth Range** | 4-7 ply | 3-6 ply | **+1-2 ply** |
| **Code Size** | ~400 LOC | ~2,600 LOC | **6.5x smaller** |

### Position-Specific Results

1. **Starting Position**: 7.47x more nodes, 6.17x NPS
2. **Italian Game Middlegame**: 4.44x more nodes, 4.97x NPS
3. **Center Tension**: 1.37x more nodes, 6.87x NPS
4. **King & Pawn Endgame**: 34.99x more nodes, 3.42x NPS (!)
5. **Complex Middlegame**: 3.44x more nodes, 4.79x NPS

## Architectural Decisions

### What Was Kept

#### 1. **Alpha-Beta Pruning (Negamax Framework)**
- Core search algorithm retained
- Essential for reducing search space
- Clean negamax implementation
- **Cost**: Minimal (fundamental algorithm)

#### 2. **Iterative Deepening**
- Allows best move at any time
- Provides depth flexibility
- Progressive refinement
- **Cost**: ~5% overhead (worthwhile for time management)

#### 3. **Basic Move Ordering**
- Captures first (MVV-LVA)
- Simple but effective
- No move scoring tables or complex heuristics
- **Cost**: Minimal (just capture detection and sorting)

#### 4. **Material Evaluation**
- Standard piece values (P=100, N=320, B=330, R=500, Q=900)
- Essential for chess understanding
- **Cost**: Trivial (simple lookups)

#### 5. **Piece-Square Tables (PST)**
- Basic positional awareness
- Encourages center control, piece activity
- Pre-computed tables for fast lookup
- **Cost**: Small (~200 bytes of data, O(1) lookup)

#### 6. **Time Management**
- Uses 80% of allocated time
- Periodic time checks during search
- Completes depth levels cleanly
- **Cost**: Minimal (simple time checks)

### What Was Removed

#### 1. **Transposition Table** ❌
- **V7P3R Impact**: 28% hit rate, 40k+ entries
- **Cost**: Hash computation + table lookups on every node
- **Overhead**: ~20-30% per node
- **VPR Decision**: Too expensive for node throughput goal
- **Trade-off**: Accept searching duplicate positions for speed

#### 2. **Killer Moves & History Heuristic** ❌
- **V7P3R Impact**: Track successful non-capture moves
- **Cost**: Move tracking, scoring, sorting
- **Overhead**: ~10-15% per node
- **VPR Decision**: Simple MVV-LVA sufficient
- **Trade-off**: Less optimal move ordering for simpler logic

#### 3. **PV Following System** ❌
- **V7P3R Impact**: Instant moves when opponent follows prediction
- **Cost**: FEN storage, prediction tracking, state management
- **Overhead**: ~5% general overhead
- **VPR Decision**: Optimization not worth complexity
- **Trade-off**: No instant moves, but consistent fast search

#### 4. **Nudge System** ❌
- **V7P3R Impact**: 1176+ analyzed positions, opening guidance
- **Cost**: Position matching, database lookups, move boosting
- **Overhead**: ~15-20% in opening positions
- **VPR Decision**: Too specialized and costly
- **Trade-off**: Weaker opening play for consistent speed

#### 5. **Advanced Pawn Evaluation** ❌
- **V7P3R Impact**: Passed pawns, isolated pawns, doubled pawns, chains
- **Cost**: Complex pawn structure analysis per evaluation
- **Overhead**: ~30-40% of evaluation time
- **VPR Decision**: PST pawn values sufficient
- **Trade-off**: Less sophisticated pawn play for speed

#### 6. **King Safety Evaluation** ❌
- **V7P3R Impact**: Pawn shield, open files near king, attackers
- **Cost**: Attack detection, pattern matching
- **Overhead**: ~20-25% of evaluation time
- **VPR Decision**: PST king values adequate
- **Trade-off**: Less defensive awareness for speed

#### 7. **Quiescence Search** ❌
- **V7P3R Impact**: Tactical stability, horizon effect mitigation
- **Cost**: Additional search nodes for captures/checks
- **Overhead**: ~30-50% more nodes
- **VPR Decision**: Depth compensation better than quiescence
- **Trade-off**: Some tactical blindness for deeper strategic search

#### 8. **Evaluation Caching** ❌
- **V7P3R Impact**: ~60% cache hit rate
- **Cost**: Hash computation, cache management
- **Overhead**: ~10-15% (diminishes with smaller cache)
- **VPR Decision**: Fast evaluation makes caching unnecessary
- **Trade-off**: Redundant evaluations accepted

#### 9. **Bitboard Operations for Tactics** ❌
- **V7P3R Impact**: Fork, pin, skewer detection
- **Cost**: Complex bitboard analysis
- **Overhead**: ~25-30% of evaluation time
- **VPR Decision**: Too costly for node throughput
- **Trade-off**: Tactical blindness for depth

## Code Structure

### File Organization

```
src/
├── vpr.py           # Main VPR engine (~400 lines)
├── vpr_uci.py       # UCI interface (~150 lines)
testing/
└── test_vpr_comparison.py  # Performance benchmark
```

### Core Classes

#### `VPREngine`
The main engine class containing:
- `__init__()` - Initialize piece values and PST
- `search()` - Root search with iterative deepening
- `_negamax()` - Recursive alpha-beta search
- `_order_moves_simple()` - Basic move ordering
- `_evaluate_position()` - Material + PST evaluation
- `_evaluate_side()` - Per-side evaluation
- `new_game()` - Reset state
- `get_engine_info()` - Engine metadata

### Piece-Square Tables

Simple 8x8 arrays for each piece type:
- **Pawn**: Encourage advancement and center control
- **Knight**: Prefer center, avoid edges
- **Bishop**: Favor long diagonals
- **Rook**: Prefer 7th rank and open files
- **Queen**: Slight center preference
- **King**: Prefer castled position (middlegame focused)

Values are from white's perspective, flipped for black using index transformation: `63 - square`

## Search Algorithm

### Negamax with Alpha-Beta

```python
def _negamax(board, depth, alpha, beta, target_time):
    # 1. Check time limit (every 1000 nodes)
    # 2. Terminal conditions (depth=0, game over)
    # 3. Generate and order moves (captures first)
    # 4. Recursive search with alpha-beta pruning
    # 5. Return best score from current player's perspective
```

**Key Features:**
- Pure negamax (no special-case root search)
- Alpha-beta pruning for efficiency
- MVV-LVA capture ordering
- Time checking every 1000 nodes
- Mate distance scoring

### Iterative Deepening

```python
for depth in 1..max_depth:
    # Check time before starting depth
    for move in legal_moves:
        # Search to current depth
        # Track best move
    # Report results via UCI
    # Break if time exceeded
```

**Benefits:**
- Best move always available
- Progressive refinement
- Natural time management
- UCI info output per depth

## Evaluation Function

### Components (in order of importance)

1. **Material Balance** (90% of evaluation)
   - Piece values: P=100, N=320, B=330, R=500, Q=900
   - Simple piece counting

2. **Piece Positioning** (10% of evaluation)
   - Piece-square table lookup
   - Encourages good piece placement
   - No complex position analysis

3. **Special Cases**
   - Checkmate: -900,000 (distance adjusted)
   - Stalemate: 0
   - Draw by insufficient material: 0

### Evaluation Speed

- **Average**: ~50,000-100,000 evaluations/second
- **No caching**: Every position evaluated from scratch
- **No complex patterns**: Just piece lookups
- **Perspective flip**: Simple index math for black

## UCI Interface

Minimal UCI implementation supporting:
- `uci` - Engine identification
- `isready` - Ready check
- `ucinewgame` - New game
- `position` - Set position (startpos/fen + moves)
- `go` - Start search (movetime, depth, wtime/btime)
- `quit` - Exit
- `stop` - Acknowledge (no mid-search stopping)

**Not Supported:**
- `setoption` - No configurable options
- Advanced `go` parameters (nodes, infinite, searchmoves)
- Mid-search stopping

## Use Cases

### 1. **Maximum Depth Tactical Puzzles**
VPR excels at puzzles where seeing 1-2 plies deeper reveals the solution:
- Forced mate sequences
- Long tactical combinations
- Deep calculation puzzles

### 2. **Endgame Positions**
With fewer pieces, VPR's simple evaluation is often sufficient:
- King and pawn endgames (34x more nodes!)
- Simple piece endgames
- Tablebase-style calculation

### 3. **Baseline for Feature Testing**
Perfect for measuring impact of specific features:
- Add transposition table → measure impact
- Add killer moves → measure impact
- Add quiescence → measure impact

### 4. **Educational Purposes**
Clean, simple implementation for learning:
- Basic chess engine structure
- Alpha-beta pruning example
- UCI protocol implementation
- Piece-square table concept

## Limitations

### 1. **Tactical Blindness**
Without quiescence search, VPR can:
- Miss hanging pieces at horizon
- Misjudge complex tactical sequences
- Undervalue quiet tactical moves

### 2. **Weak Opening Play**
Simple evaluation means:
- No opening book knowledge
- Weak understanding of pawn structures
- Limited strategic planning

### 3. **King Safety Issues**
Without king safety evaluation:
- May expose king unnecessarily
- Weak defense against attacks
- Poor castling decisions

### 4. **Positional Weakness**
Minimal positional understanding:
- No pawn structure concepts
- Limited piece coordination
- Weak prophylactic thinking

### 5. **Repetition Blindness**
No position history:
- Can't avoid repetitions
- May repeat positions searching for better
- No draw awareness except stalemate

## Comparison with V7P3R

### V7P3R Strengths
- **Sophisticated evaluation**: Understands complex positions
- **Transposition table**: Avoids redundant searches
- **Opening knowledge**: Nudge system guides early game
- **Tactical awareness**: Bitboard tactics + quiescence
- **Production ready**: Tournament tested and stable

### VPR Strengths
- **Raw speed**: 5x faster NPS
- **Search depth**: 10x more nodes, 1-2 plies deeper
- **Code simplicity**: 6x smaller codebase
- **Predictable**: No complex state or caching
- **Educational**: Easy to understand and modify

### When to Use Each

**Use V7P3R when:**
- Playing rated games or tournaments
- Opening play is important
- Complex positional understanding needed
- Tactical accuracy is critical
- Production stability required

**Use VPR when:**
- Maximum search depth is priority
- Position is tactical and concrete
- Endgame with clear evaluation
- Testing feature impact
- Learning engine architecture

## Future Enhancements (Optional)

### Potential Additions (in priority order)

1. **Minimal Quiescence Search** (depth 1-2)
   - Cost: ~20% more nodes
   - Benefit: Reduces tactical blindness
   - Decision: Worth considering

2. **Simple Repetition Detection**
   - Cost: ~5% overhead
   - Benefit: Avoid draw loops
   - Decision: Probably worth it

3. **Endgame Piece-Square Tables**
   - Cost: Minimal (just different tables)
   - Benefit: Better king activity in endgame
   - Decision: Low-hanging fruit

4. **Null Move Pruning**
   - Cost: Minimal (just null move search)
   - Benefit: 20-30% search reduction in some positions
   - Decision: Worth experimenting

5. **Tiny Transposition Table** (1000 entries)
   - Cost: Small overhead
   - Benefit: Catch obvious repetitions
   - Decision: Maybe in VPR v2.0

### Definitely Not Adding

- Full transposition table (defeats the purpose)
- Complex evaluation terms (against philosophy)
- Machine learning (too complex)
- Opening books (external dependency)

## Development Notes

### Building
No build required - pure Python implementation:
```bash
python src/vpr_uci.py
```

### Testing
```bash
# Performance comparison
python testing/test_vpr_comparison.py

# Quick test
python -c "from src.vpr import VPREngine; import chess; e = VPREngine(); print(e.search(chess.Board(), 2.0))"
```

### UCI Usage
```bash
# Via UCI interface
python src/vpr_uci.py

# Example commands
uci
isready
position startpos moves e2e4
go movetime 3000
quit
```

## Conclusion

VPR demonstrates that aggressive simplification can yield dramatic performance improvements in chess engines. By removing ~80% of V7P3R's features, VPR achieves 10x more node throughput and reaches 1-2 plies deeper in the same time.

The trade-off is clear: VPR sacrifices positional understanding, tactical accuracy, and opening knowledge for raw search depth. This makes it:
- **Excellent** for deep tactical puzzles and endgames
- **Good** as a baseline for feature testing
- **Poor** for complex strategic positions
- **Weak** in the opening

For users seeking maximum search depth at the cost of evaluation sophistication, VPR delivers. For those needing well-rounded chess strength, V7P3R remains the better choice.

**Key Insight**: In chess engines, there's no free lunch. Every feature has a cost. VPR proves that sometimes, less is more - if raw depth is your goal.

---

**VPR v1.0 - Pat Snyder - 2024**
*"See farther, not clearer"*
