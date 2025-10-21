# VPR Chess Engine

**VPR** - An experimental barebones rebuild of the V7P3R chess engine, stripped down to minimal heuristics for maximum search depth.

> *"See farther, not clearer"*

## Quick Stats

- **Nodes Searched**: 10x more than V7P3R
- **Search Speed**: 5x faster NPS (17,000+ vs 4,000)
- **Code Size**: 400 lines (vs 2,600 in V7P3R)
- **Search Depth**: +1-2 plies deeper in same time
- **Evaluation**: Material + Piece-Square Tables only

## What is VPR?

VPR is an experimental chess engine built on one principle: **raw search depth beats sophisticated evaluation** (sometimes). By removing 80% of V7P3R's features, VPR achieves 10x node throughput at the cost of positional understanding.

### What VPR Has

✅ Alpha-beta negamax search  
✅ Iterative deepening  
✅ Basic move ordering (MVV-LVA)  
✅ Material evaluation  
✅ Piece-square tables  
✅ Time management  
✅ UCI protocol support  

### What VPR Doesn't Have

❌ Transposition table  
❌ Killer moves / history heuristic  
❌ Quiescence search  
❌ Advanced pawn evaluation  
❌ King safety evaluation  
❌ Tactical pattern detection  
❌ PV following  
❌ Nudge system  
❌ Evaluation caching  

## Performance Results

Benchmark on 5 test positions (3 second time limit):

| Position | VPR Nodes | V7P3R Nodes | Improvement |
|----------|-----------|-------------|-------------|
| Starting position | 39,754 | 5,323 | **7.47x** |
| Italian middlegame | 36,092 | 8,120 | **4.44x** |
| Center tension | 36,721 | 26,750 | **1.37x** |
| K+P endgame | 75,394 | 2,155 | **34.99x** 🔥 |
| Complex middlegame | 39,334 | 11,439 | **3.44x** |

**Average**: 10.34x more nodes, 5.24x higher NPS

## Installation

### Requirements
- Python 3.8+
- python-chess library

```bash
pip install python-chess==1.999
```

### Running VPR

```bash
# UCI interface (for GUIs)
python src/vpr_uci.py

# Direct Python usage
python -c "
from src.vpr import VPREngine
import chess
engine = VPREngine()
move = engine.search(chess.Board(), time_limit=3.0)
print(f'Best move: {move}')
"
```

### Running Performance Tests

```bash
# Compare VPR vs V7P3R
python testing/test_vpr_comparison.py
```

## Usage Examples

### 1. UCI Interface (Arena, Cutechess, etc.)

```
uci
uciok
id name VPR v1.0
id author Pat Snyder

position startpos moves e2e4 e7e5
go movetime 3000

info depth 4 score cp 25 nodes 15000 time 1500 nps 10000 pv g1f3
bestmove g1f3
```

### 2. Python Integration

```python
from src.vpr import VPREngine
import chess

# Create engine
engine = VPREngine()

# Search a position
board = chess.Board()
best_move = engine.search(board, time_limit=5.0)

# Get engine info
info = engine.get_engine_info()
print(f"Searched {info['nodes_searched']} nodes")

# New game
engine.new_game()
```

### 3. Custom Time Controls

```python
# Fixed depth search
move = engine.search(board, depth=6)

# Time-based search
move = engine.search(board, time_limit=10.0)
```

## When to Use VPR

### ✅ Good Use Cases

- **Tactical Puzzles**: Deep calculation reveals solution
- **Endgames**: Simple evaluation sufficient (34x speedup!)
- **Feature Testing**: Measure impact of individual features
- **Learning**: Clean, simple engine architecture
- **Speed Chess**: Fast NPS important
- **Depth-Critical Positions**: Extra plies reveal winning move

### ❌ Poor Use Cases

- **Opening Phase**: Weak positional understanding
- **Strategic Positions**: Can't evaluate complex plans
- **King Attacks**: No king safety evaluation
- **Rated Games**: Less stable than full engine
- **Positional Endgames**: Weak pawn structure understanding

## Architecture

### File Structure

```
src/
├── vpr.py           (~400 lines) - Main engine
└── vpr_uci.py       (~150 lines) - UCI interface

testing/
└── test_vpr_comparison.py - Performance benchmark

docs/
└── VPR_Design_Document.md - Detailed design doc
```

### Core Algorithm

```python
def search(board, time_limit):
    # Iterative deepening from depth 1..8
    for depth in range(1, 9):
        for move in legal_moves:
            score = -negamax(board, depth-1, -∞, +∞)
            # Track best move
        # Return if time exceeded

def negamax(board, depth, alpha, beta):
    if depth == 0:
        return evaluate(board)  # Material + PST
    
    for move in ordered_moves():  # Captures first (MVV-LVA)
        score = -negamax(board, depth-1, -beta, -alpha)
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Cutoff
    
    return alpha
```

### Evaluation Function

```python
def evaluate(board):
    score = 0
    for piece in board:
        # Material value
        score += PIECE_VALUES[piece.type]
        
        # Position value (piece-square table)
        score += PST[piece.type][piece.square]
    
    return score  # From current player's perspective
```

**That's it!** No caching, no hashing, no complex heuristics.

## Configuration

VPR has minimal configuration (by design):

```python
# In vpr.py __init__()
self.default_depth = 8  # Maximum search depth
self.piece_values = {   # Standard piece values
    PAWN: 100, KNIGHT: 320, BISHOP: 330,
    ROOK: 500, QUEEN: 900
}
```

Piece-square tables can be tuned in `_init_piece_square_tables()`.

## Comparison with V7P3R

| Feature | VPR | V7P3R v12.x |
|---------|-----|-------------|
| **Lines of Code** | ~400 | ~2,600 |
| **Nodes/Second** | 17,677 | 3,970 |
| **Search Depth** | 4-7 ply | 3-6 ply |
| **Evaluation** | Material + PST | 9 components |
| **Move Ordering** | MVV-LVA only | TT + Killers + History |
| **Transposition Table** | No | 40k entries |
| **Quiescence** | No | Yes |
| **Opening Knowledge** | No | Nudge system |
| **Tournament Ready** | No | Yes |
| **Educational Value** | High | Medium |

## Known Limitations

1. **Tactical Horizon Effects**: No quiescence search means can miss captures
2. **Weak Opening Play**: No opening knowledge or strategic understanding
3. **King Safety**: May expose king to attacks
4. **Repetition Blindness**: Can't detect or avoid draw by repetition
5. **Positional Weakness**: Limited understanding of pawn structures
6. **No History**: Doesn't learn from searched positions

## Future Possibilities

Potential enhancements (in order of priority):

1. **Minimal Quiescence** (1-2 ply) - Reduce tactical blindness
2. **Repetition Detection** - Avoid draw loops  
3. **Endgame PST** - Better king activity
4. **Null Move Pruning** - 20-30% speedup
5. **Tiny TT** (1000 entries) - Catch repetitions

*Note*: Adding too much defeats the purpose!

## Development

### Running Tests

```bash
# Performance comparison
python testing/test_vpr_comparison.py

# Quick functionality test
python -c "
import sys
sys.path.insert(0, 'src')
from vpr import VPREngine
import chess

engine = VPREngine()
board = chess.Board()
move = engine.search(board, time_limit=2.0)
print(f'✓ Engine works! Best move: {move}')
print(f'Nodes: {engine.nodes_searched:,}')
"
```

### Testing with Arena GUI

1. Open Arena Chess GUI
2. Add engine: `python /path/to/vpr_uci.py`
3. Set engine name: "VPR v1.0"
4. Play or analyze!

### Debugging

Enable verbose output in `vpr.py`:

```python
# In _negamax(), add:
if depth > 0 and self.nodes_searched % 10000 == 0:
    print(f"info string depth={depth} nodes={self.nodes_searched}")
```

## Contributing

VPR is intentionally minimal. Contributions should:
- Maintain simplicity (avoid feature creep)
- Improve speed without adding complexity
- Fix bugs without adding overhead
- Enhance documentation

## License

Same license as V7P3R (see main repository).

## Credits

- **Author**: Pat Snyder
- **Based on**: V7P3R Chess Engine
- **Inspiration**: Simple chess engines (Sunfish, TSCP, etc.)

## FAQ

**Q: Why is it called VPR?**  
A: VPR stands for "V7P3R" with the vowels removed - representing the stripped-down nature of the engine.

**Q: Is VPR stronger than V7P3R?**  
A: No. VPR searches deeper but evaluates positions less accurately. Against V7P3R, VPR would likely lose in most positions due to weaker evaluation.

**Q: What's VPR's ELO rating?**  
A: Estimated 1200-1500. Strong tactical calculation but weak positional play.

**Q: Should I use VPR in tournaments?**  
A: No. Use V7P3R v12.x for serious play. VPR is experimental.

**Q: Can I add [feature X]?**  
A: You can, but consider if it aligns with VPR's goal of maximum simplicity. Fork for heavy modifications!

**Q: Why no transposition table?**  
A: TT lookups cost ~20-30% overhead per node. VPR prefers raw speed over avoiding repeated work.

**Q: How much stronger could VPR be with just quiescence?**  
A: Estimated +200-300 ELO, but at cost of 30-50% more nodes. Still evaluating trade-off.

## See Also

- [VPR Design Document](docs/VPR_Design_Document.md) - Detailed design rationale
- [V7P3R Main Engine](src/v7p3r.py) - Full-featured version
- [Performance Tests](testing/test_vpr_comparison.py) - Benchmark suite

---

**VPR v1.0** - Experimental Maximum Depth Engine  
*When you absolutely, positively need to see 8 plies deep.*

For questions or feedback: See main V7P3R repository
