# V7P3R Chess Engine

**Latest Release: V16.1** - November 19, 2025  
A UCI-compatible chess engine combining material safety, positional play, deep opening theory, and perfect endgames.

## 🎯 Current Version: V16.1

**V16.1** represents a major enhancement over V16.0, adding three critical improvements:

### V16.1 Key Enhancements

1. **🔥 Deep Opening Repertoire (15 moves)**
   - 52+ positions programmed with center-control focus
   - White: Italian Game (Giuoco Piano), Queen's Gambit Declined, King's Indian Attack
   - Black: Sicilian Najdorf, King's Indian Defense, French Defense, Caro-Kann
   - Smooth transition from opening to middlegame

2. **⚡ Middlegame Transition Nudges**
   - Rook activity: +20cp (open files), +10cp (semi-open files)
   - King safety: +10cp per pawn shield
   - Pawn structure: +30cp (passed pawns), -20cp (doubled pawns)
   - Intelligent piece placement bonuses

3. **♟️ Syzygy Tablebase Integration**
   - Perfect 6-piece endgame play (when tablebases available)
   - WDL (Win/Draw/Loss) probing for guaranteed optimal moves
   - Graceful fallback to heuristic search if tablebases unavailable

4. **🐛 Critical Bug Fix**
   - Fixed "no move found" bug in drawn positions (K vs K, insufficient material)
   - Changed from `is_game_over()` to specific `is_checkmate()` / `is_stalemate()` checks
   - **This bug caused Arena's "illegal move" error in V16.0**

### V16.1 Architecture

**Core Formula:**
```
Evaluation = (PST × 60%) + (Material × 40%) + Middlegame Bonuses
```

**Three-Phase Excellence:**
- **Opening (Moves 1-15):** Deep book with center control theory
- **Middlegame:** Positional bonuses for piece activity, king safety, pawn structure
- **Endgame (≤6 pieces):** Perfect play via Syzygy tablebases (or strong heuristics)

---

## 📊 Version History

### V16.1 (November 19, 2025) - Enhanced Opening + Bug Fix ✅
- Deep 52-position opening book (15 moves deep)
- Middlegame nudges: rooks, king safety, pawn structure (+70cp potential)
- Syzygy tablebase support for perfect endgames
- **Fixed critical "no move found" bug** (caused V16.0 Arena errors)
- All game phases tested and verified

### V16.0 (November 2025) - Fresh Start
- Combined MaterialOpponent + PositionalOpponent strengths
- 60% PST + 40% Material evaluation
- Pre-search move filtering (never sacrifices material)
- Castling preservation (king moves deprioritized)
- Tournament tested: 0-2 (revealed opening/endgame weaknesses)

### V14.0 (October 25, 2025) - Consolidated Performance
- Unified bitboard evaluation system
- Tactical detection integration
- Pawn structure consolidation
- Equivalent performance to V12.6 with cleaner architecture


### V12.6 (October 4, 2025) - Tournament Champion ✅
- **Engine Battle 20251004:** 7.0/10 points (2nd place)
- **Regression Battle 20251004:** 9.0/12 points (1st place)
- 30x faster evaluation (152ms → 5ms)
- Clean codebase without nudge system overhead
- High-performance bitboard evaluation

---

## 🚀 Installation & Setup

### Requirements
- **Python 3.12+** (3.13 recommended)
- **python-chess** library: `pip install python-chess`
- **Optional:** Syzygy tablebases (3-4-5 piece) for perfect endgames

### Quick Start

```bash
# 1. Install dependencies
pip install python-chess

# 2. Test engine
cd src
python v7p3r_uci.py

# 3. In UCI interface, type:
uci
isready
position startpos
go depth 6
```

### Arena Chess GUI Setup

1. **Install Engine:**
   - Open Arena Chess GUI
   - Go to: **Engines → Install New Engine**
   - Navigate to `src/` and select: `v7p3r_uci.py`
   - Engine appears as: **V7P3R v16.1**

2. **Configure Python Path (if needed):**
   - Create `V7P3R_v161.bat` with your Python path:
   ```batch
   @echo off
   "C:\Users\YourName\AppData\Local\Programs\Python\Python313\python.exe" v7p3r_uci.py
   ```

3. **Optional: Configure Syzygy Tablebases:**
   - Download 3-4-5 piece tablebases from [Syzygy Download](http://tablebase.sesse.net/syzygy/)
   - In Arena, set UCI option: `setoption name SyzygyPath value C:\path\to\tablebases`

---

## ⚙️ UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| **MaxDepth** | spin | 6 | 1-20 | Maximum search depth (V16.1: 6-10 recommended) |
| **TTSize** | spin | 128 | 16-2048 | Transposition table size in MB |
| **SyzygyPath** | string | (empty) | - | Path to Syzygy tablebase files (V16.1) |

### Configuration Examples

```uci
# Standard tournament settings
setoption name MaxDepth value 8
setoption name TTSize value 256

# With tablebases for perfect endgames
setoption name SyzygyPath value /path/to/syzygy
setoption name MaxDepth value 10
```

---

## 🎮 Playing Style

### V16.1 Strengths

1. **🔥 Opening Mastery**
   - 52 positions deep (15 moves)
   - Center control focus (Italian, QGD, Sicilian Najdorf, KID)
   - Smooth book exit transitions

2. **⚡ Middlegame Excellence**
   - Material safety (never sacrifices without compensation)
   - Active piece placement (PST-guided)
   - Rook activity detection (+20cp open files)
   - King safety awareness (+10cp per shield pawn)
   - Pawn structure intelligence (passed pawns, doubled pawns)

3. **♟️ Endgame Mastery**
   - **With Tablebases:** Perfect 6-piece play (forced wins)
   - **Without Tablebases:** Strong PST + material heuristics
   - King centralization in endgames
   - Passed pawn promotion technique

4. **🛡️ Tactical Awareness**
   - Pre-search move filtering (material safety)
   - Quiescence search (tactical stability)
   - Mate detection (finds checkmate in 1-2 moves)
   - King safety preservation (castling prioritized)

### V16.1 Characteristics

**Target Depth:** 8-10 (standard time controls)  
**Nodes/Second:** 3,000-10,000 (position dependent)  
**Opening Book Usage:** Moves 1-15 (52+ positions)  
**Evaluation Speed:** ~5-10ms per position  

### Ideal Opponents

- **Competitive Against:** C0BR4 v3.2 (primary target), material-focused engines
- **Challenging:** Super-tactical engines, deep search specialists (>12 depth)
- **Beats:** Engines without opening books, weak endgame play

### Known Limitations

1. **Depth Limitation:** Targets 8-10 depth (not ultra-deep search)
2. **Tactical Horizon:** May miss deep (5+ move) tactical sequences
3. **Time Management:** Basic time allocation (can be optimized)
4. **No Neural Networks:** Classical evaluation only

---

## 🏗️ Technical Architecture

### V16.1 Core Systems

```
V7P3REngine (Main Class)
├── Initialization
│   ├── Opening Book (v7p3r_openings_v161.py - 52 positions)
│   ├── Transposition Table (Zobrist hashing)
│   └── Syzygy Tablebase (optional)
│
├── Search Pipeline
│   ├── get_best_move() - Entry point
│   │   ├── Tablebase Probing (≤6 pieces, perfect play)
│   │   ├── Opening Book Lookup (15 moves deep)
│   │   ├── Iterative Deepening (1→max_depth)
│   │   └── Time Management
│   │
│   ├── _search() - Alpha-beta with pruning
│   │   ├── Transposition Table Probe
│   │   ├── Null Move Pruning (depth ≥3)
│   │   ├── Move Filtering (material safety)
│   │   └── Recursive Minimax
│   │
│   └── _quiescence_search() - Tactical stability
│       └── Capture-only search to quiet positions
│
└── Evaluation System
    ├── _evaluate_position() - Main evaluator
    │   ├── PST Score (60% weight)
    │   ├── Material Score (40% weight)
    │   └── Middlegame Bonuses (V16.1)
    │
    └── _calculate_middlegame_bonuses() - V16.1 Enhancement
        ├── Rook Activity (+20cp open, +10cp semi-open)
        ├── King Safety (+10cp per shield pawn)
        ├── Pawn Structure (+30cp passed, -20cp doubled)
        └── Returns total bonus in centipawns
```

### File Structure

```
v7p3r-chess-engine/
├── src/
│   ├── v7p3r.py                    # Main engine (V16.1)
│   ├── v7p3r_uci.py                # UCI protocol interface
│   └── v7p3r_openings_v161.py      # Opening book (52 positions)
│
├── testing/
│   ├── test_v161_game_phases.py    # Comprehensive phase testing
│   └── test_no_move_bug.py         # Bug diagnostic tests
│
├── docs/                           # Project documentation
├── build/                          # PyInstaller builds
└── README.md                       # This file
```

### Code Statistics (V16.1)

- **v7p3r.py:** ~870 lines (core engine)
- **v7p3r_openings_v161.py:** ~400 lines (opening book)
- **v7p3r_uci.py:** ~180 lines (UCI interface)
- **Total:** ~1,450 lines of production code

---

## 🔬 Performance Benchmarks

### V16.1 Test Results (November 19, 2025)

| Test Category | Result | Details |
|--------------|--------|---------|
| **Opening Book** | ✅ Pass | 52 positions loaded, all book moves valid |
| **Middlegame Bonuses** | ✅ Pass | +10 to +30cp applied correctly |
| **Endgame (No TB)** | ✅ Pass | Strong heuristic play (KR vs K, KQ vs K) |
| **Mate Finding** | ✅ Pass | Mate in 1 found instantly (29999cp) |
| **Tactical Awareness** | ⚠️ Partial | Finds forks/checks, depth-limited on deep tactics |
| **Bug Fix** | ✅ Fixed | "No move found" bug eliminated |

### Benchmark Positions

**Starting Position** (depth 6, 1s):
- Move: d4
- Eval: 0cp (balanced)
- Nodes: ~267

**Complex Middlegame** (depth 2, 0.06s):
- Move: Qd3
- Eval: +46cp (White advantage)
- Rook bonus applied: +0cp (no open files yet)

**K vs K Draw** (depth 2, 0.005s):
- Move: Ke5
- Eval: +24cp (centeralization)
- **Previously failed** (V16.0 bug) ✅ Now fixed

---

## 🐛 Known Issues & Fixes

### V16.1 Bug Fix: "No Move Found"

**Issue:** V16.0 returned `None` in drawn-by-insufficient-material positions (K vs K, KN vs K), causing Arena to flag "illegal move."

**Root Cause:** Used `board.is_game_over()` which returns `True` for:
- ✓ Checkmate/Stalemate (correct)
- ✗ Insufficient material draws (incorrect - still have moves)
- ✗ 50-move rule (incorrect - still have moves)

**Fix Applied:**
```python
# Before (buggy):
if board.is_game_over():
    return None

# After (fixed):
if board.is_checkmate():
    return -MATE_SCORE + ply, None
if board.is_stalemate():
    return 0, None
# Continue searching in all other cases
```

**Validation:** ✅ All drawn positions now return legal moves

---

## 📈 Development Roadmap

### V16.2 (Planned)
- [ ] Enhanced time management (increment handling)
- [ ] Opening book expansion (100+ positions)
- [ ] Middlegame bonus tuning (match testing)
- [ ] Performance profiling and optimization

### V17.0 (Future)
- [ ] Neural network evaluation exploration
- [ ] Deep tactical search extensions
- [ ] Advanced endgame heuristics (7-piece positions)
- [ ] Multi-threading support

---

## 🏆 Tournament Results

### V16.1 (Testing Phase)
- **Status:** Ready for tournament testing
- **Target Opponent:** C0BR4 v3.2
- **Expected Performance:** >50% win rate vs baseline

### V16.0 (November 2025)
- **Record:** 0-2 (vs V14.1, V12.6)
- **Issue:** Arena flagged "illegal move" (now fixed in V16.1)
- **Analysis:** Weak opening book (2 positions), no endgame tablebases

### V12.6 (October 2025) - Champion
- **Engine Battle 20251004:** 7.0/10 points (2nd place)
- **Regression Battle 20251004:** 9.0/12 points (1st place vs all V7P3R versions)
- **Performance:** 30x faster than V12.2 baseline

---

## 🤝 Contributing

This is a personal development project, but feedback and testing are welcome!

**Testing Help:**
- Run V16.1 in tournaments and report results
- Compare vs other engines (C0BR4, Stockfish levels)
- Report any UCI compatibility issues
- Share interesting game PGNs

**Contact:** Open issues in the repository or submit pull requests

---

## 📜 License

Personal project - educational use encouraged

---

## 🙏 Acknowledgments

- **python-chess** library by Niklas Fiekas
- **Syzygy Tablebases** by Ronald de Man
- **Arena Chess GUI** for tournament testing
- Chess programming community for evaluation techniques

---

**V16.1 Status:** ✅ Production Ready  
**Primary Target:** Beat C0BR4 v3.2  
**Development Focus:** Opening mastery + Perfect endgames + Material safety