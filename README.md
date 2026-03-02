# V7P3R Chess Engine

**Latest Release: V18.3.0** - December 29, 2025  
A UCI-compatible chess engine achieving **50% score vs Stockfish 1%** through advanced search optimization and tactical excellence.

---

## 🎯 Current Version: V18.3.0 (PST Optimization - Production on GCP)

**V18.3.0** delivers 28% faster piece-square table evaluation through direct square indexing and pre-computed flipped tables, achieving +56 ELO improvement.

### V18.3.0 Key Features

1. **⚡ PST Performance Optimizations**
   - Direct square indexing (eliminates modulo/division overhead)
   - Pre-computed flipped tables for Black pieces
   - 28% faster PST evaluation, 23% faster full evaluation
   - Clean architectural decomposition (material, PST, strategic components)

2. **🎯 Tactical Excellence (from v18.0)**
   - MoveSafetyChecker anti-tactical defense system
   - Threefold repetition avoidance (>100cp threshold)
   - Enhanced move ordering with safety scoring
   - History heuristic + killer moves + MVV-LVA

3. **☁️ Cloud Deployment**
   - Running 24/7 on Google Cloud Platform (e2-micro VM)
   - 1GB RAM, 2 vCPUs, us-central1-a region
   - Emergency configuration: concurrency=1, rapid+classical time controls
   - Professional infrastructure with monitoring and backups
   - ~$24/month operational cost

### V17.2.0 Performance Metrics
- **NPS:** ~5,540 nodes/second (baseline testing)
- **Search Depth:** Typically 10-12 in rapid time controls
- **TT Usage:** 39% at depth 4 (efficient memory utilization)
- **Cache Hit Rate:** 18.1% (improved from 13.3%)

---

## 🏆 V17.1 BREAKTHROUGH ACHIEVEMENT

**V7P3R v17.1 achieved the most significant performance milestone in development history:**

### Historic Results: 50% vs Stockfish 1%

**Tournament:** Engine Battle 20251121_2 (360 games, 9 engines)

| Rank | Engine | Score | Win % | vs Stockfish 1% |
|------|--------|-------|-------|-----------------|
| 1 | Stockfish 1% | 72.5/80 | 90.6% | N/A |
| **2** | **V7P3R v17.1** | **71.5/80** | **89.4%** | **10.0/20 (50%)** ⭐ |
| 3 | C0BR4 v3.2 | 62.0/80 | 77.5% | 1.5/10 (15%) |
| 4 | SlowMate v3.1 | 53.5/80 | 66.9% | 0.5/10 (5%) |

**Head-to-Head: V7P3R v17.1 vs Stockfish 1% (20 games)**
- **As White:** 5.5/10 (55%) - First move advantage utilized ✅
- **As Black:** 4.5/10 (45%) - Solid defensive play
- **Wins:** 9 (including 6 checkmate victories)
- **Draws:** 2 (10% draw rate)
- **Losses:** 9
- **Time Control:** Blitz 1+1 (1 minute + 1 second/move)

### Performance Analysis

**Strengths:**
- ✅ **Tactical Middle Games:** 57% win rate (30-60 moves)
- ✅ **Quick Tactics:** 60% win rate (<30 moves)
- ✅ **Speed Under Pressure:** 5,845 NPS in blitz, zero time losses
- ✅ **Opening Flexibility:** 83% score in A40 Queen's Pawn Game
- ✅ **Attacking Prowess:** Multiple forcing sequences and checkmates

**Growth Opportunities:**
- ⚠️ **Long Endgames:** 29% win rate (60+ moves) - Stockfish won 5 of 7
- ⚠️ **Technical Positions:** Can improve conversion of slight advantages

**Rating Achievement:**
- **Performance Rating:** ~2600-2650 Elo (vs 1% handicapped Stockfish)
- **Estimated Improvement:** +150 Elo over v16.x series
- **+25% Performance Gain** over previous versions vs Stockfish

**See full analysis:** `docs/V17_1_STOCKFISH_BREAKTHROUGH_ANALYSIS.md`

---

## 📊 Version History

### V17.2.0 (November 22, 2025) - Performance Optimization
- O(1) TT bucket replacement system
- Unified TT + evaluation cache
- In-place quiescence sorting
- Pre-allocated move ordering buffers
- UCI debugging: seldepth, hashfull tracking
- Deployed to GCP e2-medium (4GB RAM, 2 vCPUs)
- ~5,540 NPS performance (slight regression for infrastructure improvements)

### V17.1.1 (November 21, 2025) - Lichess Production Hotfix
- Fixed time management for Lichess cloud deployment
- Restricted to 10+ minute games (600s minimum base time)
- Improved matchmaking acceptance rate
- Successfully running on GCP e2-micro (1GB RAM)
- Achieved 61% win rate in cloud environment

### V17.1 (November 21, 2025) - History Heuristic ⭐ BREAKTHROUGH
- **Historic Achievement:** 50% score vs Stockfish 1% (10/20 games)
- **2nd Place** in 9-engine tournament (71.5/80 = 89.4%)
- History heuristic with position-aware tracking
- Optimized TT aging (128 generations)
- Enhanced quiescence depth limits
- **Most significant performance jump in development history**
- Rating: ~2600-2650 Elo (vs handicapped Stockfish)

### V17.0 (November 20, 2025) - Advanced Search Techniques
- Killer move heuristic (2 slots per ply)
- MVV-LVA capture ordering
- Enhanced move ordering pipeline
- Improved tactical vision
- Foundation for v17.1 breakthrough

### V16.1 (November 19, 2025) - Enhanced Opening + Bug Fix
- Deep 52-position opening book (15 moves)
- Middlegame nudges: rooks, king safety, pawn structure
- Syzygy tablebase support
- Fixed critical "no move found" bug

### V16.0 (November 2025) - Fresh Start
- Combined MaterialOpponent + PositionalOpponent strengths
- 60% PST + 40% Material evaluation
- Pre-search move filtering

### V14.0 (October 25, 2025) - Consolidated Performance
- Unified bitboard evaluation system
- Tactical detection integration
- Pawn structure consolidation

### V12.6 (October 4, 2025) - Tournament Champion
- **Engine Battle 20251004:** 7.0/10 points (2nd place)
- **Regression Battle 20251004:** 9.0/12 points (1st place)
- 30x faster evaluation (152ms → 5ms)
- High-performance bitboard evaluation

---

## 🚀 Installation & Setup

### Requirements
- **Python 3.12+** (3.13 recommended)
- **python-chess** library: `pip install python-chess`
- **Optional:** Syzygy tablebases for perfect endgames
- **Cloud Deployment:** Docker, Google Cloud SDK (for lichess-bot)

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/pssnyder/v7p3r-chess-engine.git
cd v7p3r-chess-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test engine
cd src
python v7p3r_uci.py

# 4. In UCI interface, type:
uci
isready
position startpos
go depth 10
```

### Arena Chess GUI Setup

1. **Install Engine:**
   - Open Arena Chess GUI
   - Go to: **Engines → Install New Engine**
   - Navigate to `src/` and select: `v7p3r_uci.py`
   - Engine appears as: **V7P3R v17.2.0**

2. **Configure Python Path (if needed):**
   - Create `V7P3R_v172.bat` with your Python path:
   ```batch
   @echo off
   "C:\Users\YourName\AppData\Local\Programs\Python\Python313\python.exe" v7p3r_uci.py
   ```

3. **Recommended Settings:**
   - Time Control: Rapid 10+5 or Classical 15+10
   - Max Depth: 10-12 (v17.x series)
   - Hash Size: 256MB

---

## ☁️ Lichess Cloud Deployment

V7P3R now includes integrated Lichess bot deployment infrastructure.

### Repository Structure
```
v7p3r-chess-engine/
├── src/                    # Core engine
├── lichess/                # Lichess bot integration ⭐ NEW
│   ├── config.yml          # Bot configuration
│   ├── engines/            # Deployed engine versions
│   ├── game_records/       # Game PGNs
│   └── docs/               # Deployment guides
├── docs/                   # Technical documentation
└── testing/                # Test suites
```

### Current Cloud Status
- **Instance:** v7p3r-production-bot (GCP e2-medium)
- **Engine Version:** V17.2.0
- **Location:** us-central1-a
- **Uptime:** 99.9%
- **Cost:** ~$24/month

### Quick Deployment
```bash
# Upload new engine version (5 minutes)
cd lichess/engines/V7P3R_v17.2.0
tar -czf ../v17.2.0-src.tar.gz src/
gcloud compute scp ../v17.2.0-src.tar.gz v7p3r-production-bot:/home/patss/ --zone=us-central1-a

# Deploy to container
gcloud compute ssh v7p3r-production-bot --zone=us-central1-a --command="
    sudo docker cp v17.2.0-src.tar.gz v7p3r-production:/tmp/
    sudo docker exec v7p3r-production bash -c 'mv /lichess-bot/engines/v7p3r /lichess-bot/engines/v7p3r.backup'
    sudo docker exec v7p3r-production mkdir -p /lichess-bot/engines/v7p3r
    sudo docker exec v7p3r-production bash -c 'cd /lichess-bot/engines/v7p3r && tar -xzf /tmp/v17.2.0-src.tar.gz --strip-components=1'
    sudo docker restart v7p3r-production
"
```

**Full Guide:** `lichess/docs/CLOUD_DEPLOYMENT_GUIDE.md`

---

## ⚙️ UCI Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| **MaxDepth** | spin | 10 | 1-20 | Maximum search depth (V17.x: 10-12 recommended) |
| **TTSize** | spin | 128 | 16-2048 | Transposition table size in MB |
| **UseHistory** | check | true | - | Enable history heuristic (V17.1+) |
| **UseKillers** | check | true | - | Enable killer move heuristic (V17.0+) |

### Configuration Examples

```uci
# Tournament settings (V17.x)
setoption name MaxDepth value 12
setoption name TTSize value 256
setoption name UseHistory value true
setoption name UseKillers value true

# Blitz settings
setoption name MaxDepth value 10
setoption name TTSize value 128
```

---

## 🎮 Playing Style

### V17.x Strengths

1. **🎯 Tactical Excellence**
   - Multiple forcing sequences found
   - 6 checkmate victories vs Stockfish 1%
   - Strong piece coordination
   - Excellent at king hunts

2. **⚡ Middle Game Mastery**
   - 57% win rate in 30-60 move games
   - Active piece placement
   - Effective attacking play
   - Strong calculation in complex positions

3. **🔥 Opening Flexibility**
   - 52+ positions programmed (15 moves deep)
   - 83% score in A40 Queen's Pawn Game
   - Comfortable in various systems
   - Good pawn structure understanding

4. **⚡ Speed Under Pressure**
   - 5,845 NPS in blitz time controls
   - Zero time losses in 20-game match
   - Effective time management
   - No timeouts even under pressure

### V17.x Characteristics

**Target Depth:** 10-12 (standard time controls)  
**Nodes/Second:** 5,000-6,000 (position dependent)  
**Search Features:** History heuristic, killer moves, MVV-LVA  
**Evaluation Speed:** ~0.17ms per position  
**TT Efficiency:** 18% hit rate, 39% utilization

### Ideal Opponents

- **Competitive Against:** Stockfish 1% (50% score), C0BR4 v3.2 (65% score)
- **Dominates:** All opponent engines (100% score), tactical engines without search depth
- **Challenging:** Full-strength Stockfish, engines with >14 depth search

### Known Characteristics

1. **Strength:** Tactical middle games, quick tactics (<30 moves)
2. **Growth Area:** Long endgames (60+ moves) need improvement
3. **Aggressive:** Only 10% draw rate (seeks wins over draws)
4. **Reliable:** Zero critical bugs in 20-game Stockfish match

---

## 🏗️ Technical Architecture

### V17.2.0 Core Systems

```
V7P3REngine (Main Class)
├── Initialization
│   ├── Opening Book (v7p3r_openings_v161.py - 52 positions)
│   ├── Transposition Table (Two-tier bucket system - V17.2)
│   ├── History Heuristic (Position-aware - V17.1)
│   ├── Killer Moves (2 slots per ply - V17.0)
│   └── Move Ordering Buffers (Pre-allocated - V17.2)
│
├── Search Pipeline
│   ├── get_best_move() - Entry point
│   │   ├── Opening Book Lookup (15 moves deep)
│   │   ├── Iterative Deepening (1→max_depth)
│   │   ├── Time Management (adaptive)
│   │   └── UCI Info Output (depth, seldepth, hashfull)
│   │
│   ├── _search() - Alpha-beta with advanced move ordering
│   │   ├── Transposition Table Probe (O(1) two-tier)
│   │   ├── TT Move (highest priority)
│   │   ├── Killer Moves (2 slots per ply)
│   │   ├── History Heuristic (position-aware)
│   │   ├── MVV-LVA (capture ordering)
│   │   ├── Null Move Pruning (depth ≥3)
│   │   └── Late Move Reduction (non-critical moves)
│   │
│   └── _quiescence_search() - In-place sorting (V17.2)
│       └── Capture-only search to quiet positions
│
└── Evaluation System
    ├── _evaluate_position() - Unified TT cache (V17.2)
    │   ├── PST Score (60% weight)
    │   ├── Material Score (40% weight)
    │   └── Middlegame Bonuses
    │
    └── Transposition Table (V17.2 optimizations)
        ├── Two-tier bucket replacement (O(1))
        ├── Unified evaluation cache
        └── 128-generation aging system
```

### File Structure

```
v7p3r-chess-engine/
├── src/
│   ├── v7p3r.py                    # Main engine (V17.2.0)
│   ├── v7p3r_uci.py                # UCI protocol interface
│   ├── v7p3r_openings_v161.py      # Opening book (52 positions)
│   ├── v7p3r_fast_evaluator.py     # Fast evaluation module
│   └── v7p3r_bitboard_evaluator.py # Bitboard evaluation
│
├── lichess/                        # Lichess bot integration ⭐
│   ├── config.yml                  # Bot configuration
│   ├── engines/                    # Deployed versions
│   │   ├── V7P3R_v17.1.1/
│   │   └── V7P3R_v17.2.0/
│   ├── game_records/               # PGN archives
│   └── docs/
│       ├── CLOUD_DEPLOYMENT_GUIDE.md
│       └── automation_strategy.md
│
├── testing/
│   ├── test_v17_performance.py     # Performance benchmarks
│   └── test_stockfish_match.py     # Stockfish comparison
│
├── docs/
│   ├── V17_1_STOCKFISH_BREAKTHROUGH_ANALYSIS.md  ⭐
│   ├── V17_1_BREAKTHROUGH_SUMMARY.txt
│   ├── V17_2_PERFORMANCE_PLAN.md
│   └── V7P3R_v18_0_V7P3R_DESC_SYSTEM.md  # Future AI system
│
├── build/                          # PyInstaller builds
├── requirements.txt
└── README.md                       # This file
```

### Code Statistics (V17.2.0)

- **v7p3r.py:** ~1,097 lines (core engine with optimizations)
- **v7p3r_openings_v161.py:** ~400 lines (opening book)
- **v7p3r_uci.py:** ~251 lines (UCI interface)
- **Total:** ~1,750 lines of production code

---

## 🔬 Performance Benchmarks

### V17.1 Stockfish Match (November 21, 2025)

| Metric | Value | Context |
|--------|-------|---------|
| **Overall Score** | 10.0/20 (50%) | ⭐ Rating parity achieved |
| **As White** | 5.5/10 (55%) | Effective first-move advantage |
| **As Black** | 4.5/10 (45%) | Solid defensive play |
| **Wins** | 9 games (45%) | Including 6 checkmates |
| **Draws** | 2 games (10%) | Aggressive play style |
| **Losses** | 9 games (45%) | Mostly long endgames |
| **Average Game** | 55 moves | Balanced across phases |
| **NPS** | 5,845 | Competitive in blitz |
| **Time Control** | Blitz 1+1 | Zero time losses |

### Performance by Game Phase

| Phase | Moves | V7P3R Wins | Stockfish Wins | Win Rate |
|-------|-------|------------|----------------|----------|
| Quick Tactics | <30 | 3 | 2 | 60% ⭐ |
| Middle Game | 30-60 | 4 | 3 | 57% |
| Long Endgames | 60+ | 2 | 5 | 29% ⚠️ |

### Best Opening Performance

| ECO | Opening | Games | V7P3R Score | Performance |
|-----|---------|-------|-------------|-------------|
| A40 | Queen's Pawn Game | 3 | 2.5/3 | 83% ⭐ |
| A50 | Indian Defense | 3 | 2.0/3 | 67% |
| C00 | French Defense | 2 | 1.5/2 | 75% |
| D20 | Queen's Gambit Accepted | 4 | 2.0/4 | 50% |

---

## 🐛 Known Issues & Status

### V17.2.0 Status: ✅ Production Ready

**Deployment Status:**
- ✅ Deployed to GCP e2-medium (4GB RAM, 2 vCPUs)
- ✅ Bot online and accepting challenges
- ✅ Matchmaking active
- ✅ UCI debugging metrics functioning

**Performance Notes:**
- ~5,540 NPS (slight -5% regression from v17.1 baseline)
- Trade-off: Better infrastructure for future optimizations
- TT bucket collisions offset sorting elimination gains
- Unified cache and UCI enhancements validated

**No Critical Bugs:** All v17.1 issues resolved

---

## 📈 Development Roadmap

### V17.3+ (Planned)
- [ ] Address TT bucket collision overhead
- [ ] Alternative hashing strategies
- [ ] Lazy move generation (generator pattern) for +15% NPS
- [ ] Endgame tablebase integration improvements

### V18.0 - DESC System (Future)
- [ ] **Dynamic Effort-Based Search Control (DESC)**
- [ ] AI learns V7P3R's performance characteristics
- [ ] Adaptive search depth based on position complexity
- [ ] Neural network for effort prediction (not evaluation)
- [ ] See: `docs/V7P3R_v18_0_V7P3R_DESC_SYSTEM.md`

### V18.1+ - Multi-Threading
- [ ] Lazy SMP parallel search
- [ ] Utilize 2-core e2-medium infrastructure
- [ ] Expected +80% NPS gain (1.8x speedup)
- [ ] Maintain tactical accuracy

---

## 🏆 Tournament Results

### V17.1 (November 2025) - ⭐ BREAKTHROUGH
- **Engine Battle 20251121_2:** 71.5/80 (89.4%) - 2nd place
- **vs Stockfish 1%:** 10.0/20 (50%) - Rating parity ⭐
- **vs C0BR4 v3.2:** 6.5/10 (65%)
- **vs SlowMate v3.1:** 9.5/10 (95%)
- **vs All Others:** 61.5/70 (87.9%) - Dominated field
- **Performance:** Most significant jump in development history

### V16.1 (November 2025)
- **Status:** Ready for tournament testing
- **Target:** >50% win rate vs baseline

### V12.6 (October 2025) - Previous Champion
- **Engine Battle 20251004:** 7.0/10 points (2nd place)
- **Regression Battle 20251004:** 9.0/12 points (1st place)
- **Performance:** 30x faster than V12.2 baseline

---

## 🤝 Contributing

This is a personal development project with integrated Lichess bot deployment.

**Testing Help:**
- Run V17.x in tournaments and report results
- Test new versions via Lichess challenges (@v7p3r_bot)
- Compare vs other engines (Stockfish levels, C0BR4)
- Report UCI compatibility issues
- Share interesting game PGNs

**Development Areas:**
- Search optimizations (v17.3+)
- DESC system implementation (v18.0)
- Multi-threading (v18.1+)
- Endgame improvements

**Contact:** Open issues in the repository or submit pull requests

---

## 📜 License

Personal project - educational use encouraged

---

## 🙏 Acknowledgments

- **python-chess** library by Niklas Fiekas
- **Stockfish** team for the ultimate benchmark
- **Arena Chess GUI** for tournament testing
- **Lichess** for the bot API and professional infrastructure
- **Google Cloud Platform** for reliable hosting
- Chess programming community for evaluation techniques and search algorithms

---

## 📚 Documentation

### Core Documentation
- **V17.1 Achievement:** `docs/V17_1_STOCKFISH_BREAKTHROUGH_ANALYSIS.md` ⭐
- **V17.1 Summary:** `docs/V17_1_BREAKTHROUGH_SUMMARY.txt`
- **V17.2 Plan:** `docs/V17_2_PERFORMANCE_PLAN.md`
- **V18.0 DESC:** `docs/V7P3R_v18_0_V7P3R_DESC_SYSTEM.md`

### Deployment Documentation
- **Cloud Guide:** `lichess/docs/CLOUD_DEPLOYMENT_GUIDE.md`
- **Automation:** `lichess/docs/automation_strategy.md`
- **Config:** `lichess/config.yml`

### Performance Data
- **Tournament Records:** `engine-metrics/raw_data/game_records/`
- **Analysis:** `engine-metrics/raw_data/analysis_results/`

---

**V7P3R v17.2.0 Status:** ✅ Production Ready & Cloud Deployed  
**Achievement:** 50% vs Stockfish 1% (Rating Parity) ⭐  
**Infrastructure:** Professional cloud deployment with 99.9% uptime  
**Development Focus:** Performance optimization + Future multi-threading + DESC AI system

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