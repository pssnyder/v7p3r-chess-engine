# 🎯 V7P3R V14.2 Step-by-Step Workflow

## 1. Engine Initialization
```
When V7P3R starts up:
→ Creates main V7P3REngine instance
→ Initializes unified bitboard evaluator system
→ Sets up transposition table for move caching
→ Configures search parameters (default depth, time limits)
→ Initializes performance optimization caches (V14.2):
  • Bishop value cache for dynamic piece valuation
  • Game phase cache for position classification
  • Search depth tracking for performance monitoring
→ Ready to receive UCI commands
```

## 2. Position Setup
```
When given a chess position:
→ Receives FEN string or move sequence from UCI
→ Creates python-chess Board object
→ Validates position legality
→ Detects game phase (opening/middlegame/endgame) with caching
→ Ready for optimized move search
```

## 3. Move Search Process (The Core Engine Loop)

### Step 3a: Game Phase Detection (V14.2 NEW)
```
→ Calculate material hash for cache lookup
→ If cached: Return stored game phase
→ If not cached: Analyze material and move count
  • Opening: < 8 moves + material ≥ 5000
  • Endgame: Material ≤ 2500
  • Middlegame: Everything else
→ Cache result for future use
→ Set phase-specific search parameters
```

### Step 3b: Advanced Time Management (V14.2 ENHANCED)
```
→ Calculate base time allocation
→ Apply game phase factors:
  • Opening: 0.8x (faster, rely on patterns)
  • Middlegame: 1.3x (complex tactics need time)
  • Endgame: 1.1x (precision needed)
→ Detect critical positions:
  • In check
  • Major piece captures available
  • Few legal moves (≤8)
→ Apply critical position bonus: 1.4x time
→ Set target depth based on phase:
  • Opening: 8 ply
  • Middlegame: 10 ply (target)
  • Endgame: 12 ply
```

### Step 3c: Move Generation
```
→ Generate all legal moves for current position
→ Typically 20-40 moves in opening/middlegame
→ Each move represents a possible choice
```

### Step 3d: Streamlined Move Ordering (V14.2 OPTIMIZED)
```
For each legal move, assign to category:

1. 🏆 TT Move (from transposition table)
   → Previous best move from identical position

2. 💥 Captures (MVV-LVA + Tactical Bonus)
   → Most Valuable Victim - Least Valuable Attacker
   → Uses CACHED dynamic bishop values (325/275)
   → Enhanced with bitboard tactical detection

3. ⚡ Checks
   → Moves that give check to opponent king
   → Enhanced with tactical pattern bonus

4. 🎯 Killer Moves
   → Previously successful moves at this depth
   → Fast lookup using pre-built set

5. 🚀 Development Moves
   → Knights/bishops from starting squares
   → 50 point bonus for piece activity

6. 👑 Pawn Advances
   → Forward pawn moves
   → 10 point base bonus

7. 🧠 Tactical Moves
   → High bitboard tactical score (>20)
   → Advanced pattern recognition

8. 😴 Quiet Moves
   → All remaining moves
   → Sorted by history heuristic

REMOVED FROM V14.1: Expensive threat detection
PERFORMANCE: ~6x faster than V14.1 (0.6ms vs 3-4ms)
```

### Step 3e: Iterative Deepening with Enhanced Monitoring (V14.2)
```
Start with depth 1, increase until time limit:

For each depth level:
→ Apply time prediction and management
→ Search all moves at current depth
→ Track iteration completion time
→ Monitor depth achievement
→ Apply dynamic time adjustment based on score stability
→ Store depth achieved for performance analysis

Enhanced Features:
→ Conservative iteration prediction (2.5x last iteration time)
→ Score stability detection (±50cp = stable, ±200cp = unstable)
→ Dynamic time reallocation based on position stability
→ Comprehensive depth distribution tracking
```

### Step 3f: Alpha-Beta Search with Game Phase Awareness
```
For each move in optimized order:
→ Apply move to board
→ Recursively search opponent response
→ Use alpha-beta pruning for efficiency
→ Apply Late Move Reduction (LMR) for deep moves
→ Use cached evaluations when possible
→ Phase-aware evaluation adjustments

V14.2 Enhancements:
→ Cached dynamic piece values for speed
→ Game phase context in evaluation
→ Reduced overhead from move ordering
→ Better time management prevents search interruption
```

### Step 3g: Enhanced Quiescence Search (V14.2)
```
When reaching depth limit, analyze tactical moves:

Game Phase Adaptive Limits:
→ Opening: 4 ply quiescence (fast tactical resolution)
→ Middlegame: 6 ply quiescence (standard tactical depth)
→ Endgame: 8 ply quiescence (precise endgame calculation)

Enhanced Move Selection:
→ All captures (with delta pruning)
→ All checks
→ Endgame: Pawn promotions + king moves (when depth > -3)

Performance Optimizations:
→ Delta pruning: Skip captures that can't improve alpha + 200cp
→ Search width limiting: Max 8 moves in very deep quiescence
→ Good trade bonus: +50 for captures with victim ≥ attacker value
→ Endgame check bonus: +25 additional points for checks

Tactical Move Ordering:
→ MVV-LVA with cached dynamic values
→ Promotion moves get highest priority
→ Endgame-specific adjustments
```

## 4. Position Evaluation with Game Phase Detection

### Step 4a: Game Phase-Specific Evaluation
```
Based on detected game phase:

Opening Phase:
→ Emphasize development and king safety
→ Faster evaluation (fewer complex calculations)
→ Pattern-based adjustments

Middlegame Phase:
→ Full tactical analysis
→ Complete material counting with dynamic values
→ Advanced positional factors

Endgame Phase:
→ King activity becomes crucial
→ Pawn promotion potential
→ Precise material evaluation
→ Deeper tactical calculation
```

### Step 4b: Cached Dynamic Piece Values (V14.2)
```
For each piece type:
→ Standard values: P=100, N=300, R=500, Q=900
→ Dynamic Bishop Valuation:
  • Check cache first (board material hash)
  • Bishop pair: 325 each (better than knights)
  • Single bishop: 275 each (worse than knight)
  • Cache result for repeated access

Performance: O(1) lookup instead of O(n) calculation
```

### Step 4c: Unified Bitboard Evaluation
```
→ Consolidated evaluation system from V14.0
→ Advanced tactical pattern detection
→ Positional factor analysis
→ King safety assessment
→ All enhanced with game phase awareness
```

## 5. Best Move Selection and Performance Monitoring

### Step 5a: Move Selection
```
→ Choose move with highest evaluation score
→ Apply final time management checks
→ Record depth achieved for this move
→ Update performance statistics
```

### Step 5b: Performance Monitoring (V14.2 NEW)
```
Track comprehensive performance metrics:
→ Search depth achieved per move
→ Cache hit rates (bishop values, game phases)
→ Nodes per second calculation
→ Time allocation effectiveness
→ Depth distribution analysis
→ Game phase transition tracking

Generate performance reports:
→ Average search depth
→ Cache efficiency
→ Search speed metrics
→ Depth consistency analysis
```

## 6. UCI Output with Enhanced Information
```
Standard UCI output:
→ info depth X score cp Y nodes Z time T nps N pv [moves]

V14.2 Enhanced output:
→ Game phase information
→ Target depth vs achieved depth
→ Iteration timing data
→ Performance metrics
→ Cache utilization stats
```

## 7. Key Performance Improvements in V14.2

### Overhead Elimination
- ❌ **Removed**: Expensive per-move threat detection
- ❌ **Removed**: Board copying in move ordering  
- ❌ **Removed**: Complex 10-category move ordering
- ✅ **Added**: Efficient caching systems
- ✅ **Result**: 6x faster move ordering

### Smart Optimizations
- ✅ **Game Phase Detection**: Adaptive evaluation and search
- ✅ **Enhanced Quiescence**: Deeper, smarter tactical search
- ✅ **Advanced Time Management**: Consistent 10-ply targeting
- ✅ **Performance Monitoring**: Continuous optimization data

### Expected Results
- 🎯 **Performance Recovery**: V14.0 level or better (70-75% tournament score)
- 🎯 **Consistent Depth**: 10-ply in standard time controls
- 🎯 **Better Tactics**: Enhanced quiescence search
- 🎯 **Adaptive Play**: Game phase-specific strategies

---
*V14.2 represents a performance-first optimization focused on eliminating V14.1's overhead while adding intelligent enhancements that improve search effectiveness without computational cost.*