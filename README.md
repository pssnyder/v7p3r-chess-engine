# V7P3R Chess Engine v10.0

A high-performance UCI chess engine featuring unified search architecture and advanced tactical evaluation.

## Current Version: v10.0 (Tournament Ready)

**Built**: August 31, 2025  
**Status**: 🏆 **Tournament Ready**  
**Architecture**: Unified Search with Complete Feature Set

### 🎯 Tournament Performance Metrics
- **Puzzle Accuracy**: 94.8% (417/440 tactical puzzles solved)
- **Top-5 Hit Rate**: 97.7% (finds best move in top 5 candidates)
- **Tactical Solution Rate**: 87.5% (complex position resolution)
- **Time Management**: Robust (timeout issues resolved)
- **UCI Compliance**: Full tournament standard

## 🚀 V10.0 Major Achievements

### **Unified Search Architecture** ✅
**The Core Innovation**: Single search function containing ALL advanced features:
- **Alpha-Beta Pruning**: Negamax framework with proper bounds
- **Transposition Table**: Zobrist hashing with 35k+ entries, 22% hit rate
- **Killer Moves**: Depth-specific non-capture move optimization
- **History Heuristic**: Move success tracking with depth weighting
- **Advanced Move Ordering**: TT move → Captures → Killers → History → Quiet
- **Quiescence Search**: Tactical stability in volatile positions

### **Performance Optimization**
- **Search Speed**: ~11k NPS with full intelligence features
- **Evaluation Cache**: 32k+ positions cached with strategic hit optimization
- **Memory Management**: Efficient transposition and evaluation caching
- **Time Management**: Robust cutoff handling for tournament play

### **Tactical Intelligence**
- **Bitboard Evaluation**: High-speed position assessment
- **Pattern Recognition**: Knight forks, pins, tactical motifs
- **PV Following**: Principal variation optimization
- **Mate Scoring**: Proper mate-in-N evaluation
- **King Safety**: Attack counting and defensive evaluation

## 🏗️ Technical Architecture

### Core Engine Components
- **`v7p3r.py`**: Main engine with unified search (731 lines)
- **`v7p3r_uci.py`**: UCI interface for tournament play (143 lines)
- **`v7p3r_bitboard_evaluator.py`**: High-performance position evaluation
- **`v7p3r_bitboard.py`**: Bitboard operations and utilities
- **`v7p3r_scoring_calculation.py`**: Advanced scoring algorithms

### Search Features
```python
def _unified_search(board, depth, alpha, beta):
    """Single search function with ALL features:
    - Transposition table lookups
    - Killer move ordering
    - History heuristic
    - Quiescence search
    - Proper mate scoring
    """
```

### Advanced Heuristics
- **Transposition Table**: Fixed-seed Zobrist for reproducible results
- **Killer Moves**: 2 killers per depth, non-capture beta cutoffs
- **History Heuristic**: Depth-weighted move success tracking
- **Move Ordering**: Complete priority system for optimal search

## 📊 Version Evolution

| Version | Key Features | Tournament Score | Notes |
|---------|-------------|------------------|-------|
| **v10.0** | Unified Search, Complete Features | **🎯 Tournament Ready** | Current release |
| v9.6 | Search unification, Time fixes | 94.8% puzzle accuracy | Puzzle validation |
| v9.2 | UCI improvements | 36.3% vs field | Infrastructure focus |
| v7.0 | Original foundation | 79.5% vs field | Proven baseline |

### Historical Tournament Results
- **v7.0**: 79.5% win rate (17.5/22 games) - Best historical performance
- **v9.2**: 36.3% win rate - UCI reliability but chess regression
- **v10.0**: Built on v7.0 foundation + v9.x technical advances

## 🎮 Usage

### Tournament Play (UCI)
```bash
# Direct UCI interface
./V7P3R_v10.0.exe

# Arena GUI integration
Engine: V7P3R v10.0
Protocol: UCI
Author: Pat Snyder
```

### Development/Testing
```python
# Interactive play
python src/play_chess.py

# Puzzle validation
python testing/v7p3r_puzzle_analyzer.py

# Engine testing
python testing/comprehensive_engine_test.py
```

## 🏆 Tournament Configuration

**Engine Specifications**:
- **Type**: UCI Chess Engine
- **Language**: Python 3.12 (PyInstaller executable)
- **Search Depth**: Up to 6 plies with iterative deepening
- **Time Control**: Adaptive with robust timeout handling
- **Memory**: Transposition table + evaluation cache

**Validated Features**:
- ✅ UCI protocol compliance
- ✅ Stable time management (no timeouts)
- ✅ Tactical accuracy (94.8% puzzle success)
- ✅ Zero known crashes in testing
- ✅ Ready for Arena tournaments and engine competitions

## 📈 Development Philosophy

**"Unified Architecture + Proven Chess Knowledge"**

V10.0 combines the architectural advances of v9.x with the proven chess strength of v7.0:
- **Foundation**: v7.0's tournament-proven evaluation (79.5% win rate)
- **Architecture**: v9.x unified search with all advanced features
- **Performance**: Optimized for tournament reliability and tactical accuracy

**Next Steps**: Tournament deployment, ELO rating establishment, competitive validation

---

**Ready for**: Arena tournaments, engine competitions, rating matches, and chess AI research.
