# V7P3R v18.0 - Dynamic Effort-Based Search Control (DESC) System
**Planning Document - Pre-Implementation Design**  
**Target Version:** v18.0+  
**Status:** Concept & Requirements Phase  
**Date:** November 22, 2025

---

## Executive Summary

### Vision Statement

Transform V7P3R from a static heuristic engine into an **intelligent, self-aware agent** that dynamically manages its own computational effort based on positional complexity and learned performance characteristics.

**Core Innovation:** Instead of using AI to learn chess strategy (like NNUE), use AI to learn **V7P3R's own performance profile** - teaching the engine when to "think hard" vs "think fast" based on its strengths and weaknesses.

### Key Principle

**"Efficiency over Exhaustion"**  
Focus computational effort where positions demand complexity; simplify quickly where confidence is high.

### The Paradigm Shift

**From Static Execution:**
```
"If this AND this BUT NOT that → Score X"
"Repeat exhaustively for all positions"
```

**To Dynamic Agency:**
```
"These details matter, these don't"
"This is my strength - process quickly"
"This confuses me - allocate more time"
"Only do the work I need, skip redundant checks"
"Proceed with evaluated options unless situation drastically changes"
```

---

## Problem Context

### Current State (v17.x)

V7P3R v17.1 achieved **50% score vs Stockfish 1%** - a breakthrough performance. However, analysis reveals:

**Strengths:**
- ✅ Tactical middle games (57% win rate, 30-60 moves)
- ✅ Quick tactical sequences (60% win rate, <30 moves)
- ✅ Speed under pressure (5,845 NPS in blitz)

**Weaknesses:**
- ⚠️ Long endgames (29% win rate, 60+ moves) - Stockfish won 5 of 7
- ⚠️ Complexity paradox: Advanced heuristics cost more in NPS than they return in accuracy
- ⚠️ Static search depth regardless of position complexity

### The Complexity Paradox

**Historical Evidence:**
- **v9.x:** Performance degradation from excessive complexity checks
- **v14.2:** `gives_check()` function cost analysis showed high overhead
- **v15.5:** Evaluation volatility in unstable positions
- **v17.1:** Perfect tactical vision but resource waste in simple positions

**Core Issue:** V7P3R treats all positions equally, spending the same effort on "book positions" as on complex tactical battles.

---

## The DESC System Concept

### What is DESC?

**Dynamic Effort-Based Search Control (DESC)** is an AI-augmented meta-intelligence layer that predicts the computational cost and required accuracy for each position, then dynamically adjusts search parameters.

### How DESC Differs from NNUE

| Aspect | NNUE | DESC (V7P3R) |
|--------|------|--------------|
| **Purpose** | Learn chess evaluation | Learn engine performance |
| **Training Data** | Game outcomes | V7P3R profiling metrics |
| **Output** | Position score (cp) | Positional Effort Score (PES) |
| **Use Case** | Replace static eval | Control search depth/time |
| **Goal** | Better chess understanding | Better resource allocation |

### Core Philosophy

**Do not change V7P3R's chess strategy.**  
- Same openings
- Same piece calculations  
- Same positional insights

**Change only how much EFFORT V7P3R invests in each position:**
- Think dynamically when needed
- Speed up when confident
- Manage risk in weaknesses

---

## The Positional Effort Score (PES)

### Definition

**PES** is a single, normalized value (0.0 to 1.0) that predicts:
1. **Cost:** Time/resources required for V7P3R to search this position
2. **Risk:** Accuracy required to avoid blunders

**Formula:** `PES = f(Positional_Features, Historical_Performance)`

### PES Input Features

The AI model correlates easily measurable position characteristics with V7P3R's actual observed performance.

#### Category A: Positional Complexity Features
*"What is the board doing?"*

| Feature | Metric | V7P3R Context & Rationale |
|---------|--------|---------------------------|
| **Move Options** | Legal Move Count | High count = High decision width = High effort required |
| **Tactical Density** | # of Captures & Checks | High forcing moves = High volatility/risk = High accuracy needed |
| **Structural Complexity** | # of Doubled/Isolated Pawns | More complex positional features to evaluate (v12.3 refactoring impact) |
| **King Safety** | # of Attackers on King Zone | High threat level = Increased complexity and priority |
| **Piece Cohesion** | # of Undefended Pieces | Low cohesion = High blunder risk (v15.5 analysis showed struggles) |
| **Material Balance** | Absolute Material Difference | Large imbalances change evaluation complexity |
| **Pawn Structure** | Pawn Chain Integrity Score | Complex pawn structures increase heuristic cost |

#### Category B: Engine Effort Features  
*"How hard is V7P3R working?"*

| Feature | Metric | V7P3R Context & Rationale |
|---------|--------|---------------------------|
| **Search Efficiency** | Transposition Table Hit Rate | High hit rate = Position is "known" (Low effort). Low = Novel position (High effort) |
| **Evaluation Cost** | Time to Execute `_evaluate_position()` | Direct measure of computational cost for heuristic chain |
| **Pruning Effectiveness** | Move Ordering Pruning % | Low pruning = High search effort (v13.x analysis critical) |
| **Evaluation Volatility** | Score Change (Depth N vs N+1) | Large swings = Unstable/unclear position (High effort/risk - v15.5) |
| **Node Expansion Rate** | Nodes/Second (NPS) | Real-time performance indicator |
| **TT Collisions** | Hash Collision Rate | High collisions = Wasted effort recomputing |

### PES Interpretation Scale

| PES Range | Classification | Interpretation |
|-----------|----------------|----------------|
| **0.0 - 0.2** | Very Low | Simple, well-understood position. Minimal effort needed. |
| **0.2 - 0.4** | Low | Standard position. Trust heuristics, use default search. |
| **0.4 - 0.6** | Medium | Balanced complexity. Full search with all heuristics. |
| **0.6 - 0.8** | High | Complex/tactical position. Increase depth and time. |
| **0.8 - 1.0** | Very High | Critical position or V7P3R weakness. Maximum resources or fallback mode. |

---

## Implementation Phases

### Phase 1: Data Instrumentation & Profiling
**Goal:** Generate training dataset correlating positions with V7P3R's actual performance

#### Task 1.1: Augment V7P3R Engine
**Location:** `src/v7p3r.py`

**Modifications:**
1. **In `_evaluate_position()`:**
   ```python
   # Record evaluation metrics
   start_time = time.perf_counter()
   score = self._calculate_score(board)
   eval_time = time.perf_counter() - start_time
   
   # Log feature data at root node (ply 0)
   if self.current_ply == 0:
       self.profiler.record_position({
           'fen': board.fen(),
           'legal_moves': len(list(board.legal_moves)),
           'captures': len([m for m in board.legal_moves if board.is_capture(m)]),
           'checks': len([m for m in board.legal_moves if board.gives_check(m)]),
           'eval_time': eval_time,
           'tt_hit_rate': self.tt_hits / (self.tt_hits + self.tt_misses),
           'score': score,
           # ... additional features
       })
   ```

2. **In `_search()`:**
   ```python
   # Track search performance metrics
   nodes_before = self.nodes_searched
   pruned_moves = 0
   
   # ... search logic ...
   
   if ply == 0:
       self.profiler.record_search({
           'nodes': self.nodes_searched - nodes_before,
           'nps': self.calculate_nps(),
           'pruning_rate': pruned_moves / total_moves,
           'eval_volatility': abs(current_score - previous_score)
       })
   ```

#### Task 1.2: Create Profiling Mode
**New File:** `testing/ai_profiler.py`

**Features:**
- UCI command: `go profile <n_positions>`
- Executes search at depths 1-6 for each position
- Records all Category A + B features
- Exports to `data/effort_dataset.csv`

**Usage:**
```bash
python testing/ai_profiler.py --pgn lichess/game_records/*.pgn --output data/effort_dataset.csv
```

#### Task 1.3: Dataset Generation
**Data Sources:**
1. Existing Lichess game records (736+ games)
2. Tactical puzzle databases (Lichess puzzles)
3. Opening book positions
4. Endgame tablebase positions

**Target Size:** 1-5 million position samples

**Output Schema:**
```csv
fen,legal_moves,captures,checks,doubled_pawns,isolated_pawns,attackers_king,undefended_pieces,
tt_hit_rate,eval_time_us,pruning_rate,eval_volatility,nps,depth_reached,score,result
```

---

### Phase 2: AI Model Training
**Goal:** Train lightweight model to predict PES from position features

#### Task 2.1: Define Target Variable

**Primary Target:** `log(avg_time_per_node)`
- Log-scale ensures faster computation is rewarded
- Normalizes extreme outliers (very fast vs very slow positions)

**Alternative Targets (for ensemble):**
- `eval_time_us` - Direct evaluation cost
- `nodes_searched / time_allocated` - Search efficiency
- `pruning_rate_inverse` - Difficulty of pruning

#### Task 2.2: Model Architecture Selection

**Recommended Approach:** Dual Model System

**Model 1: Fast Classifier (Primary)**
- **Type:** Gradient Boosted Tree (XGBoost or LightGBM)
- **Purpose:** Classify position into PES bins (0.0-0.2, 0.2-0.4, etc.)
- **Inference Time:** <0.5 ms on CPU
- **Accuracy Target:** 80%+ classification accuracy

**Model 2: Precise Regressor (Secondary)**
- **Type:** Shallow Neural Network (2-3 layers, 32-64 neurons)
- **Purpose:** Fine-tune PES within classified bin
- **Inference Time:** <1 ms on CPU
- **Use Case:** Critical positions (PES > 0.6)

**Architecture Rationale:**
- Mirrors NNUE philosophy: Speed > Perfection
- Classifier quickly bins 80% of positions
- Regressor adds precision only when needed

**Neural Network Structure:**
```
Input Layer:   [Position Features] → 128 neurons
Hidden Layer 1: ReLU → 64 neurons  
Hidden Layer 2: ReLU → 32 neurons
Output Layer:   Sigmoid → PES (0.0 to 1.0)
```

#### Task 2.3: Training Pipeline
**New File:** `ai/train_desc_model.py`

**Steps:**
1. Load `effort_dataset.csv`
2. Feature engineering:
   - Normalize all numeric features
   - One-hot encode categorical features (rare)
   - Create interaction features (e.g., `captures × king_attackers`)
3. Train/validation split (80/20)
4. Train classifier (XGBoost):
   ```python
   import xgboost as xgb
   
   model = xgb.XGBClassifier(
       max_depth=6,
       n_estimators=100,
       learning_rate=0.1,
       objective='multi:softmax',
       num_class=5  # 5 PES bins
   )
   ```
5. Train regressor (PyTorch):
   ```python
   class PESRegressor(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(input_features, 128)
           self.fc2 = nn.Linear(128, 64)
           self.fc3 = nn.Linear(64, 32)
           self.output = nn.Linear(32, 1)
           
       def forward(self, x):
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = F.relu(self.fc3(x))
           return torch.sigmoid(self.output(x))
   ```

#### Task 2.4: Model Export & Deployment
**Export Formats:**
1. **XGBoost Classifier:** JSON format (for fast loading)
2. **NN Regressor:** ONNX format (cross-platform, optimized)
3. **Fallback:** Pickle files (Python-specific)

**Deployment Location:** `ai/models/desc_v1.0/`

**Required Files:**
- `desc_classifier.json` (XGBoost)
- `desc_regressor.onnx` (Neural network)
- `feature_scaler.pkl` (Normalization parameters)
- `model_metadata.json` (Version, training date, accuracy metrics)

---

### Phase 3: DESC Controller Integration
**Goal:** Use predicted PES to dynamically control search parameters

#### Task 3.1: Create DESC Controller Module
**New File:** `src/v7p3r_desc_controller.py`

**Core Responsibilities:**
1. Load trained AI models
2. Extract position features from board state
3. Predict PES in <1ms
4. Return search control parameters

**Class Structure:**
```python
class DESCController:
    def __init__(self, model_path='ai/models/desc_v1.0/'):
        self.classifier = xgb.Booster()
        self.classifier.load_model(f'{model_path}/desc_classifier.json')
        self.regressor = onnx.load(f'{model_path}/desc_regressor.onnx')
        self.scaler = pickle.load(open(f'{model_path}/feature_scaler.pkl', 'rb'))
        
    def extract_features(self, board: chess.Board) -> np.array:
        """Extract all Category A + B features from position"""
        features = {
            'legal_moves': len(list(board.legal_moves)),
            'captures': sum(1 for m in board.legal_moves if board.is_capture(m)),
            'checks': sum(1 for m in board.legal_moves if board.gives_check(m)),
            # ... all features from PES Input table
        }
        return self.scaler.transform([list(features.values())])
        
    def predict_pes(self, board: chess.Board) -> float:
        """Predict Positional Effort Score"""
        features = self.extract_features(board)
        
        # Fast classification
        pes_bin = self.classifier.predict(features)[0]
        
        # Refine with regressor if high complexity
        if pes_bin >= 3:  # PES > 0.6
            pes_precise = self.regressor.run(None, {'input': features})[0]
            return float(pes_precise)
        
        # Return bin center for low/medium complexity
        return (pes_bin * 0.2) + 0.1
        
    def get_search_params(self, pes: float) -> dict:
        """Return dynamic search control parameters"""
        if pes < 0.2:
            return self._turbo_mode()
        elif pes < 0.4:
            return self._fast_mode()
        elif pes < 0.6:
            return self._standard_mode()
        elif pes < 0.8:
            return self._deep_mode()
        else:
            return self._critical_mode()
```

#### Task 3.2: Define Search Control Strategies

**Strategy Table:**

| PES Range | Mode | LMR Adjustment | Time Allocation | Evaluation Heuristics | Min/Max Depth |
|-----------|------|----------------|-----------------|----------------------|---------------|
| **0.0-0.2** | Turbo | +2x reduction (start at move 3) | 0.5x normal, enable PV early exit | Disable: mobility, advanced pawn | Min: 4, Max: 8 |
| **0.2-0.4** | Fast | +1x reduction (start at move 5) | 0.75x normal | Disable: mobility only | Min: 5, Max: 10 |
| **0.4-0.6** | Standard | Default (start at move 8) | 1.0x normal | Enable all | Min: 6, Max: 12 |
| **0.6-0.8** | Deep | Reduce by 50% (less aggressive) | 1.25x normal | Enable all + extensions | Min: 8, Max: 14 |
| **0.8-1.0** | Critical | Disable LMR completely | 1.5x normal OR hard cap at 5s | Fallback: disable expensive heuristics | Min: 6, Max: 10 |

**Implementation:**
```python
def _turbo_mode(self):
    return {
        'lmr_threshold': 3,  # Start LMR earlier
        'lmr_reduction': 2,  # More aggressive reduction
        'time_factor': 0.5,
        'min_depth': 4,
        'max_depth': 8,
        'enable_heuristics': ['basic_material', 'king_safety', 'simple_pawn'],
        'enable_extensions': [],
        'enable_early_exit': True
    }

def _critical_mode(self):
    return {
        'lmr_threshold': 999,  # Effectively disable
        'lmr_reduction': 0,
        'time_factor': 1.5,
        'time_hard_cap': 5.0,  # seconds
        'min_depth': 6,
        'max_depth': 10,
        'enable_heuristics': ['basic_material', 'king_safety'],  # Minimal set
        'enable_extensions': ['check', 'capture', 'single_reply'],
        'enable_early_exit': False
    }
```

#### Task 3.3: Integrate into V7P3R Search
**Location:** `src/v7p3r.py` - `_search()` method

**Integration Points:**

**1. Pre-Search (Root Node):**
```python
def _search(self, board, depth, alpha, beta, ply=0):
    # DESC integration at root
    if ply == 0:
        pes = self.desc_controller.predict_pes(board)
        self.search_params = self.desc_controller.get_search_params(pes)
        
        # Log PES for analysis
        self.uci_info(f"DESC PES: {pes:.2f} Mode: {self.search_params['mode']}")
```

**2. LMR Decision:**
```python
# Use DESC-controlled LMR threshold
if (move_count >= self.search_params['lmr_threshold'] and 
    depth >= 3 and 
    not board.is_capture(move) and 
    not board.gives_check(move)):
    
    reduction = self.search_params['lmr_reduction']
    score = -self._search(board, depth - 1 - reduction, -beta, -alpha, ply + 1)
```

**3. Time Management:**
```python
def _should_stop_search(self):
    elapsed = time.time() - self.search_start_time
    
    # DESC-controlled time limits
    time_budget = self.base_time * self.search_params['time_factor']
    
    if 'time_hard_cap' in self.search_params:
        time_budget = min(time_budget, self.search_params['time_hard_cap'])
    
    return elapsed >= time_budget
```

**4. Selective Evaluation:**
```python
def _evaluate_position(self, board):
    score = 0
    
    # Always execute enabled heuristics
    for heuristic in self.search_params['enable_heuristics']:
        score += self.heuristics[heuristic](board)
    
    # Skip disabled expensive heuristics in turbo/critical modes
    return score
```

---

### Phase 4: Validation & Iteration
**Goal:** Prove DESC improves performance and playing strength

#### Validation Metrics

**1. Effective NPS (eNPS)**
```
eNPS = (Nodes Searched / Time Spent) × Search Quality Factor

Search Quality Factor = (Positions Evaluated at Target Depth / Total Positions)
```

**Target:** +15% improvement over v17.1 baseline (5,845 → 6,720 NPS)

**2. Time Savings Rate**
```
Time Savings = (Moves resolved in <0.2s) / (Total Moves) × 100%
```

**Target:** +10% increase (v17.1 baseline TBD)

**3. Resource Allocation Efficiency**
```
Efficiency = (Time spent in High PES) / (Time spent in Low PES)
```

**Target:** Ratio > 3:1 (spend 3x more time on complex positions)

**4. Depth Consistency**
```
Critical Position Depth = Avg depth reached when PES > 0.6
```

**Target:** Consistent depth 8+ in critical positions

**5. Evaluation Stability**
```
Volatility = StdDev(Score changes in positions where PES < 0.4)
```

**Target:** <50 centipawn volatility in simple positions

**6. Blunder Rate**
```
Blunder Rate = (Moves with eval drop > 200cp) / (Total Moves) × 100%
```

**Target:** <2% blunder rate (maintain v17.1 tactical sharpness)

**7. Tournament Strength**
```
Elo Gain = V18.0_DESC rating - V17.1 rating
```

**Target:** +50 Elo minimum (proof of concept), +100 Elo ideal

#### Validation Test Suite
**New File:** `testing/test_desc_validation.py`

**Test Categories:**

**1. PES Prediction Accuracy**
```python
def test_pes_prediction_speed():
    """Verify PES inference <1ms"""
    assert predict_time < 0.001  # 1ms

def test_pes_classification_accuracy():
    """Verify 80%+ accuracy on validation set"""
    assert accuracy > 0.80
```

**2. Search Control Validation**
```python
def test_turbo_mode_speed():
    """Simple positions should search faster"""
    simple_positions = load_simple_test_set()
    for pos in simple_positions:
        time_v17 = benchmark_v17(pos)
        time_v18 = benchmark_v18_desc(pos)
        assert time_v18 < time_v17 * 0.8  # 20% faster

def test_deep_mode_depth():
    """Complex positions should reach deeper"""
    tactical_positions = load_tactical_test_set()
    for pos in tactical_positions:
        depth_v17 = benchmark_v17_depth(pos)
        depth_v18 = benchmark_v18_desc_depth(pos)
        assert depth_v18 >= depth_v17 + 1  # +1 depth minimum
```

**3. Playing Strength Validation**
```python
def test_tournament_vs_v17():
    """DESC should improve tournament performance"""
    results = run_tournament(
        engines=['V7P3R_v17.1', 'V7P3R_v18.0_DESC'],
        games=100,
        time_control='blitz 1+1'
    )
    assert results['V7P3R_v18.0_DESC']['elo'] > results['V7P3R_v17.1']['elo'] + 25
```

**4. Regression Testing**
```python
def test_no_tactical_regression():
    """DESC should not harm tactical accuracy"""
    tactical_suite = load_tactical_puzzles()
    
    accuracy_v17 = solve_rate_v17(tactical_suite)
    accuracy_v18 = solve_rate_v18_desc(tactical_suite)
    
    assert accuracy_v18 >= accuracy_v17 * 0.95  # Allow 5% tolerance
```

#### Continuous Feedback Loop

**Stage 1: Initial Deployment (v18.0-alpha)**
- Enable DESC logging in all games
- Record: PES, actual time spent, depth reached, move quality
- Compare predictions vs reality

**Stage 2: Model Refinement (v18.1)**
- Retrain DESC models using real-world data from v18.0-alpha
- Adjust PES thresholds based on observed performance
- Fine-tune search control parameters

**Stage 3: Optimization (v18.2)**
- Implement PES caching (similar FENs get cached PES)
- Add PES gradient analysis (is complexity increasing/decreasing?)
- Introduce adaptive thresholds (learn optimal PES bins per time control)

**Stage 4: Advanced Features (v18.3+)**
- Multi-model ensemble (different models for opening/middle/endgame)
- Opponent modeling (adjust PES based on opponent's playing style)
- Self-play reinforcement (DESC learns from its own tournament games)

---

## Technical Architecture

### File Structure
```
v7p3r-chess-engine/
├── src/
│   ├── v7p3r.py                    # Core engine (DESC integration points)
│   ├── v7p3r_desc_controller.py    # NEW: DESC control module
│   ├── v7p3r_uci.py                # UCI interface (DESC commands)
│   └── ... (existing modules)
│
├── ai/
│   ├── models/
│   │   └── desc_v1.0/
│   │       ├── desc_classifier.json
│   │       ├── desc_regressor.onnx
│   │       ├── feature_scaler.pkl
│   │       └── model_metadata.json
│   │
│   ├── train_desc_model.py         # NEW: Training pipeline
│   └── evaluate_desc_model.py      # NEW: Model evaluation
│
├── data/
│   ├── effort_dataset.csv          # NEW: Training data
│   └── desc_validation_set.csv     # NEW: Validation data
│
├── testing/
│   ├── ai_profiler.py              # NEW: Data collection tool
│   ├── test_desc_validation.py     # NEW: DESC test suite
│   └── benchmark_desc.py           # NEW: Performance comparison
│
└── docs/
    ├── V7P3R_v18_0_V7P3R_DESC_SYSTEM.md  # THIS DOCUMENT
    └── V18_0_DESC_IMPLEMENTATION_LOG.md  # NEW: Implementation progress
```

### Dependencies

**New Python Packages:**
```
xgboost>=1.7.0          # Gradient boosting for classifier
lightgbm>=4.0.0         # Alternative GBM (optional)
torch>=2.0.0            # Neural network regressor
onnx>=1.14.0            # Model export/inference
onnxruntime>=1.15.0     # Fast ONNX inference
scikit-learn>=1.3.0     # Feature scaling, metrics
pandas>=2.0.0           # Dataset manipulation
numpy>=1.24.0           # Numerical operations
matplotlib>=3.7.0       # Visualization (optional)
```

**Update `requirements.txt`:**
```bash
# Add to existing requirements
xgboost>=1.7.0
torch>=2.0.0
onnx>=1.14.0
onnxruntime>=1.15.0
```

### UCI Interface Extensions

**New UCI Commands:**

```
setoption name UseDesc value true|false
  Enable/disable DESC system (default: false for compatibility)

setoption name DescVerbose value true|false
  Enable verbose DESC logging (PES values, mode selection)

setoption name DescModelPath value <path>
  Override default model path (default: ai/models/desc_v1.0/)

go profile <n>
  Run profiling mode: analyze N positions and export training data
  
info desc pes <value> mode <mode>
  Output PES prediction and selected mode during search
```

**Example UCI Session:**
```
uci
setoption name UseDesc value true
setoption name DescVerbose value true
position startpos moves e2e4
go movetime 5000

info desc pes 0.45 mode Standard
info depth 8 seldepth 10 score cp 35 nodes 48521 nps 9704 time 5000 pv e7e5
bestmove e7e5
```

---

## Risk Mitigation

### Potential Risks & Mitigation Strategies

**Risk 1: DESC Overhead Negates Performance Gains**
- **Mitigation:** PES prediction must complete in <1ms (0.017% of 1s move time)
- **Validation:** Benchmark DESC overhead separately
- **Fallback:** Implement DESC caching for repeated positions

**Risk 2: Model Overfits to Training Data**
- **Mitigation:** Use diverse dataset (openings, middle games, endgames, puzzles)
- **Validation:** Cross-validation with held-out games
- **Fallback:** Regularization (dropout, L2 penalty) in NN training

**Risk 3: DESC Misclassifies Critical Positions**
- **Mitigation:** Conservative PES thresholds (bias toward deeper search when uncertain)
- **Validation:** Test on tactical puzzle suite (must maintain 95%+ solve rate)
- **Fallback:** Manual override for known critical patterns (e.g., all checks get PES > 0.6)

**Risk 4: Training Data Bias (Lichess games only)**
- **Mitigation:** Include tactical puzzles, opening theory, endgame positions
- **Validation:** Test DESC performance across different game phases
- **Fallback:** Phase-specific models (opening, middle, endgame DESC models)

**Risk 5: DESC Reduces Tactical Sharpness**
- **Mitigation:** Deep mode (PES > 0.6) must search deeper than v17.1 baseline
- **Validation:** Regression testing on tactical benchmarks
- **Fallback:** Disable DESC in critical time controls (e.g., bullet < 1 minute)

**Risk 6: Model Deployment Complexity**
- **Mitigation:** Use standard formats (ONNX) with mature libraries
- **Validation:** Cross-platform testing (Windows, Linux, cloud)
- **Fallback:** Provide pure-Python fallback implementation

---

## Success Criteria

### Minimum Viable Product (MVP) - v18.0-alpha

**Must Have:**
- ✅ PES prediction in <1ms
- ✅ 5 search control modes implemented (Turbo, Fast, Standard, Deep, Critical)
- ✅ No regression in tactical accuracy (<5% tolerance)
- ✅ Measurable performance improvement (any positive eNPS gain)

**Should Have:**
- ✅ +10% time savings in simple positions
- ✅ +1 average depth in complex positions
- ✅ UCI integration for DESC control

**Nice to Have:**
- ✅ Verbose logging for analysis
- ✅ Real-time PES visualization (GUI integration)

### Production Release - v18.0

**Performance Targets:**
- ✅ +15% eNPS improvement (5,845 → 6,720 NPS)
- ✅ +50 Elo gain vs v17.1
- ✅ Maintain 50% score vs Stockfish 1%

**Quality Targets:**
- ✅ <2% blunder rate
- ✅ 95%+ tactical puzzle solve rate
- ✅ Consistent depth 8+ in critical positions

**Operational Targets:**
- ✅ Stable in 100+ game tournament
- ✅ Compatible with existing UCI tools
- ✅ Documented for user configuration

---

## Future Enhancements (v18.x Series)

### v18.1: Adaptive Thresholds
- PES thresholds learn optimal values from real games
- Time control-specific models (blitz vs rapid vs classical)
- Opponent strength adjustment (play safer vs stronger opponents)

### v18.2: Multi-Phase Models
- Separate DESC models for opening/middle/endgame
- Phase transition detection
- Endgame tablebase integration with DESC

### v18.3: Self-Learning Feedback
- DESC retrains on V7P3R's own tournament games
- Continuous improvement loop
- Automated A/B testing of DESC parameters

### v18.4: Opponent Modeling
- Track opponent patterns (aggressive, defensive, tactical)
- Adjust PES based on opponent's likely responses
- Exploit known weaknesses

### v18.5: Hardware Optimization
- GPU acceleration for NN inference (optional)
- Multi-threaded DESC for parallel search
- SIMD optimizations for feature extraction

---

## Development Timeline (Estimated)

### Phase 1: Data Collection (2-3 weeks)
- Week 1: Implement profiling instrumentation
- Week 2: Run profiler on game databases
- Week 3: Data cleaning and feature engineering

### Phase 2: Model Development (3-4 weeks)
- Week 1: Train initial classifier
- Week 2: Train regressor and ensemble
- Week 3: Model evaluation and tuning
- Week 4: Export and integration prep

### Phase 3: Integration (2-3 weeks)
- Week 1: Implement DESC controller module
- Week 2: Integrate into V7P3R search
- Week 3: UCI extensions and testing

### Phase 4: Validation (2-3 weeks)
- Week 1: Unit tests and benchmarks
- Week 2: Tournament testing (100+ games)
- Week 3: Analysis and refinement

### Total Timeline: 9-13 weeks (2.25 - 3.25 months)

---

## References & Further Reading

### NNUE Background
- **Stockfish NNUE Documentation:** https://github.com/official-stockfish/nnue-pytorch
- **Original NNUE Paper:** Yu Nasu (2018) - "Efficiently Updatable Neural Network"
- **NNUE Training Guide:** https://github.com/glinscott/nnue-pytorch/wiki

### Chess Engine Optimization
- **Chessprogramming Wiki - Search:** https://www.chessprogramming.org/Search
- **Late Move Reductions (LMR):** https://www.chessprogramming.org/Late_Move_Reductions
- **Evaluation Tuning:** https://www.chessprogramming.org/Automated_Tuning

### Machine Learning for Games
- **AlphaZero Paper:** Silver et al. (2017) - "Mastering Chess without Human Knowledge"
- **Leela Chess Zero:** https://lczero.org/ (NNUE alternative using deep RL)
- **XGBoost Documentation:** https://xgboost.readthedocs.io/

### V7P3R Historical Context
- **v17.1 Breakthrough Analysis:** `docs/V17_1_STOCKFISH_BREAKTHROUGH_ANALYSIS.md`
- **v17.2 Performance Plan:** `docs/V17_2_PERFORMANCE_PLAN.md`
- **Performance Metrics:** `engine-metrics/raw_data/analysis_results/`

---

## Appendix A: Feature Engineering Details

### Derived Features (Computed from Raw Data)

**Positional Tension:**
```
tension_score = (num_checks + num_captures) / legal_moves
```

**King Danger Score:**
```
king_danger = attackers_on_king_zone × (8 - king_pawn_shelter)
```

**Material Imbalance Complexity:**
```
material_complexity = abs(material_diff) / total_material
```

**Search Efficiency Index:**
```
efficiency = (tt_hit_rate × nps) / (eval_time_us + 1)
```

### Feature Normalization

**Min-Max Scaling:**
```python
feature_scaled = (feature - feature_min) / (feature_max - feature_min)
```

**Z-Score Standardization:**
```python
feature_z = (feature - feature_mean) / feature_std
```

**Log Transformation (for time features):**
```python
time_log = log(time_us + 1)  # +1 prevents log(0)
```

---

## Appendix B: Model Hyperparameters

### XGBoost Classifier (Recommended Configuration)

```python
xgb_params = {
    'max_depth': 6,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'objective': 'multi:softmax',
    'num_class': 5,  # 5 PES bins
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'random_state': 42
}
```

### Neural Network Regressor (Recommended Configuration)

```python
nn_params = {
    'input_size': 15,  # Number of features
    'hidden_layers': [128, 64, 32],
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50,
    'optimizer': 'adam',
    'loss': 'mse'
}
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **DESC** | Dynamic Effort-Based Search Control - The AI augmentation system |
| **PES** | Positional Effort Score - Predicted computational cost (0.0 to 1.0) |
| **LMR** | Late Move Reduction - Reduced search depth for unlikely moves |
| **TT** | Transposition Table - Hash table of previously evaluated positions |
| **NPS** | Nodes Per Second - Search speed metric |
| **eNPS** | Effective NPS - Quality-adjusted search speed |
| **NNUE** | Efficiently Updatable Neural Network - Modern chess evaluation AI |
| **UCI** | Universal Chess Interface - Standard engine communication protocol |
| **GBM** | Gradient Boosted Machine - XGBoost/LightGBM models |
| **ONNX** | Open Neural Network Exchange - Cross-platform model format |

---

**Document Status:** Complete - Ready for Review  
**Next Steps:** Review with stakeholders → Begin Phase 1 implementation  
**Questions/Feedback:** Contact development team or create GitHub issue

