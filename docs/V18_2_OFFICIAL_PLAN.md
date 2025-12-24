# V7P3R v18.2 Official Development Plan
**Status**: Planning Phase  
**Created**: December 22, 2025  
**Last Updated**: December 22, 2025  

---

## Executive Summary

### Current State (Experimental v18.2.0)
V18.2.0 was created by combining v18.0's tactical safety system with v18.1's evaluation tuning, expecting 65-70% performance vs v17.1. **Tournament results showed critical regression:**

- **Performance**: 33% vs v17.1 (0W-2L-4D in 6 games)
- **Draw Rate**: 67% (4 draws in 6 games) vs expected 25-30%
- **Issue**: Passive play accepting draws with material/positional advantages

### Root Causes Identified
1. **Threefold Repetition Threshold Too Conservative**: 100cp accepts draws at +1 pawn advantage
2. **Evaluator Architecture Coupling**: Static dual-evaluator system with inconsistent behavior
3. **Game Phase Detection Fragmentation**: 5+ different phase thresholds across codebase
4. **Component Activation Inefficiency**: 42% of bitboard components always active regardless of game phase

### Strategic Direction
This document establishes the official v18.2 development path:
1. **v18.2.1** (Immediate): Fix draw threshold (100cp → 25cp) - *Expected: 50%+ vs v17.1*
2. **v18.2.2** (Short-term): Tuning pass based on v18.2.1 tournament data
3. **v19.0** (Long-term): Phase-aware modular evaluation architecture

---

## Part 1: Experimental v18.2.0 Post-Mortem

### What We Built
Combined two proven improvements:
- **v18.0 Tactical Safety** (58% vs v17.1, +56 ELO)
  - MoveSafetyChecker: -520cp hanging piece penalty
  - Threefold repetition avoidance at 100cp threshold
  - Safety-scored move ordering
  
- **v18.1 Evaluation Tuning** (64% vs v17.1, +100 ELO)
  - King safety: -100cp high-value attackers, -80cp center king
  - Passed pawns: Exponential scaling (20 × 2^advancement)
  - Bishop pair: +50cp explicit bonus
  - King centralization: +40-70cp endgame bonus

### Tournament Results (12 games total)

#### vs v17.1 (6 games)
- **Record**: 0W-2L-4D (33% score)
- **Performance**: REGRESSION (-200+ ELO vs expected +150 ELO)
- **Draw Rate**: 67% (expected 25-30%)
- **Critical Games**:
  - Game 2: Drew at +1.10cp (threefold accepted with slight advantage)
  - Game 4: Drew at +0.75cp (repetition at near-equal position)
  - Game 6: Drew at +0.95cp (threefold with tempo advantage)

#### vs v18.0 (6 games)
- **Record**: 0W-0L-6D (50% score)
- **Performance**: Even (expected 55-60% with evaluation improvements)
- **Draw Rate**: 100% (critical failure)
- **Pattern**: All games reached repetition thresholds

### What Went Wrong

#### Primary Issue: Draw Acceptance Too Passive
```python
# Current code (v7p3r.py line 427-433)
if current_eval > 100:  # Accept draws at +1 pawn advantage
    if position.can_claim_threefold_repetition():
        continue  # Skip moves that lead to repetition
```

**Impact Analysis**:
- 100cp = +1 pawn in closed position
- v18.2 accepting draws at +1.10cp, +0.95cp, +0.75cp
- Should fight for wins up to +2.5 pawns (250cp threshold recommended)
- **67% draw rate** indicates engine avoiding winning attempts

#### Secondary Issues Discovered

**1. Evaluator Selection Coupling** (v7p3r.py lines 230-269)
```python
# One-time selection at engine initialization
if use_fast_evaluator:
    self.fast_evaluator = V7P3RFastEvaluator(self.board)
else:
    self.bitboard_evaluator = V7P3RScoringCalculationBitboard(self.board)
```

**Problem**: Static choice prevents dynamic evaluation strategy
- Fast evaluator: 40x faster, enables depth 6-8
- Bitboard evaluator: Comprehensive, depth 4-6
- No switching based on position type or time pressure

**2. Hybrid Evaluation Inconsistency** (v7p3r.py lines 564-568)
```python
# Calls bitboard tactics even when using fast evaluator
if hasattr(self, 'bitboard_evaluator'):
    self.bitboard_evaluator.detect_tactical_themes(self.board)
```

**Problem**: Evaluator coupling breaks abstraction
- Fast eval selected → still calls bitboard tactical detection
- Inconsistent component activation
- Performance unpredictability

**3. Phase Detection Fragmentation**

Multiple inconsistent thresholds across files:

| File | Endgame Threshold | Opening Threshold |
|------|-------------------|-------------------|
| Fast Evaluator | Material < 800cp OR no queens | Move < 10 |
| Bitboard Evaluator | Pieces ≤ 8 OR material < 2000cp | Pieces ≥ 20 |
| Component 1 | Material < 1300cp | Move < 12 |
| Component 2 | Pieces ≤ 6 | N/A |

**Problem**: Same position classified differently by different components
- No single source of truth for game phase
- Components make independent phase decisions
- Evaluation inconsistency across search tree

**4. Component Activation Waste**

Bitboard evaluator component analysis:
- **47.4% Phase-Specific**: Only run in appropriate phase (9/19 components)
- **42.1% Always Active**: Run regardless of phase (8/19 components)
- **10.5% Unknown**: Unclear activation pattern (2/19 components)

**Problem**: Wasted computation in inappropriate phases
- King safety runs in pure endgames (kings only)
- Endgame logic runs in opening (all pieces present)
- ~40% performance overhead from inappropriate activation

### What Worked

#### Tactical Safety System (v18.0)
- ✅ MoveSafetyChecker functional and effective
- ✅ Hanging piece detection preventing blunders
- ✅ Integrated into move ordering successfully
- **Evidence**: 3/3 unit tests passing, no tactical errors in tournament games

#### Evaluation Improvements (v18.1)
- ✅ King safety penalties correctly applied
- ✅ Passed pawn exponential scaling working
- ✅ Bishop pair bonus functioning
- ✅ King centralization bonuses active
- **Evidence**: 5/5 unit tests passing, evaluation scoring as designed

### Lessons Learned

1. **Draw Thresholds Critical**: Even small threshold errors (100cp vs 25cp) cause massive regression
2. **Integration Risk**: Combining proven systems can introduce emergent behaviors
3. **Static Architecture Limits**: One-time evaluator selection prevents position-adaptive strategy
4. **Phase Fragmentation Costly**: Inconsistent phase detection wastes 40%+ computation
5. **Testing Must Include Draw Behavior**: Win/loss testing insufficient, draw rate analysis required

---

## Part 2: V18.2.1 - Immediate Draw Fix

### Objective
Fix passive draw acceptance to restore competitive play while preserving v18.0 + v18.1 improvements.

### Changes

#### 1. Lower Threefold Repetition Threshold
```python
# File: src/v7p3r.py (line 427)
# BEFORE:
if current_eval > 100:  # Too conservative
    if position.can_claim_threefold_repetition():
        continue

# AFTER:
if current_eval > 25:  # More aggressive, fight harder
    if position.can_claim_threefold_repetition():
        continue
```

**Rationale**:
- 25cp = small positional advantage (tempo, pawn structure)
- Will accept draws when truly equal or slightly worse
- Will fight for wins with any meaningful advantage
- Aligns with competitive engine philosophy

#### 2. Version Updates
- `src/v7p3r.py`: Header version v18.2.0 → v18.2.1
- `src/v7p3r_uci.py`: UCI name + header version v18.2.0 → v18.2.1
- `CHANGELOG.md`: Add v18.2.1 entry with rationale
- `deployment_log.json`: Add v18.2.1 entry with testing status

### Testing Plan

#### Phase 1: Regression Tests (5 games)
- **Opponent**: v17.1 (stable baseline)
- **Time Control**: 5min+4s blitz
- **Success Criteria**: 
  - No crashes or errors
  - Draw rate < 60% (improvement from 67%)
  - Tactical safety still active (no hanging pieces)

#### Phase 2: Performance Benchmark (25 games)
- **Opponent**: v17.1 (stable baseline)
- **Time Control**: 5min+4s blitz
- **Success Criteria**:
  - Win rate ≥ 50% (vs current 33%)
  - Draw rate ≤ 40% (vs current 67%)
  - Average depth ≥ 5.5 (verify no performance regression)
  - Blunders/game ≤ 6.0

#### Phase 3: Cross-Version Validation (10 games each)
- **v18.2.1 vs v18.0**: Expect 52-55% (evaluation improvements should show)
- **v18.2.1 vs v18.1**: Expect 52-55% (tactical safety should show)
- **Success**: Beats both predecessors, confirming combined system works

### Expected Results
- **Draw rate**: 67% → 30-40% (more aggressive play)
- **Win rate vs v17.1**: 33% → 50-60% (restored competitiveness)
- **Performance**: Should match v18.1 (64% vs v17.1) or better

### Deployment Timeline
1. **Implementation**: 5 minutes (threshold change + version updates)
2. **Phase 1 Testing**: 1 hour (5 games + analysis)
3. **Phase 2 Testing**: 4 hours (25 games + analysis)
4. **Phase 3 Testing**: 4 hours (20 games + analysis)
5. **Total**: ~9 hours to validated v18.2.1

### Rollback Plan
If v18.2.1 performs worse than experimental v18.2.0:
- Revert to v18.1.0 (last stable version, 64% vs v17.1)
- Document failure in `deployment_log.json`
- Investigate alternative draw thresholds (15cp, 35cp, 50cp)

---

## Part 3: V18.2.x - Iterative Tuning Phase

### Strategy
Use v18.2.1 tournament data to identify remaining tuning opportunities before major architecture changes.

### Potential Tuning Areas (Priority Order)

#### 1. Draw Threshold Fine-Tuning
**Status**: Dependent on v18.2.1 results
- If draw rate still high (>40%): Lower to 15cp
- If draw rate too low (<20%): Raise to 35cp
- If win rate vs v17.1 < 50%: Consider removing threefold check entirely

#### 2. Evaluation Weight Calibration
**Hypothesis**: Component weights may need rebalancing after tactical safety integration

**Candidates for Adjustment**:
- King safety penalties (-100cp, -80cp) may be too harsh with tactical safety
- Passed pawn exponential may need dampening factor
- Bishop pair +50cp may interact poorly with material evaluation

**Testing Approach**:
- Run 10-game tournaments with ±20% weight adjustments
- Measure win rate change vs v18.2.1 baseline
- Accept changes with ≥5% improvement

#### 3. Move Ordering Integration
**Hypothesis**: Safety scores may conflict with traditional move ordering

**Current Order** (v7p3r.py lines 807-843):
1. Hash move
2. Captures (MVV-LVA + safety score)
3. Killers (+ safety score)
4. Checks (+ safety score)
5. Quiet moves (+ safety score)

**Potential Issues**:
- Safety penalty may suppress good tactical moves
- Hanging detection may be too conservative in forcing variations
- Capture ordering may prioritize safe but weak captures over strong unsafe ones

**Testing**:
- Analyze top 3 candidate moves in 100 critical positions
- Check if best move is being pruned by safety scoring
- Measure tactical success rate (% of tactics found)

#### 4. Time Management Interaction
**Status**: Not yet tested with v18.2 system

**Concerns**:
- Does MoveSafetyChecker overhead impact time management?
- Do evaluation improvements change optimal depth targets?
- Does threefold checking slow down time-critical positions?

**Testing**:
- Run 10 bullet games (1min+2s) to check time forfeits
- Run 10 rapid games (15min+10s) to check depth reached
- Compare time usage vs v18.1 (should be similar)

### V18.2.x Success Criteria
**Before proceeding to v19.0 architecture overhaul:**
- ✅ Win rate vs v17.1 ≥ 60% (stable improvement)
- ✅ Draw rate ≤ 35% (appropriate balance)
- ✅ No major regressions vs v18.0 or v18.1 in any time control
- ✅ At least 100 tournament games played (statistical confidence)
- ✅ All unit tests passing (regression prevention)

---

## Part 4: V19.0 - Phase-Aware Modular Architecture

### Vision Statement
Transform V7P3R from dual-evaluator static system to **dynamic, phase-aware modular evaluation** with per-search component selection and time-pressure adaptability.

### Architectural Problems to Solve

#### Problem 1: Static Evaluator Selection
**Current**: Chosen once at engine startup, never changes
```python
if use_fast_evaluator:  # Decided at init, permanent
    self.fast_evaluator = V7P3RFastEvaluator(self.board)
```

**Desired**: Per-position evaluator selection
```python
# Select evaluator based on position characteristics
if time_pressure or simple_position:
    return fast_profile.evaluate(board)
elif tactical_position:
    return tactical_profile.evaluate(board)
else:
    return comprehensive_profile.evaluate(board)
```

#### Problem 2: Phase Detection Fragmentation
**Current**: Each component detects phase independently (5+ thresholds)

**Desired**: Central phase detector with single source of truth
```python
class GamePhaseDetector:
    def detect_phase(self, board) -> GamePhase:
        """Single authoritative phase classification"""
        # Returns: OPENING, MIDDLEGAME_COMPLEX, MIDDLEGAME_SIMPLE, ENDGAME_COMPLEX, ENDGAME_SIMPLE
```

#### Problem 3: Component Coupling
**Current**: Components scattered across fast_evaluator, bitboard_evaluator, move_safety
- 8 components in fast evaluator
- 19 components in bitboard evaluator
- 3 components in move safety
- **Total**: 30 components with unclear relationships

**Desired**: Centralized component registry with metadata
```python
COMPONENT_REGISTRY = {
    'material_counting': {
        'phases': [ALL],
        'cost': 'negligible',
        'time_pressure_skip': False
    },
    'king_safety_attackers': {
        'phases': [MIDDLEGAME_COMPLEX],
        'cost': 'medium',
        'time_pressure_skip': True
    },
    # ... 28 more components
}
```

#### Problem 4: Wasteful Component Activation
**Current**: 42% of components always active regardless of phase

**Desired**: Phase-appropriate component selection
```python
# In opening, run only: material, PST, development, king safety
# In middlegame, run: material, PST, tactics, king safety, passed pawns
# In endgame, run: material, king centralization, passed pawns, pawn structure
```

### Proposed Architecture

#### New Module: `v7p3r_game_phase.py`
**Purpose**: Single source of truth for game phase classification

```python
from enum import Enum
from dataclasses import dataclass

class GamePhase(Enum):
    OPENING = "opening"
    MIDDLEGAME_COMPLEX = "middlegame_complex"  # ≥3 pieces per side
    MIDDLEGAME_SIMPLE = "middlegame_simple"    # 2-3 pieces per side
    ENDGAME_COMPLEX = "endgame_complex"        # 2 pieces per side
    ENDGAME_SIMPLE = "endgame_simple"          # ≤1 piece per side

@dataclass
class PhaseMetrics:
    phase: GamePhase
    material_count: int
    piece_count: int
    queens_present: bool
    move_number: int
    
class GamePhaseDetector:
    """Unified game phase detection with consistent thresholds"""
    
    OPENING_THRESHOLD = 12  # moves
    ENDGAME_MATERIAL_THRESHOLD = 1300  # centipawns
    SIMPLE_PIECE_THRESHOLD = 3  # pieces per side
    
    def detect_phase(self, board: chess.Board) -> PhaseMetrics:
        """
        Single authoritative phase classification.
        
        Phase Logic:
        1. Opening: move_number < 12 AND pieces ≥ 12
        2. Endgame: material < 1300cp OR (pieces ≤ 4 AND no queens)
        3. Middlegame: Everything else
        4. Simple vs Complex: Based on piece count threshold
        """
        # Implementation here
```

**Benefits**:
- ✅ Single source of truth (no inconsistency)
- ✅ Easy to test and validate
- ✅ Can be enhanced independently
- ✅ Supports future ML-based phase classification

#### New Module: `v7p3r_eval_registry.py`
**Purpose**: Component metadata and activation rules

```python
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class EvaluationComponent:
    name: str
    function: Callable
    phases: List[GamePhase]
    cost: str  # 'negligible', 'low', 'medium', 'high'
    time_pressure_skip: bool
    description: str
    
class ComponentRegistry:
    """Central registry of all evaluation components with metadata"""
    
    def __init__(self):
        self.components = {}
        
    def register(self, component: EvaluationComponent):
        """Add component to registry"""
        self.components[component.name] = component
        
    def get_components_for_phase(self, phase: GamePhase, 
                                  time_pressure: bool = False) -> List[EvaluationComponent]:
        """
        Return appropriate components for given phase and time pressure.
        
        Time Pressure Mode:
        - Skip all components with cost='high' 
        - Skip all components with time_pressure_skip=True
        - Returns minimal component set for fastest evaluation
        """
        # Filter by phase and time pressure
        
# Register all 30 components with metadata
registry = ComponentRegistry()

# Material (always active, negligible cost)
registry.register(EvaluationComponent(
    name='material_counting',
    function=count_material,
    phases=[GamePhase.OPENING, GamePhase.MIDDLEGAME_COMPLEX, 
            GamePhase.MIDDLEGAME_SIMPLE, GamePhase.ENDGAME_COMPLEX, 
            GamePhase.ENDGAME_SIMPLE],
    cost='negligible',
    time_pressure_skip=False,
    description='Basic material counting (P=100, N=320, B=330, R=500, Q=900)'
))

# King safety (middlegame only, medium cost)
registry.register(EvaluationComponent(
    name='king_safety_attackers',
    function=evaluate_king_safety_attackers,
    phases=[GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE],
    cost='medium',
    time_pressure_skip=True,  # Skip in time pressure
    description='High-value attacker penalty (-100cp per Q/R in king zone)'
))

# ... 28 more components
```

**Benefits**:
- ✅ Self-documenting component inventory
- ✅ Easy to add/remove components
- ✅ Clear cost/benefit tradeoffs
- ✅ Supports A/B testing (disable specific components)
- ✅ Foundation for ML-based component selection

#### New Module: `v7p3r_modular_evaluator.py`
**Purpose**: Unified evaluator with dynamic component selection

```python
class ModularEvaluator:
    """
    Unified evaluator that selects components per-search based on 
    game phase and time pressure.
    """
    
    def __init__(self, board: chess.Board):
        self.board = board
        self.phase_detector = GamePhaseDetector()
        self.registry = ComponentRegistry()
        
        # Evaluation profiles (pre-selected component sets)
        self.profiles = {
            'comprehensive': self._build_comprehensive_profile(),
            'fast': self._build_fast_profile(),
            'tactical': self._build_tactical_profile(),
            'endgame': self._build_endgame_profile()
        }
        
    def evaluate(self, board: chess.Board, time_pressure: bool = False) -> int:
        """
        Main evaluation entry point.
        
        Strategy:
        1. Detect game phase
        2. Select appropriate evaluation profile
        3. Run only components relevant to phase
        4. Cache phase detection for search tree (persist across moves)
        """
        
        # Detect phase ONCE per search (not per node)
        phase_metrics = self.phase_detector.detect_phase(board)
        
        # Select profile based on phase and time pressure
        if time_pressure:
            profile = self.profiles['fast']
        elif phase_metrics.phase in [GamePhase.ENDGAME_COMPLEX, GamePhase.ENDGAME_SIMPLE]:
            profile = self.profiles['endgame']
        elif self._is_tactical_position(board):
            profile = self.profiles['tactical']
        else:
            profile = self.profiles['comprehensive']
            
        # Run selected components
        score = 0
        for component in profile:
            score += component.function(board, phase_metrics)
            
        return score
        
    def _build_comprehensive_profile(self) -> List[EvaluationComponent]:
        """All components except those marked for specific phases only"""
        # Include: material, PST, king safety, passed pawns, tactics, etc.
        
    def _build_fast_profile(self) -> List[EvaluationComponent]:
        """Minimal component set for 40x speed (current fast evaluator)"""
        # Include: material, PST, basic middlegame bonuses
        
    def _build_tactical_profile(self) -> List[EvaluationComponent]:
        """Tactical-focused components for sharp positions"""
        # Include: material, king safety, tactical detection, passed pawns
        
    def _build_endgame_profile(self) -> List[EvaluationComponent]:
        """Endgame-focused components"""
        # Include: material, king centralization, passed pawns, pawn structure
```

**Benefits**:
- ✅ **Per-search component selection** (not per-ply, efficient)
- ✅ **Profile-based evaluation** (pre-selected component sets)
- ✅ **Time-pressure adaptability** (automatic fast mode)
- ✅ **Phase-appropriate evaluation** (no wasted computation)
- ✅ **Easy to extend** (add new profiles for specific position types)

### Migration Strategy

#### Phase 1: Create Infrastructure (No Behavior Change)
**Goal**: Establish new modules without changing engine behavior

**Tasks**:
1. Create `v7p3r_game_phase.py` with unified phase detection
2. Create `v7p3r_eval_registry.py` with component metadata
3. Create `v7p3r_modular_evaluator.py` with profile system
4. **Validation**: Unit tests for phase detection, component registry loading

**Success Criteria**:
- ✅ All new modules load without errors
- ✅ Phase detection matches current fast evaluator thresholds (for compatibility)
- ✅ Component registry contains all 30 components
- ✅ Modular evaluator exists but not yet connected to engine

#### Phase 2: Parallel Evaluation (A/B Testing)
**Goal**: Run modular evaluator alongside current system for validation

**Tasks**:
1. Add modular evaluator to `v7p3r.py` (parallel to current evaluators)
2. Run both evaluators on same positions
3. Compare scores (should be identical initially)
4. Identify any score discrepancies

**Validation**:
- Run 1000 random positions through both evaluators
- Measure score difference (target: <5cp average)
- Fix any discrepancies before proceeding

**Success Criteria**:
- ✅ Modular evaluator scores match current evaluator scores
- ✅ Performance within 10% of current system
- ✅ No crashes or errors in 1000+ position test

#### Phase 3: Gradual Component Migration
**Goal**: Move components one-by-one to modular system

**Order** (lowest risk first):
1. Material counting (trivial, all phases)
2. Piece-square tables (trivial, all phases)
3. King centralization (endgame only, isolated)
4. Passed pawns (multiple phases, medium complexity)
5. King safety (middlegame only, medium complexity)
6. Tactical detection (complex, high risk)

**Per-Component Process**:
1. Migrate component to modular system
2. Run unit tests (component should work identically)
3. Run 10-game tournament (performance should be identical)
4. If passes, proceed to next component
5. If fails, rollback and investigate

**Success Criteria**:
- ✅ Each component migrates without regression
- ✅ Cumulative performance stays within 2% of baseline
- ✅ All unit tests continue passing

#### Phase 4: Switch to Modular Evaluator
**Goal**: Make modular evaluator the primary evaluation system

**Tasks**:
1. Remove `use_fast_evaluator` boolean from `v7p3r.py`
2. Remove direct calls to `V7P3RFastEvaluator` and `V7P3RScoringCalculationBitboard`
3. Route all evaluation through `ModularEvaluator.evaluate()`
4. Enable profile-based evaluation (comprehensive, fast, tactical, endgame)

**Validation**:
- Run 50-game tournament vs v18.2.x (should be identical or better)
- Run 50-game tournament vs v17.1 (should maintain ≥60% win rate)
- Check performance metrics (depth, time usage, blunders)

**Success Criteria**:
- ✅ Performance identical or better than v18.2.x
- ✅ Win rate vs v17.1 ≥ 60%
- ✅ No increase in blunders or time forfeits
- ✅ Code cleaner (evaluator selection logic simplified)

#### Phase 5: Enable Advanced Features
**Goal**: Activate phase-aware and time-pressure features

**Tasks**:
1. Enable per-search profile selection (not per-ply)
2. Enable time-pressure fast mode (skip high-cost components)
3. Enable tactical position detection (use tactical profile)
4. Add evaluation profile logging (track which profiles used)

**New Capabilities**:
- Automatic fast mode in time pressure (skip king safety, tactics)
- Tactical profile for sharp positions (prioritize king safety, threats)
- Endgame profile for simplified positions (skip opening bonuses, tactics)
- Per-search caching (evaluate phase once, persist through tree)

**Validation**:
- Run 20 bullet games (1min+2s): Check time forfeit rate
- Run 20 tactical puzzles: Check tactics found rate
- Run 20 endgame positions: Check conversion rate

**Success Criteria**:
- ✅ Bullet games: Time forfeit rate < 10% (vs baseline)
- ✅ Tactical puzzles: ≥90% success rate (vs baseline)
- ✅ Endgame positions: ≥80% conversion rate (vs baseline)
- ✅ Performance gain: +5-10% vs v18.2.x (from efficiency)

### V19.0 Expected Benefits

#### Performance Improvements
- **Depth**: +0.5-1.5 ply (from phase-appropriate component selection)
- **Speed**: 10-20% faster (from skipping irrelevant components)
- **Time Pressure**: Better bullet/blitz performance (automatic fast mode)
- **Consistency**: Fewer evaluation glitches (single phase authority)

#### Code Quality Improvements
- **Maintainability**: Single evaluator vs dual system
- **Testability**: Components isolated and unit-testable
- **Extensibility**: Easy to add new components/profiles
- **Clarity**: Self-documenting component registry

#### Future Capabilities Enabled
- **ML-Based Component Selection**: Replace hand-coded profiles with learned profiles
- **Position-Type Specialization**: Custom profiles for specific position types
- **Dynamic Component Weighting**: Adjust weights based on success rate
- **A/B Testing Framework**: Easy to test component effectiveness

### V19.0 Timeline Estimate
- **Phase 1** (Infrastructure): 4-6 hours coding + 2 hours testing = 6-8 hours
- **Phase 2** (A/B Testing): 2 hours coding + 4 hours testing = 6 hours
- **Phase 3** (Migration): 8 hours coding + 12 hours testing (6 components) = 20 hours
- **Phase 4** (Switch): 2 hours coding + 8 hours testing = 10 hours
- **Phase 5** (Advanced Features): 4 hours coding + 6 hours testing = 10 hours
- **Total**: ~52-54 hours (1-2 weeks of focused development)

### Risks and Mitigation

#### Risk 1: Performance Regression During Migration
**Mitigation**: Parallel evaluation and per-component validation
- Keep old evaluators as fallback
- Require identical scores before switching
- Gradual rollout with rollback capability

#### Risk 2: Component Interaction Bugs
**Mitigation**: Extensive unit testing and tournament validation
- Test each component in isolation
- Test component combinations (pairwise)
- Run 100+ game tournaments at each phase

#### Risk 3: Increased Code Complexity
**Mitigation**: Clean abstractions and documentation
- Clear separation of concerns (phase, registry, evaluator)
- Comprehensive docstrings and examples
- Maintain simple API for external callers

#### Risk 4: Breaking Existing Behavior
**Mitigation**: Regression test suite
- Preserve all v18.2.x unit tests
- Add new tests for modular system
- Require 100% test pass rate before deployment

---

## Part 5: Long-Term Vision (v20.0+)

### DESC System Integration
**DESC** = Dynamic Evaluation System Controller (ML-enhanced component selection)

**Current Plan** (v19.0): Hand-coded component selection based on phase
**Future Vision** (v20.0+): ML model predicts optimal component set for position

#### DESC Architecture
```python
class DESCController:
    """
    Machine learning model that predicts which evaluation components
    to activate for a given position.
    
    Training Data:
    - 10,000+ positions with full component evaluation
    - Stockfish ground truth scores
    - Component contribution to accuracy
    
    Model Output:
    - Component activation probability (0.0-1.0 per component)
    - Threshold: Activate if probability > 0.7
    
    Expected Benefit:
    - 20-30% faster (better component selection than hand-coded rules)
    - More accurate (components activated based on actual relevance)
    """
```

#### Training Procedure
1. **Data Collection** (v19.x versions):
   - Log every position evaluated
   - Record: FEN, phase, components activated, score, Stockfish score
   - Collect 10,000+ diverse positions (opening, middlegame, endgame)

2. **Feature Engineering**:
   - Position features: material, piece count, king safety metrics, pawn structure
   - Phase features: move number, piece distribution
   - Component features: historical accuracy, cost

3. **Model Training**:
   - Architecture: Feedforward neural network or gradient boosting
   - Input: 100+ position features
   - Output: 30 component activation probabilities
   - Loss: Weighted combination of speed and accuracy

4. **Deployment**:
   - Replace hand-coded profile selection with ML prediction
   - Fallback to hand-coded rules if ML model unavailable
   - Continuous learning (update model based on tournament results)

#### Expected Timeline
- **v19.0-v19.x** (2025 Q1): Collect training data, validate modular architecture
- **v20.0** (2025 Q2): Train initial DESC model, deploy to testing
- **v20.x** (2025 Q2-Q3): Iterate on model, collect more data, improve accuracy
- **v21.0** (2025 Q4): Production deployment with mature DESC system

### Other Future Enhancements

#### 1. Opening Book Integration
**Status**: Partially implemented (v17.1+)
**Needed**: Better integration with modular evaluator
- Use opening book profile (disable most evaluation)
- Transition smoothly to middlegame evaluation
- Learn opening book success rates

#### 2. Endgame Tablebase Support
**Status**: Not implemented
**Vision**: Pre-calculate won/drawn/lost endgames
- Syzygy tablebase integration
- Disable evaluation when tablebase available
- Perfect endgame play (3-5 pieces)

#### 3. Time Manager Evolution
**Status**: v14.1 smart time management active
**Vision**: Integrate with modular evaluator
- Use evaluation profile to estimate time per move
- Fast profile → less time, comprehensive → more time
- Dynamic time allocation based on position complexity

#### 4. Neural Network Evaluation
**Status**: Research phase (v7p3r-chess-ai project)
**Vision**: NNUE-style evaluation as alternative profile
- Train deep neural network on Stockfish data
- Use as "neural" evaluation profile
- Hybrid: Use modular eval for tactics, NNUE for positional

---

## Part 6: Testing and Validation Standards

### Unit Test Requirements

#### Required Test Coverage
- **Game Phase Detection**: ≥95% branch coverage
- **Component Registry**: 100% component coverage (all 30 registered)
- **Modular Evaluator**: ≥90% branch coverage
- **Evaluation Profiles**: Each profile tested independently

#### Test Categories
1. **Phase Detection Tests**: Verify consistent classification
   - Opening positions (move < 12)
   - Middlegame positions (standard material, various piece counts)
   - Endgame positions (low material, king activity)
   - Edge cases (unusual material imbalances)

2. **Component Tests**: Verify each component works correctly
   - Isolated component tests (one component at a time)
   - Known position tests (verify expected scores)
   - Edge case tests (empty board, one king, etc.)

3. **Profile Tests**: Verify profiles select correct components
   - Fast profile: Only low-cost components
   - Tactical profile: Includes tactical detection
   - Endgame profile: Includes king centralization
   - Comprehensive profile: Includes all appropriate components

4. **Integration Tests**: Verify complete evaluation pipeline
   - Score consistency (same position → same score)
   - Performance consistency (depth, time usage)
   - Regression tests (v18.2.x positions still score correctly)

### Tournament Test Requirements

#### Standard Tournament Protocol
All new versions must pass:

1. **Regression Tests** (5 games vs v17.1)
   - No crashes or errors
   - Draw rate within acceptable range
   - Basic functionality verified

2. **Performance Benchmark** (50 games vs v17.1)
   - Win rate ≥ 50% (vs v17.1 baseline)
   - Draw rate ≤ 40%
   - Blunders/game ≤ 6.0
   - Time forfeit rate < 10%

3. **Cross-Version Validation** (20 games each)
   - vs v18.0: Verify evaluation improvements show
   - vs v18.1: Verify tactical safety shows
   - vs v18.2.x: Verify no regression

4. **Time Control Validation** (10 games each)
   - Bullet (1min+2s): Time forfeit rate < 15%
   - Blitz (5min+4s): Standard performance
   - Rapid (15min+10s): Maximum depth/accuracy

#### Acceptance Criteria
**Before production deployment:**
- ✅ 100% regression test pass rate
- ✅ Win rate vs v17.1 ≥ 60%
- ✅ Draw rate ≤ 35%
- ✅ No critical errors in 100+ games
- ✅ Performance metrics meet targets
- ✅ All unit tests passing

### Performance Metrics

#### Track These Metrics
- **Win Rate**: % of games won (target: ≥60% vs v17.1)
- **Draw Rate**: % of games drawn (target: 25-35%)
- **Blunders/Game**: Average blunders per game (target: ≤6.0)
- **Average Depth**: Mean search depth (target: ≥5.5)
- **Time Forfeit Rate**: % of games lost on time (target: <10%)
- **ACPL** (Average Centipawn Loss): Mean error vs Stockfish (target: <150cp)

#### Statistical Confidence
- **Minimum Sample Size**: 50 games (for ±10% confidence interval)
- **Preferred Sample Size**: 100 games (for ±7% confidence interval)
- **Production Validation**: 200+ games (for ±5% confidence interval)

### Regression Prevention

#### Known Failure Patterns (Must Test)
Based on historical issues (see `version_management.instructions.md`):

1. **v17.4 Endgame Failure**: Mate-in-3 miss
   - **Test**: Include mate-in-3 positions in regression suite
   
2. **v17.0 Black-Side Weakness**: Color imbalance
   - **Test**: 50/50 White/Black split in tournaments
   
3. **v17.7 Rapid Regression**: Draw acceptance too conservative
   - **Test**: Monitor draw rate in all time controls
   
4. **v18.2.0 Draw Rate Spike**: Threefold threshold too high
   - **Test**: Track threefold repetition rate separately

#### Continuous Monitoring
- Deploy new versions to testing environment first
- Monitor first 24 hours for anomalies
- Compare statistics against baseline continuously
- Rollback immediately if critical regression detected

---

## Part 7: Implementation Roadmap

### Timeline Overview

```
December 2025:
├── Week 4 (Dec 22-28): v18.2.1 Implementation & Testing
│   ├── Dec 22: Implement draw threshold fix
│   ├── Dec 23-24: Regression tests (5 games)
│   ├── Dec 25-26: Performance benchmark (50 games)
│   ├── Dec 27-28: Cross-version validation (20 games)
│   └── Dec 28: v18.2.1 deployment decision

January 2026:
├── Week 1-2 (Jan 1-15): v18.2.x Tuning Phase
│   ├── Week 1: Analyze v18.2.1 results, identify tuning needs
│   ├── Week 2: Implement v18.2.2+ improvements, test each
│   └── Target: Stable v18.2.x with 60%+ vs v17.1
│
├── Week 3 (Jan 16-22): v19.0 Planning & Design
│   ├── Finalize modular architecture design
│   ├── Create detailed component migration plan
│   ├── Set up development branch
│   └── Review plan with stakeholders
│
└── Week 4 (Jan 23-31): v19.0 Phase 1 (Infrastructure)
    ├── Create v7p3r_game_phase.py
    ├── Create v7p3r_eval_registry.py
    ├── Create v7p3r_modular_evaluator.py
    └── Unit tests for all new modules

February 2026:
├── Week 1 (Feb 1-7): v19.0 Phase 2 (A/B Testing)
│   ├── Parallel evaluation implementation
│   ├── 1000-position validation
│   └── Score discrepancy analysis
│
├── Week 2-3 (Feb 8-21): v19.0 Phase 3 (Component Migration)
│   ├── Week 2: Migrate material, PST, king centralization
│   ├── Week 3: Migrate passed pawns, king safety, tactics
│   └── 10-game validation per component
│
├── Week 4 (Feb 22-28): v19.0 Phase 4 (Switch to Modular)
│   ├── Remove dual evaluator code
│   ├── Route all evaluation through modular system
│   ├── 50-game validation vs v18.2.x
│   └── 50-game validation vs v17.1

March 2026:
├── Week 1 (Mar 1-7): v19.0 Phase 5 (Advanced Features)
│   ├── Enable per-search profile selection
│   ├── Enable time-pressure mode
│   ├── Enable tactical position detection
│   └── Validation across time controls
│
├── Week 2-3 (Mar 8-21): v19.0 Tournament Validation
│   ├── 100-game tournament vs v18.2.x
│   ├── 100-game tournament vs v17.1
│   ├── Cross-time-control validation
│   └── Statistical analysis
│
└── Week 4 (Mar 22-31): v19.0 Production Deployment
    ├── Final regression test suite
    ├── Documentation updates
    ├── Deployment to production
    └── Monitoring and validation

April 2026+:
└── DESC System Development (v20.0)
    ├── Data collection (v19.x versions)
    ├── Model training
    ├── Testing and iteration
    └── Production deployment (Q2-Q3 2026)
```

### Milestone Definitions

#### Milestone 1: v18.2.1 Deployed (Dec 28, 2025)
**Success Criteria**:
- ✅ Draw threshold fixed (100cp → 25cp)
- ✅ Win rate vs v17.1 ≥ 50%
- ✅ Draw rate ≤ 40%
- ✅ No regressions vs v18.0/v18.1

#### Milestone 2: v18.2.x Stable (Jan 15, 2026)
**Success Criteria**:
- ✅ Win rate vs v17.1 ≥ 60%
- ✅ Draw rate 25-35%
- ✅ 100+ tournament games played
- ✅ Ready for long-term use

#### Milestone 3: v19.0 Infrastructure Complete (Jan 31, 2026)
**Success Criteria**:
- ✅ All new modules created and tested
- ✅ Component registry complete (30 components)
- ✅ Phase detector validated
- ✅ Modular evaluator exists (not yet connected)

#### Milestone 4: v19.0 Modular Evaluator Active (Feb 28, 2026)
**Success Criteria**:
- ✅ All components migrated
- ✅ Performance identical or better than v18.2.x
- ✅ Dual evaluator code removed
- ✅ Code cleaner and more maintainable

#### Milestone 5: v19.0 Production Ready (Mar 31, 2026)
**Success Criteria**:
- ✅ Advanced features enabled
- ✅ Win rate vs v17.1 ≥ 65%
- ✅ Performance gain +5-10% vs v18.2.x
- ✅ 200+ tournament games validated
- ✅ Deployed to production

#### Milestone 6: DESC System Active (Q3 2026)
**Success Criteria**:
- ✅ ML model trained on 10,000+ positions
- ✅ Component selection automated
- ✅ Performance +20-30% vs hand-coded rules
- ✅ Continuous learning active

### Resource Requirements

#### Development Time
- **v18.2.1**: 1-2 days (implementation + testing)
- **v18.2.x**: 1-2 weeks (iterative tuning)
- **v19.0**: 1-2 months (architecture overhaul)
- **v20.0 DESC**: 2-3 months (ML development)

#### Testing Resources
- **Local Testing**: Arena GUI, Python scripts
- **Tournament Engines**: v17.1 (baseline), v18.0, v18.1, v18.2.x
- **Compute**: Local development machine (sufficient for 100-game tournaments)
- **Validation**: Stockfish 17.1 (ground truth analysis)

#### Documentation
- **This Plan**: Living document, updated with learnings
- **CHANGELOG.md**: Detailed version history
- **deployment_log.json**: Production deployment tracking
- **Component Documentation**: Inline docstrings + registry metadata

---

## Part 8: Success Criteria & Metrics

### V18.2.1 Success Criteria
**Target Date**: December 28, 2025

| Metric | Target | Baseline (v18.2.0) | Status |
|--------|--------|-------------------|--------|
| Win Rate vs v17.1 | ≥50% | 33% | ⏳ Pending |
| Draw Rate | ≤40% | 67% | ⏳ Pending |
| Blunders/Game | ≤6.0 | Unknown | ⏳ Pending |
| Time Forfeit Rate | <10% | 0% | ⏳ Pending |
| Games Tested | 50+ | 12 | ⏳ Pending |

### V18.2.x Success Criteria
**Target Date**: January 15, 2026

| Metric | Target | Baseline (v17.1) | Status |
|--------|--------|------------------|--------|
| Win Rate vs v17.1 | ≥60% | 50% (self) | ⏳ Pending |
| Draw Rate | 25-35% | ~30% | ⏳ Pending |
| Average Depth | ≥5.5 | ~5.0 | ⏳ Pending |
| ACPL | <150cp | ~160cp | ⏳ Pending |
| Games Tested | 100+ | 100+ | ⏳ Pending |

### V19.0 Success Criteria
**Target Date**: March 31, 2026

| Metric | Target | Baseline (v18.2.x) | Status |
|--------|--------|-------------------|--------|
| Win Rate vs v17.1 | ≥65% | 60% | ⏳ Pending |
| Performance vs v18.2.x | +5-10% | 50% (self) | ⏳ Pending |
| Depth Improvement | +0.5-1.5 ply | ±0 | ⏳ Pending |
| Code Complexity | Lower | Baseline | ⏳ Pending |
| Component Count | 30 registered | 30 scattered | ⏳ Pending |
| Games Tested | 200+ | 100+ | ⏳ Pending |

### Key Performance Indicators (KPIs)

#### Competitive Strength
- **Primary**: Win rate vs v17.1 (stable baseline)
- **Secondary**: ELO rating estimate
- **Tertiary**: Performance vs Stockfish handicapped (depth limit)

#### Reliability
- **Primary**: Time forfeit rate
- **Secondary**: Crash/error rate
- **Tertiary**: Consistent performance across time controls

#### Code Quality
- **Primary**: Unit test coverage %
- **Secondary**: Code duplication metrics
- **Tertiary**: Documentation completeness

#### Efficiency
- **Primary**: Average search depth
- **Secondary**: Nodes per second
- **Tertiary**: Evaluation speed (positions/second)

---

## Part 9: Risk Management

### High-Priority Risks

#### Risk 1: v18.2.1 Draw Fix Insufficient
**Probability**: Medium (30%)  
**Impact**: High (delays v19.0 timeline)

**Scenario**: Lowering threshold to 25cp still results in >40% draw rate

**Mitigation**:
- Have alternative thresholds ready (15cp, 35cp, 50cp)
- Test multiple thresholds in parallel tournaments
- Consider disabling threefold check entirely for testing

**Contingency**: If draw rate still high, investigate move ordering or evaluation issues

#### Risk 2: v19.0 Migration Performance Regression
**Probability**: Medium (40%)  
**Impact**: High (architecture redesign required)

**Scenario**: Modular evaluator 10%+ slower than current system

**Mitigation**:
- Parallel evaluation during development (catch issues early)
- Per-component performance profiling
- Optimize hot paths before full deployment

**Contingency**: Keep dual evaluator code as fallback, optimize modular system before switch

#### Risk 3: Component Interaction Bugs
**Probability**: High (60%)  
**Impact**: Medium (testing will catch)

**Scenario**: Components interfere when combined (e.g., king safety + passed pawns double-count)

**Mitigation**:
- Extensive unit tests for component combinations
- Regression suite with known positions
- Gradual migration (one component at a time)

**Contingency**: Isolate problematic components, fix interactions before proceeding

### Medium-Priority Risks

#### Risk 4: Tournament Testing Insufficient Sample Size
**Probability**: Medium (30%)  
**Impact**: Medium (false confidence)

**Scenario**: 50-game tournaments show improvement but not statistically significant

**Mitigation**:
- Use consistent baselines (v17.1, v18.0, v18.1)
- Run larger sample sizes when critical (100+ games)
- Track confidence intervals explicitly

**Contingency**: Require 100-game minimum for production decisions

#### Risk 5: Time Pressure Mode Breaks Gameplay
**Probability**: Low (20%)  
**Impact**: High (bullet/blitz unplayable)

**Scenario**: Fast mode too aggressive, causes tactical blunders in time pressure

**Mitigation**:
- Test time pressure mode extensively in bullet
- Keep safety checks even in fast mode
- Gradual reduction of components (not all-or-nothing)

**Contingency**: Disable time pressure mode if forfeit rate spikes

### Low-Priority Risks

#### Risk 6: DESC Training Data Collection Slow
**Probability**: Medium (40%)  
**Impact**: Low (v19.0 still valuable without DESC)

**Scenario**: Takes longer than expected to collect 10,000 positions

**Mitigation**:
- Start data collection in v19.0 development
- Use automated self-play for position generation
- Leverage existing game archives

**Contingency**: Delay DESC to v20.1+, v19.0 still valuable

---

## Part 10: Conclusion & Next Steps [UPDATED 2025-12-23]

### STRATEGIC PIVOT: Modular Evaluation NOW

**Tournament data reveals critical insight:**
- v18.0 has **140+ ELO swing** between rapid (+56) and blitz (-85)
- Root cause: **Evaluation overhead compounds in faster time controls**
- v18.0 depth decline: 3.977-3.994 in long blitz games (vs expected 4.0+)
- Both v18.0 and v18.2 have **100cp threefold threshold** (causing 55-67% draw rates)

**Conclusion**: Adding more heuristics to v18.2 will only worsen time control issues.

### New Direction: v18.2 = Modular Evaluation Foundation

**Skip incremental fixes** (v18.2.1, v18.2.2, v18.2.x) and **implement selective evaluation NOW**:

1. **v18.2** (NEW SCOPE): Pre-search evaluation profile system
   - Target: Maintain v18.2 strength while fixing time control issues
   - Timeline: 1-2 weeks
   
2. **v18.3+** (Iteration): Expand modular components
   - Target: Add new heuristics without performance penalty
   - Timeline: Ongoing

3. **v19.0** (Future): ML-based component selection (DESC system)
   - Target: Automated profile optimization
   - Timeline: Q2 2026

### Immediate Next Steps [UPDATED 2025-12-23]

**New Path: Implement Modular Evaluation in v18.2**

1. **Design Position Context System** (Dec 23, TODAY):
   - Create pre-search input calculation module
   - Define position metrics: time pressure, game phase, material imbalance, piece types
   - Design evaluation profile selection logic
   - Document module metadata structure

2. **Catalog & Classify Existing Evaluations** (Dec 23-24):
   - Inventory all 30+ evaluation components (fast eval, bitboard, safety checker)
   - Tag each with: required inputs, cost, criticality, game phases
   - Identify redundancies and overlaps
   - Group into logical modules

3. **Implement Core Infrastructure** (Dec 24-26):
   - Create `v7p3r_position_context.py`: Calculate inputs once per search
   - Create `v7p3r_eval_modules.py`: Modular evaluation components
   - Create `v7p3r_eval_selector.py`: Profile selection based on context
   - Integrate with existing search loop

4. **Migration & Testing** (Dec 26-28):
   - Migrate critical evaluations to modular system
   - Test parity with v18.2 baseline
   - Validate performance improvement in blitz
   - 25-game tournament: blitz AND rapid

### Long-Term Vision
By March 2026, V7P3R will have:
- ✅ Stable competitive performance (65%+ vs v17.1 baseline)
- ✅ Clean modular architecture (30 registered components)
- ✅ Phase-aware evaluation (no wasted computation)
- ✅ Time-pressure adaptability (better bullet/blitz)
- ✅ Foundation for ML enhancement (DESC system ready)

### Commitment to Quality
Every version will be:
- **Tested**: ≥50 tournament games before deployment
- **Documented**: CHANGELOG + deployment_log entries
- **Validated**: Unit tests + regression tests passing
- **Monitored**: First 24-48 hours closely watched
- **Reversible**: Rollback plan documented

---

## Appendices

### Appendix A: Component Inventory
**Current Count**: 30 components across 3 files

#### Fast Evaluator (8 components)
1. Material counting
2. Piece-square tables
3. Rook on open file bonus
4. King safety (pawn shield)
5. Passed pawn detection
6. Opening detection
7. Endgame detection
8. Phase-based score adjustments

#### Bitboard Evaluator (19 components)
1. Material counting
2. Piece-square tables
3. Center control
4. Development tracking
5. King safety (attackers)
6. King safety (center penalty)
7. King centralization (endgame)
8. Passed pawn exponential
9. Bishop pair bonus
10. Doubled pawns penalty
11. Isolated pawns penalty
12. Backward pawns penalty
13. Pawn chain bonus
14. Rook on open file
15. Rook on 7th rank
16. Fork detection
17. Pin detection
18. Skewer detection
19. Knight outpost bonus

#### Move Safety (3 components)
1. Hanging piece detection
2. Immediate capture threats
3. Check exposure penalty

### Appendix B: Game Phase Thresholds
**Current (Inconsistent)**:

| Component | Endgame Trigger | Opening Trigger |
|-----------|----------------|-----------------|
| Fast Eval | Material < 800cp OR no queens | Move < 10 |
| Bitboard Eval | Pieces ≤ 8 OR material < 2000cp | Pieces ≥ 20 |
| King Centralization | Material < 1300cp | N/A |
| Passed Pawn Bonus | No endgame check | N/A |

**Proposed (Unified)**:

| Phase | Definition | Material Range | Piece Count Range | Move Range |
|-------|-----------|----------------|-------------------|------------|
| Opening | Early game | >2000cp | ≥12 pieces | <12 moves |
| Middlegame Complex | Standard middle | 1300-2000cp | 7-11 pieces | ≥12 moves |
| Middlegame Simple | Simplified middle | 1300-2000cp | 4-6 pieces | ≥12 moves |
| Endgame Complex | Early endgame | 800-1300cp | 3-6 pieces | Any |
| Endgame Simple | Late endgame | <800cp | ≤2 pieces | Any |

### Appendix C: Evaluation Profile Specifications

#### Fast Profile (Time Pressure)
**Target Speed**: 40x faster than comprehensive  
**Components** (6 total):
1. Material counting (negligible cost)
2. Piece-square tables (low cost)
3. Passed pawn detection (low cost)
4. Basic king safety (low cost)
5. Phase detection (negligible cost)
6. Score interpolation (negligible cost)

**Use Cases**:
- Time pressure (< 10 seconds remaining)
- Bullet games (1min+2s)
- Deep search nodes (depth > 8)

#### Tactical Profile (Sharp Positions)
**Target**: Maximize tactical accuracy  
**Components** (12 total):
- All Fast Profile components +
- Fork detection (medium cost)
- Pin detection (medium cost)
- Skewer detection (medium cost)
- King safety attackers (medium cost)
- Hanging piece detection (medium cost)
- Immediate capture threats (low cost)

**Use Cases**:
- Many checks available
- Material imbalance
- Kings exposed
- Tactical puzzles

#### Endgame Profile (Simplified Positions)
**Target**: Maximize conversion rate  
**Components** (10 total):
- Material counting
- King centralization (high bonus)
- Passed pawn exponential
- Pawn structure (doubled, isolated, backward)
- Pawn chain bonus
- Rook on 7th rank
- King safety (minimal)
- No tactical detection (low piece count)

**Use Cases**:
- Material < 1300cp
- Pieces ≤ 6
- Pawn endgames
- King and pawn endings

#### Comprehensive Profile (Default)
**Target**: Maximum accuracy  
**Components**: All 30 components  
**Use Cases**:
- Standard positions
- Opening/middlegame
- Ample time available
- Critical decisions

### Appendix D: Version History Quick Reference

| Version | Date | Status | Key Features | Performance vs v17.1 |
|---------|------|--------|--------------|---------------------|
| v18.2.1 | Dec 2025 | Planned | Draw threshold fix (25cp) | Expected: 50-60% |
| v18.2.0 | Dec 2025 | Experimental | Tactical + Positional combined | 33% (FAILURE) |
| v18.1.0 | Dec 2025 | Tested | Evaluation tuning | 64% (+100 ELO) |
| v18.0.0 | Dec 2025 | Tested | Tactical safety | 58% (+56 ELO) |
| v17.8.0 | Dec 2025 | Production | Repetition threshold 50cp | 52% |
| v17.7.0 | Dec 2025 | Rolled Back | Repetition threshold 200cp | Rapid regression |
| v17.1.0 | Nov 2025 | Stable | PV instant move disabled | Baseline (50%) |
| v14.1.0 | Oct 2025 | Stable | Smart time management | 25-day deployment |

---

**Document Control**  
**Version**: 1.0  
**Author**: V7P3R Development Team  
**Last Updated**: December 22, 2025  
**Next Review**: After v18.2.1 tournament results (Dec 28, 2025)  
**Status**: APPROVED FOR IMPLEMENTATION  

---
