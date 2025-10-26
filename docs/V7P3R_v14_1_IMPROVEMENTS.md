# V7P3R V14.1 Planned Improvements

## Implementation Status

### ✅ Ready for Implementation (V14.1)

#### 1. Workflow Documentation Cleanup
- **Status**: COMPLETED
- **Description**: Fixed markdown formatting in V7P3R_v14_WORKFLOW.md
- **Impact**: Better documentation readability and structure

#### 2. Enhanced Move Ordering with Threat Detection
- **Status**: IN PROGRESS
- **New Priority Order**:
  1. **Threats (NEW!)** - Pieces attacked by lower-value pieces
  2. **Castling** - King safety moves
  3. **Checks** - Putting opponent king in check
  4. **Captures** - Taking opponent pieces (MVV-LVA)
  5. **Development** - Moving pieces to better squares
  6. **Pawn Advancement** - Safe pawn moves
  7. **Quiet Moves** - Other positional moves
- **Implementation**: Add threat detection using bitboard analysis
- **Benefits**: Prioritizes defending valuable pieces under attack

#### 3. Dynamic Bishop Valuation
- **Status**: PLANNED
- **Logic**: 
  - Two bishops present: Bishop = 325 points (both bishops)
  - One bishop remaining: Bishop = 275 points
  - Knight maintains 300 points always
- **Philosophy**: Two bishops > two knights, one bishop < one knight
- **Implementation**: Dynamic evaluation based on remaining pieces
- **Performance**: Minimal impact - simple conditional logic

---

## 📋 Under Consideration (Future Versions)

### 4. Multi-PV Principal Variation Following
- **Concept**: Store multiple promising variations for instant play
- **Benefit**: Combat unpredictability, faster response to expected moves
- **Complexity**: Moderate - needs tracking multiple PV lines
- **Risk**: Memory usage increase, complexity in PV management
- **Investigation Needed**:
  - How many PV lines to track (2-3 reasonable)
  - When to activate multi-PV vs single PV
  - Performance impact on search time
  - Interaction with existing killer move heuristic

**Current Status**: Similar functionality exists with killer moves and history heuristic. Need to determine if multi-PV adds significant value beyond current caching mechanisms.

### 5. Performance Optimizations (NPS Increases)
- **Deep Tree Pruning**: More aggressive pruning at deeper levels
- **Game Phase Dynamic Evaluation**: 
  - Opening: Focus on development, castling
  - Middlegame: Full evaluation enabled
  - Endgame: Disable castling checks, focus on king activity
- **Complexity**: High - requires game phase detection
- **Risk**: Losing chess strength through aggressive pruning
- **Investigation Needed**:
  - Game phase boundaries (move count? material count?)
  - Which evaluations to disable in each phase
  - Performance vs strength trade-offs

### 6. Advanced Time Management
- **Compute Complexity Factors**: Adjust time based on position complexity
- **Intelligent Search Cutoffs**: Position-dependent depth limits
- **Enhanced Quiescence**: More quiescence in deeper searches
- **Target**: Reach 10-ply search depth consistently
- **Complexity**: Very High - requires position complexity analysis
- **Risk**: Time management bugs can lose games on time
- **Investigation Needed**:
  - How to measure position complexity quickly
  - Dynamic time allocation strategies
  - Balancing depth vs breadth in complex positions

---

## 🎯 V14.1 Implementation Plan

### Phase 1: Core Improvements (Current Sprint)
1. ✅ Fix workflow documentation formatting
2. 🔄 Implement threat detection in move ordering
3. 📋 Add dynamic bishop valuation
4. 🧪 Test performance vs V14.0

### Phase 2: Validation
- Run regression battle vs V12.6 and V14.0
- Measure NPS performance impact
- Validate chess strength improvements
- Document any behavioral changes

### Phase 3: Future Considerations Review
- Analyze multi-PV benefits vs complexity
- Prototype game phase detection
- Research time management improvements
- Plan V14.2+ roadmap based on V14.1 results

---

## 🔍 Key Questions for Review

1. **Multi-PV**: Do we need this beyond current killer moves + history heuristic?
2. **Game Phase**: How to detect phases without expensive calculation?
3. **Pruning**: How aggressive can we be without losing tactical awareness?
4. **Time Management**: Balance between depth and move quality?

These considerations require careful prototyping to avoid the performance regressions seen in V13.x series.