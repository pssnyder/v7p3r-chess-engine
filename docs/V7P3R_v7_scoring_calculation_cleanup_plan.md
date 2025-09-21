# V7P3R Scoring Calculation Cleanup Plan

## Current Mess:
- `v7p3r_scoring_calculation.py` (v7.0 - simple, fast, tournament winner)
- `v7p3r_scoring_calculation_v93.py` (complex, slow - currently being used!)
- `v7p3r_scoring_calculation_v94_alpha.py` (our attempt to fix v9.3)
- `v7p3r_scoring_calculation_v94_beta.py` (refined attempt)

## The Problem:
**The current engine (`v7p3r.py`) is importing and using v9.3 scoring which is SLOW and COMPLEX!**

```python
from v7p3r_scoring_calculation_v93 import V7P3RScoringCalculationV93  # PROBLEM!
self.scoring_calculator = V7P3RScoringCalculationV93(self.piece_values)  # SLOW!
```

## Performance Analysis:

### V7.0 Scoring (FAST - tournament winner):
- ~150 lines of code
- 7 simple evaluation components
- Material, King Safety, Development, Castling, Rook Coordination, Center Control, Endgame
- NO complex heuristics, NO opening penalties, NO tactical search
- **Result: 79.5% tournament win rate, fast NPS**

### V9.3 Scoring (SLOW - current):
- ~559 lines of code 
- Complex piece-square tables
- Developmental heuristics with move counting
- Early game penalties with multiple sub-functions
- Tactical awareness, pin detection, fork detection
- **Result: Terrible NPS, over-engineered**

## Root Cause of NPS Problem:
**Every node evaluation in v9.3 does:**
1. Complex piece-square table lookups
2. Developmental heuristic calculations
3. Early game penalty calculations (checking move counts, piece positions)
4. Tactical pattern detection
5. Multiple helper function calls

**V7.0 evaluation was SIMPLE:**
1. Count material
2. Basic king safety check
3. Simple development bonus (just check if piece off back rank)
4. Basic castling/center control
5. Done.

## Solution:
1. **Backup current versions** to development folder
2. **Replace `v7p3r_scoring_calculation.py` with the version that should be used**
3. **Update `v7p3r.py` to import the correct class**
4. **Delete the versioned files** to eliminate confusion
5. **Test performance improvement**

## Decision Point:
Which scoring calculation should be the "one true version"?

**Option A: Go back to v7.0** (simple, fast, proven winner)
**Option B: Use v9.4-beta** (enhanced but try to keep it fast)
**Option C: Create new "v10" version** (v7.0 base + minimal tactical improvements)

## Recommendation:
**Start with v7.0 scoring to get NPS back up, then add minimal improvements**
