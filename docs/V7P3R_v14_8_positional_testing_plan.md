# V14.8 Positional Testing Plan

**Date**: October 31, 2025  
**Version**: V7P3R v14.8  
**Purpose**: Comprehensive positional analysis before Arena tournament testing

---

## Testing Rationale

Before releasing V14.8 into Arena tournament testing against V14.0, V12.6, and V14.3, we need to:

1. **Validate tactical strength** - Ensure simplified V14.8 hasn't lost tactical sharpness
2. **Identify tuning opportunities** - Find specific weaknesses to address
3. **Measure time management** - Verify 55% time usage is competitive
4. **Assess position types** - Understand where V14.8 excels vs struggles
5. **Estimate rating range** - Predict tournament performance

---

## Test Configuration

### Universal Puzzle Analyzer Parameters

```
Engine: V7P3R v14.8 (via v7p3r_v14_8.bat wrapper)
Puzzles: 100
Rating Range: 1200-2000
Time Per Position: 20 seconds (suggested, not enforced)
Time Control: 30+2 (matches Arena tournament format)
Comparison Engine: Stockfish (top 5 moves)
```

### What Gets Measured

**Sequence Metrics:**
- **Linear Accuracy**: Simple % of positions solved correctly
- **Weighted Accuracy**: Later positions in sequences weighted higher (exponential: 1, 1.5, 2.25, 3.375...)
- **Perfect Sequences**: Puzzles where engine found ALL moves in solution
- **Position Depth Performance**: Accuracy degradation through sequence (position 1, 2, 3, etc.)

**Tactical Themes:**
- Performance by theme (crushing, hangingPiece, mate, fork, pin, skewer, etc.)
- Identifies strong themes (potential strengths)
- Identifies weak themes (tuning opportunities)

**Time Management:**
- Average time per move
- Time management score (0-1, measures adherence to suggested 20s)
- Exceeded suggestions count
- Time pressure incidents

**Rating Estimation:**
- Average rating of perfect sequences
- Average rating of high-accuracy puzzles (≥80%)
- Estimated V7P3R rating range

**Stockfish Comparison:**
- Rank of V7P3R's move in Stockfish's top 5
- Scoring: 5pts (1st), 4pts (2nd), 3pts (3rd), 2pts (4th), 1pt (5th), 0pts (not in top 5)
- Average Stockfish score per position

---

## Expected Outcomes & Tuning Insights

### If Linear Accuracy < 40%
**Issue**: Fundamental evaluation problems  
**Action**: Review material values, basic piece-square tables

### If Weighted Accuracy << Linear Accuracy
**Issue**: Performance degrades in complex sequences  
**Action**: Improve search depth, time allocation for difficult positions

### If Perfect Sequence Rate < 20%
**Issue**: Consistency problems, missing moves in sequences  
**Action**: Review move ordering, ensure best moves searched first

### If Time Management Score < 0.6
**Issue**: Poor time discipline  
**Action**: Adjust time allocation algorithm, emergency stop thresholds

### If Time Exceeded Rate > 50%
**Issue**: Consistently overshooting time suggestions  
**Action**: More aggressive time cutoffs, better iterative deepening

### Theme-Specific Issues
- **Weak on "mate" theme**: King safety evaluation needs tuning
- **Weak on "hangingPiece"**: Need minimal blunder prevention (root-level only)
- **Weak on "endgame"**: Endgame evaluation may need adjustment
- **Weak on "opening"**: Opening book or early-game piece values
- **Weak on "middlegame"**: Core tactical evaluation issues

### Position Depth Performance
- **Position 1 strong, Position 3+ weak**: Search depth insufficient for deep calculation
- **Consistent across depths**: Good sign, engine is stable
- **Erratic performance**: Move ordering or time management issues

---

## Post-Test Analysis Workflow

### Step 1: Review Overall Metrics
```
Target: Weighted Accuracy ≥50%
Target: Perfect Sequence Rate ≥25%
Target: Time Management Score ≥0.7
```

### Step 2: Identify Weakest Themes
- Sort themes by weighted accuracy (ascending)
- Focus on themes with ≥10 puzzles (statistically significant)
- Target bottom 3-5 themes for improvement

### Step 3: Analyze Position Depth Curve
- Compare Position 1 accuracy vs Position 3+ accuracy
- If drop >20%: Depth/time management issue
- If drop <10%: Stable, good sign

### Step 4: Time Management Review
- Average time per move vs suggested 20s
- If avg >25s: Too slow, needs cutoffs
- If avg <10s: Too fast, potentially missing tactics
- Target: 15-20s average (good depth without excessive time)

### Step 5: Rating Estimation
- Check average rating of perfect sequences
- If avg perfect rating <1400: Concerning, below expected
- If avg perfect rating 1600-1800: Good, competitive
- If avg perfect rating >1900: Excellent, strong tactical play

### Step 6: Stockfish Comparison
- Average Stockfish rank of chosen moves
- If avg rank ≤2: Engine choosing top moves, excellent
- If avg rank 3-4: Decent, finding reasonable moves
- If avg rank >4 or many 0-rank: Evaluation issues

---

## Tuning Priorities Based on Results

### Priority 1: Critical Issues (Must Fix Before Arena)
- Perfect sequence rate <15%
- Time management score <0.5
- Average Stockfish rank >4
- Weighted accuracy <35%

### Priority 2: Important Improvements (Should Fix)
- Specific weak themes with >20% accuracy gap vs strongest theme
- Position depth drop >20%
- Time exceeded rate >60%
- Rating estimation <1400

### Priority 3: Nice-to-Have Optimizations (Optional)
- Fine-tune strong themes to be even stronger
- Optimize time allocation per position type
- Improve consistency (reduce variance)

---

## V14.8 Baseline Expectations

Based on V14.8's simplified architecture (disabled safety filtering, V14.0 foundation):

**Realistic Targets:**
- Linear Accuracy: 45-55%
- Weighted Accuracy: 40-50%
- Perfect Sequence Rate: 20-30%
- Time Management Score: 0.6-0.8
- Average Rating (perfect): 1500-1700
- Average Stockfish Rank: 2.5-3.5

**Concerning Thresholds:**
- Linear Accuracy: <40%
- Weighted Accuracy: <35%
- Perfect Sequence Rate: <15%
- Time Management Score: <0.5
- Average Rating (perfect): <1400
- Average Stockfish Rank: >4.5

**Exceptional Performance:**
- Linear Accuracy: >60%
- Weighted Accuracy: >55%
- Perfect Sequence Rate: >35%
- Time Management Score: >0.8
- Average Rating (perfect): >1800
- Average Stockfish Rank: <2.0

---

## Integration with Arena Testing

After positional testing and any tuning:

1. **If Results Good (meet realistic targets)**:
   - Proceed directly to Arena tournament
   - Expect 55-65% overall performance
   - V14.8 should be competitive with V14.0

2. **If Results Mixed (some targets missed)**:
   - Make targeted improvements to weak themes
   - Quick validation test (10-20 puzzles)
   - Then proceed to Arena

3. **If Results Poor (many targets missed)**:
   - Consider creating V14.9 with specific fixes
   - Re-test with positional analyzer
   - Delay Arena until results improve

---

## Test Execution Status

**Started**: October 31, 2025  
**Status**: Running (100 puzzles @ 20s/position = ~30-45 minutes expected)  
**Output**: Will generate JSON results file with comprehensive analysis  
**Report**: Automated report with all metrics above will be printed and saved

---

## Next Actions After Results

1. **Review generated JSON file** - Contains all raw data
2. **Analyze printed report** - Summary metrics and insights
3. **Document findings** - Create v14_8_positional_results_analysis.md
4. **Make tuning decisions** - Based on weakest areas
5. **Implement fixes** (if critical issues found) - Create V14.8.1 or proceed
6. **Arena tournament** - Final validation against other versions
