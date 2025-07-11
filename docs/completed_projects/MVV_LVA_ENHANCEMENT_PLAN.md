# MVV-LVA and Move Ordering Enhancement Plan

## Current State
- Unknown - MVV-LVA implementation in `v7p3r_mvv_lva.py`

## Enhancement Goals
1. Verify comprehensive piece value tables

2. Improve Move Ordering
   - Enhance `_order_move_score` to use full MVV-LVA
   - Better integrate with search

3. Position Evaluation Integration
   - Properly weight MVV-LVA in overall evaluation
   - Add tactical pattern recognition
   - Consider piece mobility in ordering

## Implementation Steps
### Step 1: Verification
1. Verify piece value tables
   - Ensure all pieces have correct values
   - Add missing game phase PSTs (e.g., bishops, knights)

### Step 2: MVV-LVA Enhancement
1. Update `_order_move_score`:
   - Use full MVV-LVA values
   - Check for consistency with standard chess piece values

### Step 3: Integration
1. Update evaluation weights and verify scoring
2. Add pattern recognition
3. Add mobility scoring

## Testing Plan
1. Query the available positions in the puzzle_db using `v7p3r_live_tuner.py` to test positions for:
   - Capture sequences
   - Tactical patterns
   - Quiet position improvement

2. Compare move ordering efficiency:
   - Node count reduction
   - Best move finding speed
   - Tactical accuracy

3. Complete test game using `v7p3r_play.py`:
   - Run default config, unless custom config is needed
   - Run with all enhancements enabled
   - Check chess_metrics database for results

## Success Metrics
1. 20% reduction in nodes searched
2. 90% first-move accuracy in tactical positions
3. Improved capture sequencing in complex positions
4. 50%+ overall puzzle solve success rate (51/100 puzzles)
5. Move durations under 30 seconds
6. 10%+ win rate vs Stockfish