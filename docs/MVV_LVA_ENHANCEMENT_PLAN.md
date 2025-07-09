# MVV-LVA and Move Ordering Enhancement Plan

## Current State
- Multiple MVV-LVA implementations across different files
- Basic move ordering in `v7p3r_ordering.py`
- Sophisticated but disconnected MVV-LVA in `v7p3r_rules.py`
- Basic MVV-LVA in `v7p3r_score.py`

## Enhancement Goals
1. Consolidate MVV-LVA Logic
   - Move primary MVV-LVA implementation to `v7p3r_score.py`
   - Remove duplicate implementations
   - Add comprehensive piece value tables

2. Improve Move Ordering
   - Enhance `_order_move_score` to use full MVV-LVA
   - Add history heuristic
   - Add killer move heuristic
   - Better integrate with search

3. Position Evaluation Integration
   - Properly weight MVV-LVA in overall evaluation
   - Add tactical pattern recognition
   - Consider piece mobility in ordering

## Implementation Steps

### Phase 1: MVV-LVA Consolidation
1. Update `v7p3r_score.py`:
   ```python
   def _calculate_mvv_lva_score(self, move, board):
       # Enhanced MVV-LVA with:
       # - Piece values from config
       # - Safety evaluation
       # - Position context
   ```

2. Update `v7p3r_ordering.py`:
   ```python
   def _order_move_score(self, board, move):
       # Use scoring calculator's MVV-LVA
       # Add history and killer moves
       # Better prioritize tactical moves
   ```

### Phase 2: Move Ordering Enhancement
1. Add History Table:
   ```python
   class v7p3rOrdering:
       def __init__(self):
           self.history_table = {}
           self.killer_moves = [[]] # 2D array for different depths
   ```

2. Improve Move Selection:
   ```python
   def order_moves(self, board, moves, depth):
       # 1. Hash move (if available)
       # 2. Winning captures (by MVV-LVA)
       # 3. Killer moves
       # 4. History moves
       # 5. Quiet moves
   ```

### Phase 3: Integration
1. Update evaluation weights
2. Add pattern recognition
3. Add mobility scoring

## Testing Plan
1. Create test positions for:
   - Capture sequences
   - Tactical patterns
   - Quiet position improvement

2. Compare move ordering efficiency:
   - Node count reduction
   - Best move finding speed
   - Tactical accuracy

## Success Metrics
1. 20% reduction in nodes searched
2. 90% first-move accuracy in tactical positions
3. Improved capture sequencing in complex positions
