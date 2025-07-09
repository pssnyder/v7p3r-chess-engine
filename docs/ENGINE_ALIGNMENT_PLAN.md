# V7P3R Engine Alignment Plan

## Phase 1: Core Move Selection Flow
Aligning the engine's move selection process with the design document's flow:

1. Short Circuit Implementation
   - Implement checkmate detection within 5 moves
   - Add stalemate prevention logic
   - Add principle variation (PV) early exit conditions

2. Evaluation Hierarchy
   - Reorganize evaluation order:
     1. Checkmate evaluation
     2. Stalemate evaluation
     3. Primary scoring (Material, PST, MVV-LVA)
     4. Secondary scoring (Castling rights)

3. Search Algorithm Enhancement
   - Implement proper node limiting based on PV scores
   - Add dynamic depth adjustment
   - Enhance move ordering with MVV-LVA
   - Implement proper quiescence search

## Files to Modify

1. `v7p3r_play.py`
   - Implement proper game loop flow
   - Add move validation and fallback logic
   - Integrate metrics recording

2. `v7p3r_search.py`
   - Implement proper PV handling
   - Add node limiting
   - Add dynamic depth adjustment
   - Enhance short circuit conditions

3. `v7p3r_score.py`
   - Reorganize evaluation hierarchy
   - Implement proper MVV-LVA
   - Add castling evaluation

4. `v7p3r_rules.py`
   - Enhance checkmate detection
   - Add stalemate prevention
   - Implement draw prevention

## Implementation Steps

1. Update Search Logic
```python
def search(self, board, color):
    # 1. Check book moves first
    book_move = self.get_book_move(board)
    if book_move:
        return book_move
        
    # 2. Check for immediate checkmate
    mate_move = self.find_checkmate_in_n(board, 5)
    if mate_move:
        return mate_move
        
    # 3. Start iterative deepening if enabled
    if self.use_iterative_deepening:
        return self.iterative_deepening_search(board, color)
        
    # 4. Standard search with move ordering
    return self.ordered_depth_search(board, color)
```

2. Update Evaluation Order
```python
def evaluate_position(self, board):
    # 1. Check terminal conditions
    if self.is_checkmate_in_n(board, 5):
        return float('inf')
    if self.is_stalemate(board):
        return float('-inf')
        
    # 2. Primary scoring
    material_score = self.evaluate_material(board)
    position_score = self.evaluate_piece_squares(board)
    capture_score = self.evaluate_captures(board)
    
    # 3. Secondary scoring
    castling_score = self.evaluate_castling_state(board)
    
    return self.combine_scores([
        material_score,
        position_score,
        capture_score,
        castling_score
    ])
```

## Testing Plan

1. Create specific test positions for:
   - Checkmate detection
   - Stalemate prevention
   - Capture evaluation
   - Castling evaluation

2. Compare move selection against known good moves in critical positions

3. Validate metrics and game recording functionality

## Success Criteria

1. Engine consistently identifies checkmates within 5 moves
2. Engine properly prevents stalemates when possible
3. Engine maintains material advantage in winning positions
4. Engine successfully records all required metrics
5. Engine achieves target win rate against Stockfish level 4
