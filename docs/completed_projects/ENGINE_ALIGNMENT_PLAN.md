# V7P3R Engine Alignment Plan

## Progress Update
- ✓ MVV-LVA implementation completed and tested
- ✓ Move ordering centralized in v7p3r_ordering.py
- ✓ Tempo and risk management modularized in v7p3r_tempo.py
- ✓ Test suite expanded for new modules
- ✓ Search integration with move ordering completed
- ✓ Iterative deepening search with PV tracking added
- ✓ Type safety improvements in test files
- ⧖ Alpha-beta root search implementation in progress
- ⧖ Search depth adaptation logic in progress

## Phase 1: Core Move Selection Flow
Aligning the engine's move selection process with the design document's flow:

1. Short Circuit Implementation
   - ✓ Implement checkmate detection within 5 moves
   - ✓ Add stalemate prevention logic
   - ✓ Add principle variation (PV) early exit conditions
   - Next: Integrate with tempo assessment for improved draw prevention

2. Evaluation Hierarchy
   - ✓ Checkmate evaluation
   - ✓ Stalemate evaluation
   - ✓ Primary scoring (Material, PST, MVV-LVA)
   - ✓ Secondary scoring (Castling rights)
   - Next: Add tempo scoring to evaluation hierarchy

3. Search Algorithm Enhancement
   - ✓ Implement proper node limiting based on PV scores
   - ✓ Add dynamic depth adjustment
   - ✓ Enhance move ordering with MVV-LVA
   - ✓ Implement proper quiescence search
   - ✓ Initialize iterative deepening framework
   - ✓ Add PV move tracking
   - ⧖ Complete alpha-beta root search
   - ⧖ Fine-tune search depth adaptation
   - Next: Finalize tempo integration into search

4. Tempo and Risk Management (New)
   - ✓ Modularize tempo and risk logic
   - ✓ Implement game phase detection
   - ✓ Add position history tracking
   - ✓ Create tempo scoring system
   - Next: Integrate with search and evaluation

## Integration Steps

1. Search Integration
   ```python
   def search(self, board, color):
       # 1. Check book moves first
       book_move = self.get_book_move(board)
       if book_move:
           return book_move
           
       # 2. Check for immediate threats
       mate_move = self.find_checkmate_in_n(board, 5)
       if mate_move:
           return mate_move
       
       # 3. Assess position's tempo characteristics
       tempo_score = self.tempo.assess_tempo(board, color)
       self.tempo.update_position_score(board, tempo_score)
       
       # 4. Start iterative deepening with tempo awareness
       if self.use_iterative_deepening:
           return self.iterative_deepening_search(board, color)
           
       # 5. Standard search with enhanced move ordering
       return self.ordered_depth_search(board, color)
   ```

2. Evaluation Integration
   ```python
   def evaluate_position(self, board):
       # 1. Check terminal conditions
       if self.is_checkmate_in_n(board, 5):
           return float('inf')
       if self.is_stalemate(board):
           return float('-inf')
           
       # 2. Primary scoring with tempo
       material_score = self.evaluate_material(board)
       position_score = self.evaluate_piece_squares(board)
       capture_score = self.evaluate_captures(board)
       tempo_score = self.tempo.assess_tempo(board, self.color)
       
       # 3. Secondary scoring
       castling_score = self.evaluate_castling_state(board)
       
       # 4. Risk assessment
       risk_score = (
           self.tempo.assess_checkmate_threats(board, self.color) +
           self.tempo.assess_stalemate_threats(board, self.color) +
           self.tempo.assess_drawish_positions(board)
       )
       
       return self.combine_scores([
           material_score,
           position_score,
           capture_score,
           tempo_score,
           castling_score,
           risk_score
       ])
   ```

3. Move Ordering Integration
   ```python
   def order_moves(self, board, moves, history):
       # Update history scores with tempo consideration
       if self.use_tempo:
           for move in moves:
               board_copy = board.copy()
               board_copy.push(move)
               tempo_score = self.tempo.assess_tempo(board_copy, board.turn)
               history[move] = history.get(move, 0) + tempo_score
               
       # Sort moves by combined score
       return sorted(moves, key=lambda m: (
           self.get_mvv_lva_score(board, m) +  # Tactical score
           history.get(m, 0) +                 # History score
           self.get_tempo_score(board, m)      # Tempo score
       ), reverse=True)
   ```

4. Position History Integration
   ```python
   def evaluate_and_update(self, board, color):
       # Get position evaluation
       score = self.evaluate_position(board)
       
       # Update tempo tracking
       self.tempo.update_position_score(board, score)
       
       # Check for zugzwang potential
       if self.tempo.assess_zugzwang(board, color) < -0.5:
           score *= 0.8  # Reduce score for positions with zugzwang risk
           
       return score
   ```

## Testing Plan

1. Create specific test positions for:
   - ✓ Checkmate detection
   - ✓ Stalemate prevention
   - ✓ Capture evaluation
   - ✓ Castling evaluation
   - ✓ Tempo evaluation
   - ✓ Type-safe move handling
   - ✓ PV move tracking
   - ✓ MVV-LVA integration
   - ⧖ Alpha-beta search correctness
   - ⧖ Iterative deepening stability
   - New: Combined position evaluation with tempo

2. Compare move selection against known good moves in critical positions:
   - Opening development positions
   - Zugzwang positions
   - Draw prevention scenarios
   - Complex tactical positions

3. Validate metrics and game recording functionality:
   - Add tempo metrics tracking
   - Record game phase transitions
   - Track position repetition statistics

## Success Criteria

1. ✓ Engine consistently identifies checkmates within 5 moves
2. ✓ Engine properly prevents stalemates when possible
3. ✓ Engine maintains material advantage in winning positions
4. ✓ Type-safe move handling in all code paths
5. ✓ Proper move ordering with MVV-LVA integration
6. ⧖ Iterative deepening search completes within time limits
7. ⧖ Alpha-beta search produces correct PV lines
8. Engine successfully maintains tempo advantage
9. Engine effectively prevents drawish positions
10. Engine achieves target win rate against Stockfish level 4
