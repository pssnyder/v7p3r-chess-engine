# Search Perspective Consistency Fixes

## Problem Description

The search functions in V7P3R Chess Engine had inconsistencies in how they handled evaluation scores from different perspectives. The engine was sometimes returning scores that were not correctly adjusted for the player's perspective, which could lead to the engine making suboptimal move choices, especially when playing as Black.

## Analysis

Upon examining the code, we found three key issues:

1. The `evaluate_position_from_perspective` method in `v7p3rScore` would only provide scores for the player whose turn it was, returning 0 for the non-active player.

2. The search functions (`_minimax_search`, `_negamax_search`, and `_quiescence_search`) had inconsistent handling of perspective-based scores, particularly when negating values.

3. Quiescence search had potential for infinite recursion and used complex logic for score adjustment.

## Fixes Implemented

### 1. Fixed Perspective-Based Evaluation

Modified `evaluate_position_from_perspective` to correctly calculate scores from either player's perspective regardless of whose turn it is. This is done by calculating the scores for both players and then returning the difference from the requested perspective.

```python
def evaluate_position_from_perspective(self, board: chess.Board, color: Optional[chess.Color] = chess.WHITE) -> float:
    """Calculate position evaluation from specified player's perspective by delegating to scoring_calculator."""
    self.color_name = "White" if color == chess.WHITE else "Black"
    
    if not isinstance(color, chess.Color) or not board.is_valid():
        if self.monitoring_enabled and self.logger:
            self.logger.error(f"[Error] Invalid input for evaluation from perspective. Player: {self.color_name}, FEN: {board.fen() if hasattr(board, 'fen') else 'N/A'}")
        return 0.0

    # Calculate direct scores for both sides
    white_score = self.calculate_score(board=board, color=chess.WHITE)
    black_score = self.calculate_score(board=board, color=chess.BLACK)
    
    # Convert to score from the requested perspective
    if color == chess.WHITE:
        score = white_score - black_score
    else:
        score = black_score - white_score

    if self.monitoring_enabled and self.logger:
        self.logger.info(f"{self.color_name}'s perspective: {score:.3f} | FEN: {board.fen()}")
    self.score_dataset['evaluation'] = score
    return score
```

### 2. Simplified Minimax Search

Simplified and standardized the evaluation in minimax search to ensure consistent perspective handling:

```python
# Always evaluate from the current perspective
eval_result = self.scoring_calculator.evaluate_position_from_perspective(board, self.current_perspective)

# For minimax, if we're minimizing (opponent's turn), we need to negate the score
# The evaluation is already from our perspective, so we only negate for minimizing player
if not maximizing_player:
    eval_result = -eval_result
```

### 3. Simplified Negamax Search

Maintained the negamax algorithm's perspective handling, but removed quiescence search at terminal nodes to reduce complexity:

```python
# Always evaluate from the current perspective
eval_result = self.scoring_calculator.evaluate_position_from_perspective(board, self.current_perspective)

# In negamax, we need to flip the sign if it's not the current player's turn
if board.turn != self.current_perspective:
    eval_result = -eval_result
```

### 4. Limited Quiescence Search Depth

Modified quiescence search to use a strict depth limit to prevent infinite recursion:

```python
# Depth control - strict limit to prevent infinite recursion
max_q_depth = 2  # Hard limit of 2 plies for quiescence search
if current_ply >= max_q_depth:
    return stand_pat_score
```

## Testing

Created and ran both simplified and comprehensive test scripts to verify the correctness of search perspective handling. Tests confirmed that:

1. Direct evaluations from White's perspective show positive values for positions good for White and negative for positions good for Black.
2. Direct evaluations from Black's perspective show positive values for positions good for Black and negative for positions good for White.
3. Search functions (minimax, negamax, quiescence) maintain the correct perspective.

## Conclusion

The fixes ensure that V7P3R will always select moves based on perspective-correct scores, regardless of which color it plays as. This will improve the engine's performance, especially when playing as Black.

## Future Improvements

1. Consider adding more comprehensive perspective tests as part of the engine's test suite.
2. Improve handling of logging file access issues.
3. Further optimize quiescence search for more stable evaluations at terminal nodes.
