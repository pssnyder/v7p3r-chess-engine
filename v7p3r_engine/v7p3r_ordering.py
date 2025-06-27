
class v7p3rOrdering:
    """Class for move ordering in the V7P3R chess engine."""
    def order_moves(self, board: chess.Board, moves, depth: int = 0):
        """Order moves for better alpha-beta pruning efficiency"""
        if isinstance(moves, chess.Move):
            moves = [moves]

        if not moves or not isinstance(board, chess.Board) or not board.is_valid():
            if self.logger:
                self.logger.error(f"Invalid input to order_moves: board type {type(board)} | FEN: {board.fen() if hasattr(board, 'fen') else 'N/A'}")
            return []
        
        if not moves or not isinstance(board, chess.Board) or not board.is_valid():
            if self.logger:
                self.logger.error(f"Invalid input to order_moves: board type {type(board)} | FEN: {board.fen() if hasattr(board, 'fen') else 'N/A'}")
            return []

        move_scores = []
        
        for move in moves:
            if not board.is_legal(move):
                if self.logger:
                    self.logger.warning(f"Illegal move passed to order_moves: {move} | FEN: {board.fen()}")
                continue
            
            score = self._order_move_score(board, move, depth)
            move_scores.append((move, score))

        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # max_moves_to_evaluate can come from v7p3r_config_data or game_settings_config_data (performance section)
        # self.engine_config should have this resolved value
        max_moves_to_evaluate = self.engine_config.get('max_moves_evaluated', None)
        
        if max_moves_to_evaluate is not None and max_moves_to_evaluate > 0:
            if self.logging_enabled and self.logger:
                self.logger.debug(f"Limiting ordered moves from {len(move_scores)} to {max_moves_to_evaluate}")
            move_scores = move_scores[:max_moves_to_evaluate]

        if self.logging_enabled and self.logger:
            self.logger.debug(f"Ordered moves at depth {depth}: {[f'{move} ({score:.2f})' for move, score in move_scores]} | FEN: {board.fen()}")
        
        return [move for move, _ in move_scores]

    def _order_move_score(self, board: chess.Board, move: chess.Move, depth: int = 0) -> float:
        """Calculate a score for a move for ordering purposes."""
        score = 0.0

        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.is_checkmate():
            temp_board.pop()
            # Get checkmate move bonus from the current ruleset
            return self.scoring_calculator.rules('checkmate_move_bonus', 1000000.0)
        
        if temp_board.is_check(): # Check after move is made
            score += self.scoring_calculator.rules('check_move_bonus', 10000.0)
        temp_board.pop() # Pop before is_capture check on original board state

        if board.is_capture(move):
            score += self.scoring_calculator.rules('capture_move_bonus', 4000.0)
            victim_type = board.piece_type_at(move.to_square)
            aggressor_type = board.piece_type_at(move.from_square)
            if victim_type and aggressor_type:
                score += (self.piece_values.get(victim_type, 0) * 10) - self.piece_values.get(aggressor_type, 0)

        if depth < len(self.killer_moves) and move in self.killer_moves[depth]:
            score += self.scoring_calculator.rules('killer_move_bonus', 2000.0)
        
        if move.promotion:
            score += self.scoring_calculator.rules('promotion_move_bonus', 3000.0)
            if move.promotion == chess.QUEEN:
                score += self.piece_values.get(chess.QUEEN, 9.0) * 100 # Ensure piece_values is used

        return score
    