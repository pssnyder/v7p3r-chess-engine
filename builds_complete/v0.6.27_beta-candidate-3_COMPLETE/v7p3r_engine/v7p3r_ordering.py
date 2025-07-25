import chess
import logging

class v7p3rOrdering:
    """Class for move ordering in the V7P3R chess engine."""
    def __init__(self, engine_config: dict, scoring_calculator, logger: logging.Logger):
        self.engine_config = engine_config
        self.scoring_calculator = scoring_calculator
        self.logger = logger or logging.getLogger('v7p3r_engine_logger')

    def order_moves(self, board: chess.Board, moves, depth: int = 0) -> list:
        """Order moves for better alpha-beta pruning efficiency"""
        if isinstance(moves, chess.Move):
            moves = [moves]
        else:
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
        
        if self.logger:
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
                score += (self.engine_config.get('piece_values', {}).get(victim_type, 0) * 10) - self.engine_config.get('piece_values', {}).get(aggressor_type, 0)

        if move.promotion:
            score += self.scoring_calculator.rules('promotion_move_bonus', 3000.0)
            if move.promotion == chess.QUEEN:
                score += self.engine_config.get('piece_values', {}).get(chess.QUEEN, 9.0) * 100 # Ensure piece_values is used

        return score
