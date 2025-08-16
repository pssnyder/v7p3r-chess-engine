# v7p3r_see.py
"""
Static Exchange Evaluation (SEE) for V7P3R Chess Engine
"""
from v7p3r_utils import PIECE_VALUES

class v7p3rSEE:
    def __init__(self):
        self.piece_values = PIECE_VALUES

    def evaluate(self, board, move):
        """
        Evaluate the static exchange for a given move.
        Returns the evaluation score of the exchange.
        """
        if not board.is_valid_move(move):
            return 0
        
        # Perform SEE
        return self.static_exchange_evaluation(board, move)

    def static_exchange_evaluation(self, board, move):
        """
        Perform the static exchange evaluation for the given move.
        This is a simplified version of SEE that evaluates the exchange based on material balance.
        """
        if not board.is_capture(move):
            return 0
        
        capturing_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)
        
        if not capturing_piece or not captured_piece:
            return 0
        
        # Basic material value evaluation
        capturing_value = self.piece_values.get(capturing_piece)
        captured_value = self.piece_values.get(captured_piece)
        if capturing_value is None or captured_value is None:
            return 0
        return capturing_value - captured_value