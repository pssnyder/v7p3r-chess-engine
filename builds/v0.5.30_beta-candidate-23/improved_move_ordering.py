
# improved_move_ordering.py
# Enhanced move ordering for more efficient search

import chess

class MoveOrdering:
    def __init__(self):
        self.killer_moves = {}  # Store killer moves by depth
        self.history_table = {}  # Store history heuristic scores
        self.counter_moves = {}  # Store counter moves

        # Initialize history table for all piece types and squares
        for piece_type in range(1, 7):  # 1-6 for all piece types
            for from_sq in range(64):
                for to_sq in range(64):
                    self.history_table[(piece_type, from_sq, to_sq)] = 0

    def order_moves(self, board, moves, hash_move=None, depth=0):
        """
        Order moves for better alpha-beta pruning efficiency

        Args:
            board: The current chess board state
            moves: List of legal moves to order
            hash_move: Move from transposition table if available
            depth: Current search depth

        Returns:
            Ordered list of moves
        """
        # Store move scores for later sorting
        move_scores = []

        for move in moves:
            # Calculate score for move
            score = self._score_move(board, move, hash_move, depth)
            move_scores.append((move, score))

        # Sort moves by score in descending order
        move_scores.sort(key=lambda x: x[1], reverse=True)

        # Return just the moves, not the scores
        return [move for move, _ in move_scores]

    def _score_move(self, board, move, hash_move, depth):
        """Score a move for ordering"""
        # Base score
        score = 0

        # 1. Hash move gets highest priority
        if hash_move and move == hash_move:
            return 10000000

        # 2. Captures scored by MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
        if board.is_capture(move):
            victim_type = board.piece_at(move.to_square).piece_type
            aggressor_type = board.piece_at(move.from_square).piece_type

            # Most valuable victim (queen=9, rook=5, etc.) minus least valuable aggressor
            victim_value = self._get_piece_value(victim_type)
            aggressor_value = self._get_piece_value(aggressor_type)

            # MVV-LVA formula: 10 * victim_value - aggressor_value
            score = 1000000 + 10 * victim_value - aggressor_value

            # Bonus for promotions
            if move.promotion:
                score += 900000  # High score for promotions

            return score

        # 3. Killer moves (non-capture moves that caused a beta cutoff)
        if depth in self.killer_moves and move in self.killer_moves[depth]:
            return 900000

        # 4. Counter moves (moves that are good responses to the previous move)
        last_move = board.peek() if board.move_stack else None
        if last_move:
            counter_key = (last_move.from_square, last_move.to_square)
            if counter_key in self.counter_moves and self.counter_moves[counter_key] == move:
                return 800000

        # 5. History heuristic (moves that have caused cutoffs in similar positions)
        piece_type = board.piece_at(move.from_square).piece_type
        history_key = (piece_type, move.from_square, move.to_square)
        history_score = self.history_table.get(history_key, 0)

        # 6. Promotions (already handled in captures, but add for non-capture promotions)
        if move.promotion:
            score += 700000

        # 7. Checks
        board.push(move)
        gives_check = board.is_check()
        board.pop()

        if gives_check:
            score += 600000

        # Add history score
        score += history_score

        return score

    def update_killer_move(self, move, depth):
        """Update killer move table with a move that caused a beta cutoff"""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = [move]
        elif move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            # Keep only the best 2 killer moves per depth
            self.killer_moves[depth] = self.killer_moves[depth][:2]

    def update_history_score(self, board, move, depth):
        """Update history heuristic score for a move that caused a beta cutoff"""
        piece_type = board.piece_at(move.from_square).piece_type
        history_key = (piece_type, move.from_square, move.to_square)

        # Update history score using depth-squared bonus
        self.history_table[history_key] += depth * depth

    def update_counter_move(self, last_move, current_move):
        """Update counter move table"""
        if last_move:
            counter_key = (last_move.from_square, last_move.to_square)
            self.counter_moves[counter_key] = current_move

    def _get_piece_value(self, piece_type):
        """Get standard piece value"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100  # High value for king captures
        }
        return piece_values.get(piece_type, 0)
