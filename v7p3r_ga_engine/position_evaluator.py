"""
Compares v7p3r and Stockfish evaluations for positions.
"""

import chess
import random
import csv

class PositionEvaluator:
    def __init__(self, stockfish_config, v7p3r_score_class=None):
        """
        stockfish_config: dict for StockfishHandler
        v7p3r_score_class: class or function to evaluate positions with a ruleset
        """
        from v7p3r_engine.stockfish_handler import StockfishHandler
        self.stockfish = StockfishHandler(stockfish_config)
        self.v7p3r_score_class = v7p3r_score_class

    def load_positions(self, source="random", count=100):
        """
        Load FEN positions from a CSV file or generate random legal positions.
        If source is a file, expects one FEN per line or in a column named 'FEN'.
        """
        positions = []
        if source == "random":
            for _ in range(count):
                board = chess.Board()
                # Play a random number of random legal moves
                for _ in range(random.randint(5, 40)):
                    if board.is_game_over():
                        break
                    move = random.choice(list(board.legal_moves))
                    board.push(move)
                positions.append(board.fen())
        elif source.endswith(".csv"):
            with open(source, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "FEN" in row:
                        positions.append(row["FEN"])
                    if len(positions) >= count:
                        break
        else:
            # Assume it's a text file with one FEN per line
            with open(source, "r", encoding="utf-8") as f:
                for line in f:
                    fen = line.strip()
                    if fen:
                        positions.append(fen)
                    if len(positions) >= count:
                        break
        return positions[:count]

    def evaluate_ruleset(self, ruleset, positions):
        """
        Evaluate each position using v7p3rScore (with ruleset) and Stockfish.
        Returns two lists: v7p3r_evals, stockfish_evals
        """
        v7p3r_evals = []
        stockfish_evals = []
        # You must provide a v7p3r_score_class with an evaluate_position(board) method
        if self.v7p3r_score_class is None:
            raise ValueError("v7p3r_score_class must be provided for evaluation.")
        for fen in positions:
            board = chess.Board(fen)
            # Instantiate the scoring calculator with the ruleset for each evaluation
            v7p3r_score = self.v7p3r_score_class(ruleset).evaluate_position(board)
            # Use Stockfish's static evaluation from the side to move's perspective
            stockfish_score = self.stockfish.evaluate_position_from_perspective(board, board.turn)
            v7p3r_evals.append(v7p3r_score)
            stockfish_evals.append(stockfish_score)
        return v7p3r_evals, stockfish_evals

    def calculate_fitness(self, v7p3r_evals, stockfish_evals):
        """
        Calculate fitness as the negative mean squared error between v7p3r and Stockfish evals.
        Higher fitness = closer to Stockfish.
        """
        if not v7p3r_evals or not stockfish_evals:
            return float('-inf')
        mse = sum((a - b) ** 2 for a, b in zip(v7p3r_evals, stockfish_evals)) / len(v7p3r_evals)
        return -mse
