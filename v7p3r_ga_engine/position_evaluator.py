"""
Compares v7p3r and Stockfish evaluations for positions with CUDA acceleration.
"""

import chess
import random
import csv
from typing import List, Tuple, Optional
from cuda_accelerator import CUDAAccelerator, NeuralNetworkEvaluator

class PositionEvaluator:
    def __init__(self, stockfish_config, v7p3r_score_class=None, use_cuda=True, 
                 use_nn_evaluator=False, nn_model_path=None):
        """
        stockfish_config: dict for StockfishHandler
        v7p3r_score_class: class or function to evaluate positions with a ruleset
        use_cuda: Enable CUDA acceleration for fitness calculations
        use_nn_evaluator: Use neural network instead of Stockfish for reference evaluations
        nn_model_path: Path to pre-trained neural network model
        """
        from v7p3r_engine.stockfish_handler import StockfishHandler
        self.stockfish = StockfishHandler(stockfish_config)
        self.v7p3r_score_class = v7p3r_score_class
        
        # Initialize CUDA acceleration
        self.cuda_accelerator = CUDAAccelerator(use_cuda=use_cuda)
        
        # Initialize neural network evaluator if requested
        self.nn_evaluator = None
        if use_nn_evaluator and nn_model_path:
            try:
                self.nn_evaluator = NeuralNetworkEvaluator(nn_model_path, self.cuda_accelerator)
                print("[PositionEvaluator] Neural network evaluator initialized")
            except Exception as e:
                print(f"[PositionEvaluator] Failed to initialize NN evaluator: {e}")
                print("[PositionEvaluator] Falling back to Stockfish")
        
        self.evaluation_cache = {}  # Cache for position evaluations
        self.cache_hits = 0
        self.cache_misses = 0

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
        Evaluate each position using v7p3rScore (with ruleset) and reference evaluator (Stockfish/NN).
        Returns two lists: v7p3r_evals, reference_evals
        Uses CUDA acceleration and caching for improved performance.
        """
        v7p3r_evals = []
        reference_evals = []
        
        # You must provide a v7p3r_score_class with an evaluate_position(board) method
        if self.v7p3r_score_class is None:
            raise ValueError("v7p3r_score_class must be provided for evaluation.")
        
        # Create a minimal engine config with the ruleset
        engine_config = {
            'verbose_output': False,
            'engine_ruleset': 'ga_test_ruleset'
        }
        
        # Create a real PST and logger for v7p3rScore
        import logging
        from v7p3r_engine.v7p3r_pst import v7p3rPST
        logger = logging.getLogger("ga_test")
        
        # Create a real PST object
        pst = v7p3rPST()
        
        # Instantiate v7p3rScore with proper parameters
        scorer = self.v7p3r_score_class(engine_config, pst, logger)
        # Manually set the ruleset
        scorer.rules = ruleset
        
        # Batch evaluate all positions
        print(f"      Evaluating {len(positions)} positions (GPU-accelerated)...")
        
        # Convert positions to boards
        boards = [chess.Board(fen) for fen in positions]
        
        # Get reference evaluations (Stockfish or Neural Network)
        if self.nn_evaluator and self.nn_evaluator.model:
            print(f"      Using neural network for reference evaluations...")
            reference_evals = self.nn_evaluator.batch_evaluate_positions(boards)
        else:
            print(f"      Using Stockfish for reference evaluations...")
            reference_evals = self.cuda_accelerator.parallel_stockfish_evaluation(
                boards, self.stockfish)
        
        # Evaluate with v7p3r (this part is harder to parallelize due to complex logic)
        for i, board in enumerate(boards):
            # Check cache first
            board_key = (board.fen(), str(sorted(ruleset.items())))
            if board_key in self.evaluation_cache:
                v7p3r_evals.append(self.evaluation_cache[board_key])
                self.cache_hits += 1
                continue
            
            self.cache_misses += 1
            try:
                v7p3r_score = scorer.evaluate_position(board)
                v7p3r_evals.append(v7p3r_score)
                # Cache the result
                self.evaluation_cache[board_key] = v7p3r_score
            except Exception as e:
                print(f"      Position {i+1}: v7p3r ERROR = {e}")
                v7p3r_evals.append(0.0)
        
        # Display cache statistics
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            cache_rate = (self.cache_hits / total_requests) * 100
            print(f"      Cache hit rate: {cache_rate:.1f}% ({self.cache_hits}/{total_requests})")
        
        return v7p3r_evals, reference_evals
    
    def batch_stockfish_evaluation(self, boards):
        """Evaluate multiple positions with Stockfish in batch for efficiency."""
        stockfish_evals = []
        for board in boards:
            try:
                score = self.stockfish.evaluate_position_from_perspective(board, board.turn)
                stockfish_evals.append(score)
            except Exception as e:
                stockfish_evals.append(0.0)
        return stockfish_evals

    def calculate_fitness(self, v7p3r_evals, reference_evals):
        """
        Calculate fitness as the negative mean squared error between v7p3r and reference evals.
        Higher fitness = closer to reference. Uses CUDA acceleration.
        """
        if not v7p3r_evals or not reference_evals:
            return float('-inf')
        
        # Use CUDA-accelerated fitness calculation
        return self.cuda_accelerator.batch_fitness_calculation(v7p3r_evals, reference_evals)
    
    def get_performance_stats(self):
        """Get performance statistics for monitoring."""
        memory_usage = self.cuda_accelerator.get_memory_usage()
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_size': len(self.evaluation_cache),
            'memory_usage': memory_usage
        }
    
    def clear_cache(self):
        """Clear evaluation cache and GPU memory."""
        self.evaluation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cuda_accelerator.clear_cache()
        print("      Caches cleared")
