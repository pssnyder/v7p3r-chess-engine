"""Performance benchmark tests for v7p3r chess engine.
Tests search speed, move generation efficiency, and evaluation performance."""

import os
import sys
import time
import chess
import unittest
from typing import List, Dict
from statistics import mean, stdev

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_search import v7p3rSearch
from v7p3r_score import v7p3rScore
from v7p3r_time import v7p3rTime
from v7p3r_pst import v7p3rPST
from v7p3r_rules import v7p3rRules

class TestV7P3RPerformance(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.pst = v7p3rPST()
        self.rules = v7p3rRules(pst=self.pst)
        self.score_calculator = v7p3rScore(rules_manager=self.rules, pst=self.pst)
        self.time_manager = v7p3rTime()
        self.search_engine = v7p3rSearch(
            scoring_calculator=self.score_calculator,
            time_manager=self.time_manager
        )
        
    def test_move_generation_speed(self):
        """Test move generation performance."""
        board = chess.Board()
        iterations = 1000
        times = []
        
        # Measure move generation time
        for _ in range(iterations):
            start_time = time.time()
            list(board.legal_moves)  # Generate moves
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_time = mean(times)
        std_dev = stdev(times)
        
        # Move generation should be fast (< 1ms on average)
        self.assertLess(avg_time, 0.001)
        print(f"\nMove generation: {avg_time*1000:.3f}ms ±{std_dev*1000:.3f}ms")
        
    def test_evaluation_speed(self):
        """Test position evaluation performance."""
        board = chess.Board()
        iterations = 1000
        times = []
        
        # Measure evaluation time
        for _ in range(iterations):
            start_time = time.time()
            self.score_calculator.evaluate_position(board, chess.WHITE)
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_time = mean(times)
        std_dev = stdev(times)
        
        # Position evaluation should be relatively fast (< 5ms)
        self.assertLess(avg_time, 0.005)
        print(f"Position evaluation: {avg_time*1000:.3f}ms ±{std_dev*1000:.3f}ms")
        
    def test_search_speed(self):
        """Test search performance at different depths."""
        board = chess.Board()
        depths = [1, 2, 3]  # Test reasonable depths
        
        for depth in depths:
            self.search_engine.depth = depth
            start_time = time.time()
            move = self.search_engine.search(board, chess.WHITE)
            end_time = time.time()
            
            search_time = end_time - start_time
            print(f"Depth {depth} search: {search_time*1000:.3f}ms")
            
            # Each depth should complete within reasonable time
            self.assertLess(search_time, 5.0 * depth)
            
    def test_quiescence_search_speed(self):
        """Test quiescence search performance."""
        # Position with captures available
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
        iterations = 100
        times = []
        
        # Measure quiescence search time
        for _ in range(iterations):
            start_time = time.time()
            self.search_engine._quiescence_search(board, float('-inf'), float('inf'), chess.WHITE)
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_time = mean(times)
        std_dev = stdev(times)
        
        # Quiescence search should be relatively fast (< 50ms)
        self.assertLess(avg_time, 0.05)
        print(f"Quiescence search: {avg_time*1000:.3f}ms ±{std_dev*1000:.3f}ms")
        
    def test_move_ordering_efficiency(self):
        """Test move ordering efficiency."""
        board = chess.Board()
        cutoff_counts = 0
        iterations = 100
        
        # Count beta cutoffs (good ordering should produce more cutoffs)
        for _ in range(iterations):
            self.search_engine.depth = 3
            alpha = float('-inf')
            beta = float('inf')
            score = self.search_engine._negamax(board, 3, alpha, beta, chess.WHITE)
            if score >= beta:
                cutoff_counts += 1
                
        cutoff_ratio = cutoff_counts / iterations
        print(f"Beta cutoff ratio: {cutoff_ratio:.2%}")
        
        # Good move ordering should achieve at least 70% cutoffs
        self.assertGreater(cutoff_ratio, 0.70)

if __name__ == '__main__':
    unittest.main(verbosity=2)
