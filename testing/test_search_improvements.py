"""Test suite for v7p3r search and evaluation improvements.

Tests the following components:
1. Checkmate detection within 5 moves
2. Evaluation hierarchy (checkmate > stalemate > material > positional)
3. MVV-LVA move ordering
4. Quiescence search
"""

import sys
import os
import chess
import time
import unittest
from typing import List, Optional

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_config import v7p3rConfig
from v7p3r_play import v7p3rChess
from v7p3r_score import v7p3rScore
from v7p3r_search import v7p3rSearch
from v7p3r_mvv_lva import v7p3rMVVLVA
from v7p3r_quiescence import v7p3rQuiescence

class TestSearchImprovements(unittest.TestCase):
    def setUp(self):
        """Initialize test components with test config"""
        self.config = v7p3rConfig()
        self.engine_config = self.config.get_engine_config()
        self.engine_config.update({
            'depth': 4,
            'max_depth': 5,
            'use_quiescence': True,
            'use_move_ordering': True,
            'use_mvv_lva': True
        })
        
        # Initialize engine components
        self.engine = v7p3rChess('test_search_config.json')
        self.search = self.engine.engine.search_engine
        self.evaluator = self.engine.engine.scoring_calculator
        self.mvv_lva = self.engine.engine.search_engine.move_organizer.mvv_lva
        
    def test_checkmate_detection(self):
        """Test checkmate detection in simple positions"""
        # Fool's mate position (checkmate in 1)
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        move = self.search.find_checkmate_in_n(board, 1)
        self.assertIsNotNone(move, "Failed to detect checkmate in 1")
        
        # Scholar's mate position (checkmate in 1)
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/2B5/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 1 4")
        move = self.search.find_checkmate_in_n(board, 1)
        self.assertIsNotNone(move, "Failed to detect Scholar's mate")
        
    def test_evaluation_hierarchy(self):
        """Test evaluation prioritization"""
        # Normal position
        board = chess.Board()
        normal_score = self.evaluator.evaluate_position(board)
        
        # Checkmate position (should return infinite score)
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        mate_score = self.evaluator.evaluate_position(board)
        self.assertGreater(abs(mate_score), abs(normal_score), "Checkmate score not properly prioritized")
        
        # Stalemate position
        board = chess.Board("5bnr/4pppp/8/8/8/8/8/7k w - - 0 1")
        stale_score = self.evaluator.evaluate_position(board)
        self.assertEqual(stale_score, 0.0, "Stalemate score should be 0")
        
    def test_mvv_lva_ordering(self):
        """Test Most Valuable Victim - Least Valuable Attacker ordering"""
        # Position with multiple captures available
        board = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq - 0 1")
        moves = list(board.legal_moves)
        
        # Get ordered moves
        ordered_moves = self.engine.engine.move_organizer.order_moves(board, moves)
        
        # Verify captures are ordered by MVV-LVA
        for i in range(len(ordered_moves) - 1):
            if board.is_capture(ordered_moves[i]) and board.is_capture(ordered_moves[i + 1]):
                score1 = self.mvv_lva.calculate_mvv_lva_score(ordered_moves[i], board)
                score2 = self.mvv_lva.calculate_mvv_lva_score(ordered_moves[i + 1], board)
                self.assertGreaterEqual(score1, score2, "Captures not properly ordered by MVV-LVA")
                
    def test_quiescence_search(self):
        """Test quiescence search in tactical positions"""
        # Position with forced capture sequence (Black to move, can start taking pieces)
        board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1")
        
        # Test without quiescence - just evaluate the position
        self.engine.engine_config['use_quiescence'] = False
        score_without_q = self.evaluator.evaluate_position(board)
        nodes_without_q = 1  # Just one evaluation
        
        # Test with quiescence - evaluate position then search captures
        self.engine.engine_config['use_quiescence'] = True
        score_with_q = self.search.quiescence.search(board, float('-inf'), float('inf'))
        nodes_with_q = getattr(self.search.quiescence, 'nodes_searched', 0)
        
        # Quiescence search should evaluate more positions due to checking captures
        self.assertGreater(nodes_with_q, nodes_without_q,
                         "Quiescence search should evaluate more positions")

if __name__ == '__main__':
    unittest.main()
