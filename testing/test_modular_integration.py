#!/usr/bin/env python3
"""
Integration test for modular evaluation system with V7P3R engine.

Tests that:
1. Engine initializes with modular components
2. Context is calculated before search
3. Profile is selected correctly
4. Dynamic threefold threshold works
5. Engine still produces valid moves
"""

import unittest
import chess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


class TestModularIntegration(unittest.TestCase):
    """Test modular evaluation integration with engine"""
    
    def setUp(self):
        """Create engine instance for each test"""
        self.engine = V7P3REngine(use_fast_evaluator=True)
    
    def test_engine_initialization(self):
        """Test that engine initializes with modular components"""
        self.assertIsNotNone(self.engine.context_calculator)
        self.assertIsNotNone(self.engine.profile_selector)
        self.assertFalse(self.engine.use_modular_evaluation)  # Parallel testing mode
    
    def test_search_basic_move(self):
        """Test that search still produces valid moves"""
        board = chess.Board()
        move = self.engine.search(board, time_limit=1.0)
        
        self.assertIsNotNone(move)
        self.assertIn(move, board.legal_moves)
    
    def test_context_calculated(self):
        """Test that context is calculated during search"""
        board = chess.Board()
        move = self.engine.search(board, time_limit=1.0)
        
        # After root search, engine should have context
        self.assertHasAttr(self.engine, 'current_context')
        self.assertHasAttr(self.engine, 'current_profile')
        
        # Context should have valid data
        self.assertIsNotNone(self.engine.current_context)
        self.assertGreaterEqual(self.engine.current_context.time_remaining, 0)
    
    def assertHasAttr(self, obj, attr):
        """Helper to check attribute exists"""
        self.assertTrue(hasattr(obj, attr), f"Object missing attribute: {attr}")
    
    def test_profile_selection(self):
        """Test that profile is selected correctly"""
        board = chess.Board()
        move = self.engine.search(board, time_limit=5.0)
        
        # Should have selected a profile
        self.assertIsNotNone(self.engine.current_profile)
        self.assertGreater(self.engine.current_profile.module_count, 0)
        
        # Profile name should be valid
        valid_profiles = ["DESPERATE", "EMERGENCY", "FAST", "TACTICAL", "ENDGAME", "COMPREHENSIVE"]
        self.assertIn(self.engine.current_profile.name, valid_profiles)
    
    def test_time_pressure_profile(self):
        """Test emergency profile under time pressure"""
        board = chess.Board()
        move = self.engine.search(board, time_limit=0.5)  # Very short time
        
        # Should select fast or emergency profile
        self.assertIn(self.engine.current_profile.name, ["EMERGENCY", "FAST"])
    
    def test_desperate_profile(self):
        """Test desperate profile when down material"""
        # Down a queen
        board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
        move = self.engine.search(board, time_limit=2.0)
        
        # Should select desperate profile
        self.assertEqual(self.engine.current_profile.name, "DESPERATE")
        self.assertLess(self.engine.current_profile.module_count, 15)
    
    def test_comprehensive_profile(self):
        """Test profile selection for normal position with good time"""
        # Use position not in opening book
        board = chess.Board("rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
        move = self.engine.search(board, time_limit=10.0)  # Long time
        
        # Should select a valid profile
        valid_profiles = ["COMPREHENSIVE", "TACTICAL", "FAST", "EMERGENCY"]
        self.assertIn(self.engine.current_profile.name, valid_profiles)
        self.assertGreater(self.engine.current_profile.module_count, 0)
    
    def test_dynamic_threefold_threshold(self):
        """Test that dynamic threshold is calculated"""
        from v7p3r_eval_selector import get_threefold_threshold
        
        # Equal position
        board = chess.Board()
        move = self.engine.search(board, time_limit=1.0)
        threshold = get_threefold_threshold(self.engine.current_context)
        self.assertEqual(threshold, 0)  # Never accept draw when equal
        
        # Winning position
        board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")
        move = self.engine.search(board, time_limit=1.0)
        threshold = get_threefold_threshold(self.engine.current_context)
        self.assertGreater(threshold, 0)  # Should have higher threshold
    
    def test_search_multiple_positions(self):
        """Test engine handles different position types"""
        test_positions = [
            chess.Board(),  # Starting
            chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),  # Endgame
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),  # After e4
        ]
        
        for board in test_positions:
            move = self.engine.search(board, time_limit=1.0)
            self.assertIsNotNone(move)
            self.assertIn(move, board.legal_moves)


if __name__ == '__main__':
    unittest.main()
