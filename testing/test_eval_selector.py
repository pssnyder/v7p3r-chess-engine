#!/usr/bin/env python3
"""
Unit tests for EvaluationProfileSelector.
"""

import unittest
import chess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_eval_selector import (
    EvaluationProfileSelector, EvaluationProfile,
    select_evaluation_profile, get_threefold_threshold
)
from v7p3r_position_context import (
    PositionContextCalculator, GamePhase, MaterialBalance
)


class TestEvaluationProfileSelector(unittest.TestCase):
    """Test evaluation profile selection logic"""
    
    def setUp(self):
        """Initialize calculator and selector for each test"""
        self.calculator = PositionContextCalculator()
        self.selector = EvaluationProfileSelector()
    
    def test_desperate_profile(self):
        """Test desperate profile when down material"""
        # Down a queen (-900cp)
        board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
        context = self.calculator.calculate(board)
        
        profile = self.selector.select_profile(context)
        
        self.assertEqual(profile.name, EvaluationProfile.DESPERATE)
        self.assertLess(profile.module_count, 15)
        self.assertIn("tactical recovery", profile.reason.lower())
    
    def test_emergency_profile(self):
        """Test emergency profile in time pressure"""
        board = chess.Board()
        context = self.calculator.calculate(board, time_remaining=15.0, time_per_move=2.0)
        
        profile = self.selector.select_profile(context)
        
        self.assertEqual(profile.name, EvaluationProfile.EMERGENCY)
        self.assertLess(profile.module_count, 10)
        self.assertIn("pressure", profile.reason.lower())
    
    def test_fast_profile(self):
        """Test fast profile for fast time control"""
        board = chess.Board()
        context = self.calculator.calculate(board, time_remaining=60.0, time_per_move=3.0)
        
        profile = self.selector.select_profile(context)
        
        self.assertEqual(profile.name, EvaluationProfile.FAST)
        self.assertGreater(profile.module_count, 10)
        self.assertLess(profile.module_count, 25)
    
    def test_tactical_profile(self):
        """Test tactical profile for exposed king"""
        # King exposed, no pawn shield
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        context = self.calculator.calculate(board, time_remaining=300.0, time_per_move=10.0)
        
        profile = self.selector.select_profile(context)
        
        # Should be tactical due to king exposure
        self.assertEqual(profile.name, EvaluationProfile.TACTICAL)
    
    def test_endgame_profile(self):
        """Test endgame profile"""
        # Simple pawn endgame with pawn shield (no king exposure)
        board = chess.Board("8/8/8/4k3/4p3/8/3PPP2/4K3 w - - 0 1")
        context = self.calculator.calculate(board, time_remaining=300.0, time_per_move=10.0)
        
        profile = self.selector.select_profile(context)
        
        self.assertEqual(profile.name, EvaluationProfile.ENDGAME)
        self.assertIn("endgame", profile.reason.lower())
    
    def test_comprehensive_profile(self):
        """Test comprehensive profile for normal position"""
        board = chess.Board()
        context = self.calculator.calculate(board, time_remaining=300.0, time_per_move=10.0)
        
        profile = self.selector.select_profile(context)
        
        # Starting position should be comprehensive (not tactical, not endgame)
        self.assertEqual(profile.name, EvaluationProfile.COMPREHENSIVE)
        self.assertGreater(profile.module_count, 15)
    
    def test_profile_priority_desperate_over_emergency(self):
        """Test that desperate takes priority over emergency"""
        # Down material AND time pressure
        board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
        context = self.calculator.calculate(board, time_remaining=10.0, time_per_move=1.0)
        
        profile = self.selector.select_profile(context)
        
        # DESPERATE should take priority
        self.assertEqual(profile.name, EvaluationProfile.DESPERATE)
    
    def test_dynamic_threefold_threshold_equal(self):
        """Test threefold threshold for equal position"""
        board = chess.Board()
        context = self.calculator.calculate(board)
        
        threshold = self.selector.get_dynamic_threefold_threshold(context)
        
        # Equal position should never accept draw
        self.assertEqual(threshold, 0)
    
    def test_dynamic_threefold_threshold_slight(self):
        """Test threefold threshold for slight advantage"""
        # Up a pawn (+100cp)
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        context = self.calculator.calculate(board)
        
        threshold = self.selector.get_dynamic_threefold_threshold(context)
        
        # Slight advantage should be very aggressive (10cp)
        self.assertEqual(threshold, 10)
    
    def test_dynamic_threefold_threshold_winning(self):
        """Test threefold threshold for winning position"""
        # Up a rook (+500cp)
        board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
        context = self.calculator.calculate(board)
        
        threshold = self.selector.get_dynamic_threefold_threshold(context)
        
        # Winning should still avoid draws (25cp)
        self.assertEqual(threshold, 25)
    
    def test_dynamic_threefold_threshold_crushing(self):
        """Test threefold threshold for crushing advantage"""
        # Up a queen (+900cp)
        board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")
        context = self.calculator.calculate(board)
        
        threshold = self.selector.get_dynamic_threefold_threshold(context)
        
        # Crushing can afford to repeat (50cp)
        self.assertEqual(threshold, 50)
    
    def test_cost_estimation(self):
        """Test that cost estimation is reasonable"""
        board = chess.Board()
        context = self.calculator.calculate(board)
        
        # Emergency should be cheapest
        emergency_ctx = self.calculator.calculate(board, 10.0, 1.0)
        emergency = self.selector.select_profile(emergency_ctx)
        
        # Comprehensive should be most expensive
        comprehensive_ctx = self.calculator.calculate(board, 300.0, 30.0)
        comprehensive = self.selector.select_profile(comprehensive_ctx)
        
        self.assertLess(emergency.estimated_cost_ms, comprehensive.estimated_cost_ms)
    
    def test_convenience_functions(self):
        """Test convenience wrapper functions"""
        board = chess.Board()
        context = self.calculator.calculate(board)
        
        # Test select_evaluation_profile
        profile = select_evaluation_profile(context)
        self.assertIsNotNone(profile)
        self.assertIsNotNone(profile.name)
        
        # Test get_threefold_threshold
        threshold = get_threefold_threshold(context)
        self.assertIsInstance(threshold, int)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 50)
    
    def test_tactical_detection_material_advantage(self):
        """Test tactical profile when we have advantage"""
        # Up a bishop, should use tactics to convert
        board = chess.Board("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1")
        context = self.calculator.calculate(board, 300.0, 10.0)
        
        profile = self.selector.select_profile(context)
        
        # Should be tactical (converting advantage) or endgame
        self.assertIn(profile.name, [EvaluationProfile.TACTICAL, EvaluationProfile.ENDGAME])
    
    def test_module_filtering(self):
        """Test that modules are properly filtered by context"""
        # Endgame position
        board = chess.Board("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
        context = self.calculator.calculate(board)
        
        profile = self.selector.select_profile(context)
        
        # Should NOT include opening modules
        module_names = [m.name for m in profile.modules]
        self.assertNotIn("development", module_names)
        self.assertNotIn("center_control", module_names)


if __name__ == '__main__':
    unittest.main()
