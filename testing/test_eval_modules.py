#!/usr/bin/env python3
"""
Unit tests for EvaluationModule registry and selection logic.
"""

import unittest
import chess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_eval_modules import (
    MODULE_REGISTRY, get_module, is_module_relevant, 
    get_active_modules, get_desperate_modules, get_emergency_modules,
    EvaluationCost, EvaluationCriticality
)
from v7p3r_position_context import (
    PositionContextCalculator, GamePhase, MaterialBalance
)


class TestEvaluationModules(unittest.TestCase):
    """Test evaluation module registry and selection"""
    
    def setUp(self):
        """Initialize calculator for each test"""
        self.calculator = PositionContextCalculator()
    
    def test_registry_completeness(self):
        """Test that registry is populated"""
        self.assertGreater(len(MODULE_REGISTRY), 25, 
                          "Should have 25+ modules in registry")
        
        # Essential modules must exist
        essential = ["material_counter", "piece_square_tables", "move_safety_checker"]
        for name in essential:
            module = get_module(name)
            self.assertIsNotNone(module, f"Essential module '{name}' missing")
            self.assertEqual(module.criticality, EvaluationCriticality.ESSENTIAL)
    
    def test_get_module(self):
        """Test module lookup by name"""
        material = get_module("material_counter")
        self.assertIsNotNone(material)
        self.assertEqual(material.name, "material_counter")
        self.assertEqual(material.criticality, EvaluationCriticality.ESSENTIAL)
        
        # Non-existent module
        fake = get_module("does_not_exist")
        self.assertIsNone(fake)
    
    def test_piece_requirements(self):
        """Test piece requirement filtering"""
        board = chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1")  # Only kings
        context = self.calculator.calculate(board)
        
        # Bishop pair should NOT be relevant (no bishops)
        bishop_pair = get_module("bishop_pair")
        self.assertFalse(is_module_relevant(bishop_pair, context))
        
        # Add bishops
        board = chess.Board("8/8/8/8/8/8/8/2B1K1Bk w - - 0 1")
        context = self.calculator.calculate(board)
        self.assertTrue(is_module_relevant(bishop_pair, context))
    
    def test_phase_requirements(self):
        """Test game phase filtering"""
        # Opening position
        board = chess.Board()
        context = self.calculator.calculate(board)
        self.assertEqual(context.game_phase, GamePhase.OPENING)
        
        development = get_module("development")
        self.assertTrue(is_module_relevant(development, context))
        
        # Endgame position (development not relevant)
        board = chess.Board("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
        context = self.calculator.calculate(board)
        self.assertEqual(context.game_phase, GamePhase.ENDGAME_SIMPLE)
        self.assertFalse(is_module_relevant(development, context))
    
    def test_desperate_mode_filtering(self):
        """Test desperate mode skips strategic evaluations"""
        # Down a queen (-900cp)
        board = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
        context = self.calculator.calculate(board)
        
        # Should be CRUSHING material imbalance
        self.assertEqual(context.material_balance, MaterialBalance.CRUSHING)
        self.assertLess(context.material_diff_cp, -300)
        
        # Strategic modules should be skipped
        bishop_pair = get_module("bishop_pair")
        self.assertFalse(is_module_relevant(bishop_pair, context))
        
        doubled_pawns = get_module("doubled_pawns")
        self.assertFalse(is_module_relevant(doubled_pawns, context))
        
        # Tactical modules should still be active
        hanging_pieces = get_module("hanging_pieces")
        self.assertTrue(is_module_relevant(hanging_pieces, context))
    
    def test_time_pressure_filtering(self):
        """Test time pressure skips expensive modules"""
        board = chess.Board()
        context = self.calculator.calculate(board, time_remaining=15.0, time_per_move=2.0)
        
        # Time pressure should be active
        self.assertTrue(context.time_pressure)
        
        # High-cost modules should be skipped
        piece_mobility = get_module("piece_mobility")
        self.assertFalse(is_module_relevant(piece_mobility, context))
        
        # Low-cost essentials should remain
        material = get_module("material_counter")
        self.assertTrue(is_module_relevant(material, context))
    
    def test_desperate_profile(self):
        """Test desperate module set"""
        desperate = get_desperate_modules()
        
        # Should be small (10-12 modules)
        self.assertGreaterEqual(len(desperate), 8)
        self.assertLessEqual(len(desperate), 15)
        
        # Must include essentials
        names = [m.name for m in desperate]
        self.assertIn("material_counter", names)
        self.assertIn("hanging_pieces", names)
        self.assertIn("capture_priority", names)
        
        # Should NOT include strategic evaluations
        self.assertNotIn("bishop_pair", names)
        self.assertNotIn("doubled_pawns", names)
    
    def test_emergency_profile(self):
        """Test emergency module set (time pressure)"""
        emergency = get_emergency_modules()
        
        # Should be minimal (5-8 modules)
        self.assertGreaterEqual(len(emergency), 4)
        self.assertLessEqual(len(emergency), 10)
        
        # Must include absolute essentials
        names = [m.name for m in emergency]
        self.assertIn("material_counter", names)
        self.assertIn("piece_square_tables", names)
        self.assertIn("move_safety_checker", names)
    
    def test_active_modules_starting_position(self):
        """Test module selection for starting position"""
        board = chess.Board()
        context = self.calculator.calculate(board)
        
        active = get_active_modules(context)
        
        # Should have many active modules (not desperate, not time pressure)
        self.assertGreater(len(active), 10)
        
        # Should include opening-specific modules
        names = [m.name for m in active]
        self.assertIn("development", names)
        self.assertIn("center_control", names)
    
    def test_active_modules_endgame(self):
        """Test module selection for endgame"""
        board = chess.Board("8/8/8/8/8/3r4/4P3/4K2R w - - 0 1")
        context = self.calculator.calculate(board)
        
        active = get_active_modules(context)
        names = [m.name for m in active]
        
        # Should include endgame modules
        self.assertIn("king_centralization", names)
        
        # Should NOT include opening modules
        self.assertNotIn("development", names)
    
    def test_module_cost_distribution(self):
        """Test that we have good cost distribution"""
        by_cost = {}
        for module in MODULE_REGISTRY:
            cost = module.cost
            by_cost[cost] = by_cost.get(cost, 0) + 1
        
        # Should have more cheap modules than expensive
        negligible = by_cost.get(EvaluationCost.NEGLIGIBLE, 0)
        low = by_cost.get(EvaluationCost.LOW, 0)
        high = by_cost.get(EvaluationCost.HIGH, 0)
        
        self.assertGreater(negligible + low, high, 
                          "Should have more cheap modules than expensive")
    
    def test_criticality_distribution(self):
        """Test criticality distribution"""
        by_crit = {}
        for module in MODULE_REGISTRY:
            crit = module.criticality
            by_crit[crit] = by_crit.get(crit, 0) + 1
        
        # Should have some essential modules
        essential = by_crit.get(EvaluationCriticality.ESSENTIAL, 0)
        self.assertGreaterEqual(essential, 3, "Need at least 3 essential modules")
        
        # Should have some optional modules
        optional = by_crit.get(EvaluationCriticality.OPTIONAL, 0)
        self.assertGreater(optional, 0, "Should have some optional modules")


if __name__ == '__main__':
    # Print summary first
    from v7p3r_eval_modules import print_module_summary
    print_module_summary()
    print("\n" + "="*60 + "\n")
    
    unittest.main()
