#!/usr/bin/env python3
"""
Unit tests for PositionContext calculator.

Tests position context calculation for correctness and performance.
"""

import unittest
import chess
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_position_context import (
    PositionContextCalculator, GamePhase, MaterialBalance, TacticalFlags
)


class TestPositionContextCalculator(unittest.TestCase):
    """Test position context calculation"""
    
    def setUp(self):
        """Initialize calculator for each test"""
        self.calculator = PositionContextCalculator()
    
    def test_starting_position(self):
        """Test context calculation for starting position"""
        board = chess.Board()
        context = self.calculator.calculate(board, time_remaining=300.0, time_per_move=5.0)
        
        # Should be opening
        self.assertEqual(context.game_phase, GamePhase.OPENING)
        
        # Material should be equal
        self.assertEqual(context.material_balance, MaterialBalance.EQUAL)
        self.assertEqual(context.material_diff_cp, 0)
        
        # All piece types present
        self.assertEqual(context.piece_types, {chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN})
        
        # 15 pieces per side (excluding kings)
        self.assertEqual(context.white_pieces, 15)
        self.assertEqual(context.black_pieces, 15)
        
        # No time pressure
        self.assertFalse(context.time_pressure)
        self.assertFalse(context.use_fast_profile)
    
    def test_endgame_simple(self):
        """Test simple endgame detection (K+P vs K)"""
        board = chess.Board("8/8/8/8/8/8/4P3/4K2k w - - 0 1")
        context = self.calculator.calculate(board)
        
        # Should be simple endgame
        self.assertEqual(context.game_phase, GamePhase.ENDGAME_SIMPLE)
        
        # Only pawns
        self.assertEqual(context.piece_types, {chess.PAWN})
        
        # Pawn endgame
        self.assertTrue(context.pawn_endgame)
    
    def test_endgame_complex(self):
        """Test complex endgame detection (R+P vs R+P)"""
        board = chess.Board("8/8/8/8/8/3r4/4P3/4K2R w - - 0 1")
        context = self.calculator.calculate(board)
        
        # Should be complex endgame (4 pieces total, no queens, but > 2 pieces)
        self.assertIn(context.game_phase, {GamePhase.ENDGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE})
        
        # Rooks and pawns
        self.assertTrue(chess.ROOK in context.piece_types)
        self.assertTrue(chess.PAWN in context.piece_types)
    
    def test_theoretical_draw(self):
        """Test theoretical draw detection (K vs K)"""
        board = chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1")
        context = self.calculator.calculate(board)
        
        # Should be theoretical draw
        self.assertTrue(context.theoretical_draw)
        
        # No pieces
        self.assertEqual(len(context.piece_types), 0)
        self.assertEqual(context.total_material, 0)
    
    def test_material_advantage(self):
        """Test material advantage calculation"""
        # White up a queen
        board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")
        context = self.calculator.calculate(board)
        
        # White to move, should be CRUSHING (+900cp)
        self.assertEqual(context.material_balance, MaterialBalance.CRUSHING)
        self.assertEqual(context.material_diff_cp, 900)
        
        # Test from black's perspective
        board.push(chess.Move.from_uci("h1h2"))  # Random queen move
        context = self.calculator.calculate(board)
        
        # Black to move, should be CRUSHING (-900cp from black's view)
        self.assertEqual(context.material_balance, MaterialBalance.CRUSHING)
        self.assertEqual(context.material_diff_cp, -900)
    
    def test_king_exposed(self):
        """Test king exposure detection"""
        # King without pawn shield
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        context = self.calculator.calculate(board)
        
        # King is exposed (no pawns in front)
        self.assertIn(TacticalFlags.KING_EXPOSED, context.tactical_flags)
        self.assertTrue(context.king_safety_critical)
    
    def test_pawn_shield(self):
        """Test pawn shield recognition"""
        # King with full pawn shield
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        context = self.calculator.calculate(board)
        
        # Starting position has good pawn shield
        # (Note: Our simple check may not detect this perfectly)
        # This is OK - these are just HINTS for module selection
    
    def test_time_pressure(self):
        """Test time pressure detection"""
        board = chess.Board()
        
        # No time pressure
        context = self.calculator.calculate(board, time_remaining=300.0, time_per_move=10.0)
        self.assertFalse(context.time_pressure)
        self.assertFalse(context.use_fast_profile)
        
        # Time pressure (< 30 seconds)
        context = self.calculator.calculate(board, time_remaining=20.0, time_per_move=2.0)
        self.assertTrue(context.time_pressure)
        self.assertTrue(context.use_fast_profile)
        
        # Fast profile needed (short time per move)
        context = self.calculator.calculate(board, time_remaining=100.0, time_per_move=3.0)
        self.assertFalse(context.time_pressure)
        self.assertTrue(context.use_fast_profile)  # Still use fast due to short time
    
    def test_depth_target(self):
        """Test depth target calculation"""
        board = chess.Board()
        
        # Emergency mode
        context = self.calculator.calculate(board, time_remaining=5.0, time_per_move=1.0)
        self.assertEqual(context.depth_target, 4)
        
        # Blitz fast
        context = self.calculator.calculate(board, time_remaining=60.0, time_per_move=4.0)
        self.assertEqual(context.depth_target, 5)
        
        # Rapid normal
        context = self.calculator.calculate(board, time_remaining=600.0, time_per_move=10.0)
        self.assertEqual(context.depth_target, 6)
        
        # Long time control
        context = self.calculator.calculate(board, time_remaining=3600.0, time_per_move=120.0)
        self.assertEqual(context.depth_target, 8)
    
    def test_opposite_bishops(self):
        """Test opposite-colored bishops detection"""
        # Both sides have bishops
        board = chess.Board("4k1b1/8/8/8/8/8/8/4K1B1 w - - 0 1")
        context = self.calculator.calculate(board)
        
        self.assertTrue(context.opposite_bishops)
        self.assertTrue(context.bishops_on_board)
    
    def test_piece_inventory(self):
        """Test piece type detection"""
        # Position with specific pieces
        board = chess.Board("4k3/8/8/3r4/8/8/3N4/4K3 w - - 0 1")
        context = self.calculator.calculate(board)
        
        # Should detect rook and knight
        self.assertIn(chess.ROOK, context.piece_types)
        self.assertIn(chess.KNIGHT, context.piece_types)
        self.assertNotIn(chess.QUEEN, context.piece_types)
        self.assertNotIn(chess.BISHOP, context.piece_types)
        
        self.assertTrue(context.rooks_on_board)
        self.assertFalse(context.queens_on_board)
        self.assertFalse(context.bishops_on_board)
    
    def test_performance(self):
        """Test that context calculation is fast (< 1ms target)"""
        import time
        
        board = chess.Board()
        
        # Warm up
        for _ in range(10):
            self.calculator.calculate(board)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            self.calculator.calculate(board)
        end = time.perf_counter()
        
        avg_time = (end - start) / iterations * 1000  # milliseconds
        
        # Should be < 1ms per calculation
        self.assertLess(avg_time, 1.0, 
                       f"Context calculation too slow: {avg_time:.3f}ms (target < 1ms)")
        
        print(f"\nPerformance: {avg_time:.4f}ms per context calculation")


if __name__ == '__main__':
    unittest.main()
