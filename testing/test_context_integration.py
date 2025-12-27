#!/usr/bin/env python3
"""
Integration tests for PositionContext with V7P3R engine.

Tests that the new context calculator:
1. Works alongside existing engine code
2. Produces consistent results
3. Doesn't cause performance regressions
4. Correctly identifies game phases and material balance
"""

import unittest
import chess
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import both old engine and new context calculator
from v7p3r_position_context import PositionContextCalculator, GamePhase, MaterialBalance
import v7p3r


class TestContextIntegration(unittest.TestCase):
    """Integration tests for PositionContext with existing engine"""
    
    def setUp(self):
        """Initialize calculator and engine"""
        self.calculator = PositionContextCalculator()
        self.engine = v7p3r.V7P3REngine()
    
    def test_engine_still_runs(self):
        """Verify engine still functions after importing context module"""
        board = chess.Board()
        
        # Engine should find a move
        move = self.engine.search(board, time_limit=1.0, depth=3)
        
        self.assertIsNotNone(move)
        self.assertIsInstance(move, chess.Move)
        self.assertTrue(board.is_legal(move))
    
    def test_context_with_engine_positions(self):
        """Test context calculation on positions the engine will evaluate"""
        test_positions = [
            # Starting position
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", GamePhase.OPENING, MaterialBalance.EQUAL),
            
            # Middlegame/opening transition (move 6 is still early)
            ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6", GamePhase.OPENING, MaterialBalance.EQUAL),
            
            # Endgame simple (R+P vs R)
            ("8/8/8/8/8/3r4/4P3/4K2R w - - 0 1", GamePhase.ENDGAME_COMPLEX, MaterialBalance.SLIGHT_ADVANTAGE),
            
            # Winning position (up a queen)
            ("4k3/8/8/8/8/8/8/4K2Q w - - 0 1", GamePhase.ENDGAME_SIMPLE, MaterialBalance.CRUSHING),
        ]
        
        for fen, expected_phase, expected_balance in test_positions:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                context = self.calculator.calculate(board)
                
                self.assertEqual(context.game_phase, expected_phase,
                               f"Phase mismatch for {fen}: got {context.game_phase}, expected {expected_phase}")
                self.assertEqual(context.material_balance, expected_balance,
                               f"Balance mismatch for {fen}: got {context.material_balance}, expected {expected_balance}")
    
    def test_context_performance_overhead(self):
        """Ensure context calculation adds minimal overhead"""
        board = chess.Board()
        
        # Benchmark context calculation
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            context = self.calculator.calculate(board)
        context_time = (time.perf_counter() - start) * 1000  # milliseconds
        
        # Should be < 50ms for 1000 iterations (0.05ms each)
        self.assertLess(context_time, 50.0,
                       f"Context calculation too slow: {context_time:.2f}ms for {iterations} iterations")
        
        print(f"\nContext calculation: {context_time:.2f}ms for {iterations} iterations ({context_time/iterations:.4f}ms each)")
    
    def test_tactical_game_positions(self):
        """Test context on tactical positions from real games"""
        # Position with king exposure
        fen_exposed_king = "r1bq1rk1/ppp2ppp/2n5/3np3/1b1P4/2NB1N2/PPP2PPP/R1BQK2R w KQ - 0 8"
        board = chess.Board(fen_exposed_king)
        context = self.calculator.calculate(board)
        
        # Should detect some tactical themes
        self.assertGreater(len(context.tactical_flags), 0, "Should detect tactical flags in complex position")
        
        # Verify engine can still evaluate this position
        move = self.engine.search(board, time_limit=1.0, depth=3)
        self.assertIsNotNone(move)
    def test_time_pressure_context(self):
        """Test context calculation with different time controls"""
        board = chess.Board()
        
        # Blitz (5min+4s, ~4s per move)
        context_blitz = self.calculator.calculate(board, time_remaining=300.0, time_per_move=4.0)
        self.assertTrue(context_blitz.use_fast_profile)
        self.assertEqual(context_blitz.depth_target, 5)
        
        # Rapid (15min+10s, ~10s per move)
        context_rapid = self.calculator.calculate(board, time_remaining=900.0, time_per_move=10.0)
        self.assertFalse(context_rapid.use_fast_profile)
        self.assertEqual(context_rapid.depth_target, 6)
        
        # Emergency (< 30s)
        context_emergency = self.calculator.calculate(board, time_remaining=15.0, time_per_move=2.0)
        self.assertTrue(context_emergency.time_pressure)
        self.assertTrue(context_emergency.use_fast_profile)
        self.assertEqual(context_emergency.depth_target, 4)
    
    def test_endgame_transitions(self):
        """Test context as game transitions from middlegame to endgame"""
        # Middlegame with queens (move 15, most pieces traded)
        fen_mg = "r3k2r/ppp2ppp/2n5/3q4/3Q4/2N5/PPP2PPP/R3K2R w KQkq - 0 15"
        board_mg = chess.Board(fen_mg)
        context_mg = self.calculator.calculate(board_mg)
        # 10 pieces total, has queens, move 15 → should be middlegame
        self.assertIn(context_mg.game_phase, {GamePhase.MIDDLEGAME_SIMPLE, GamePhase.MIDDLEGAME_COMPLEX})
        self.assertTrue(context_mg.queens_on_board)
        
        # After queen trade → endgame (move 20 to ensure past opening)
        fen_eg = "r3k2r/ppp2ppp/2n5/8/8/2N5/PPP2PPP/R3K2R w KQkq - 0 20"
        board_eg = chess.Board(fen_eg)
        context_eg = self.calculator.calculate(board_eg)
        # 8 pieces, no queens → should be endgame or simple middlegame
        self.assertIn(context_eg.game_phase, {GamePhase.ENDGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE})
        self.assertFalse(context_eg.queens_on_board)
    
    def test_material_calculation_accuracy(self):
        """Verify material calculation matches expected values"""
        # Up a pawn (+100cp)
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        context = self.calculator.calculate(board)
        self.assertEqual(context.material_diff_cp, 100)
        self.assertEqual(context.material_balance, MaterialBalance.SLIGHT_ADVANTAGE)
        
        # Up a knight (+320cp)
        board = chess.Board("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1")
        context = self.calculator.calculate(board)
        self.assertEqual(context.material_diff_cp, 320)
        self.assertEqual(context.material_balance, MaterialBalance.ADVANTAGE)
        
        # Up a rook (+500cp)
        board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
        context = self.calculator.calculate(board)
        self.assertEqual(context.material_diff_cp, 500)
        self.assertEqual(context.material_balance, MaterialBalance.WINNING)
        
        # Up a queen (+900cp)
        board = chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1")
        context = self.calculator.calculate(board)
        self.assertEqual(context.material_diff_cp, 900)
        self.assertEqual(context.material_balance, MaterialBalance.CRUSHING)
    
    def test_no_conflicts_with_existing_evaluation(self):
        """Ensure context module doesn't interfere with current evaluation"""
        positions = [
            chess.Board(),  # Starting position
            chess.Board("4k3/8/8/8/8/8/8/4K2Q w - - 0 1"),  # Simple endgame
            chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"),  # Open game
        ]
        
        for board in positions:
            with self.subTest(fen=board.fen()):
                # Calculate context
                context = self.calculator.calculate(board)
                
                # Engine should still evaluate normally
                move = self.engine.search(board, time_limit=1.0, depth=3)
                
                self.assertIsNotNone(move)
                self.assertTrue(board.is_legal(move))
    
    def test_context_diagnostic_output(self):
        """Generate diagnostic output for various positions"""
        print("\n" + "="*70)
        print("POSITION CONTEXT DIAGNOSTIC OUTPUT")
        print("="*70)
        
        test_cases = [
            ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("Sicilian Dragon", "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 1"),
            ("Rook Endgame", "8/8/8/8/8/3r4/4P3/4K2R w - - 0 1"),
            ("Queen vs Pawns", "4k3/pppppppp/8/8/8/8/8/4K2Q w - - 0 1"),
            ("Theoretical Draw", "8/8/8/8/8/8/8/4K2k w - - 0 1"),
        ]
        
        for name, fen in test_cases:
            board = chess.Board(fen)
            context = self.calculator.calculate(board)
            
            print(f"\n{name}:")
            print(f"  FEN: {fen}")
            print(f"  Phase: {context.game_phase.value}")
            print(f"  Material: {context.material_balance.value} ({context.material_diff_cp:+d}cp)")
            print(f"  Total Material: {context.total_material}cp")
            print(f"  Pieces: {context.white_pieces} vs {context.black_pieces}")
            print(f"  Piece Types: {', '.join(chess.piece_name(pt) for pt in context.piece_types)}")
            print(f"  Tactical Flags: {', '.join(f.value for f in context.tactical_flags) if context.tactical_flags else 'None'}")
            print(f"  King Safety Critical: {context.king_safety_critical}")
            print(f"  Endgame Flags: pawn={context.pawn_endgame}, pure_piece={context.pure_piece_endgame}, theoretical_draw={context.theoretical_draw}")
            print(f"  Depth Target: {context.depth_target}")


if __name__ == '__main__':
    # Run with verbose output to see diagnostic information
    unittest.main(verbosity=2)
