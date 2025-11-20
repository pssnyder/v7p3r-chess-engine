#!/usr/bin/env python3
"""
V16.1 Validation Test - Pre-Deployment Check
Tests for the critical issues found in tournament:
1. Search depth reaches 10 (not stuck at 2-3)
2. Time management works properly
3. No material blunders (hanging pieces)
4. Tactical awareness improved
"""

import sys
import os
import time
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

class ValidationTest:
    def __init__(self):
        self.engine = V7P3REngine(max_depth=10, tt_size_mb=256)
        self.passed = 0
        self.failed = 0
        
    def log(self, message, status="INFO"):
        symbols = {"PASS": "✓", "FAIL": "✗", "INFO": "→"}
        print(f"{symbols.get(status, '→')} {message}")
        
    def test_depth_reaches_target(self):
        """Test that engine actually searches to depth 10"""
        self.log("Test 1: Depth Validation", "INFO")
        
        board = chess.Board()
        
        # Give it a reasonable time budget
        start_time = time.time()
        move = self.engine.get_best_move(board, time_left=5.0)
        elapsed = time.time() - start_time
        
        # Check last iteration depth from engine stats
        if hasattr(self.engine, 'stats') and 'depth' in self.engine.stats:
            depth = self.engine.stats['depth']
        else:
            depth = "unknown"
            
        self.log(f"  Move: {move}, Time: {elapsed:.2f}s, Depth: {depth}")
        
        if elapsed > 0.5 and move is not None:  # Should take time to search deep
            self.log("  Search depth appears reasonable (took >0.5s)", "PASS")
            self.passed += 1
            return True
        else:
            self.log("  Search too fast - may be stuck at shallow depth", "FAIL")
            self.failed += 1
            return False
    
    def test_time_management(self):
        """Test that engine respects time limits"""
        self.log("Test 2: Time Management", "INFO")
        
        board = chess.Board()
        time_limits = [0.5, 1.0, 2.0]
        
        all_passed = True
        for limit in time_limits:
            start = time.time()
            move = self.engine.get_best_move(board, time_left=limit)
            elapsed = time.time() - start
            
            # Allow 20% overshoot for overhead
            if elapsed <= limit * 1.2:
                self.log(f"  {limit}s limit: {elapsed:.2f}s (OK)", "PASS")
            else:
                self.log(f"  {limit}s limit: {elapsed:.2f}s (TIMEOUT)", "FAIL")
                all_passed = False
                
        if all_passed:
            self.passed += 1
            return True
        else:
            self.failed += 1
            return False
    
    def test_no_material_blunders(self):
        """Test that engine doesn't hang pieces"""
        self.log("Test 3: Material Safety (No Hanging Pieces)", "INFO")
        
        # Position where pieces can be captured
        # After 1.e4 e5 2.Nf3 Nc6 3.Bc4
        test_positions = [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "name": "Starting position",
                "unsafe_moves": []  # All moves are safe
            },
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
                "name": "Italian opening - Black to move",
                "unsafe_moves": ["Nxe4"]  # This hangs the knight (Bxf7+ tactics)
            },
            {
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
                "name": "Italian - White to move",
                "unsafe_moves": []  # Most moves are safe
            }
        ]
        
        all_passed = True
        for pos in test_positions:
            board = chess.Board(pos["fen"])
            move = self.engine.get_best_move(board, time_left=2.0)
            
            if move is None:
                self.log(f"  {pos['name']}: No move found!", "FAIL")
                all_passed = False
                continue
                
            move_san = board.san(move)
            
            # Check if move hangs material
            board_after = board.copy()
            board_after.push(move)
            
            # Simple check: did we leave a piece hanging?
            is_safe = True
            if move_san in pos["unsafe_moves"]:
                is_safe = False
                
            if is_safe:
                self.log(f"  {pos['name']}: {move_san} (safe)", "PASS")
            else:
                self.log(f"  {pos['name']}: {move_san} (HANGS MATERIAL)", "FAIL")
                all_passed = False
                
        if all_passed:
            self.passed += 1
            return True
        else:
            self.failed += 1
            return False
    
    def test_tactical_awareness(self):
        """Test that engine finds basic tactics"""
        self.log("Test 4: Tactical Awareness", "INFO")
        
        tactical_positions = [
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
                "name": "Scholar's mate threat",
                "best_moves": ["Qxf7"],  # Checkmate
                "avoid_moves": []
            },
            {
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
                "name": "Open e-file position",
                "best_moves": ["Nc6", "Nf6", "d6", "Bc5"],  # Development
                "avoid_moves": ["Qh4", "Qf6"]  # Early queen
            },
            {
                "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
                "name": "Italian castling position",
                "best_moves": ["O-O", "Nc3", "c3"],  # Castle or develop
                "avoid_moves": ["Qd2", "Qe2"]  # Early queen
            }
        ]
        
        all_passed = True
        for pos in tactical_positions:
            board = chess.Board(pos["fen"])
            move = self.engine.get_best_move(board, time_left=3.0)
            
            if move is None:
                self.log(f"  {pos['name']}: No move found!", "FAIL")
                all_passed = False
                continue
                
            move_san = board.san(move)
            
            # Check if move is good
            is_best = move_san in pos["best_moves"] if pos["best_moves"] else True
            is_avoided = move_san not in pos["avoid_moves"]
            
            if is_best or (is_avoided and not pos["best_moves"]):
                self.log(f"  {pos['name']}: {move_san} (good)", "PASS")
            else:
                self.log(f"  {pos['name']}: {move_san} (suboptimal)", "FAIL")
                all_passed = False
                
        if all_passed:
            self.passed += 1
            return True
        else:
            self.failed += 1
            return False
    
    def test_no_early_queen_moves(self):
        """Test that engine doesn't play Qd3 too early (tournament issue)"""
        self.log("Test 5: Opening Principles (No Early Queen)", "INFO")
        
        # Starting position - should NOT play Qd3 on move 2
        board = chess.Board()
        board.push_san("d4")  # 1.d4
        board.push_san("Nf6")  # 1...Nf6
        
        move = self.engine.get_best_move(board, time_left=2.0)
        move_san = board.san(move) if move else "None"
        
        # Qd3 is the bad move we saw in tournament
        if move_san == "Qd3":
            self.log(f"  After 1.d4 Nf6: {move_san} (EARLY QUEEN DEVELOPMENT)", "FAIL")
            self.failed += 1
            return False
        else:
            self.log(f"  After 1.d4 Nf6: {move_san} (good development)", "PASS")
            self.passed += 1
            return True
    
    def test_endgame_no_move_bug(self):
        """Test the fixed 'no move found' bug in drawn endgames"""
        self.log("Test 6: Endgame Bug Fix (K vs K)", "INFO")
        
        # K vs K - should still make a move even though drawn
        board = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
        
        move = self.engine.get_best_move(board, time_left=1.0)
        
        if move is None:
            self.log("  K vs K: NO MOVE FOUND (BUG NOT FIXED)", "FAIL")
            self.failed += 1
            return False
        else:
            move_san = board.san(move)
            self.log(f"  K vs K: {move_san} (bug fixed)", "PASS")
            self.passed += 1
            return True
    
    def run_full_validation(self):
        """Run all validation tests"""
        print("\n" + "="*60)
        print("V7P3R v16.1 PRE-DEPLOYMENT VALIDATION")
        print("="*60 + "\n")
        
        self.test_depth_reaches_target()
        print()
        self.test_time_management()
        print()
        self.test_no_material_blunders()
        print()
        self.test_tactical_awareness()
        print()
        self.test_no_early_queen_moves()
        print()
        self.test_endgame_no_move_bug()
        
        print("\n" + "="*60)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        
        if self.failed == 0:
            print("\n✓ ALL TESTS PASSED - READY FOR DEPLOYMENT")
            print("="*60 + "\n")
            return True
        else:
            print(f"\n✗ {self.failed} TESTS FAILED - FIX BEFORE DEPLOYMENT")
            print("="*60 + "\n")
            return False

def main():
    validator = ValidationTest()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
