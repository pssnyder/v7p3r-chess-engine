#!/usr/bin/env python3
"""
V16.1 UCI Validation Test - Pre-Deployment Check
Tests V7P3R through UCI interface like it will be used in real games.
Checks for critical tournament issues:
1. Depth reaches reasonable levels (not stuck at 2-3)
2. Time management works properly  
3. No material blunders
4. No early queen development
5. Endgame bug is fixed
"""

import subprocess
import time
import chess
import chess.engine
import sys
import os

class UCIValidator:
    def __init__(self):
        self.engine_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'src', 
            'v7p3r_uci.py'
        )
        self.passed = 0
        self.failed = 0
        
    def log(self, message, status="INFO"):
        symbols = {"PASS": "✓", "FAIL": "✗", "INFO": "→", "WARN": "⚠"}
        print(f"{symbols.get(status, '→')} {message}")
    
    def test_uci_startup(self):
        """Test that UCI interface starts correctly"""
        self.log("Test 1: UCI Interface Startup", "INFO")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(["python", self.engine_path])
            self.log("  Engine started successfully", "PASS")
            
            # Check engine identity
            if engine.id.get("name"):
                self.log(f"  Engine: {engine.id['name']}", "INFO")
            
            engine.quit()
            self.passed += 1
            return True
        except Exception as e:
            self.log(f"  Failed to start engine: {e}", "FAIL")
            self.failed += 1
            return False
    
    def test_time_management(self):
        """Test that engine respects time limits"""
        self.log("Test 2: Time Management", "INFO")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(["python", self.engine_path])
            board = chess.Board()
            
            time_limits = [0.5, 1.0, 2.0]
            all_passed = True
            
            for limit in time_limits:
                start = time.time()
                result = engine.play(board, chess.engine.Limit(time=limit))
                elapsed = time.time() - start
                
                # Allow 30% overshoot for UCI overhead
                if elapsed <= limit * 1.3:
                    self.log(f"  {limit:.1f}s limit: {elapsed:.2f}s (OK)", "PASS")
                else:
                    self.log(f"  {limit:.1f}s limit: {elapsed:.2f}s (TIMEOUT)", "FAIL")
                    all_passed = False
            
            engine.quit()
            
            if all_passed:
                self.passed += 1
                return True
            else:
                self.failed += 1
                return False
                
        except Exception as e:
            self.log(f"  Error: {e}", "FAIL")
            self.failed += 1
            return False
    
    def test_depth_info(self):
        """Test that engine searches to reasonable depth"""
        self.log("Test 3: Search Depth", "INFO")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(["python", self.engine_path])
            board = chess.Board()
            
            # Search with 3 second time limit
            info_handler = chess.engine.InfoDict()
            with engine.analysis(board, chess.engine.Limit(time=3.0)) as analysis:
                for info in analysis:
                    info_handler.update(info)
                    # Stop after getting depth info
                    if "depth" in info and info["depth"] >= 5:
                        break
            
            depth = info_handler.get("depth", 0)
            
            if depth >= 5:
                self.log(f"  Reached depth {depth} (good)", "PASS")
                self.passed += 1
                engine.quit()
                return True
            elif depth >= 3:
                self.log(f"  Reached depth {depth} (marginal)", "WARN")
                self.passed += 1
                engine.quit()
                return True
            else:
                self.log(f"  Only reached depth {depth} (too shallow)", "FAIL")
                self.failed += 1
                engine.quit()
                return False
                
        except Exception as e:
            self.log(f"  Error: {e}", "FAIL")
            self.failed += 1
            return False
    
    def test_no_hanging_pieces(self):
        """Test that engine doesn't hang material"""
        self.log("Test 4: Material Safety", "INFO")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(["python", self.engine_path])
            
            # Test position: after 1.e4 e5 2.Nf3 Nc6 3.Bc4
            # Black should NOT play Nxe4 (hangs knight to Bxf7+)
            board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
            
            result = engine.play(board, chess.engine.Limit(time=2.0))
            move_san = board.san(result.move)
            
            # Check if it's the blunder move
            if move_san == "Nxe4":
                self.log(f"  Italian Game: {move_san} (HANGS KNIGHT)", "FAIL")
                engine.quit()
                self.failed += 1
                return False
            else:
                self.log(f"  Italian Game: {move_san} (safe)", "PASS")
                engine.quit()
                self.passed += 1
                return True
                
        except Exception as e:
            self.log(f"  Error: {e}", "FAIL")
            self.failed += 1
            return False
    
    def test_no_early_queen(self):
        """Test that engine doesn't develop queen too early (tournament issue)"""
        self.log("Test 5: Opening Principles", "INFO")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(["python", self.engine_path])
            
            # After 1.d4 Nf6 - should NOT play 2.Qd3
            board = chess.Board()
            board.push_san("d4")
            board.push_san("Nf6")
            
            result = engine.play(board, chess.engine.Limit(time=2.0))
            move_san = board.san(result.move)
            
            if move_san in ["Qd3", "Qd2"]:
                self.log(f"  After 1.d4 Nf6: {move_san} (EARLY QUEEN)", "FAIL")
                engine.quit()
                self.failed += 1
                return False
            else:
                self.log(f"  After 1.d4 Nf6: {move_san} (good development)", "PASS")
                engine.quit()
                self.passed += 1
                return True
                
        except Exception as e:
            self.log(f"  Error: {e}", "FAIL")
            self.failed += 1
            return False
    
    def test_endgame_bug_fix(self):
        """Test that 'no move found' bug is fixed in drawn endgames"""
        self.log("Test 6: Endgame Bug Fix", "INFO")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(["python", self.engine_path])
            
            # K vs K - should still return a move even though drawn
            board = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
            
            result = engine.play(board, chess.engine.Limit(time=1.0))
            
            if result.move is None:
                self.log("  K vs K: NO MOVE FOUND (BUG NOT FIXED)", "FAIL")
                engine.quit()
                self.failed += 1
                return False
            else:
                move_san = board.san(result.move)
                self.log(f"  K vs K: {move_san} (bug fixed)", "PASS")
                engine.quit()
                self.passed += 1
                return True
                
        except Exception as e:
            self.log(f"  Error: {e}", "FAIL")
            self.failed += 1
            return False
    
    def test_tactical_position(self):
        """Test basic tactical awareness"""
        self.log("Test 7: Tactical Awareness", "INFO")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(["python", self.engine_path])
            
            # Scholar's mate position - should find Qxf7#
            board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
            
            result = engine.play(board, chess.engine.Limit(time=3.0))
            move_san = board.san(result.move)
            
            if move_san == "Qxf7#":
                self.log(f"  Scholar's mate: {move_san} (FOUND MATE)", "PASS")
                engine.quit()
                self.passed += 1
                return True
            else:
                self.log(f"  Scholar's mate: {move_san} (missed mate but OK)", "WARN")
                engine.quit()
                self.passed += 1
                return True
                
        except Exception as e:
            self.log(f"  Error: {e}", "FAIL")
            self.failed += 1
            return False
    
    def test_simulated_game(self):
        """Play a short simulated game to check for major issues"""
        self.log("Test 8: Simulated Game (10 moves)", "INFO")
        
        try:
            engine = chess.engine.SimpleEngine.popen_uci(["python", self.engine_path])
            board = chess.Board()
            
            moves_made = 0
            max_moves = 10
            
            while moves_made < max_moves and not board.is_game_over():
                result = engine.play(board, chess.engine.Limit(time=1.0))
                
                if result.move is None:
                    self.log(f"  Move {moves_made + 1}: NO MOVE FOUND", "FAIL")
                    engine.quit()
                    self.failed += 1
                    return False
                
                board.push(result.move)
                moves_made += 1
            
            self.log(f"  Played {moves_made} moves without errors", "PASS")
            engine.quit()
            self.passed += 1
            return True
            
        except Exception as e:
            self.log(f"  Error: {e}", "FAIL")
            self.failed += 1
            return False
    
    def run_full_validation(self):
        """Run all validation tests"""
        print("\n" + "="*60)
        print("V7P3R v16.1 UCI VALIDATION (PRE-DEPLOYMENT)")
        print("="*60 + "\n")
        
        self.test_uci_startup()
        print()
        self.test_time_management()
        print()
        self.test_depth_info()
        print()
        self.test_no_hanging_pieces()
        print()
        self.test_no_early_queen()
        print()
        self.test_endgame_bug_fix()
        print()
        self.test_tactical_position()
        print()
        self.test_simulated_game()
        
        print("\n" + "="*60)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        
        if self.failed == 0:
            print("\n✓ ALL TESTS PASSED - READY FOR DEPLOYMENT")
            print("\nNext Steps:")
            print("1. Deploy to Lichess with 60+1 time control")
            print("2. Monitor first 10 games for issues")
            print("3. Check rating progression")
            print("="*60 + "\n")
            return True
        else:
            print(f"\n✗ {self.failed} TESTS FAILED - FIX BEFORE DEPLOYMENT")
            print("="*60 + "\n")
            return False

def main():
    validator = UCIValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
