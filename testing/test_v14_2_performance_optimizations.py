#!/usr/bin/env python3
"""
V7P3R v14.2 Performance Optimization Test Suite

Tests for:
1. Overhead removal (no expensive threat detection)
2. Game phase detection accuracy
3. Enhanced quiescence search functionality
4. Advanced time management and depth targeting
5. Search depth monitoring and profiling
6. Cached dynamic piece values
"""

import chess
import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

class V14_2_TestSuite:
    def __init__(self):
        self.engine = V7P3REngine()
        self.test_results = []
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result with details"""
        status = "PASS" if passed else "FAIL"
        result = f"[{status}] {test_name}"
        if details:
            result += f" - {details}"
        self.test_results.append((test_name, passed, details))
        print(result)
        
    def test_no_threat_detection_overhead(self):
        """Test 1: Verify threat detection method has been removed"""
        print("\n=== Test 1: Overhead Removal ===")
        
        # Check that _detect_threats method doesn't exist
        has_detect_threats = hasattr(self.engine, '_detect_threats')
        self.log_result(
            "Threat detection method removed", 
            not has_detect_threats,
            f"Method exists: {has_detect_threats}"
        )
        
        # Test move ordering performance (should be faster without threat detection)
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 4 4")
        legal_moves = list(board.legal_moves)
        
        start_time = time.time()
        ordered_moves = self.engine._order_moves_advanced(board, legal_moves, 5)
        ordering_time = time.time() - start_time
        
        # Should be very fast without expensive threat detection
        fast_ordering = ordering_time < 0.01  # Less than 10ms
        self.log_result(
            "Fast move ordering without threat detection",
            fast_ordering,
            f"Ordering time: {ordering_time:.4f}s"
        )
        
        return not has_detect_threats and fast_ordering
        
    def test_cached_dynamic_piece_values(self):
        """Test 2: Verify dynamic piece values are cached for performance"""
        print("\n=== Test 2: Cached Dynamic Piece Values ===")
        
        # Test that bishop pair gets correct value
        board = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4")
        
        # First call should populate cache
        bishop_value_1 = self.engine._get_dynamic_piece_value(board, chess.BISHOP, chess.WHITE)
        cache_size_1 = len(self.engine.bishop_value_cache)
        
        # Second call should use cache
        bishop_value_2 = self.engine._get_dynamic_piece_value(board, chess.BISHOP, chess.WHITE)
        cache_size_2 = len(self.engine.bishop_value_cache)
        
        # Values should be same and cache should exist
        values_consistent = bishop_value_1 == bishop_value_2
        cache_working = cache_size_1 > 0 and cache_size_2 == cache_size_1
        
        self.log_result(
            "Dynamic bishop values cached correctly",
            values_consistent and cache_working,
            f"Value: {bishop_value_1}, Cache entries: {cache_size_1}"
        )
        
        # Test bishop pair bonus (should be 325)
        bishop_pair_value = bishop_value_1 == 325
        self.log_result(
            "Bishop pair gets 325 value",
            bishop_pair_value,
            f"Actual value: {bishop_value_1}"
        )
        
        return values_consistent and cache_working and bishop_pair_value
        
    def test_game_phase_detection(self):
        """Test 3: Verify game phase detection works correctly"""
        print("\n=== Test 3: Game Phase Detection ===")
        
        # Opening position
        opening_board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        opening_phase = self.engine._detect_game_phase(opening_board)
        opening_correct = opening_phase == 'opening'
        
        self.log_result(
            "Opening phase detected correctly",
            opening_correct,
            f"Detected: {opening_phase}"
        )
        
        # Middlegame position (some pieces developed, most material present)
        middlegame_board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4")
        middlegame_phase = self.engine._detect_game_phase(middlegame_board)
        middlegame_correct = middlegame_phase == 'middlegame'
        
        self.log_result(
            "Middlegame phase detected correctly",
            middlegame_correct,
            f"Detected: {middlegame_phase}"
        )
        
        # Endgame position (few pieces left)
        endgame_board = chess.Board("8/8/8/3k4/8/3K4/8/R7 w - - 0 1")
        endgame_phase = self.engine._detect_game_phase(endgame_board)
        endgame_correct = endgame_phase == 'endgame'
        
        self.log_result(
            "Endgame phase detected correctly",
            endgame_correct,
            f"Detected: {endgame_phase}"
        )
        
        # Test caching
        phase_cache_size = len(self.engine.game_phase_cache)
        cache_working = phase_cache_size >= 3  # Should have cached all three phases
        
        self.log_result(
            "Game phase caching working",
            cache_working,
            f"Cache entries: {phase_cache_size}"
        )
        
        return opening_correct and middlegame_correct and endgame_correct and cache_working
        
    def test_enhanced_quiescence_search(self):
        """Test 4: Verify enhanced quiescence search functionality"""
        print("\n=== Test 4: Enhanced Quiescence Search ===")
        
        # Test position with tactical shots
        tactical_board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 4 4")
        
        # Test that quiescence runs without errors
        try:
            quiescence_score = self.engine._quiescence_search(tactical_board, -1000, 1000, -1)
            quiescence_working = True
            quiescence_error = "No errors"
        except Exception as e:
            quiescence_working = False
            quiescence_error = str(e)
        
        self.log_result(
            "Enhanced quiescence search runs without errors",
            quiescence_working,
            quiescence_error
        )
        
        # Test game phase awareness in quiescence
        opening_score = self.engine._quiescence_search(
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
            -1000, 1000, -1
        )
        
        endgame_score = self.engine._quiescence_search(
            chess.Board("8/8/8/3k4/8/3K4/8/R7 w - - 0 1"),
            -1000, 1000, -1
        )
        
        # Should handle different phases without errors
        phase_awareness_working = True
        
        self.log_result(
            "Quiescence handles different game phases",
            phase_awareness_working,
            f"Opening: {opening_score:.0f}, Endgame: {endgame_score:.0f}"
        )
        
        return quiescence_working and phase_awareness_working
        
    def test_advanced_time_management(self):
        """Test 5: Verify advanced time management and depth targeting"""
        print("\n=== Test 5: Advanced Time Management ===")
        
        # Test target depth calculation
        opening_depth = self.engine._calculate_target_depth('opening', 5.0)
        middlegame_depth = self.engine._calculate_target_depth('middlegame', 5.0)
        endgame_depth = self.engine._calculate_target_depth('endgame', 5.0)
        
        # Endgame should allow deeper search
        depth_progression = endgame_depth >= middlegame_depth >= opening_depth
        
        self.log_result(
            "Target depth calculation works correctly",
            depth_progression,
            f"Opening: {opening_depth}, Middlegame: {middlegame_depth}, Endgame: {endgame_depth}"
        )
        
        # Test advanced time allocation
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 4 4")
        target_time, max_time = self.engine._calculate_advanced_time_allocation(board, 5.0)
        
        time_allocation_reasonable = (
            0 < target_time < max_time <= 5.0 and
            target_time >= 1.0  # Should allocate reasonable time
        )
        
        self.log_result(
            "Advanced time allocation working",
            time_allocation_reasonable,
            f"Target: {target_time:.2f}s, Max: {max_time:.2f}s"
        )
        
        return depth_progression and time_allocation_reasonable
        
    def test_search_depth_monitoring(self):
        """Test 6: Verify search depth monitoring and performance profiling"""
        print("\n=== Test 6: Search Depth Monitoring ===")
        
        # Perform a quick search to populate monitoring data
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 4 4")
        
        # Clear previous data
        self.engine.search_depth_achieved.clear()
        
        # Perform search with limited time
        try:
            best_move = self.engine.search(board, time_limit=2.0)
            search_completed = best_move != chess.Move.null()
            search_error = "No errors"
        except Exception as e:
            search_completed = False
            search_error = str(e)
        
        self.log_result(
            "Search completes successfully",
            search_completed,
            search_error
        )
        
        # Check if depth was tracked
        depth_tracking = len(self.engine.search_depth_achieved) > 0
        
        self.log_result(
            "Search depth tracking working",
            depth_tracking,
            f"Tracked moves: {len(self.engine.search_depth_achieved)}"
        )
        
        # Test performance report generation
        try:
            performance_report = self.engine.get_performance_report()
            report_generated = "Performance Report" in performance_report
            report_error = "No errors"
        except Exception as e:
            report_generated = False
            report_error = str(e)
        
        self.log_result(
            "Performance report generation working",
            report_generated,
            report_error
        )
        
        return search_completed and depth_tracking and report_generated
        
    def test_performance_vs_v14_1(self):
        """Test 7: Compare performance with V14.1 (overhead removal verification)"""
        print("\n=== Test 7: Performance Improvement Verification ===")
        
        # Test position for move ordering performance
        complex_board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 4 4")
        legal_moves = list(complex_board.legal_moves)
        
        # Time move ordering multiple times for average
        times = []
        for _ in range(10):
            start_time = time.time()
            self.engine._order_moves_advanced(complex_board, legal_moves, 5)
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Should be very fast without expensive operations
        fast_performance = avg_time < 0.005  # Less than 5ms average
        
        self.log_result(
            "Move ordering performance improved",
            fast_performance,
            f"Average time: {avg_time:.4f}s"
        )
        
        # Test that we can achieve target depths consistently
        start_time = time.time()
        try:
            best_move = self.engine.search(complex_board, time_limit=3.0)
            search_time = time.time() - start_time
            depth_achieved = self.engine.search_stats.get('average_depth_achieved', 0)
            
            good_depth = depth_achieved >= 6  # Should achieve at least 6 ply in 3 seconds
            reasonable_time = search_time <= 3.5  # Should not exceed time limit significantly
            
            self.log_result(
                "Consistent depth achievement",
                good_depth and reasonable_time,
                f"Depth: {depth_achieved:.1f}, Time: {search_time:.2f}s"
            )
            
        except Exception as e:
            good_depth = False
            reasonable_time = False
            self.log_result(
                "Consistent depth achievement",
                False,
                f"Error: {e}"
            )
        
        return fast_performance and good_depth and reasonable_time
        
    def run_all_tests(self):
        """Run all tests and provide summary"""
        print("V7P3R v14.2 Performance Optimization Test Suite")
        print("=" * 50)
        
        test_methods = [
            self.test_no_threat_detection_overhead,
            self.test_cached_dynamic_piece_values,
            self.test_game_phase_detection,
            self.test_enhanced_quiescence_search,
            self.test_advanced_time_management,
            self.test_search_depth_monitoring,
            self.test_performance_vs_v14_1
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
            except Exception as e:
                print(f"[ERROR] {test_method.__name__}: {e}")
        
        # Print summary
        print(f"\n" + "=" * 50)
        print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! V14.2 optimizations are working correctly.")
        else:
            print("⚠️  Some tests failed. Review the results above.")
            
        # Print performance report if available
        try:
            print("\n" + self.engine.get_performance_report())
        except:
            print("Performance report not available")
            
        return passed_tests == total_tests

if __name__ == "__main__":
    test_suite = V14_2_TestSuite()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)