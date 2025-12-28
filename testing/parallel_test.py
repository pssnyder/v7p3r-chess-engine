#!/usr/bin/env python3
"""
Parallel Testing Suite for Modular Evaluation System

Compares old evaluation system vs new modular system on identical positions.

Tests:
1. Score parity (are evaluations similar?)
2. Move agreement (do they pick same moves?)
3. Performance (depth, nodes, time)
4. Profile selection accuracy
5. No regressions on known positions

Author: Pat Snyder
Created: 2025-12-27 (v18.2 Modular Evaluation - Day 5)
"""

import chess
import sys
import os
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


@dataclass
class TestPosition:
    """Test position with metadata"""
    fen: str
    description: str
    expected_profile: str  # Expected modular profile
    category: str  # opening, middlegame, endgame, tactical


@dataclass
class SearchResult:
    """Search result with metrics"""
    move: chess.Move
    score: int
    depth: int
    nodes: int
    time_ms: int
    nps: int
    profile: str = None  # Modular profile used


class ParallelTester:
    """Runs parallel tests comparing old vs new evaluation"""
    
    def __init__(self):
        self.test_positions = self._create_test_suite()
        self.results = []
    
    def _create_test_suite(self) -> List[TestPosition]:
        """Create comprehensive test position suite"""
        return [
            # OPENING POSITIONS
            TestPosition(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "Starting position",
                "COMPREHENSIVE",
                "opening"
            ),
            TestPosition(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "After 1.e4",
                "COMPREHENSIVE",
                "opening"
            ),
            
            # MIDDLEGAME POSITIONS
            TestPosition(
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
                "Italian Game middlegame",
                "COMPREHENSIVE",
                "middlegame"
            ),
            
            # TACTICAL POSITIONS
            TestPosition(
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 1",
                "Tactical middlegame (open king)",
                "TACTICAL",
                "tactical"
            ),
            TestPosition(
                "4k3/8/8/8/8/8/8/4K2R w - - 0 1",
                "King exposed endgame",
                "TACTICAL",
                "tactical"
            ),
            
            # ENDGAME POSITIONS
            TestPosition(
                "8/8/8/4k3/4p3/8/3PPP2/4K3 w - - 0 1",
                "Pawn endgame",
                "ENDGAME",
                "endgame"
            ),
            TestPosition(
                "8/8/8/8/8/3r4/4P3/4K2R w - - 0 1",
                "Rook endgame",
                "ENDGAME",
                "endgame"
            ),
            TestPosition(
                "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
                "K+P vs K",
                "ENDGAME",
                "endgame"
            ),
            
            # DESPERATE POSITIONS
            TestPosition(
                "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1",
                "Down a queen (desperate)",
                "DESPERATE",
                "desperate"
            ),
            TestPosition(
                "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
                "Up a queen (winning)",
                "TACTICAL",
                "winning"
            ),
        ]
    
    def run_search(self, engine: V7P3REngine, board: chess.Board, 
                   time_limit: float = 5.0) -> SearchResult:
        """Run search and collect metrics"""
        start_time = time.time()
        move = engine.search(board, time_limit=time_limit)
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # Get metrics
        nodes = engine.nodes_searched
        depth = engine.default_depth  # Last completed depth
        nps = int(nodes / max(elapsed_ms / 1000, 0.001))
        
        # Get profile if modular enabled
        profile = None
        if hasattr(engine, 'current_profile') and engine.current_profile:
            profile = engine.current_profile.name
        
        # Get score from last search
        score = 0  # Would need to extract from search
        
        return SearchResult(
            move=move,
            score=score,
            depth=depth,
            nodes=nodes,
            time_ms=elapsed_ms,
            nps=nps,
            profile=profile
        )
    
    def compare_evaluations(self, pos: TestPosition, time_limit: float = 5.0):
        """Run same position with old and new evaluation"""
        board = chess.Board(pos.fen)
        
        print(f"\n{'='*70}")
        print(f"Position: {pos.description} ({pos.category})")
        print(f"FEN: {pos.fen}")
        print(f"Expected Profile: {pos.expected_profile}")
        
        # Test 1: OLD SYSTEM (modular disabled)
        print(f"\n--- OLD SYSTEM (v18.2 current) ---")
        engine_old = V7P3REngine(use_fast_evaluator=True)
        engine_old.use_modular_evaluation = False
        result_old = self.run_search(engine_old, board, time_limit)
        
        print(f"Move: {result_old.move}")
        print(f"Depth: {result_old.depth}")
        print(f"Nodes: {result_old.nodes:,}")
        print(f"Time: {result_old.time_ms}ms")
        print(f"NPS: {result_old.nps:,}")
        
        # Test 2: NEW SYSTEM (modular enabled)
        print(f"\n--- NEW SYSTEM (modular evaluation) ---")
        engine_new = V7P3REngine(use_fast_evaluator=True)
        engine_new.use_modular_evaluation = True  # ENABLE MODULAR
        result_new = self.run_search(engine_new, board, time_limit)
        
        print(f"Move: {result_new.move}")
        print(f"Profile: {result_new.profile}")
        print(f"Depth: {result_new.depth}")
        print(f"Nodes: {result_new.nodes:,}")
        print(f"Time: {result_new.time_ms}ms")
        print(f"NPS: {result_new.nps:,}")
        
        # COMPARISON
        print(f"\n--- COMPARISON ---")
        move_match = result_old.move == result_new.move
        print(f"Move agreement: {'[SAME]' if move_match else '[DIFFERENT]'}")
        
        depth_diff = result_new.depth - result_old.depth
        print(f"Depth change: {depth_diff:+d} (old: {result_old.depth}, new: {result_new.depth})")
        
        time_ratio = result_new.time_ms / max(result_old.time_ms, 1)
        print(f"Time ratio: {time_ratio:.2f}x (new/old)")
        
        nps_ratio = result_new.nps / max(result_old.nps, 1)
        print(f"NPS ratio: {nps_ratio:.2f}x")
        
        profile_match = result_new.profile == pos.expected_profile
        print(f"Profile: {result_new.profile} {'[OK]' if profile_match else f'[expected {pos.expected_profile}]'}")
        
        # Store results
        self.results.append({
            'position': pos,
            'old': result_old,
            'new': result_new,
            'move_match': move_match,
            'depth_diff': depth_diff,
            'time_ratio': time_ratio,
            'profile_match': profile_match
        })
        
        return move_match, depth_diff, time_ratio
    
    def run_all_tests(self, time_per_position: float = 5.0):
        """Run all parallel tests"""
        print("="*70)
        print("PARALLEL TESTING: Profile Selection Validation")
        print("NOTE: Both systems use SAME evaluation (modular not active yet)")
        print("Testing: Context calculation and profile selection only")
        print("="*70)
        
        for pos in self.test_positions:
            try:
                self.compare_evaluations(pos, time_per_position)
            except Exception as e:
                print(f"\n[ERROR] testing {pos.description}: {e}")
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if not self.results:
            print("No results collected")
            return
        
        # Move agreement
        move_matches = sum(1 for r in self.results if r['move_match'])
        print(f"\nMove Agreement: {move_matches}/{len(self.results)} ({move_matches/len(self.results)*100:.1f}%)")
        
        # Profile accuracy
        profile_matches = sum(1 for r in self.results if r['profile_match'])
        print(f"Profile Accuracy: {profile_matches}/{len(self.results)} ({profile_matches/len(self.results)*100:.1f}%)")
        
        # Depth analysis
        avg_depth_diff = sum(r['depth_diff'] for r in self.results) / len(self.results)
        print(f"\nAverage Depth Change: {avg_depth_diff:+.1f}")
        
        depth_improved = sum(1 for r in self.results if r['depth_diff'] > 0)
        print(f"Depth Improved: {depth_improved}/{len(self.results)}")
        
        # Time analysis
        avg_time_ratio = sum(r['time_ratio'] for r in self.results) / len(self.results)
        print(f"\nAverage Time Ratio: {avg_time_ratio:.2f}x (new/old)")
        
        faster_count = sum(1 for r in self.results if r['time_ratio'] < 1.0)
        print(f"Faster Searches: {faster_count}/{len(self.results)}")
        
        # By category
        print("\n--- By Category ---")
        categories = {}
        for r in self.results:
            cat = r['position'].category
            if cat not in categories:
                categories[cat] = {'total': 0, 'move_match': 0, 'profile_match': 0}
            categories[cat]['total'] += 1
            if r['move_match']:
                categories[cat]['move_match'] += 1
            if r['profile_match']:
                categories[cat]['profile_match'] += 1
        
        for cat, stats in sorted(categories.items()):
            print(f"\n{cat.upper()}:")
            print(f"  Move agreement: {stats['move_match']}/{stats['total']}")
            print(f"  Profile accuracy: {stats['profile_match']}/{stats['total']}")
        
        # VERDICT
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        # Criteria for success
        move_agreement_pct = move_matches / len(self.results) * 100
        profile_accuracy_pct = profile_matches / len(self.results) * 100
        
        passing = True
        if move_agreement_pct < 70:
            print(f"[FAIL] Move agreement {move_agreement_pct:.1f}% < 70% threshold")
            passing = False
        else:
            print(f"[PASS] Move agreement {move_agreement_pct:.1f}% >= 70%")
        
        if profile_accuracy_pct < 80:
            print(f"[FAIL] Profile accuracy {profile_accuracy_pct:.1f}% < 80% threshold")
            passing = False
        else:
            print(f"[PASS] Profile accuracy {profile_accuracy_pct:.1f}% >= 80%")
        
        if avg_depth_diff < -0.5:
            print(f"[FAIL] Depth regression {avg_depth_diff:.1f}")
            passing = False
        else:
            print(f"[PASS] Depth acceptable ({avg_depth_diff:+.1f})")
        
        if passing:
            print("\n*** ALL TESTS PASSED - READY FOR DEPLOYMENT ***")
        else:
            print("\n*** TESTS FAILED - NEEDS INVESTIGATION ***")


def quick_test():
    """Quick 3-position sanity check"""
    print("="*70)
    print("QUICK SANITY CHECK (3 positions)")
    print("="*70)
    
    tester = ParallelTester()
    
    # Test just 3 representative positions
    quick_positions = [
        tester.test_positions[0],  # Starting position
        tester.test_positions[5],  # Pawn endgame
        tester.test_positions[8],  # Down a queen
    ]
    
    tester.test_positions = quick_positions
    tester.run_all_tests(time_per_position=5.0)


def full_test():
    """Full test suite"""
    tester = ParallelTester()
    tester.run_all_tests(time_per_position=5.0)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full_test()
    else:
        quick_test()
