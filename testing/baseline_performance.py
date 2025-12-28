#!/usr/bin/env python3
"""
Baseline Performance Measurement

Establishes performance baseline BEFORE implementing Phase 2.
This ensures we can measure exact impact of each change.

Metrics:
1. Search depth by position type (opening, middlegame, endgame, desperate)
2. Move quality (tactical accuracy)
3. Time usage patterns
4. Profile selection distribution

Author: Pat Snyder
Created: 2025-12-28
"""

import chess
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


class BaselineTest:
    """Comprehensive baseline performance measurement"""
    
    def __init__(self):
        self.results = []
    
    def test_position(self, name, fen, time_limit, expected_profile):
        """Test single position and record metrics"""
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"FEN: {fen}")
        print(f"Time: {time_limit}s")
        print(f"{'='*80}")
        
        board = chess.Board(fen)
        engine = V7P3REngine(use_fast_evaluator=True)
        engine.use_modular_evaluation = True
        
        start = time.time()
        move = engine.search(board, time_limit=time_limit)
        elapsed = time.time() - start
        
        result = {
            'name': name,
            'fen': fen,
            'time_limit': time_limit,
            'move': str(move),
            'depth': engine.default_depth,
            'nodes': engine.nodes_searched,
            'time_ms': int(elapsed * 1000),
            'nps': int(engine.nodes_searched / max(elapsed, 0.001)),
            'profile': engine.current_profile.name,
            'module_count': engine.current_profile.module_count,
            'expected_profile': expected_profile
        }
        
        print(f"\nResults:")
        print(f"  Move: {result['move']}")
        print(f"  Profile: {result['profile']} ({result['module_count']} modules)")
        print(f"  Depth: {result['depth']}")
        print(f"  Nodes: {result['nodes']:,}")
        print(f"  Time: {result['time_ms']}ms")
        print(f"  NPS: {result['nps']:,}")
        
        self.results.append(result)
        return result
    
    def run_baseline_suite(self):
        """Run comprehensive baseline test suite"""
        print("\n" + "#"*80)
        print("# BASELINE PERFORMANCE MEASUREMENT")
        print("# Version: v18.2 (Phase 1 - Delegated Evaluation)")
        print("#"*80)
        
        # Test 1: Opening position (COMPREHENSIVE expected)
        self.test_position(
            "Opening - Starting",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            10.0,
            "COMPREHENSIVE"
        )
        
        # Test 2: Middlegame tactical (TACTICAL expected)
        self.test_position(
            "Middlegame - Tactical",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
            10.0,
            "TACTICAL"
        )
        
        # Test 3: Endgame (ENDGAME expected)
        self.test_position(
            "Endgame - Pawn",
            "8/8/8/4k3/4p3/8/3PPP2/4K3 w - - 0 1",
            10.0,
            "ENDGAME"
        )
        
        # Test 4: Desperate - Down Queen (DESPERATE expected)
        self.test_position(
            "Desperate - Down Queen",
            "rnbqkb1r/pppppppp/8/8/8/8/PPPP1PPP/RNBK1BNR w KQkq - 0 1",
            10.0,
            "DESPERATE"
        )
        
        # Test 5: Desperate - Down Rook (DESPERATE expected)
        self.test_position(
            "Desperate - Down Rook",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1",
            10.0,
            "DESPERATE"
        )
        
        # Test 6: Time Pressure (EMERGENCY expected)
        self.test_position(
            "Time Pressure - Opening",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            2.0,
            "EMERGENCY"
        )
        
        # Test 7: Fast Time Control (FAST expected)
        self.test_position(
            "Fast - Middlegame",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
            3.5,
            "FAST"
        )
        
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("BASELINE SUMMARY")
        print("="*80)
        
        print(f"\nTotal Positions Tested: {len(self.results)}")
        
        # Depth by profile
        print("\n--- Depth by Profile ---")
        by_profile = {}
        for r in self.results:
            profile = r['profile']
            if profile not in by_profile:
                by_profile[profile] = []
            by_profile[profile].append(r['depth'])
        
        for profile, depths in sorted(by_profile.items()):
            avg_depth = sum(depths) / len(depths)
            print(f"{profile:15s}: avg {avg_depth:.1f} (range {min(depths)}-{max(depths)})")
        
        # Overall depth
        all_depths = [r['depth'] for r in self.results]
        print(f"\nOverall Average Depth: {sum(all_depths)/len(all_depths):.2f}")
        
        # NPS by profile
        print("\n--- NPS by Profile ---")
        for profile in sorted(by_profile.keys()):
            nps_values = [r['nps'] for r in self.results if r['profile'] == profile]
            avg_nps = sum(nps_values) / len(nps_values)
            print(f"{profile:15s}: {avg_nps:,.0f} nps")
        
        # Profile selection accuracy
        print("\n--- Profile Selection Accuracy ---")
        correct = sum(1 for r in self.results if r['profile'] == r['expected_profile'])
        print(f"Correct: {correct}/{len(self.results)} ({correct/len(self.results)*100:.0f}%)")
        
        # Time usage
        print("\n--- Time Usage ---")
        for r in self.results:
            utilization = (r['time_ms'] / 1000) / r['time_limit'] * 100
            print(f"{r['name']:25s}: {r['time_ms']}ms / {r['time_limit']*1000:.0f}ms ({utilization:.0f}%)")
        
        # CRITICAL: Desperate mode depth
        print("\n" + "="*80)
        print("CRITICAL METRIC: DESPERATE MODE DEPTH")
        print("="*80)
        desperate_results = [r for r in self.results if r['profile'] == 'DESPERATE']
        if desperate_results:
            desperate_depths = [r['depth'] for r in desperate_results]
            avg_desperate_depth = sum(desperate_depths) / len(desperate_depths)
            print(f"Current DESPERATE depth: {avg_desperate_depth:.1f}")
            print(f"Phase 2 Target: {avg_desperate_depth + 2:.1f}+ (2-3 ply improvement)")
            print(f"\n[BASELINE ESTABLISHED]")
            print(f"Any Phase 2 implementation must achieve depth {avg_desperate_depth + 2:.0f}+")
            print(f"in desperate positions to be considered successful.")
        else:
            print("[WARNING] No DESPERATE positions tested")
        
        print("\n" + "="*80)
        print("BASELINE COMPLETE - Ready for Phase 2 Implementation")
        print("="*80)


if __name__ == "__main__":
    baseline = BaselineTest()
    baseline.run_baseline_suite()
