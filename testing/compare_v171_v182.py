#!/usr/bin/env python3
"""
Side-by-Side Engine Comparison: v17.1 vs v18.2 Modular

Real-world scenario testing across multiple time controls and positions.

Compares:
- v17.1 (stable baseline, proven 2+ weeks production)
- v18.2 (new modular evaluation system)

Metrics:
- Move selection
- Search depth
- Time usage
- Nodes searched
- Profile selection (v18.2)

Time Controls:
- Bullet: 1s, 2s
- Blitz: 3s, 5s, 8s
- Rapid: 15s, 30s

Author: Pat Snyder
Created: 2025-12-27
"""

import chess
import sys
import os
import time
import importlib.util
from typing import Tuple, List
from dataclasses import dataclass

# Add current src to path for v18.2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class EngineResult:
    """Result from engine search"""
    move: chess.Move
    depth: int
    nodes: int
    time_ms: int
    nps: int
    score: int = 0
    profile: str = None


@dataclass
class TestScenario:
    """Test scenario definition"""
    fen: str
    description: str
    category: str
    expected_characteristic: str  # What we're testing


class EngineComparator:
    """Side-by-side engine comparison"""
    
    def __init__(self):
        self.v171_engine = None
        self.v182_engine = None
        self.results = []
    
    def load_engines(self):
        """Load both engine versions"""
        print("Loading engines...")
        
        # Load v18.2 (current development)
        from v7p3r import V7P3REngine
        self.v182_engine = V7P3REngine(use_fast_evaluator=True)
        self.v182_engine.use_modular_evaluation = True  # ENABLE MODULAR
        print(f"  v18.2: Loaded (modular evaluation ENABLED)")
        
        # Load v17.1 from tournament engines
        v171_path = r"s:\Programming\Chess Engines\Tournament Engines\V7P3R\V7P3R_v17.1\src\v7p3r.py"
        
        if not os.path.exists(v171_path):
            print(f"  WARNING: v17.1 not found at {v171_path}")
            print(f"  Will use v18.2 with modular disabled as baseline")
            self.v171_engine = V7P3REngine(use_fast_evaluator=True)
            self.v171_engine.use_modular_evaluation = False
        else:
            # Load v17.1 module
            spec = importlib.util.spec_from_file_location("v7p3r_v171", v171_path)
            v171_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(v171_module)
            self.v171_engine = v171_module.V7P3REngine(use_fast_evaluator=True)
        
        print(f"  v17.1: Loaded (baseline)")
    
    def create_test_suite(self) -> List[TestScenario]:
        """Create comprehensive test scenarios"""
        return [
            # OPENING BOOK POSITIONS
            TestScenario(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "Starting position",
                "opening",
                "Book move consistency"
            ),
            TestScenario(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "After 1.e4",
                "opening",
                "Opening response speed"
            ),
            
            # MIDDLEGAME TACTICAL
            TestScenario(
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
                "Italian Game - tactical complexity",
                "middlegame",
                "Tactical calculation depth"
            ),
            TestScenario(
                "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
                "Spanish complex middlegame",
                "middlegame",
                "Strategic planning"
            ),
            
            # ENDGAME POSITIONS
            TestScenario(
                "8/8/8/4k3/4p3/8/3PPP2/4K3 w - - 0 1",
                "Pawn endgame - technique",
                "endgame",
                "Endgame technique depth"
            ),
            TestScenario(
                "8/8/8/8/8/3r4/4P3/4K2R w - - 0 1",
                "Rook endgame",
                "endgame",
                "Endgame conversion"
            ),
            TestScenario(
                "8/8/8/8/8/8/4P3/4K2k w - - 0 1",
                "K+P vs K - trivial win",
                "endgame",
                "Instant recognition"
            ),
            
            # DESPERATE POSITIONS
            TestScenario(
                "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1",
                "Down a queen (desperate)",
                "desperate",
                "Tactical recovery focus"
            ),
            
            # TIME PRESSURE SIMULATION
            TestScenario(
                "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
                "Complex position for time pressure",
                "middlegame",
                "Fast evaluation under pressure"
            ),
        ]
    
    def run_engine(self, engine, board: chess.Board, time_limit: float, 
                   version: str) -> EngineResult:
        """Run single engine search"""
        start = time.time()
        
        # Reset stats
        if hasattr(engine, 'nodes_searched'):
            engine.nodes_searched = 0
        
        move = engine.search(board, time_limit=time_limit)
        elapsed_ms = int((time.time() - start) * 1000)
        
        nodes = getattr(engine, 'nodes_searched', 0)
        depth = getattr(engine, 'default_depth', 0)
        nps = int(nodes / max(elapsed_ms / 1000, 0.001))
        
        profile = None
        if version == "v18.2" and hasattr(engine, 'current_profile'):
            profile = engine.current_profile.name if engine.current_profile else None
        
        return EngineResult(
            move=move,
            depth=depth,
            nodes=nodes,
            time_ms=elapsed_ms,
            nps=nps,
            profile=profile
        )
    
    def compare_on_position(self, scenario: TestScenario, time_limit: float):
        """Compare both engines on one position"""
        board = chess.Board(scenario.fen)
        
        print(f"\n{'='*80}")
        print(f"Position: {scenario.description}")
        print(f"Category: {scenario.category} | Testing: {scenario.expected_characteristic}")
        print(f"FEN: {scenario.fen}")
        print(f"Time Control: {time_limit}s")
        print(f"{'='*80}")
        
        # Run v17.1
        print(f"\n[v17.1 BASELINE]")
        result_171 = self.run_engine(self.v171_engine, board, time_limit, "v17.1")
        print(f"  Move: {result_171.move}")
        print(f"  Depth: {result_171.depth}")
        print(f"  Nodes: {result_171.nodes:,}")
        print(f"  Time: {result_171.time_ms}ms")
        print(f"  NPS: {result_171.nps:,}")
        
        # Run v18.2
        print(f"\n[v18.2 MODULAR]")
        result_182 = self.run_engine(self.v182_engine, board, time_limit, "v18.2")
        print(f"  Move: {result_182.move}")
        print(f"  Profile: {result_182.profile}")
        print(f"  Depth: {result_182.depth}")
        print(f"  Nodes: {result_182.nodes:,}")
        print(f"  Time: {result_182.time_ms}ms")
        print(f"  NPS: {result_182.nps:,}")
        
        # Analysis
        print(f"\n[COMPARISON]")
        move_match = result_171.move == result_182.move
        print(f"  Move Agreement: {'[SAME]' if move_match else '[DIFFERENT]'} {result_171.move} vs {result_182.move}")
        
        depth_diff = result_182.depth - result_171.depth
        print(f"  Depth: {result_171.depth} -> {result_182.depth} ({depth_diff:+d})")
        
        time_diff_ms = result_182.time_ms - result_171.time_ms
        time_ratio = result_182.time_ms / max(result_171.time_ms, 1)
        print(f"  Time: {result_171.time_ms}ms -> {result_182.time_ms}ms ({time_diff_ms:+d}ms, {time_ratio:.2f}x)")
        
        nodes_ratio = result_182.nodes / max(result_171.nodes, 1) if result_171.nodes > 0 else 0
        print(f"  Nodes: {result_171.nodes:,} -> {result_182.nodes:,} ({nodes_ratio:.2f}x)")
        
        nps_ratio = result_182.nps / max(result_171.nps, 1) if result_171.nps > 0 else 0
        print(f"  NPS: {result_171.nps:,} -> {result_182.nps:,} ({nps_ratio:.2f}x)")
        
        # Store for summary
        self.results.append({
            'scenario': scenario,
            'time_limit': time_limit,
            'v171': result_171,
            'v182': result_182,
            'move_match': move_match,
            'depth_diff': depth_diff,
            'time_ratio': time_ratio,
            'nodes_ratio': nodes_ratio
        })
    
    def run_time_control_sweep(self, scenario: TestScenario, 
                                time_controls: List[float]):
        """Test one position across multiple time controls"""
        print(f"\n{'#'*80}")
        print(f"# TIME CONTROL SWEEP: {scenario.description}")
        print(f"{'#'*80}")
        
        for time_limit in time_controls:
            self.compare_on_position(scenario, time_limit)
    
    def run_comprehensive_test(self):
        """Run full comparison suite"""
        scenarios = self.create_test_suite()
        
        print("="*80)
        print("COMPREHENSIVE ENGINE COMPARISON: v17.1 vs v18.2 Modular")
        print("="*80)
        print(f"Test Scenarios: {len(scenarios)}")
        print(f"Time Controls: Bullet (1s, 2s), Blitz (3s, 5s, 8s), Rapid (15s)")
        print("="*80)
        
        # Quick test: Each position at 3 time controls
        for scenario in scenarios:
            # Test at blitz time control (representative)
            self.compare_on_position(scenario, 5.0)
        
        # Print summary
        self.print_summary()
    
    def run_quick_test(self):
        """Quick sanity check - 3 positions, 3 time controls each"""
        scenarios = self.create_test_suite()
        
        print("="*80)
        print("QUICK COMPARISON TEST: v17.1 vs v18.2 Modular")
        print("="*80)
        
        # Test 3 representative positions at 3 time controls
        test_positions = [
            scenarios[0],  # Starting position
            scenarios[2],  # Middlegame tactical
            scenarios[4],  # Endgame
        ]
        
        time_controls = [2.0, 5.0, 10.0]  # Bullet, Blitz, Rapid
        
        for scenario in test_positions:
            for time_limit in time_controls:
                self.compare_on_position(scenario, time_limit)
        
        self.print_summary()
    
    def print_summary(self):
        """Print comprehensive summary"""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Move agreement
        move_matches = sum(1 for r in self.results if r['move_match'])
        print(f"\nMove Agreement: {move_matches}/{len(self.results)} ({move_matches/len(self.results)*100:.1f}%)")
        
        # Depth analysis
        depth_diffs = [r['depth_diff'] for r in self.results]
        avg_depth = sum(depth_diffs) / len(depth_diffs)
        improved = sum(1 for d in depth_diffs if d > 0)
        worse = sum(1 for d in depth_diffs if d < 0)
        
        print(f"\nDepth Performance:")
        print(f"  Average change: {avg_depth:+.2f}")
        print(f"  Improved: {improved}/{len(self.results)}")
        print(f"  Worse: {worse}/{len(self.results)}")
        print(f"  Same: {len(self.results)-improved-worse}/{len(self.results)}")
        
        # Time analysis
        time_ratios = [r['time_ratio'] for r in self.results if r['time_ratio'] > 0]
        if time_ratios:
            avg_time = sum(time_ratios) / len(time_ratios)
            faster = sum(1 for t in time_ratios if t < 1.0)
            print(f"\nTime Performance:")
            print(f"  Average ratio: {avg_time:.2f}x (new/old)")
            print(f"  Faster: {faster}/{len(time_ratios)}")
        
        # By category
        print(f"\n--- By Category ---")
        by_category = {}
        for r in self.results:
            cat = r['scenario'].category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)
        
        for cat, results in sorted(by_category.items()):
            matches = sum(1 for r in results if r['move_match'])
            avg_depth = sum(r['depth_diff'] for r in results) / len(results)
            print(f"\n{cat.upper()}:")
            print(f"  Tests: {len(results)}")
            print(f"  Move agreement: {matches}/{len(results)} ({matches/len(results)*100:.0f}%)")
            print(f"  Avg depth change: {avg_depth:+.2f}")
        
        # Verdict
        print("\n" + "="*80)
        print("VERDICT")
        print("="*80)
        
        agreement_pct = move_matches / len(self.results) * 100
        
        if agreement_pct >= 80 and avg_depth >= -0.3:
            print("[EXCELLENT] v18.2 maintains v17.1 quality with modular system")
        elif agreement_pct >= 70 and avg_depth >= -0.5:
            print("[GOOD] v18.2 comparable to v17.1, acceptable for testing")
        else:
            print("[NEEDS WORK] Significant differences from v17.1")
        
        print(f"\nMove Agreement: {agreement_pct:.1f}%")
        print(f"Depth Change: {avg_depth:+.2f}")


def quick_comparison():
    """Quick 3x3 test (3 positions, 3 time controls)"""
    comparator = EngineComparator()
    comparator.load_engines()
    comparator.run_quick_test()


def full_comparison():
    """Full test suite"""
    comparator = EngineComparator()
    comparator.load_engines()
    comparator.run_comprehensive_test()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full_comparison()
    else:
        quick_comparison()
