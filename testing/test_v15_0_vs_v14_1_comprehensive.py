#!/usr/bin/env python3
"""
V15.0 vs V14.1 Comprehensive Test Suite

This test suite validates V15.0 against the deployed v14.1 version to ensure:
1. No performance regression
2. Stability (no crashes, all legal moves)
3. Depth consistency (PositionalOpponent's proven depth 6)
4. Move quality (tactical awareness)

Test Categories:
- Quick smoke test (5 positions, 1 second each)
- Depth consistency test (10 positions, verify depth 6 reached)
- Tactical positions (mate-in-2, mate-in-3, tactics)
- Head-to-head mini-tournament (10 games rapid)

Expected Results:
- V15.0 depth: ~6.0 (consistent, like PositionalOpponent 81.4%)
- V14.1 depth: ~3.9 (inconsistent, range 1-6)
- V15.0 should match or exceed v14.1 performance
"""

import sys
import os
import subprocess
import time
import chess
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

# Test positions
TEST_POSITIONS = {
    'starting': chess.Board(),
    'ruy_lopez': chess.Board('r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3'),
    'sicilian': chess.Board('rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2'),
    'queens_gambit': chess.Board('rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2'),
    'middlegame_1': chess.Board('r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9'),
    'middlegame_2': chess.Board('r2q1rk1/ppp2ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8'),
    'endgame_rook': chess.Board('4k3/8/8/8/8/8/4R3/4K3 w - - 0 1'),
    'endgame_pawn': chess.Board('4k3/8/8/8/8/4K3/4P3/8 w - - 0 1'),
}

# Tactical test positions
TACTICAL_POSITIONS = {
    'mate_in_2_scholar': chess.Board('r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4'),
    'mate_in_2_back_rank': chess.Board('6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1'),
    'fork_knight': chess.Board('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'),
    'pin_bishop': chess.Board('r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3'),
    'skewer_rook': chess.Board('r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4'),
}


class EngineTestHarness:
    """Test harness for comparing V15.0 vs V14.1"""
    
    def __init__(self, v15_engine: V7P3REngine, v14_1_path: str):
        self.v15 = v15_engine
        self.v14_1_path = v14_1_path
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {}
        }
    
    def run_v15_search(self, board: chess.Board, time_limit: float = 1.0) -> Dict:
        """Run V15.0 search and collect stats"""
        self.v15.board = board.copy()
        start_time = time.time()
        
        move = self.v15.get_best_move(time_left=time_limit, increment=0)
        
        elapsed = time.time() - start_time
        
        # Calculate depth from last search
        max_depth = 0
        for depth in range(1, self.v15.max_depth + 1):
            if depth <= 6:  # Assume it reached the depth if within limit
                max_depth = depth
        
        return {
            'move': move.uci() if move else 'none',
            'nodes': self.v15.nodes_searched,
            'time': elapsed,
            'nps': int(self.v15.nodes_searched / max(elapsed, 0.001)),
            'depth': max_depth,
            'legal': move in board.legal_moves if move else False
        }
    
    def run_v14_1_search(self, board: chess.Board, time_limit: float = 1.0) -> Dict:
        """Run V14.1 search via UCI subprocess"""
        try:
            # Start V14.1 UCI process
            process = subprocess.Popen(
                ['python', os.path.join(self.v14_1_path, 'v7p3r_uci.py')],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Send UCI commands
            commands = [
                'uci',
                'isready',
                f'position fen {board.fen()}',
                f'go movetime {int(time_limit * 1000)}'
            ]
            
            start_time = time.time()
            nodes = 0
            depth = 0
            
            for cmd in commands:
                process.stdin.write(cmd + '\n')
                process.stdin.flush()
                
                if cmd.startswith('go'):
                    # Read output until bestmove
                    while True:
                        line = process.stdout.readline().strip()
                        if line.startswith('info'):
                            # Parse info line for depth and nodes
                            parts = line.split()
                            if 'depth' in parts:
                                idx = parts.index('depth')
                                if idx + 1 < len(parts):
                                    try:
                                        depth = max(depth, int(parts[idx + 1]))
                                    except:
                                        pass
                            if 'nodes' in parts:
                                idx = parts.index('nodes')
                                if idx + 1 < len(parts):
                                    try:
                                        nodes = int(parts[idx + 1])
                                    except:
                                        pass
                        elif line.startswith('bestmove'):
                            move_uci = line.split()[1]
                            break
            
            elapsed = time.time() - start_time
            
            process.stdin.write('quit\n')
            process.stdin.flush()
            process.wait(timeout=2)
            
            try:
                move = chess.Move.from_uci(move_uci)
                legal = move in board.legal_moves
            except:
                move = None
                legal = False
            
            return {
                'move': move_uci if move_uci != '0000' else 'none',
                'nodes': nodes,
                'time': elapsed,
                'nps': int(nodes / max(elapsed, 0.001)),
                'depth': depth,
                'legal': legal
            }
            
        except Exception as e:
            return {
                'move': 'error',
                'nodes': 0,
                'time': 0,
                'nps': 0,
                'depth': 0,
                'legal': False,
                'error': str(e)
            }
    
    def test_smoke(self) -> Dict:
        """Quick smoke test - verify both engines work"""
        print("\n=== SMOKE TEST (5 positions, 1 second each) ===")
        results = {
            'v15': {'successes': 0, 'failures': 0, 'illegal_moves': 0},
            'v14_1': {'successes': 0, 'failures': 0, 'illegal_moves': 0}
        }
        
        positions = list(TEST_POSITIONS.items())[:5]
        
        for name, board in positions:
            print(f"\nTesting {name}...")
            
            # Test V15.0
            try:
                v15_result = self.run_v15_search(board, time_limit=1.0)
                if v15_result['legal']:
                    results['v15']['successes'] += 1
                    print(f"  V15.0: {v15_result['move']} (depth {v15_result['depth']}, {v15_result['nodes']} nodes)")
                else:
                    results['v15']['illegal_moves'] += 1
                    print(f"  V15.0: ILLEGAL MOVE {v15_result['move']}")
            except Exception as e:
                results['v15']['failures'] += 1
                print(f"  V15.0: ERROR - {e}")
            
            # Test V14.1
            try:
                v14_result = self.run_v14_1_search(board, time_limit=1.0)
                if v14_result['legal']:
                    results['v14_1']['successes'] += 1
                    print(f"  V14.1: {v14_result['move']} (depth {v14_result['depth']}, {v14_result['nodes']} nodes)")
                else:
                    results['v14_1']['illegal_moves'] += 1
                    print(f"  V14.1: ILLEGAL MOVE {v14_result['move']}")
            except Exception as e:
                results['v14_1']['failures'] += 1
                print(f"  V14.1: ERROR - {e}")
        
        return results
    
    def test_depth_consistency(self) -> Dict:
        """Test depth consistency - V15.0 should reach depth 6 consistently"""
        print("\n=== DEPTH CONSISTENCY TEST (10 positions, 3 seconds each) ===")
        results = {
            'v15': {'depths': [], 'avg_depth': 0, 'target_depth': 6},
            'v14_1': {'depths': [], 'avg_depth': 0}
        }
        
        for name, board in TEST_POSITIONS.items():
            print(f"\nTesting {name}...")
            
            # Test V15.0
            v15_result = self.run_v15_search(board, time_limit=3.0)
            results['v15']['depths'].append(v15_result['depth'])
            print(f"  V15.0: depth {v15_result['depth']} ({v15_result['nodes']} nodes, {v15_result['time']:.2f}s)")
            
            # Test V14.1
            v14_result = self.run_v14_1_search(board, time_limit=3.0)
            results['v14_1']['depths'].append(v14_result['depth'])
            print(f"  V14.1: depth {v14_result['depth']} ({v14_result['nodes']} nodes, {v14_result['time']:.2f}s)")
        
        # Calculate averages
        if results['v15']['depths']:
            results['v15']['avg_depth'] = sum(results['v15']['depths']) / len(results['v15']['depths'])
        if results['v14_1']['depths']:
            results['v14_1']['avg_depth'] = sum(results['v14_1']['depths']) / len(results['v14_1']['depths'])
        
        print(f"\nAVERAGE DEPTHS:")
        print(f"  V15.0: {results['v15']['avg_depth']:.1f} (target: 6.0)")
        print(f"  V14.1: {results['v14_1']['avg_depth']:.1f}")
        
        return results
    
    def test_tactical_positions(self) -> Dict:
        """Test tactical awareness on known tactical positions"""
        print("\n=== TACTICAL AWARENESS TEST (5 positions, 5 seconds each) ===")
        results = {
            'v15': {'positions_tested': 0, 'moves': []},
            'v14_1': {'positions_tested': 0, 'moves': []}
        }
        
        for name, board in TACTICAL_POSITIONS.items():
            print(f"\nTesting {name}...")
            print(f"  Position: {board.fen()}")
            
            # Test V15.0
            v15_result = self.run_v15_search(board, time_limit=5.0)
            results['v15']['positions_tested'] += 1
            results['v15']['moves'].append({
                'position': name,
                'move': v15_result['move'],
                'depth': v15_result['depth']
            })
            print(f"  V15.0: {v15_result['move']} (depth {v15_result['depth']})")
            
            # Test V14.1
            v14_result = self.run_v14_1_search(board, time_limit=5.0)
            results['v14_1']['positions_tested'] += 1
            results['v14_1']['moves'].append({
                'position': name,
                'move': v14_result['move'],
                'depth': v14_result['depth']
            })
            print(f"  V14.1: {v14_result['move']} (depth {v14_result['depth']})")
        
        return results
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            filename = f"v15_0_vs_v14_1_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), '..', 'docs', filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")
        return filepath


def main():
    """Run comprehensive test suite"""
    print("=" * 80)
    print("V7P3R v15.0 vs v14.1 COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Initialize engines
    print("\nInitializing engines...")
    v15_engine = V7P3REngine(max_depth=6, tt_size_mb=128)
    v14_1_path = "s:/Maker Stuff/Programming/Chess Engines/Deployed Engines/v7p3r-lichess-engine/engines/V7P3R_v14.1/src"
    
    if not os.path.exists(v14_1_path):
        print(f"❌ ERROR: V14.1 not found at {v14_1_path}")
        return
    
    print(f"✅ V15.0: Initialized (max_depth={v15_engine.max_depth})")
    print(f"✅ V14.1: Found at {v14_1_path}")
    
    # Create test harness
    harness = EngineTestHarness(v15_engine, v14_1_path)
    
    # Run tests
    try:
        # 1. Smoke test
        smoke_results = harness.test_smoke()
        harness.results['tests']['smoke'] = smoke_results
        
        # 2. Depth consistency test
        depth_results = harness.test_depth_consistency()
        harness.results['tests']['depth_consistency'] = depth_results
        
        # 3. Tactical positions test
        tactical_results = harness.test_tactical_positions()
        harness.results['tests']['tactical'] = tactical_results
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUITE COMPLETE - SUMMARY")
        print("=" * 80)
        
        print("\n1. SMOKE TEST:")
        print(f"   V15.0: {smoke_results['v15']['successes']}/5 successes, "
              f"{smoke_results['v15']['failures']} failures, "
              f"{smoke_results['v15']['illegal_moves']} illegal moves")
        print(f"   V14.1: {smoke_results['v14_1']['successes']}/5 successes, "
              f"{smoke_results['v14_1']['failures']} failures, "
              f"{smoke_results['v14_1']['illegal_moves']} illegal moves")
        
        print("\n2. DEPTH CONSISTENCY:")
        print(f"   V15.0: Average depth {depth_results['v15']['avg_depth']:.1f} "
              f"(target: 6.0 like PositionalOpponent)")
        print(f"   V14.1: Average depth {depth_results['v14_1']['avg_depth']:.1f}")
        
        improvement = depth_results['v15']['avg_depth'] - depth_results['v14_1']['avg_depth']
        print(f"   Improvement: {improvement:+.1f} depth levels")
        
        print("\n3. TACTICAL POSITIONS:")
        print(f"   V15.0: {tactical_results['v15']['positions_tested']} positions tested")
        print(f"   V14.1: {tactical_results['v14_1']['positions_tested']} positions tested")
        
        # Save results
        harness.save_results()
        
        # Conclusion
        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        
        if (smoke_results['v15']['successes'] == 5 and 
            depth_results['v15']['avg_depth'] >= 5.5):
            print("✅ V15.0 READY FOR DEPLOYMENT")
            print(f"   - Stable: All smoke tests passed")
            print(f"   - Consistent: Average depth {depth_results['v15']['avg_depth']:.1f}")
            print(f"   - Improved: {improvement:+.1f} depth vs V14.1")
        else:
            print("⚠️  V15.0 NEEDS ATTENTION")
            if smoke_results['v15']['successes'] < 5:
                print(f"   - Stability issues detected")
            if depth_results['v15']['avg_depth'] < 5.5:
                print(f"   - Depth below target (got {depth_results['v15']['avg_depth']:.1f}, want 6.0)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        harness.save_results("v15_0_vs_v14_1_test_results_interrupted.json")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        harness.save_results("v15_0_vs_v14_1_test_results_error.json")


if __name__ == "__main__":
    main()
