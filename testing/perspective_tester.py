#!/usr/bin/env python3
"""
V7P3R Perspective Diagnostic Tool

This tool tests V7P3R's evaluation consistency between playing as White and Black.
It feeds the same positions to the engine from both perspectives and compares:
1. Evaluation scores and their signs
2. Best move choices
3. Search depth consistency
4. Time management

Usage: python v7p3r_perspective_tester.py
"""

import subprocess
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import sys
import os

@dataclass
class EngineResponse:
    """Stores engine response data"""
    bestmove: str
    evaluation: float
    depth: int
    time_ms: int
    pv: List[str]  # Principal variation
    nodes: int

class UCIEngine:
    """Handles UCI communication with chess engine"""
    
    def __init__(self, engine_path: str, time_limit: int = 5000):
        self.engine_path = engine_path
        self.time_limit = time_limit
        self.process = None
        self.ready = False
        
    def start(self):
        """Start the engine process"""
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            # Initialize UCI
            self._send_command("uci")
            self._wait_for("uciok")
            
            # Set ready
            self._send_command("isready")
            self._wait_for("readyok")
            
            self.ready = True
            print(f"✅ Engine {self.engine_path} started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start engine {self.engine_path}: {e}")
            return False
        
        return True
    
    def stop(self):
        """Stop the engine process"""
        if self.process:
            self._send_command("quit")
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.ready = False
    
    def _send_command(self, command: str):
        """Send a command to the engine"""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
    
    def _wait_for(self, expected: str, timeout: int = 5) -> bool:
        """Wait for expected response from engine"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process and self.process.stdout:
                line = self.process.stdout.readline().strip()
                if expected in line:
                    return True
        return False
    
    def analyze_position(self, fen: str, move_time: int = 5000) -> Optional[EngineResponse]:
        """Analyze a position and return engine response"""
        if not self.ready:
            return None
        
        try:
            # Set position
            self._send_command(f"position fen {fen}")
            
            # Start search
            self._send_command(f"go movetime {move_time}")
            
            # Collect search info
            bestmove = None
            evaluation = 0.0
            depth = 0
            time_ms = 0
            pv = []
            nodes = 0
            
            start_time = time.time()
            while time.time() - start_time < (move_time / 1000 + 2):
                if self.process and self.process.stdout:
                    line = self.process.stdout.readline().strip()
                    
                    if line.startswith("info"):
                        parts = line.split()
                        try:
                            if "depth" in parts:
                                depth = int(parts[parts.index("depth") + 1])
                            if "score" in parts:
                                score_idx = parts.index("score")
                                if score_idx + 2 < len(parts) and parts[score_idx + 1] == "cp":
                                    evaluation = int(parts[score_idx + 2]) / 100.0
                                elif score_idx + 2 < len(parts) and parts[score_idx + 1] == "mate":
                                    mate_in = int(parts[score_idx + 2])
                                    evaluation = 999.0 if mate_in > 0 else -999.0
                            if "time" in parts:
                                time_ms = int(parts[parts.index("time") + 1])
                            if "nodes" in parts:
                                nodes = int(parts[parts.index("nodes") + 1])
                            if "pv" in parts:
                                pv_idx = parts.index("pv")
                                pv = parts[pv_idx + 1:]
                        except (ValueError, IndexError):
                            continue
                    
                    elif line.startswith("bestmove"):
                        bestmove = line.split()[1]
                        break
            
            if bestmove:
                return EngineResponse(
                    bestmove=bestmove,
                    evaluation=evaluation,
                    depth=depth,
                    time_ms=time_ms,
                    pv=pv,
                    nodes=nodes
                )
            
        except Exception as e:
            print(f"❌ Error analyzing position: {e}")
        
        return None

def flip_fen(fen: str) -> str:
    """
    Flip a FEN position (swap colors and board orientation)
    This allows us to test the same position from opposite perspectives
    """
    parts = fen.split()
    if len(parts) != 6:
        return fen
    
    # Flip piece placement
    ranks = parts[0].split('/')
    flipped_ranks = []
    
    for rank in reversed(ranks):
        flipped_rank = ""
        for char in rank:
            if char.isalpha():
                # Swap case (white <-> black)
                flipped_rank += char.swapcase()
            else:
                flipped_rank += char
        flipped_ranks.append(flipped_rank)
    
    # Flip active color
    active_color = "b" if parts[1] == "w" else "w"
    
    # Flip castling rights
    castling = parts[2]
    if castling != "-":
        new_castling = ""
        for char in castling:
            new_castling += char.swapcase()
        castling = new_castling
    
    # Flip en passant square
    en_passant = parts[3]
    if en_passant != "-":
        file = en_passant[0]
        rank = int(en_passant[1])
        new_rank = 9 - rank
        en_passant = f"{file}{new_rank}"
    
    return f"{'/'.join(flipped_ranks)} {active_color} {castling} {en_passant} {parts[4]} {parts[5]}"

def run_perspective_test(engine_path: str, test_positions: List[Tuple[str, str]], move_time: int = 3000):
    """Run perspective consistency tests on given positions"""
    
    print(f"\n🔍 V7P3R Perspective Diagnostic Test")
    print(f"Engine: {engine_path}")
    print(f"Time per position: {move_time}ms")
    print(f"Test positions: {len(test_positions)}")
    print("=" * 60)
    
    results = []
    
    # Start engine
    engine = UCIEngine(engine_path, move_time)
    if not engine.start():
        return None
    
    try:
        for i, (description, fen) in enumerate(test_positions, 1):
            print(f"\n📍 Test {i}: {description}")
            print(f"Position: {fen}")
            
            # Test original position (as given)
            print("  🔵 Testing original position...")
            original_response = engine.analyze_position(fen, move_time)
            
            # Test flipped position (opposite color perspective)
            flipped_fen = flip_fen(fen)
            print("  🔴 Testing flipped position...")
            print(f"  Flipped: {flipped_fen}")
            flipped_response = engine.analyze_position(flipped_fen, move_time)
            
            # Analyze results
            if original_response and flipped_response:
                print(f"\n  📊 Results Comparison:")
                print(f"    Original - Move: {original_response.bestmove:8} Eval: {original_response.evaluation:+6.2f} Depth: {original_response.depth}")
                print(f"    Flipped  - Move: {flipped_response.bestmove:8} Eval: {flipped_response.evaluation:+6.2f} Depth: {flipped_response.depth}")
                
                # Check for perspective issues
                eval_consistency = abs(original_response.evaluation + flipped_response.evaluation) < 0.5
                depth_consistency = abs(original_response.depth - flipped_response.depth) <= 1
                
                print(f"    Eval Consistency: {'✅' if eval_consistency else '❌'} (should sum close to 0)")
                print(f"    Depth Consistency: {'✅' if depth_consistency else '❌'}")
                
                if not eval_consistency:
                    print(f"    ⚠️  PERSPECTIVE ISSUE: Evals should be opposite signs but similar magnitude")
                    print(f"    ⚠️  Original: {original_response.evaluation:+.2f}, Flipped: {flipped_response.evaluation:+.2f}, Sum: {original_response.evaluation + flipped_response.evaluation:+.2f}")
                
                results.append({
                    'test': i,
                    'description': description,
                    'fen': fen,
                    'flipped_fen': flipped_fen,
                    'original': {
                        'move': original_response.bestmove,
                        'eval': original_response.evaluation,
                        'depth': original_response.depth,
                        'time': original_response.time_ms,
                        'nodes': original_response.nodes
                    },
                    'flipped': {
                        'move': flipped_response.bestmove,
                        'eval': flipped_response.evaluation,
                        'depth': flipped_response.depth,
                        'time': flipped_response.time_ms,
                        'nodes': flipped_response.nodes
                    },
                    'eval_consistency': eval_consistency,
                    'depth_consistency': depth_consistency,
                    'eval_sum': original_response.evaluation + flipped_response.evaluation
                })
            else:
                print(f"  ❌ Failed to get responses from engine")
                results.append({
                    'test': i,
                    'description': description,
                    'fen': fen,
                    'error': 'Engine communication failed'
                })
    
    finally:
        engine.stop()
    
    return results

def generate_test_positions() -> List[Tuple[str, str]]:
    """Generate test positions based on game patterns from tournament"""
    return [
        # Starting position
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        
        # Positions from problematic games
        ("Early Opening - Reti", "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1"),
        
        # Position after typical v7p3r opening moves
        ("Post c3-d4 setup", "rnbqkbnr/ppp1pppp/8/3p4/3P4/2P2N2/PP2PPPP/RNBQKB1R b KQkq - 0 3"),
        
        # Middle game position where perspective might matter
        ("Middlegame Test", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5"),
        
        # Position where black is to move (common problem area)
        ("Black to Move Test", "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 3 5"),
        
        # Endgame position
        ("Simple Endgame", "8/8/8/8/3k4/8/3K4/8 w - - 0 1"),
        
        # Position with evaluation complexity
        ("Complex Position", "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b - - 0 6"),
    ]

def save_results(results: list, filename: str = "v7p3r_perspective_test_results.json"):
    """Save test results to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def print_summary(results: list):
    """Print test summary"""
    if not results:
        print("\n❌ No results to summarize")
        return
    
    print("\n" + "=" * 60)
    print("📋 PERSPECTIVE TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len([r for r in results if 'error' not in r])
    passed_eval = len([r for r in results if 'error' not in r and r.get('eval_consistency', False)])
    passed_depth = len([r for r in results if 'error' not in r and r.get('depth_consistency', False)])
    
    print(f"Total Tests: {total_tests}")
    print(f"Evaluation Consistency: {passed_eval}/{total_tests} ({'✅' if passed_eval == total_tests else '❌'})")
    print(f"Depth Consistency: {passed_depth}/{total_tests} ({'✅' if passed_depth == total_tests else '❌'})")
    
    if total_tests > 0:
        print(f"\n🎯 Perspective Issues Found: {total_tests - passed_eval}")
        
        # Show worst offenders
        eval_issues = [r for r in results if 'error' not in r and not r.get('eval_consistency', True)]
        if eval_issues:
            print("\n⚠️  WORST PERSPECTIVE ISSUES:")
            for result in sorted(eval_issues, key=lambda x: abs(x['eval_sum']), reverse=True)[:3]:
                print(f"  • {result['description']}: Eval sum = {result['eval_sum']:+.2f}")
                print(f"    Original: {result['original']['eval']:+.2f}, Flipped: {result['flipped']['eval']:+.2f}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python v7p3r_perspective_tester.py <engine_path> [move_time_ms]")
        print("Example: python v7p3r_perspective_tester.py ./v7p3r_v7.2.exe 3000")
        return
    
    engine_path = sys.argv[1]
    move_time = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return
    
    # Generate test positions
    test_positions = generate_test_positions()
    
    # Run tests
    results = run_perspective_test(engine_path, test_positions, move_time)
    
    if results:
        # Print summary
        print_summary(results)
        
        # Save results
        save_results(results)
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        issues_found = len([r for r in results if 'error' not in r and not r.get('eval_consistency', True)])
        if issues_found > 0:
            print("  1. Check evaluation function for sign errors when playing as black")
            print("  2. Verify board representation consistency between white/black perspectives")
            print("  3. Review move generation from black's perspective")
            print("  4. Test evaluation symmetry in search algorithm")
        else:
            print("  ✅ No obvious perspective issues detected in basic tests")
            print("  Consider testing with more complex positions or longer time controls")

if __name__ == "__main__":
    main()
