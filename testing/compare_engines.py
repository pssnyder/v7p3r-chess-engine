#!/usr/bin/env python3
"""
Engine Comparison Tool - C0BR4 v2.0 vs V7P3R v6.2
Tests both engines on the same positions with identical time controls
"""

import subprocess
import time
import os
import sys
from typing import List, Tuple, Dict

class EngineComparison:
    def __init__(self):
        self.cobra_path = r"S:\Maker Stuff\Programming\Chess Engines\C0BR4 Chess Engine\cobra-chess-engine\dist\v2.0\C0BR4_v2.0.exe"
        self.v7p3r_path = r"S:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\src\v7p3r_uci.py"
        self.python_path = r"C:/Users/patss/AppData/Local/Programs/Python/Python313/python.exe"
    
    def send_uci_commands(self, engine_path: str, commands: List[str], is_python: bool = False) -> List[str]:
        """Send UCI commands to an engine and collect responses"""
        try:
            if is_python:
                process = subprocess.Popen(
                    [self.python_path, engine_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.path.dirname(engine_path)
                )
            else:
                process = subprocess.Popen(
                    [engine_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            # Send all commands
            input_text = "\n".join(commands) + "\n"
            stdout, stderr = process.communicate(input=input_text, timeout=30)
            
            return stdout.strip().split('\n')
            
        except subprocess.TimeoutExpired:
            process.kill()
            return ["ERROR: Engine timeout"]
        except Exception as e:
            return [f"ERROR: {str(e)}"]
    
    def parse_engine_response(self, output_lines: List[str]) -> Dict:
        """Parse UCI engine output to extract key information"""
        result = {
            'bestmove': None,
            'depth': 0,
            'nodes': 0,
            'time': 0,
            'nps': 0,
            'score_cp': 0,
            'pv': '',
            'engine_name': 'Unknown'
        }
        
        for line in output_lines:
            line = line.strip()
            
            if line.startswith('id name'):
                result['engine_name'] = line[8:]
            elif line.startswith('bestmove'):
                parts = line.split()
                if len(parts) >= 2:
                    result['bestmove'] = parts[1]
            elif line.startswith('info') and 'depth' in line:
                # Parse info line: info depth 4 score cp 25 nodes 1234 nps 5000 time 247 pv e2e4
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'depth' and i + 1 < len(parts):
                        try:
                            result['depth'] = int(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'nodes' and i + 1 < len(parts):
                        try:
                            result['nodes'] = int(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'time' and i + 1 < len(parts):
                        try:
                            result['time'] = int(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'nps' and i + 1 < len(parts):
                        try:
                            result['nps'] = int(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'cp' and i + 1 < len(parts):
                        try:
                            result['score_cp'] = int(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'pv' and i + 1 < len(parts):
                        result['pv'] = ' '.join(parts[i + 1:])
                        break
        
        return result
    
    def test_position(self, position: str, time_control: str, description: str) -> Tuple[Dict, Dict]:
        """Test both engines on a single position"""
        print(f"\n🎯 Testing: {description}")
        print(f"Position: {position}")
        print(f"Time Control: {time_control}")
        print("-" * 60)
        
        # Prepare UCI commands
        base_commands = [
            "uci",
            "isready",
            "ucinewgame",
            f"position {position}",
            f"go {time_control}",
            "quit"
        ]
        
        # Test C0BR4
        print("🐍 Testing C0BR4 v2.0...")
        cobra_output = self.send_uci_commands(self.cobra_path, base_commands, is_python=False)
        cobra_result = self.parse_engine_response(cobra_output)
        
        # Test V7P3R
        print("⚡ Testing V7P3R v6.2...")
        v7p3r_output = self.send_uci_commands(self.v7p3r_path, base_commands, is_python=True)
        v7p3r_result = self.parse_engine_response(v7p3r_output)
        
        return cobra_result, v7p3r_result
    
    def format_result(self, result: Dict, engine_name: str) -> str:
        """Format engine result for display"""
        return f"""
{engine_name}:
  Best Move: {result['bestmove']}
  Depth: {result['depth']}
  Nodes: {result['nodes']:,}
  Time: {result['time']}ms
  NPS: {result['nps']:,}
  Score: {result['score_cp']} cp
  PV: {result['pv']}"""
    
    def compare_results(self, cobra_result: Dict, v7p3r_result: Dict) -> str:
        """Generate comparison analysis"""
        analysis = "\n📊 Performance Comparison:"
        
        # Speed comparison
        if cobra_result['nps'] > 0 and v7p3r_result['nps'] > 0:
            speed_ratio = v7p3r_result['nps'] / cobra_result['nps']
            analysis += f"\n  V7P3R is {speed_ratio:.2f}x faster in NPS"
        
        # Time comparison  
        if cobra_result['time'] > 0 and v7p3r_result['time'] > 0:
            time_ratio = cobra_result['time'] / v7p3r_result['time']
            analysis += f"\n  V7P3R is {time_ratio:.2f}x faster in time"
        
        # Depth comparison
        depth_diff = v7p3r_result['depth'] - cobra_result['depth']
        analysis += f"\n  Depth difference: {depth_diff:+d} (V7P3R vs C0BR4)"
        
        # Move comparison
        if cobra_result['bestmove'] == v7p3r_result['bestmove']:
            analysis += f"\n  ✅ Same move selected: {cobra_result['bestmove']}"
        else:
            analysis += f"\n  ⚠️  Different moves:"
            analysis += f"\n     C0BR4: {cobra_result['bestmove']}"
            analysis += f"\n     V7P3R: {v7p3r_result['bestmove']}"
        
        # Score comparison
        score_diff = v7p3r_result['score_cp'] - cobra_result['score_cp']
        analysis += f"\n  Score difference: {score_diff:+d} cp (V7P3R vs C0BR4)"
        
        return analysis
    
    def run_comparison_suite(self):
        """Run complete comparison test suite"""
        print("🔥 C0BR4 v2.0 vs V7P3R v6.2 Engine Comparison")
        print("=" * 60)
        
        # Test positions - various game phases and complexities
        test_cases = [
            # (position, time_control, description)
            ("startpos", "movetime 1000", "Opening Position - 1 second"),
            ("startpos", "movetime 5000", "Opening Position - 5 seconds"),
            ("fen rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "movetime 3000", "Sicilian Defense - 3 seconds"),
            ("fen r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5", "movetime 2000", "Four Knights - 2 seconds"),
            ("fen r2qkb1r/ppp2ppp/2n1bn2/4p3/4P3/3P1N2/PPP1NPPP/R1BQK2R w KQkq - 0 9", "movetime 4000", "Complex Middlegame - 4 seconds"),
            ("fen 8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", "movetime 2000", "Pawn Endgame - 2 seconds"),
            ("startpos moves e2e4 e7e5 g1f3 b8c6 f1b5", "movetime 1500", "Spanish Opening - 1.5 seconds"),
        ]
        
        overall_results = []
        
        for position, time_control, description in test_cases:
            try:
                cobra_result, v7p3r_result = self.test_position(position, time_control, description)
                
                print(self.format_result(cobra_result, "C0BR4 v2.0"))
                print(self.format_result(v7p3r_result, "V7P3R v6.2"))
                print(self.compare_results(cobra_result, v7p3r_result))
                
                overall_results.append((description, cobra_result, v7p3r_result))
                
            except Exception as e:
                print(f"❌ Error testing {description}: {e}")
            
            print("\n" + "="*60)
        
        # Overall summary
        self.generate_summary(overall_results)
    
    def generate_summary(self, results: List):
        """Generate overall comparison summary"""
        print("\n🏆 OVERALL COMPARISON SUMMARY")
        print("=" * 50)
        
        if not results:
            print("No results to analyze")
            return
        
        total_tests = len(results)
        same_moves = 0
        v7p3r_faster_time = 0
        v7p3r_higher_nps = 0
        v7p3r_deeper = 0
        
        total_v7p3r_nps = 0
        total_cobra_nps = 0
        valid_nps_tests = 0
        
        for description, cobra_result, v7p3r_result in results:
            # Count same moves
            if cobra_result['bestmove'] == v7p3r_result['bestmove']:
                same_moves += 1
            
            # Count speed advantages
            if cobra_result['time'] > 0 and v7p3r_result['time'] > 0:
                if v7p3r_result['time'] < cobra_result['time']:
                    v7p3r_faster_time += 1
            
            if cobra_result['nps'] > 0 and v7p3r_result['nps'] > 0:
                if v7p3r_result['nps'] > cobra_result['nps']:
                    v7p3r_higher_nps += 1
                total_v7p3r_nps += v7p3r_result['nps']
                total_cobra_nps += cobra_result['nps']
                valid_nps_tests += 1
            
            # Count depth advantages
            if v7p3r_result['depth'] > cobra_result['depth']:
                v7p3r_deeper += 1
        
        print(f"📊 Test Results ({total_tests} positions tested):")
        print(f"  Same moves selected: {same_moves}/{total_tests} ({same_moves/total_tests*100:.1f}%)")
        print(f"  V7P3R faster time: {v7p3r_faster_time}/{total_tests} ({v7p3r_faster_time/total_tests*100:.1f}%)")
        print(f"  V7P3R higher NPS: {v7p3r_higher_nps}/{total_tests} ({v7p3r_higher_nps/total_tests*100:.1f}%)")
        print(f"  V7P3R deeper search: {v7p3r_deeper}/{total_tests} ({v7p3r_deeper/total_tests*100:.1f}%)")
        
        if valid_nps_tests > 0:
            avg_v7p3r_nps = total_v7p3r_nps / valid_nps_tests
            avg_cobra_nps = total_cobra_nps / valid_nps_tests
            nps_ratio = avg_v7p3r_nps / avg_cobra_nps if avg_cobra_nps > 0 else 0
            print(f"\n⚡ Average Performance:")
            print(f"  C0BR4 average NPS: {avg_cobra_nps:,.0f}")
            print(f"  V7P3R average NPS: {avg_v7p3r_nps:,.0f}")
            print(f"  V7P3R speed advantage: {nps_ratio:.2f}x")
        
        print(f"\n🎯 Conclusion:")
        if same_moves / total_tests >= 0.7:
            print("  ✅ Engines show good move agreement")
        else:
            print("  ⚠️  Engines show significant move differences")
        
        if v7p3r_higher_nps / total_tests >= 0.7:
            print("  ✅ V7P3R consistently faster")
        else:
            print("  ⚠️  Mixed performance results")

if __name__ == "__main__":
    print("🚀 Starting Engine Comparison Test")
    print("This will test C0BR4 v2.0 vs V7P3R v6.2 on various positions")
    print("=" * 60)
    
    comparison = EngineComparison()
    
    # Verify engines exist
    if not os.path.exists(comparison.cobra_path):
        print(f"❌ C0BR4 engine not found at: {comparison.cobra_path}")
        sys.exit(1)
    
    if not os.path.exists(comparison.v7p3r_path):
        print(f"❌ V7P3R engine not found at: {comparison.v7p3r_path}")
        sys.exit(1)
    
    try:
        comparison.run_comparison_suite()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
