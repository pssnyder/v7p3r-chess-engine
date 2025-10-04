#!/usr/bin/env python3
"""
V7P3R V12.6 vs V12.2 Comprehensive Testing Suite
Tests the clean V12.6 build against the proven V12.2 baseline
"""

import subprocess
import time
import json
import os
from pathlib import Path

class EngineComparison:
    def __init__(self):
        self.base_path = Path("s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine")
        self.v12_2_path = self.base_path / "dist" / "V7P3R_v12.2.exe"
        self.v12_6_path = self.base_path / "dist" / "V7P3R_v12.6.exe"
        self.test_results = []
        
    def send_command(self, engine_process, command):
        """Send command to engine and wait for response"""
        engine_process.stdin.write(f"{command}\n")
        engine_process.stdin.flush()
        time.sleep(0.1)  # Give engine time to process
        
    def get_engine_response(self, engine_process, command, timeout=5):
        """Send command and capture engine response"""
        engine_process.stdin.write(f"{command}\n")
        engine_process.stdin.flush()
        
        response_lines = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                engine_process.stdout.settimeout(0.1)
                line = engine_process.stdout.readline()
                if line:
                    line = line.strip()
                    response_lines.append(line)
                    if line.startswith("bestmove") or line == "uciok" or line == "readyok":
                        break
            except:
                continue
                
        return response_lines
        
    def start_engine(self, engine_path):
        """Start an engine process"""
        return subprocess.Popen(
            str(engine_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
    def test_basic_uci_compliance(self):
        """Test basic UCI protocol compliance"""
        print("=== Testing Basic UCI Compliance ===")
        
        for version, engine_path in [("V12.2", self.v12_2_path), ("V12.6", self.v12_6_path)]:
            print(f"\nTesting {version}...")
            
            try:
                engine = self.start_engine(engine_path)
                
                # Test UCI identification
                response = self.get_engine_response(engine, "uci")
                print(f"{version} UCI Response: {response}")
                
                # Test ready
                response = self.get_engine_response(engine, "isready")
                print(f"{version} Ready Response: {response}")
                
                engine.terminate()
                engine.wait()
                
            except Exception as e:
                print(f"Error testing {version}: {e}")
                
    def test_starting_position_analysis(self):
        """Test both engines analyzing starting position"""
        print("\n=== Testing Starting Position Analysis ===")
        
        test_cases = [
            {
                "name": "Quick Analysis (1s)",
                "position": "position startpos",
                "time": "go movetime 1000"
            },
            {
                "name": "Medium Analysis (3s)", 
                "position": "position startpos",
                "time": "go movetime 3000"
            },
            {
                "name": "Deep Analysis (5s)",
                "position": "position startpos", 
                "time": "go movetime 5000"
            }
        ]
        
        for test_case in test_cases:
            print(f"\n--- {test_case['name']} ---")
            
            for version, engine_path in [("V12.2", self.v12_2_path), ("V12.6", self.v12_6_path)]:
                try:
                    engine = self.start_engine(engine_path)
                    
                    # Initialize
                    self.send_command(engine, "uci")
                    time.sleep(0.2)
                    self.send_command(engine, "isready")
                    time.sleep(0.2)
                    
                    # Set position
                    self.send_command(engine, test_case["position"])
                    time.sleep(0.1)
                    
                    # Start analysis
                    start_time = time.time()
                    response = self.get_engine_response(engine, test_case["time"], timeout=10)
                    end_time = time.time()
                    
                    # Find bestmove and evaluation info
                    bestmove = None
                    best_eval = None
                    depth_reached = 0
                    nodes_searched = 0
                    
                    for line in response:
                        if line.startswith("bestmove"):
                            bestmove = line.split()[1] if len(line.split()) > 1 else "none"
                        elif line.startswith("info depth"):
                            parts = line.split()
                            try:
                                depth_idx = parts.index("depth")
                                depth_reached = max(depth_reached, int(parts[depth_idx + 1]))
                            except:
                                pass
                            try:
                                score_idx = parts.index("score")
                                if score_idx + 2 < len(parts) and parts[score_idx + 1] == "cp":
                                    best_eval = int(parts[score_idx + 2])
                            except:
                                pass
                            try:
                                nodes_idx = parts.index("nodes")
                                nodes_searched = max(nodes_searched, int(parts[nodes_idx + 1]))
                            except:
                                pass
                    
                    result = {
                        "version": version,
                        "test": test_case["name"],
                        "bestmove": bestmove,
                        "evaluation": best_eval,
                        "depth": depth_reached,
                        "nodes": nodes_searched,
                        "time": end_time - start_time
                    }
                    
                    self.test_results.append(result)
                    
                    print(f"{version}: {bestmove} (eval: {best_eval}, depth: {depth_reached}, nodes: {nodes_searched}, time: {end_time - start_time:.2f}s)")
                    
                    engine.terminate()
                    engine.wait()
                    
                except Exception as e:
                    print(f"Error testing {version} on {test_case['name']}: {e}")
                    
    def test_tactical_positions(self):
        """Test known tactical positions"""
        print("\n=== Testing Tactical Positions ===")
        
        tactical_positions = [
            {
                "name": "Simple Knight Fork",
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                "expected_type": "Knight development or center control"
            },
            {
                "name": "King Safety Test",
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",
                "expected_type": "Castling or development"
            },
            {
                "name": "Material Win",
                "fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
                "expected_type": "Center control"
            }
        ]
        
        for position in tactical_positions:
            print(f"\n--- {position['name']} ---")
            print(f"Expected: {position['expected_type']}")
            
            for version, engine_path in [("V12.2", self.v12_2_path), ("V12.6", self.v12_6_path)]:
                try:
                    engine = self.start_engine(engine_path)
                    
                    # Initialize
                    self.send_command(engine, "uci")
                    time.sleep(0.2)
                    self.send_command(engine, "isready")
                    time.sleep(0.2)
                    
                    # Set position
                    self.send_command(engine, f"position fen {position['fen']}")
                    time.sleep(0.1)
                    
                    # Analyze
                    response = self.get_engine_response(engine, "go movetime 3000", timeout=8)
                    
                    # Extract results
                    bestmove = None
                    best_eval = None
                    
                    for line in response:
                        if line.startswith("bestmove"):
                            bestmove = line.split()[1] if len(line.split()) > 1 else "none"
                        elif "score cp" in line:
                            parts = line.split()
                            try:
                                score_idx = parts.index("score")
                                if score_idx + 2 < len(parts) and parts[score_idx + 1] == "cp":
                                    best_eval = int(parts[score_idx + 2])
                            except:
                                pass
                    
                    print(f"{version}: {bestmove} (eval: {best_eval})")
                    
                    engine.terminate()
                    engine.wait()
                    
                except Exception as e:
                    print(f"Error testing {version} on {position['name']}: {e}")
                    
    def test_time_management(self):
        """Test time management under pressure"""
        print("\n=== Testing Time Management ===")
        
        time_scenarios = [
            {"name": "Rapid (100ms)", "time": 100},
            {"name": "Blitz (500ms)", "time": 500},
            {"name": "Standard (2000ms)", "time": 2000}
        ]
        
        for scenario in time_scenarios:
            print(f"\n--- {scenario['name']} ---")
            
            for version, engine_path in [("V12.2", self.v12_2_path), ("V12.6", self.v12_6_path)]:
                try:
                    engine = self.start_engine(engine_path)
                    
                    # Initialize
                    self.send_command(engine, "uci")
                    time.sleep(0.2)
                    self.send_command(engine, "isready") 
                    time.sleep(0.2)
                    
                    # Set starting position
                    self.send_command(engine, "position startpos")
                    time.sleep(0.1)
                    
                    # Test time constraint
                    start_time = time.time()
                    response = self.get_engine_response(engine, f"go movetime {scenario['time']}", timeout=5)
                    end_time = time.time()
                    
                    actual_time = (end_time - start_time) * 1000  # Convert to ms
                    
                    bestmove = None
                    for line in response:
                        if line.startswith("bestmove"):
                            bestmove = line.split()[1] if len(line.split()) > 1 else "none"
                            break
                    
                    time_accuracy = abs(actual_time - scenario['time'])
                    print(f"{version}: {bestmove} (requested: {scenario['time']}ms, actual: {actual_time:.0f}ms, diff: {time_accuracy:.0f}ms)")
                    
                    engine.terminate()
                    engine.wait()
                    
                except Exception as e:
                    print(f"Error testing {version} time management: {e}")
                    
    def generate_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # Analyze results by test type
        starting_pos_results = [r for r in self.test_results if "Analysis" in r["test"]]
        
        if starting_pos_results:
            print("\nStarting Position Analysis Summary:")
            v12_2_results = [r for r in starting_pos_results if r["version"] == "V12.2"]
            v12_6_results = [r for r in starting_pos_results if r["version"] == "V12.6"]
            
            for test_name in set(r["test"] for r in starting_pos_results):
                v12_2 = next((r for r in v12_2_results if r["test"] == test_name), None)
                v12_6 = next((r for r in v12_6_results if r["test"] == test_name), None)
                
                if v12_2 and v12_6:
                    print(f"\n{test_name}:")
                    print(f"  V12.2: {v12_2['bestmove']} (depth: {v12_2['depth']}, nodes: {v12_2['nodes']}, time: {v12_2['time']:.2f}s)")
                    print(f"  V12.6: {v12_6['bestmove']} (depth: {v12_6['depth']}, nodes: {v12_6['nodes']}, time: {v12_6['time']:.2f}s)")
                    
                    if v12_2['bestmove'] == v12_6['bestmove']:
                        print(f"  ✅ Same move chosen")
                    else:
                        print(f"  ⚠️  Different moves chosen")
                        
                    nodes_ratio = v12_6['nodes'] / max(v12_2['nodes'], 1)
                    print(f"  📊 V12.6 searched {nodes_ratio:.2f}x as many nodes as V12.2")
        
        print(f"\n📋 Total test results collected: {len(self.test_results)}")
        
        # Save detailed results
        results_file = self.base_path / "testing" / f"v12_6_vs_v12_2_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"📁 Detailed results saved to: {results_file}")
        
    def run_all_tests(self):
        """Run the complete test suite"""
        print("V7P3R V12.6 vs V12.2 COMPREHENSIVE TESTING")
        print("="*50)
        print(f"V12.2 Engine: {self.v12_2_path}")
        print(f"V12.6 Engine: {self.v12_6_path}")
        
        # Verify engines exist
        if not self.v12_2_path.exists():
            print(f"❌ V12.2 engine not found: {self.v12_2_path}")
            return
        if not self.v12_6_path.exists():
            print(f"❌ V12.6 engine not found: {self.v12_6_path}")
            return
            
        print("✅ Both engines found, starting tests...\n")
        
        try:
            self.test_basic_uci_compliance()
            self.test_starting_position_analysis()
            self.test_tactical_positions()
            self.test_time_management()
            self.generate_report()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Testing interrupted by user")
        except Exception as e:
            print(f"\n\n❌ Testing error: {e}")


if __name__ == "__main__":
    tester = EngineComparison()
    tester.run_all_tests()