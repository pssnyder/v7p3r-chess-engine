#!/usr/bin/env python3
"""
UCI Tournament Test Script for V7P3R Chess Engine
Tests all common Arena GUI time controls and tournament formats
"""

import subprocess
import time
import threading
import sys
from typing import List, Optional

class UCITester:
    def __init__(self, engine_path: str = "python v7p3r_uci.py"):
        self.engine_path = engine_path
        self.process = None
        
    def start_engine(self):
        """Start the UCI engine process"""
        self.process = subprocess.Popen(
            self.engine_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        
    def send_command(self, command: str) -> str:
        """Send a command to the engine and get response"""
        if not self.process:
            return "Engine not started"
            
        print(f">>> {command}")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        
        # Read response
        response_lines = []
        start_time = time.time()
        
        while time.time() - start_time < 10:  # 10 second timeout
            if self.process.poll() is not None:
                break
                
            # Try to read a line with timeout
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    print(f"<<< {line}")
                    response_lines.append(line)
                    
                    # Stop reading on certain commands
                    if line.startswith("uciok") or line.startswith("readyok") or line.startswith("bestmove"):
                        break
                        
            except Exception as e:
                print(f"Error reading: {e}")
                break
                
        return "\n".join(response_lines)
        
    def stop_engine(self):
        """Stop the engine process"""
        if self.process:
            try:
                self.send_command("quit")
                self.process.wait(timeout=2)
            except:
                self.process.terminate()
                
    def test_basic_uci(self):
        """Test basic UCI protocol"""
        print("\n=== Testing Basic UCI Protocol ===")
        
        response = self.send_command("uci")
        assert "uciok" in response, "Engine didn't respond with uciok"
        assert "V7P3R v5.0" in response, "Engine name not found"
        
        response = self.send_command("isready")
        assert "readyok" in response, "Engine not ready"
        
        print("✓ Basic UCI protocol working")
        
    def test_options(self):
        """Test UCI options"""
        print("\n=== Testing UCI Options ===")
        
        # Test hash size option
        self.send_command("setoption name Hash value 128")
        self.send_command("setoption name Move Overhead value 50")
        
        response = self.send_command("isready")
        assert "readyok" in response, "Engine not ready after setting options"
        
        print("✓ UCI options working")
        
    def test_position_setup(self):
        """Test position setup"""
        print("\n=== Testing Position Setup ===")
        
        # Test starting position
        self.send_command("position startpos")
        
        # Test with moves
        self.send_command("position startpos moves e2e4 e7e5 g1f3")
        
        # Test FEN position
        self.send_command("position fen rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        
        response = self.send_command("isready")
        assert "readyok" in response, "Engine not ready after position setup"
        
        print("✓ Position setup working")
        
    def test_time_control_2_1(self):
        """Test 2 minutes + 1 second increment (2/1)"""
        print("\n=== Testing 2/1 Time Control ===")
        
        self.send_command("ucinewgame")
        self.send_command("position startpos")
        
        # 2 minutes = 120,000 ms, 1 second increment = 1000 ms
        response = self.send_command("go wtime 120000 btime 120000 winc 1000 binc 1000")
        
        # Should see info output and bestmove
        assert "info depth" in response, "No depth info output"
        assert "score cp" in response, "No score output"
        assert "nodes" in response, "No nodes output"
        assert "nps" in response, "No NPS output"
        assert "pv" in response, "No PV output"
        assert "bestmove" in response, "No best move output"
        
        print("✓ 2/1 time control working")
        
    def test_time_control_5_5(self):
        """Test 5 minutes + 5 second increment (5/5)"""
        print("\n=== Testing 5/5 Time Control ===")
        
        self.send_command("ucinewgame")
        self.send_command("position startpos moves e2e4")
        
        # 5 minutes = 300,000 ms, 5 second increment = 5000 ms
        response = self.send_command("go wtime 300000 btime 300000 winc 5000 binc 5000")
        
        assert "info depth" in response, "No depth info output"
        assert "bestmove" in response, "No best move output"
        
        print("✓ 5/5 time control working")
        
    def test_bullet_60s(self):
        """Test 60-second bullet"""
        print("\n=== Testing 60s Bullet ===")
        
        self.send_command("ucinewgame")
        self.send_command("position startpos")
        
        # 60 seconds = 60,000 ms
        response = self.send_command("go wtime 60000 btime 60000")
        
        assert "info depth" in response, "No depth info output"
        assert "bestmove" in response, "No best move output"
        
        print("✓ 60s bullet working")
        
    def test_fixed_depth(self):
        """Test fixed depth search"""
        print("\n=== Testing Fixed Depth Search ===")
        
        self.send_command("ucinewgame")
        self.send_command("position startpos")
        
        response = self.send_command("go depth 4")
        
        # Should search exactly to depth 4
        assert "info depth 4" in response, "Didn't reach depth 4"
        assert "bestmove" in response, "No best move output"
        
        print("✓ Fixed depth search working")
        
    def test_node_limit(self):
        """Test node-limited search"""
        print("\n=== Testing Node-Limited Search ===")
        
        self.send_command("ucinewgame")
        self.send_command("position startpos")
        
        response = self.send_command("go nodes 1000")
        
        assert "nodes" in response, "No nodes output"
        assert "bestmove" in response, "No best move output"
        
        print("✓ Node-limited search working")
        
    def test_movetime(self):
        """Test fixed move time"""
        print("\n=== Testing Fixed Move Time ===")
        
        self.send_command("ucinewgame")
        self.send_command("position startpos")
        
        start_time = time.time()
        response = self.send_command("go movetime 2000")  # 2 seconds
        end_time = time.time()
        
        # Should take approximately 2 seconds
        elapsed = end_time - start_time
        assert 1.5 < elapsed < 3.0, f"Move time too far from 2s: {elapsed:.2f}s"
        assert "bestmove" in response, "No best move output"
        
        print(f"✓ Fixed move time working ({elapsed:.2f}s)")
        
    def test_stop_command(self):
        """Test stop command during search"""
        print("\n=== Testing Stop Command ===")
        
        self.send_command("ucinewgame")
        self.send_command("position startpos")
        
        # Start a long search
        self.send_command("go infinite")
        
        # Wait a bit then stop
        time.sleep(0.5)
        response = self.send_command("stop")
        
        # Should get a bestmove quickly
        assert "bestmove" in response, "No best move after stop"
        
        print("✓ Stop command working")
        
    def run_all_tests(self):
        """Run all tournament tests"""
        print("Starting V7P3R UCI Tournament Tests...")
        
        try:
            self.start_engine()
            
            self.test_basic_uci()
            self.test_options()
            self.test_position_setup()
            self.test_time_control_2_1()
            self.test_time_control_5_5()
            self.test_bullet_60s()
            self.test_fixed_depth()
            self.test_node_limit()
            self.test_movetime()
            self.test_stop_command()
            
            print("\n🎉 All tests passed! Engine ready for tournament play!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.stop_engine()

def main():
    """Run the UCI tournament tests"""
    tester = UCITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
