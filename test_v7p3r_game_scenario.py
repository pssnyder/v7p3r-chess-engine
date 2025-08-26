#!/usr/bin/env python3
"""
Test V7P3R v7.2 in a simulated game scenario
"""

import subprocess
import time

def test_game_scenario():
    """Test V7P3R v7.2 in a real game-like scenario"""
    print("Testing V7P3R v7.2 in game scenario...")
    print("This simulates the conditions from Engine Battle tournaments")
    
    try:
        process = subprocess.Popen(
            ['python', 'src/v7p3r_uci.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine'
        )
        
        # Simulate tournament conditions: 300+5 time control (5 minutes + 5 second increment)
        commands = [
            "uci",
            "isready",
            "ucinewgame",
            "position startpos",
            "go wtime 300000 btime 300000 winc 5000 binc 5000",  # 5 minute game
            "position startpos moves e2e4",
            "go wtime 295000 btime 295000 winc 5000 binc 5000",  # After first move
            "position startpos moves e2e4 e7e5",
            "go wtime 290000 btime 290000 winc 5000 binc 5000",  # After second move
        ]
        
        start_time = time.time()
        
        for cmd in commands:
            print(f">>> {cmd}")
            process.stdin.write(cmd + '\n')
            process.stdin.flush()
            
            if cmd.startswith("go"):
                # Wait for bestmove response
                time.sleep(0.1)
                
        # Wait a bit more then quit
        time.sleep(2)
        process.stdin.write("quit\n")
        process.stdin.flush()
        
        # Get output
        stdout, stderr = process.communicate(timeout=10)
        
        print("\nEngine output:")
        print(stdout)
        
        if stderr:
            print("Errors:")
            print(stderr)
            
        # Analyze the output for time usage
        lines = stdout.split('\n')
        move_times = []
        
        for line in lines:
            if 'time' in line and 'bestmove' not in line:
                try:
                    # Extract time from info line
                    parts = line.split()
                    if 'time' in parts:
                        time_idx = parts.index('time')
                        if time_idx + 1 < len(parts):
                            move_time = int(parts[time_idx + 1]) / 1000.0
                            move_times.append(move_time)
                except:
                    pass
        
        if move_times:
            avg_time = sum(move_times) / len(move_times)
            max_time = max(move_times)
            print(f"\nTime analysis:")
            print(f"Average thinking time: {avg_time:.2f}s")
            print(f"Maximum thinking time: {max_time:.2f}s")
            print(f"Times: {[f'{t:.2f}s' for t in move_times]}")
            
            if avg_time < 12.0:
                print("✅ SUCCESS: Competitive time management!")
            else:
                print("⚠️  Still needs optimization")
        else:
            print("Could not extract timing data")
            
    except Exception as e:
        print(f"Error testing game scenario: {e}")


if __name__ == "__main__":
    test_game_scenario()
