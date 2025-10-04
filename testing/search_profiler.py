#!/usr/bin/env python3
"""
V7P3R Search Performance Profiler
Tests actual search performance during gameplay scenarios
"""

import subprocess
import time
import threading
from typing import Dict, List

class SearchProfiler:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
    
    def test_search_time(self, position: str, time_limit: int = 5) -> Dict:
        """Test search time for a given position"""
        commands = [
            "uci",
            f"position {position}",
            f"go movetime {time_limit * 1000}",  # Convert to milliseconds
            "quit"
        ]
        
        start_time = time.time()
        result = subprocess.run(
            [self.engine_path],
            input="\n".join(commands),
            text=True,
            capture_output=True,
            timeout=time_limit + 5  # Extra buffer
        )
        actual_time = time.time() - start_time
        
        # Parse the output
        lines = result.stdout.split('\n')
        depths_reached = []
        nodes_searched = 0
        best_move = None
        
        for line in lines:
            if line.startswith('info depth'):
                try:
                    depth = int(line.split('depth')[1].split()[0])
                    depths_reached.append(depth)
                    if 'nodes' in line:
                        nodes = int(line.split('nodes')[1].split()[0])
                        nodes_searched = max(nodes_searched, nodes)
                except:
                    pass
            elif line.startswith('bestmove'):
                best_move = line.split()[1]
        
        max_depth = max(depths_reached) if depths_reached else 0
        nps = nodes_searched / actual_time if actual_time > 0 else 0
        
        return {
            'position': position,
            'time_limit': time_limit,
            'actual_time': actual_time,
            'max_depth': max_depth,
            'nodes': nodes_searched,
            'nps': nps,
            'best_move': best_move,
            'time_exceeded': actual_time > (time_limit + 1),  # 1 second tolerance
            'output': result.stdout
        }
    
    def profile_positions(self) -> None:
        """Profile various game positions"""
        print("V7P3R Search Performance Profiler")
        print("=" * 50)
        
        # Test positions of varying complexity
        positions = [
            ("startpos", "Starting Position", 5),
            ("startpos moves e2e4 e7e5", "King's Pawn Opening", 5),
            ("fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Kiwipete (Complex)", 3),
            ("fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Endgame Position", 5),
        ]
        
        print(f"\n{'Position':<20} {'Time Limit':<12} {'Actual Time':<12} {'Max Depth':<10} {'Nodes':<10} {'NPS':<12} {'Status'}")
        print("-" * 95)
        
        for pos, name, time_limit in positions:
            print(f"\n🔍 Testing: {name}")
            result = self.test_search_time(pos, time_limit)
            
            status = "✅ OK"
            if result['time_exceeded']:
                status = "⏰ TIME EXCEEDED"
            elif result['max_depth'] == 0:
                status = "❌ NO SEARCH"
            elif result['max_depth'] < 3:
                status = "⚠️ SHALLOW"
            
            print(f"{name:<20} {time_limit:<12} {result['actual_time']:<12.2f} {result['max_depth']:<10} {result['nodes']:<10} {result['nps']:<12.0f} {status}")
            
            # Show search progress for problematic cases
            if result['time_exceeded'] or result['max_depth'] < 3:
                print(f"   🔍 Search output preview:")
                output_lines = result['output'].split('\n')[:10]
                for line in output_lines:
                    if line.startswith('info'):
                        print(f"   📊 {line}")
    
    def test_time_management(self) -> None:
        """Test if engine respects time limits"""
        print(f"\n⏱️ TIME MANAGEMENT TEST")
        print("-" * 30)
        
        time_limits = [1, 3, 5, 10]  # seconds
        
        for time_limit in time_limits:
            result = self.test_search_time("startpos", time_limit)
            
            tolerance = 1.0  # 1 second tolerance
            if result['actual_time'] > (time_limit + tolerance):
                status = f"❌ EXCEEDED by {result['actual_time'] - time_limit:.1f}s"
            else:
                status = "✅ OK"
            
            print(f"Time Limit: {time_limit}s -> Actual: {result['actual_time']:.2f}s {status}")

if __name__ == "__main__":
    import sys
    
    # Default to v12.5 engine
    engine_path = sys.argv[1] if len(sys.argv) > 1 else "s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-tester/engines/V7P3R/V7P3R_v12.5.exe"
    
    profiler = SearchProfiler(engine_path)
    profiler.profile_positions()
    profiler.test_time_management()