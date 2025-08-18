#!/usr/bin/env python3
"""
Test V7P3R Engine v4.3 Time Controls - Simplified Version
Compare v4.3 simplified engine performance with previous versions.
"""

import subprocess
import time
import sys
import os

def test_engine_time_controls(exe_path, test_name):
    """Test engine response time under various time controls."""
    print(f"\n{'='*60}")
    print(f"Testing {test_name}")
    print(f"Executable: {exe_path}")
    print(f"{'='*60}")
    
    # Test scenarios with different time controls
    test_scenarios = [
        ("Quick move (1 second)", ["uci", "position startpos", "go movetime 1000", "quit"]),
        ("Blitz move (3 seconds)", ["uci", "position startpos", "go movetime 3000", "quit"]),
        ("Standard move (5 seconds)", ["uci", "position startpos", "go movetime 5000", "quit"]),
        ("Middle game position", ["uci", "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5", "go movetime 2000", "quit"]),
        ("Tactical position", ["uci", "position fen r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1", "go movetime 2000", "quit"])
    ]
    
    results = []
    
    for scenario_name, commands in test_scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # Prepare command input
        command_input = "\n".join(commands) + "\n"
        
        # Measure time
        start_time = time.time()
        
        try:
            # Run the engine
            result = subprocess.run(
                [exe_path],
                input=command_input,
                text=True,
                capture_output=True,
                timeout=30  # 30 second timeout
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output for move
            best_move = "No move found"
            for line in result.stdout.split('\n'):
                if line.startswith('bestmove'):
                    best_move = line.strip()
                    break
            
            print(f"  Time taken: {elapsed_time:.2f} seconds")
            print(f"  Best move: {best_move}")
            print(f"  Engine output (last 3 lines):")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-3:]:
                if line.strip():
                    print(f"    {line}")
            
            results.append({
                'scenario': scenario_name,
                'time': elapsed_time,
                'move': best_move,
                'success': True
            })
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            print(f"  ❌ TIMEOUT after {elapsed_time:.2f} seconds")
            results.append({
                'scenario': scenario_name,
                'time': elapsed_time,
                'move': "TIMEOUT",
                'success': False
            })
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"  ❌ ERROR: {e}")
            results.append({
                'scenario': scenario_name,
                'time': elapsed_time,
                'move': "ERROR",
                'success': False
            })
    
    return results

def compare_engines():
    """Compare v4.3 with previous versions if available."""
    print("V7P3R Engine Time Control Comparison - v4.3 Simplified")
    print("=" * 65)
    
    # Engine paths
    engines = [
        ("V7P3R v4.3 (Simplified)", "dist/V7P3R_v4.3.exe"),
        ("V7P3R v4.2 (Previous)", "build/V7P3R_v4.2/V7P3R_v4.2.exe")
    ]
    
    all_results = {}
    
    for engine_name, engine_path in engines:
        if os.path.exists(engine_path):
            results = test_engine_time_controls(engine_path, engine_name)
            all_results[engine_name] = results
        else:
            print(f"\n❌ Engine not found: {engine_path}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    if len(all_results) > 1:
        print(f"{'Scenario':<25} {'v4.3 Time':<12} {'v4.2 Time':<12} {'Improvement'}")
        print("-" * 60)
        
        v43_results = all_results.get("V7P3R v4.3 (Simplified)", [])
        v42_results = all_results.get("V7P3R v4.2 (Previous)", [])
        
        for i, v43_result in enumerate(v43_results):
            if i < len(v42_results) and v43_result['success'] and v42_results[i]['success']:
                v43_time = v43_result['time']
                v42_time = v42_results[i]['time']
                improvement = ((v42_time - v43_time) / v42_time) * 100
                
                print(f"{v43_result['scenario'][:24]:<25} {v43_time:<12.2f} {v42_time:<12.2f} {improvement:+6.1f}%")
    
    # Show v4.3 specific improvements
    print(f"\n{'='*60}")
    print("V7P3R v4.3 SIMPLIFICATION RESULTS")
    print(f"{'='*60}")
    
    if "V7P3R v4.3 (Simplified)" in all_results:
        v43_results = all_results["V7P3R v4.3 (Simplified)"]
        avg_time = sum(r['time'] for r in v43_results if r['success']) / len([r for r in v43_results if r['success']])
        success_rate = len([r for r in v43_results if r['success']]) / len(v43_results) * 100
        
        print(f"Average response time: {avg_time:.2f} seconds")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Simplifications implemented:")
        print(f"  ✅ Removed hanging piece detection from move ordering")
        print(f"  ✅ Removed mate-in-1 scanning from search")
        print(f"  ✅ Added transposition table with FEN-based hashing")
        print(f"  ✅ Streamlined move ordering with beta cutoffs")
        print(f"  ✅ Improved time management (50-node checks)")
        
        # Show unique moves (might be different due to simplifications)
        print(f"\nMoves found by v4.3:")
        for result in v43_results:
            if result['success']:
                print(f"  {result['scenario']}: {result['move']}")

if __name__ == "__main__":
    try:
        compare_engines()
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
