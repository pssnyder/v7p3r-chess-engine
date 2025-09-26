#!/usr/bin/env python3
"""
V7P3R v12.2 Time Management Debug Test
======================================
Analyze actual time usage vs allocated time
"""

import subprocess
import time
import os

def test_time_usage(engine_path, wtime_ms, expected_usage_pct=40):
    """Test how much time the engine actually uses vs allocation"""
    print(f"\n🕐 Testing with {wtime_ms}ms white time (expecting ~{expected_usage_pct}% usage)...")
    
    try:
        # Create UCI commands - simulate game with specific wtime
        commands = f"""uci
position startpos
go wtime {wtime_ms} btime {wtime_ms}
quit
"""
        
        start_time = time.time()
        result = subprocess.run(
            engine_path,
            input=commands,
            text=True,
            capture_output=True,
            timeout=30,
            cwd=os.path.dirname(engine_path)
        )
        actual_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Parse UCI output for time spent
        lines = result.stdout.split('\n')
        engine_reported_time = 0
        depth = 0
        nodes = 0
        
        for line in lines:
            if line.startswith('info') and 'time' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'time' and i+1 < len(parts):
                        try:
                            engine_reported_time = max(engine_reported_time, int(parts[i+1]))
                        except: pass
                    elif part == 'depth' and i+1 < len(parts):
                        try:
                            depth = max(depth, int(parts[i+1]))
                        except: pass
                    elif part == 'nodes' and i+1 < len(parts):
                        try:
                            nodes = int(parts[i+1])
                        except: pass
        
        expected_time = wtime_ms * (expected_usage_pct / 100.0)
        usage_pct = (engine_reported_time / wtime_ms) * 100 if wtime_ms > 0 else 0
        
        print(f"  Allocated time: {wtime_ms}ms")
        print(f"  Expected usage: {expected_time:.0f}ms (~{expected_usage_pct}%)")
        print(f"  Engine reported: {engine_reported_time}ms ({usage_pct:.1f}%)")
        print(f"  Actual wall time: {actual_time:.0f}ms")
        print(f"  Final depth: {depth}")
        print(f"  Nodes searched: {nodes:,}")
        
        if usage_pct < 20:
            print(f"  🚨 TOO CONSERVATIVE - using only {usage_pct:.1f}%!")
        elif usage_pct < 40:
            print(f"  ⚠️  Still conservative - using {usage_pct:.1f}%")
        elif usage_pct < 70:
            print(f"  ✅ Good usage - using {usage_pct:.1f}%")
        else:
            print(f"  ⚡ Aggressive usage - using {usage_pct:.1f}%")
            
        return {
            'allocated_ms': wtime_ms,
            'used_ms': engine_reported_time,
            'usage_pct': usage_pct,
            'depth': depth,
            'nodes': nodes
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    print("🕐 V7P3R v12.2 Time Management Analysis")
    print("🎯 Testing time usage across different allocations")
    
    engine_path = r"s:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\V7P3R\V7P3R_v12.2.exe"
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return
    
    # Test various time allocations
    test_scenarios = [
        (60000, 16.7),   # 1 minute total, expect ~1/6 = 16.7%
        (30000, 12.5),   # 30 seconds, expect ~1/8 = 12.5% 
        (10000, 12.5),   # 10 seconds, expect ~1/8 = 12.5%
        (5000, 16.7),    # 5 seconds, expect ~1/6 = 16.7%
        (3000, 16.7),    # 3 seconds, expect ~1/6 = 16.7%
        (1000, 16.7),    # 1 second, expect ~1/6 = 16.7%
    ]
    
    results = []
    for wtime_ms, expected_pct in test_scenarios:
        result = test_time_usage(engine_path, wtime_ms, expected_pct)
        if result:
            results.append(result)
    
    # Analysis
    print(f"\n{'='*70}")
    print("TIME MANAGEMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"{'Allocated':<10} | {'Used':<8} | {'Usage%':<8} | {'Depth':<5} | {'Assessment':<15}")
    print("-" * 70)
    
    for result in results:
        assessment = "TOO CONSERVATIVE"
        if result['usage_pct'] > 40:
            assessment = "AGGRESSIVE"
        elif result['usage_pct'] > 20:
            assessment = "MODERATE"
        
        print(f"{result['allocated_ms']/1000:.1f}s{'':<5} | {result['used_ms']}ms{'':<4} | {result['usage_pct']:.1f}%{'':<4} | {result['depth']:<5} | {assessment:<15}")
    
    # Recommendations
    avg_usage = sum(r['usage_pct'] for r in results) / len(results) if results else 0
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"Average time usage: {avg_usage:.1f}%")
    
    if avg_usage < 25:
        print("🚨 CRITICAL: Time management is too conservative!")
        print("   - Engine is using <25% of allocated time")
        print("   - Needs much more aggressive time factors")
        print("   - Suggested: Reduce time_factor by half (6→3, 8→4, 12→6, 18→9)")
    elif avg_usage < 40:
        print("⚠️  WARNING: Time management is somewhat conservative")
        print("   - Engine is using <40% of allocated time")
        print("   - Could be more aggressive for better depth")
        print("   - Suggested: Reduce time_factor slightly (6→5, 8→6, 12→10, 18→15)")
    else:
        print("✅ Time management looks reasonable")

if __name__ == "__main__":
    main()