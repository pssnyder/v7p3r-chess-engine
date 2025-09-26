#!/usr/bin/env python3
"""
V7P3R v12.2 Final Performance Comparison
========================================
Direct comparison of v12.2 vs v12.0 vs v10.8 using command-line UCI
"""

import subprocess
import time
import os

def test_engine_performance(engine_path, time_limit_ms=3000):
    """Test engine performance using direct UCI commands"""
    engine_name = os.path.basename(engine_path).replace('.exe', '')
    print(f"\n🔧 Testing {engine_name}...")
    
    try:
        # Prepare UCI commands
        commands = f"""uci
position startpos
go movetime {time_limit_ms}
quit
"""
        
        # Run engine with timeout
        result = subprocess.run(
            engine_path,
            input=commands,
            text=True,
            capture_output=True,
            timeout=time_limit_ms/1000 + 5,  # Add 5 seconds buffer
            cwd=os.path.dirname(engine_path)
        )
        
        # Parse output
        lines = result.stdout.split('\n')
        
        depth = 0
        nodes = 0
        nps = 0
        time_ms = 0
        best_move = "none"
        
        for line in lines:
            if 'id name' in line:
                print(f"  Engine: {line.split('id name')[1].strip()}")
            elif line.startswith('info'):
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'depth' and i+1 < len(parts):
                        try: depth = max(depth, int(parts[i+1]))
                        except: pass
                    elif part == 'nodes' and i+1 < len(parts):
                        try: nodes = int(parts[i+1])
                        except: pass
                    elif part == 'nps' and i+1 < len(parts):
                        try: nps = int(parts[i+1])
                        except: pass
                    elif part == 'time' and i+1 < len(parts):
                        try: time_ms = int(parts[i+1])
                        except: pass
            elif line.startswith('bestmove'):
                parts = line.split()
                if len(parts) > 1:
                    best_move = parts[1]
        
        print(f"  ✅ Success: Depth {depth}, {nodes:,} nodes, {nps:,} NPS, {time_ms}ms")
        print(f"  Best move: {best_move}")
        
        return {
            'success': True,
            'name': engine_name,
            'depth': depth,
            'nodes': nodes,
            'nps': nps,
            'time_ms': time_ms,
            'best_move': best_move
        }
        
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timeout after {time_limit_ms}ms")
        return {'success': False, 'name': engine_name, 'error': 'timeout'}
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {'success': False, 'name': engine_name, 'error': str(e)}

def print_comparison_table(results):
    """Print comparison table"""
    print(f"\n{'='*80}")
    print("V7P3R VERSION PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Version':<15} | {'Depth':<5} | {'Nodes':<10} | {'NPS':<8} | {'Time(ms)':<8} | {'Move':<8}")
    print("-" * 80)
    
    for result in results:
        if result['success']:
            print(f"{result['name']:<15} | {result['depth']:<5} | {result['nodes']:<10,} | {result['nps']:<8,} | {result['time_ms']:<8} | {result['best_move']:<8}")
        else:
            print(f"{result['name']:<15} | {'FAIL':<5} | {'FAIL':<10} | {'FAIL':<8} | {'FAIL':<8} | {'FAIL':<8}")

def main():
    print("🚀 V7P3R v12.2 Final Performance Comparison")
    print("🎯 Direct UCI command comparison test")
    
    # Engine paths
    engine_dir = r"s:\Maker Stuff\Programming\Chess Engines\Chess Engine Playground\engine-tester\engines\V7P3R"
    engines = [
        os.path.join(engine_dir, "V7P3R_v12.2.exe"),
        os.path.join(engine_dir, "V7P3R_v12.0.exe"), 
        os.path.join(engine_dir, "V7P3R_v10.8.exe")
    ]
    
    # Test each engine
    results = []
    for engine_path in engines:
        if os.path.exists(engine_path):
            result = test_engine_performance(engine_path, 3000)  # 3 seconds
            results.append(result)
        else:
            print(f"❌ Engine not found: {engine_path}")
    
    # Print comparison
    print_comparison_table(results)
    
    # Performance analysis
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['success']]
    if len(successful_results) >= 2:
        # Find v12.2 and v12.0 results
        v12_2 = next((r for r in successful_results if 'v12.2' in r['name']), None)
        v12_0 = next((r for r in successful_results if 'v12.0' in r['name']), None)
        v10_8 = next((r for r in successful_results if 'v10.8' in r['name']), None)
        
        if v12_2:
            print(f"\n🎯 V12.2 Performance:")
            print(f"  Depth: {v12_2['depth']}")
            print(f"  NPS: {v12_2['nps']:,}")
            print(f"  Nodes: {v12_2['nodes']:,}")
            
            if v12_0:
                nps_improvement = v12_2['nps'] / v12_0['nps'] if v12_0['nps'] > 0 else 1
                depth_comparison = v12_2['depth'] - v12_0['depth']
                print(f"\n📊 V12.2 vs V12.0:")
                print(f"  NPS Improvement: {nps_improvement:.1f}x")
                print(f"  Depth Difference: {depth_comparison:+d}")
            
            if v10_8:
                nps_vs_baseline = v12_2['nps'] / v10_8['nps'] if v10_8['nps'] > 0 else 1
                depth_vs_baseline = v12_2['depth'] - v10_8['depth']
                print(f"\n📊 V12.2 vs V10.8 (Tournament Champion):")
                print(f"  NPS Comparison: {nps_vs_baseline:.1f}x")
                print(f"  Depth Difference: {depth_vs_baseline:+d}")
        
        # Tournament readiness
        print(f"\n🏆 TOURNAMENT READINESS:")
        if v12_2 and v12_2['nps'] > 3000 and v12_2['depth'] >= 4:
            print("  ✅ V12.2 is TOURNAMENT READY")
            print("  ✅ Sufficient NPS for time controls")
            print("  ✅ Good depth achievement")
            print("  🎯 Ready for competitive testing vs v10.8")
        else:
            print("  ⚠️  V12.2 may need more optimization")

if __name__ == "__main__":
    main()