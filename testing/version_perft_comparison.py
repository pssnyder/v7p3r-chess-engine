#!/usr/bin/env python3
"""
V7P3R Version Comparison Perft Test
Tests multiple engine versions to compare performance and identify regressions
"""

import subprocess
import time
import os
from typing import Dict, List, Tuple

class V7P3RVersionComparer:
    def __init__(self, engine_dir: str):
        self.engine_dir = engine_dir
        self.engines = self._find_engines()
        
    def _find_engines(self) -> List[str]:
        """Find all V7P3R engines in directory"""
        engines = []
        if os.path.exists(self.engine_dir):
            for file in os.listdir(self.engine_dir):
                if file.startswith("V7P3R_") and file.endswith(".exe"):
                    engines.append(file)
        engines.sort()  # Sort by version
        return engines
    
    def test_engine_perft(self, engine_name: str, depth: int = 3) -> Dict:
        """Test single engine perft performance"""
        engine_path = os.path.join(self.engine_dir, engine_name)
        
        print(f"\n🔍 Testing {engine_name}")
        
        # Standard position for consistent testing
        commands = [
            "uci",
            "position startpos",
            f"go perft {depth}",
            "quit"
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(
                [engine_path],
                input="\n".join(commands),
                text=True,
                capture_output=True,
                timeout=20  # 20 second timeout
            )
            end_time = time.time()
            
            if result.returncode == 0:
                # Parse output for nodes
                lines = result.stdout.split('\n')
                nodes = None
                for line in lines:
                    if 'Nodes searched:' in line:
                        try:
                            nodes = int(line.split(':')[1].strip())
                        except:
                            pass
                    elif line.strip().isdigit():  # Some engines just output the number
                        nodes = int(line.strip())
                
                time_taken = end_time - start_time
                nps = nodes / time_taken if nodes and time_taken > 0 else 0
                
                return {
                    'engine': engine_name,
                    'nodes': nodes,
                    'time': time_taken,
                    'nps': nps,
                    'success': True,
                    'error': None
                }
            else:
                return {
                    'engine': engine_name,
                    'success': False,
                    'error': result.stderr or "Engine failed",
                    'time': end_time - start_time
                }
                
        except subprocess.TimeoutExpired:
            return {
                'engine': engine_name,
                'success': False,
                'error': "Timeout after 20 seconds",
                'time': 20.0
            }
        except Exception as e:
            return {
                'engine': engine_name,
                'success': False,
                'error': str(e),
                'time': 0
            }
    
    def compare_versions(self, depth: int = 3) -> None:
        """Compare performance across all engine versions"""
        print("V7P3R Engine Version Performance Comparison")
        print("=" * 60)
        print(f"Perft Depth: {depth}")
        print(f"Engines Found: {len(self.engines)}")
        
        results = []
        
        # Test key versions for comparison
        key_versions = [
            "V7P3R_v12.0.exe",  # Before v12.5 changes
            "V7P3R_v12.4.exe",  # Last stable before nudges
            "V7P3R_v12.5.exe",  # Current version with issues
        ]
        
        # Test key versions first
        for engine in key_versions:
            if engine in self.engines:
                result = self.test_engine_perft(engine, depth)
                results.append(result)
        
        # Print results
        print(f"\n{'Engine':<20} {'Nodes':<12} {'Time(s)':<8} {'NPS':<12} {'Status'}")
        print("-" * 60)
        
        baseline_nps = None
        for result in results:
            if result['success']:
                nps_str = f"{result['nps']:,.0f}"
                nodes_str = f"{result['nodes']:,}"
                time_str = f"{result['time']:.2f}"
                status = "✅ OK"
                
                # Set baseline from v12.0
                if baseline_nps is None and "v12.0" in result['engine']:
                    baseline_nps = result['nps']
                
                # Compare to baseline
                if baseline_nps and result['nps'] > 0:
                    ratio = result['nps'] / baseline_nps
                    if ratio < 0.8:
                        status = f"🐌 {ratio:.1f}x SLOWER"
                    elif ratio > 1.2:
                        status = f"🚀 {ratio:.1f}x FASTER"
                
            else:
                nodes_str = "N/A"
                time_str = f"{result['time']:.2f}"
                nps_str = "N/A"
                status = f"❌ {result['error']}"
            
            print(f"{result['engine']:<20} {nodes_str:<12} {time_str:<8} {nps_str:<12} {status}")
        
        # Performance analysis
        self._analyze_performance(results)
    
    def _analyze_performance(self, results: List[Dict]) -> None:
        """Analyze performance trends"""
        print("\n📊 PERFORMANCE ANALYSIS")
        print("=" * 40)
        
        successful_results = [r for r in results if r['success']]
        if len(successful_results) < 2:
            print("❌ Insufficient data for comparison")
            return
        
        # Find fastest and slowest
        fastest = max(successful_results, key=lambda x: x['nps'])
        slowest = min(successful_results, key=lambda x: x['nps'])
        
        print(f"🏆 Fastest: {fastest['engine']} - {fastest['nps']:,.0f} NPS")
        print(f"🐌 Slowest: {slowest['engine']} - {slowest['nps']:,.0f} NPS")
        
        if fastest['nps'] > 0 and slowest['nps'] > 0:
            ratio = fastest['nps'] / slowest['nps']
            print(f"📈 Performance Range: {ratio:.1f}x difference")
        
        # Check v12.5 specifically
        v12_5_result = next((r for r in results if "v12.5" in r['engine']), None)
        if v12_5_result and v12_5_result['success']:
            v12_0_result = next((r for r in results if "v12.0" in r['engine']), None)
            if v12_0_result and v12_0_result['success']:
                ratio = v12_5_result['nps'] / v12_0_result['nps']
                if ratio < 0.9:
                    print(f"⚠️  v12.5 is {(1/ratio):.1f}x SLOWER than v12.0 - REGRESSION DETECTED!")
                elif ratio > 1.1:
                    print(f"✅ v12.5 is {ratio:.1f}x faster than v12.0")
                else:
                    print(f"✅ v12.5 performance similar to v12.0 ({ratio:.1f}x)")

if __name__ == "__main__":
    engine_dir = "s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-tester/engines/V7P3R"
    depth = 3  # Start with depth 3 for speed
    
    comparer = V7P3RVersionComparer(engine_dir)
    comparer.compare_versions(depth)