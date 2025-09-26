#!/usr/bin/env python3
"""
V7P3R v12.2 Feature Toggle Test
Verify nudge system is disabled and measure performance impact
"""

import sys
import time
import chess
import os

sys.path.append('src')
from v7p3r import V7P3REngine

def test_nudge_system_disabled():
    """Test that nudge system is properly disabled"""
    
    print("V7P3R v12.2 Feature Toggle Test")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Check feature toggles
    print("Feature Toggle Status:")
    print(f"  ENABLE_NUDGE_SYSTEM: {engine.ENABLE_NUDGE_SYSTEM}")
    print(f"  ENABLE_PV_FOLLOWING: {engine.ENABLE_PV_FOLLOWING}")  
    print(f"  ENABLE_ADVANCED_EVALUATION: {engine.ENABLE_ADVANCED_EVALUATION}")
    
    # Verify nudge system is disabled
    if not engine.ENABLE_NUDGE_SYSTEM:
        print("\n✅ Nudge system successfully DISABLED")
        print("  - No database loading overhead")
        print("  - No position matching during search")
        print("  - Smaller memory footprint")
    else:
        print("\n❌ Nudge system still enabled - check implementation")
        return False
    
    # Test nudge methods return expected values when disabled
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    test_move = legal_moves[0]
    
    nudge_bonus = engine._get_nudge_bonus(board, test_move)
    instant_nudge = engine._check_instant_nudge_move(board)
    
    if nudge_bonus == 0.0 and instant_nudge is None:
        print("✅ Nudge methods return expected disabled values")
    else:
        print(f"❌ Nudge methods not properly disabled: bonus={nudge_bonus}, instant={instant_nudge}")
        return False
    
    return True

def benchmark_performance():
    """Benchmark search performance with nudge system disabled"""
    
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("Testing 5-second search on starting position...")
    
    start_time = time.perf_counter()
    move = engine.search(board, time_limit=5.0)
    elapsed = time.perf_counter() - start_time
    
    nps = int(engine.nodes_searched / max(elapsed, 0.001))
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Nodes: {engine.nodes_searched:,}")
    print(f"  NPS: {nps:,}")
    print(f"  Best move: {move}")
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if nps > 50000:
        print("✅ EXCELLENT performance - tournament ready")
    elif nps > 20000:
        print("✅ GOOD performance - competitive")  
    elif nps > 10000:
        print("⚠️  MODERATE performance - may need further optimization")
    elif nps > 5000:
        print("⚠️  POOR performance - significant optimization needed")
    else:
        print("🚨 CRITICAL performance - major issues remain")
    
    return nps

def test_startup_speed():
    """Test engine initialization speed without nudge database"""
    
    print("\n" + "=" * 50)
    print("STARTUP SPEED TEST")
    print("=" * 50)
    
    print("Testing engine initialization speed...")
    
    start_time = time.perf_counter()
    engine = V7P3REngine()
    elapsed = time.perf_counter() - start_time
    
    print(f"Engine initialization: {elapsed:.3f}s")
    
    if elapsed < 0.1:
        print("✅ FAST startup - good for rapid games")
    elif elapsed < 0.5:
        print("✅ GOOD startup speed")
    elif elapsed < 1.0:
        print("⚠️  SLOW startup - may affect quick games")
    else:
        print("🚨 VERY SLOW startup - optimization needed")
    
    return elapsed

def main():
    """Run complete feature toggle verification"""
    
    # Test feature toggles
    toggles_ok = test_nudge_system_disabled()
    if not toggles_ok:
        print("\n❌ Feature toggle test FAILED")
        return False
    
    # Test startup speed  
    startup_time = test_startup_speed()
    
    # Benchmark performance
    nps = benchmark_performance()
    
    # Summary
    print("\n" + "=" * 50)
    print("V12.2 TEST SUMMARY")
    print("=" * 50)
    
    print("✅ Nudge system successfully disabled")
    print(f"✅ Startup time: {startup_time:.3f}s")
    print(f"✅ Search performance: {nps:,} NPS")
    
    improvements = []
    if startup_time < 0.2:
        improvements.append("Fast startup")
    if nps > 20000:
        improvements.append("Good search speed")
    
    if improvements:
        print(f"\n🎯 Performance improvements: {', '.join(improvements)}")
    
    next_steps = []
    if nps < 20000:
        next_steps.append("Optimize evaluation functions")
    if startup_time > 0.5:
        next_steps.append("Reduce initialization overhead")
    
    if next_steps:
        print(f"\n📋 Next optimization targets: {', '.join(next_steps)}")
    else:
        print(f"\n🎉 V12.2 performance targets achieved!")
    
    return True

if __name__ == "__main__":
    main()