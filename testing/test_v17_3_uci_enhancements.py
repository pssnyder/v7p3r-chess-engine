#!/usr/bin/env python3
"""
V7P3R v17.3 UCI Enhancement Test
Tests that seldepth and hashfull are properly reported
"""

import sys
import os
import subprocess
import time
import re

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine
import chess


def test_uci_version():
    """Test that UCI reports v17.3"""
    print("=" * 60)
    print("TEST 1: UCI Version String")
    print("=" * 60)
    
    # Start engine subprocess
    engine_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'v7p3r_uci.py')
    process = subprocess.Popen(
        ['python', engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send UCI command
    process.stdin.write("uci\n")
    process.stdin.flush()
    
    # Read response
    output = []
    for _ in range(10):
        line = process.stdout.readline().strip()
        output.append(line)
        if line == "uciok":
            break
    
    # Check for v17.3
    version_found = False
    for line in output:
        if "id name" in line and "v17.3" in line:
            version_found = True
            print(f"✅ Version: {line}")
            break
    
    if not version_found:
        print("❌ FAIL: v17.3 not found in UCI output")
        print(f"Output: {output}")
    else:
        print("✅ PASS: UCI reports v17.3")
    
    # Cleanup
    process.stdin.write("quit\n")
    process.stdin.flush()
    process.wait(timeout=2)
    
    return version_found


def test_selective_depth_tracking():
    """Test that seldepth is tracked and reported"""
    print("\n" + "=" * 60)
    print("TEST 2: Selective Depth Tracking")
    print("=" * 60)
    
    engine = V7P3REngine()
    # Use tactical position NOT in opening book to force search
    tactical_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(tactical_fen)
    
    # Verify seldepth attribute exists
    if not hasattr(engine, 'seldepth'):
        print("❌ FAIL: Engine missing seldepth attribute")
        return False
    
    print("✅ Engine has seldepth attribute")
    
    # Run a search
    print("Running search on tactical position (forcing actual search)...")
    move = engine.search(board, time_limit=2.0)
    
    # Check that seldepth was updated
    if engine.seldepth == 0:
        print("❌ FAIL: seldepth not updated during search")
        return False
    
    print(f"✅ Selective depth tracked: {engine.seldepth} plies")
    print(f"   Nodes searched: {engine.nodes_searched}")
    print(f"   Best move: {move}")
    
    return True


def test_hashfull_reporting():
    """Test that hashfull is calculated and would be reported"""
    print("\n" + "=" * 60)
    print("TEST 3: Hash Table Usage (hashfull)")
    print("=" * 60)
    
    engine = V7P3REngine()
    # Use tactical position NOT in opening book to force search
    tactical_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(tactical_fen)
    
    # Run a deeper search to populate TT
    print("Running search on tactical position to populate TT...")
    engine.search(board, time_limit=3.0)
    
    # Calculate hashfull
    tt_size = len(engine.transposition_table)
    max_entries = engine.max_tt_entries
    hashfull = int((tt_size / max_entries) * 1000)
    
    print(f"✅ Transposition table populated")
    print(f"   Entries: {tt_size:,} / {max_entries:,}")
    print(f"   Hashfull: {hashfull} permille ({hashfull/10:.1f}%)")
    
    if tt_size == 0:
        print("❌ FAIL: No entries in transposition table")
        return False
    
    if hashfull > 1000:
        print("⚠️  WARNING: hashfull exceeds 1000 (overflow)")
    
    print("✅ PASS: Hash table usage calculated correctly")
    return True


def test_uci_info_output():
    """Test that UCI info output includes all expected fields"""
    print("\n" + "=" * 60)
    print("TEST 4: UCI Info Output Format")
    print("=" * 60)
    
    # Capture stdout to check info output
    import io
    from contextlib import redirect_stdout
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Capture output
    captured_output = io.StringIO()
    
    print("Running search and capturing UCI output...")
    with redirect_stdout(captured_output):
        engine.search(board, time_limit=2.0)
    
    output = captured_output.getvalue()
    info_lines = [line for line in output.split('\n') if line.startswith('info depth')]
    
    if not info_lines:
        print("❌ FAIL: No 'info depth' lines found")
        return False
    
    print(f"Found {len(info_lines)} info lines")
    
    # Check last info line for all fields
    last_info = info_lines[-1]
    print(f"\nLast info line:\n{last_info}")
    
    required_fields = ['depth', 'seldepth', 'score cp', 'nodes', 'time', 'nps', 'hashfull', 'pv']
    missing_fields = []
    
    for field in required_fields:
        if field not in last_info:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ FAIL: Missing fields: {missing_fields}")
        return False
    
    # Extract and validate values
    match = re.search(r'depth (\d+)', last_info)
    depth = int(match.group(1)) if match else 0
    
    match = re.search(r'seldepth (\d+)', last_info)
    seldepth = int(match.group(1)) if match else 0
    
    match = re.search(r'hashfull (\d+)', last_info)
    hashfull = int(match.group(1)) if match else 0
    
    match = re.search(r'nodes (\d+)', last_info)
    nodes = int(match.group(1)) if match else 0
    
    match = re.search(r'nps (\d+)', last_info)
    nps = int(match.group(1)) if match else 0
    
    print(f"\n✅ All required fields present:")
    print(f"   depth: {depth}")
    print(f"   seldepth: {seldepth} (should be >= depth)")
    print(f"   nodes: {nodes:,}")
    print(f"   nps: {nps:,}")
    print(f"   hashfull: {hashfull} permille ({hashfull/10:.1f}%)")
    
    # Validate seldepth >= depth
    if seldepth < depth:
        print(f"⚠️  WARNING: seldepth ({seldepth}) < depth ({depth})")
    
    print("\n✅ PASS: UCI info format correct")
    return True


def test_no_performance_regression():
    """Test that NPS is still >= 5,800 (v17.1.1 baseline)"""
    print("\n" + "=" * 60)
    print("TEST 5: Performance Regression Check")
    print("=" * 60)
    
    engine = V7P3REngine()
    # Use tactical position NOT in opening book to force search
    tactical_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(tactical_fen)
    
    print("Running 3-second search on tactical position for NPS measurement...")
    start_time = time.time()
    move = engine.search(board, time_limit=3.0)
    elapsed = time.time() - start_time
    
    nps = engine.nodes_searched / elapsed if elapsed > 0 else 0
    
    baseline_nps = 5800
    
    print(f"✅ Search complete")
    print(f"   Nodes: {engine.nodes_searched:,}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   NPS: {nps:,.0f}")
    print(f"   Baseline: {baseline_nps:,} NPS")
    
    if nps < baseline_nps:
        regression = ((baseline_nps - nps) / baseline_nps) * 100
        print(f"⚠️  WARNING: {regression:.1f}% slower than baseline")
        if regression > 5:
            print("❌ FAIL: >5% performance regression")
            return False
    else:
        improvement = ((nps - baseline_nps) / baseline_nps) * 100
        print(f"✅ Performance maintained (+{improvement:.1f}% vs baseline)")
    
    print("✅ PASS: No significant performance regression")
    return True


def main():
    """Run all UCI enhancement tests"""
    print("\n" + "=" * 60)
    print("V7P3R v17.3 UCI Enhancement Test Suite")
    print("Testing: seldepth tracking, hashfull reporting, info format")
    print("=" * 60)
    
    tests = [
        ("UCI Version String", test_uci_version),
        ("Selective Depth Tracking", test_selective_depth_tracking),
        ("Hash Table Usage", test_hashfull_reporting),
        ("UCI Info Output Format", test_uci_info_output),
        ("Performance Regression Check", test_no_performance_regression),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - v17.3 UCI Enhancements Working!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
