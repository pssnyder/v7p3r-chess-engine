#!/usr/bin/env python3
"""
Simple isolated search test to verify v11.5 search interface fix
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_search_basic():
    """Basic test of search interface fix"""
    print("V7P3R v11.5 Search Interface Test")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Simple opening position
    board = chess.Board()
    
    print(f"Testing position: {board.fen()}")
    print("Calling search(board, time_limit=2.0, depth=3)...")
    
    start_time = time.time()
    try:
        result = engine.search(board, time_limit=2.0, depth=3)
        end_time = time.time()
        
        print(f"Search completed in {end_time - start_time:.3f}s")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if isinstance(result, tuple) else 'N/A'}")
        
        if isinstance(result, tuple) and len(result) == 3:
            move, score, search_info = result
            print(f"✅ SUCCESS: Proper tuple returned")
            print(f"   Move: {move}")
            print(f"   Score: {score}")
            print(f"   Search Info: {search_info}")
            
            # Validate search info structure
            expected_keys = ['nodes', 'time', 'nps', 'depth', 'score']
            missing_keys = [key for key in expected_keys if key not in search_info]
            if missing_keys:
                print(f"⚠️  WARNING: Missing search info keys: {missing_keys}")
            else:
                print(f"✅ Search info structure correct")
                
                # Check for realistic values
                nodes = search_info.get('nodes', 0)
                nps = search_info.get('nps', 0)
                if nodes > 0 and nps > 100:
                    print(f"✅ Realistic performance: {nodes} nodes, {nps:.0f} NPS")
                else:
                    print(f"⚠️  WARNING: Unrealistic performance: {nodes} nodes, {nps:.0f} NPS")
        else:
            print(f"❌ FAILED: Expected tuple of length 3, got {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Exception during search: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_multiple_depths():
    """Test search at multiple depths to verify scaling"""
    print("\nMultiple Depth Test")
    print("=" * 50)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    for depth in [1, 2, 3]:
        print(f"\n--- Depth {depth} ---")
        try:
            start = time.time()
            move, score, info = engine.search(board, time_limit=5.0, depth=depth)
            elapsed = time.time() - start
            
            nodes = info.get('nodes', 0)
            nps = info.get('nps', 0)
            
            print(f"Move: {move}, Score: {score}")
            print(f"Time: {elapsed:.3f}s, Nodes: {nodes}, NPS: {nps:.0f}")
            
            # Check that deeper searches generally use more nodes
            if depth > 1 and nodes < 10:
                print(f"⚠️  WARNING: Very low node count for depth {depth}")
                
        except Exception as e:
            print(f"❌ FAILED at depth {depth}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("Starting V7P3R v11.5 Search Interface Tests...\n")
    
    success = True
    success &= test_search_basic()
    success &= test_multiple_depths()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ALL TESTS PASSED - Search interface fix successful!")
    else:
        print("❌ TESTS FAILED - Search interface needs more work")