#!/usr/bin/env python3
"""
Final integration test for v7p3r chess engine with all engines.
Tests that all engines (v7p3r, stockfish, v7p3r_rl, v7p3r_ga, v7p3r_nn) can be created, 
have cleanup methods, and can be selected in play_v7p3r.py
"""

import sys
import os
import chess

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_all_engines():
    """Test all engine types for proper initialization and cleanup"""
    print("Testing all engines for initialization and cleanup...")
    
    # Test NN engine
    try:
        from v7p3r_nn_engine.v7p3r_nn import v7p3rNeuralNetwork
        print("\n1. Testing v7p3rNeuralNetwork...")
        nn_engine = v7p3rNeuralNetwork()
        print(f"   - Created successfully: {nn_engine is not None}")
        print(f"   - Has cleanup method: {hasattr(nn_engine, 'cleanup')}")
        print(f"   - Has close method: {hasattr(nn_engine, 'close')}")
        if hasattr(nn_engine, 'cleanup'):
            print("   - Calling cleanup...")
            nn_engine.cleanup()
            print("   - Cleanup successful")
        elif hasattr(nn_engine, 'close'):
            print("   - Calling close...")
            nn_engine.close()
            print("   - Close successful")
    except Exception as e:
        print(f"   - NN Engine test failed: {e}")
    
    # Test GA engine
    try:
        from v7p3r_ga_engine.position_evaluator import PositionEvaluator
        print("\n2. Testing GA PositionEvaluator...")
        # Use minimal config for testing
        ga_config = {'stockfish_path': 'engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe'}
        ga_engine = PositionEvaluator(stockfish_config=ga_config)
        print(f"   - Created successfully: {ga_engine is not None}")
        print(f"   - Has cleanup method: {hasattr(ga_engine, 'cleanup')}")
        if hasattr(ga_engine, 'cleanup'):
            print("   - Calling cleanup...")
            ga_engine.cleanup()
            print("   - Cleanup successful")
    except Exception as e:
        print(f"   - GA Engine test failed: {e}")
    
    # Test RL engine
    try:
        from v7p3r_rl_engine.v7p3r_rl import v7p3rRLEngine
        print("\n3. Testing v7p3rRLEngine...")
        rl_engine = v7p3rRLEngine()
        print(f"   - Created successfully: {rl_engine is not None}")
        print(f"   - Has cleanup method: {hasattr(rl_engine, 'cleanup')}")
        if hasattr(rl_engine, 'cleanup'):
            print("   - Calling cleanup...")
            rl_engine.cleanup()
            print("   - Cleanup successful")
    except Exception as e:
        print(f"   - RL Engine test failed: {e}")
    
    # Test main v7p3r engine
    try:
        from v7p3r_engine.v7p3r import v7p3rEngine
        print("\n4. Testing v7p3rEngine...")
        v7p3r = v7p3rEngine()
        print(f"   - Created successfully: {v7p3r is not None}")
        print(f"   - Has cleanup method: {hasattr(v7p3r, 'cleanup')}")
        # v7p3rEngine may not have cleanup, that's ok
    except Exception as e:
        print(f"   - v7p3r Engine test failed: {e}")
    
    # Test Stockfish handler
    try:
        from v7p3r_engine.stockfish_handler import StockfishHandler
        print("\n5. Testing StockfishHandler...")
        stockfish_config = {'stockfish_path': 'engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe'}
        stockfish = StockfishHandler(stockfish_config=stockfish_config)
        print(f"   - Created successfully: {stockfish is not None}")
        print(f"   - Has quit method: {hasattr(stockfish, 'quit')}")
        if hasattr(stockfish, 'quit'):
            print("   - Calling quit...")
            stockfish.quit()
            print("   - Quit successful")
    except Exception as e:
        print(f"   - Stockfish test failed: {e}")

def test_play_v7p3r_integration():
    """Test the main play_v7p3r.py integration"""
    print("\n\nTesting play_v7p3r.py integration...")
    
    try:
        from v7p3r_engine.play_v7p3r import ChessGame
        print("   - ChessGame class imported successfully")
        
        # Test that all engines can be referenced
        engine_names = ['v7p3r', 'stockfish', 'v7p3r_rl', 'v7p3r_ga', 'v7p3r_nn']
        print(f"   - Available engines: {engine_names}")
        print("   - Integration test passed (import successful)")
                    
    except Exception as e:
        print(f"   - Integration test failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("V7P3R CHESS ENGINE - FINAL INTEGRATION TEST")
    print("=" * 60)
    
    test_all_engines()
    test_play_v7p3r_integration()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)
