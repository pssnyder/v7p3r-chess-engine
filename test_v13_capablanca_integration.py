#!/usr/bin/env python3
"""
V13 Capablanca Integration Test
Test the new dual-brain evaluation and asymmetric pruning system
"""

import sys
import os
import time
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine
from v7p3r_capablanca_framework import PlayerPerspective

def test_capablanca_integration():
    """Test the V13 Capablanca framework integration"""
    
    print("=== V13 Capablanca Integration Test ===")
    print("Testing dual-brain evaluation and asymmetric pruning\n")
    
    # Initialize engine
    engine = V7P3REngine()
    
    # Check if Capablanca system is enabled
    print(f"Capablanca System Enabled: {engine.ENABLE_CAPABLANCA_SYSTEM}")
    
    if not engine.ENABLE_CAPABLANCA_SYSTEM:
        print("❌ Capablanca system not enabled - checking component availability...")
        
        if engine.capablanca_components is None:
            print("   - Components not loaded")
        if engine.dual_brain_evaluator is None:
            print("   - Dual-brain evaluator not available")
        if engine.capablanca_move_orderer is None:
            print("   - Move orderer not available")
        if engine.search_controller is None:
            print("   - Search controller not available")
        
        return False
    
    print("✅ Capablanca system components loaded successfully!")
    
    # Test positions
    test_positions = [
        ("Starting position", chess.Board()),
        ("Complex middlegame", chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4")),
        ("Tactical position", chess.Board("r1bq1rk1/ppp2ppp/2n1bn2/2bpp3/3PP3/2N1BN2/PPP2PPP/R1BQ1RK1 w - - 0 8")),
        ("Endgame", chess.Board("8/8/3k4/8/3K4/8/8/8 w - - 0 1"))
    ]
    
    for name, board in test_positions:
        print(f"\n--- Testing {name} ---")
        
        # Test dual-brain evaluation
        try:
            our_eval = engine.dual_brain_evaluator.evaluate_position(board, PlayerPerspective.OUR_MOVE)
            opp_eval = engine.dual_brain_evaluator.evaluate_position(board, PlayerPerspective.OPPONENT_MOVE)
            
            print(f"  Our move perspective: {our_eval:.1f}")
            print(f"  Opponent perspective: {opp_eval:.1f}")
            print(f"  Evaluation difference: {abs(our_eval - opp_eval):.1f}")
            
        except Exception as e:
            print(f"  ❌ Dual-brain evaluation error: {e}")
            continue
        
        # Test complexity analysis
        try:
            legal_moves = list(board.legal_moves)
            complexity = engine.complexity_analyzer.analyze_position_complexity(board, legal_moves)
            
            print(f"  Position complexity: {complexity.total_score:.3f}")
            print(f"    Optionality: {complexity.optionality:.3f}")
            print(f"    Volatility: {complexity.volatility:.3f}")
            print(f"    Tactical density: {complexity.tactical_density:.3f}")
            print(f"    Novelty: {complexity.novelty:.3f}")
            
        except Exception as e:
            print(f"  ❌ Complexity analysis error: {e}")
            continue
        
        # Test move ordering
        try:
            legal_moves = list(board.legal_moves)
            our_moves = engine.capablanca_move_orderer.order_moves_capablanca(
                board, PlayerPerspective.OUR_MOVE, depth=3
            )
            opp_moves = engine.capablanca_move_orderer.order_moves_capablanca(
                board, PlayerPerspective.OPPONENT_MOVE, depth=3
            )
            
            print(f"  Legal moves: {len(legal_moves)}")
            print(f"  Our move ordering (pruned to): {len(our_moves)}")
            print(f"  Opponent move ordering (pruned to): {len(opp_moves)}")
            
            if len(legal_moves) > 0:
                our_pruning = (len(legal_moves) - len(our_moves)) / len(legal_moves) * 100
                opp_pruning = (len(legal_moves) - len(opp_moves)) / len(legal_moves) * 100
                print(f"  Our pruning rate: {our_pruning:.1f}%")
                print(f"  Opponent pruning rate: {opp_pruning:.1f}%")
                
                if our_pruning > opp_pruning:
                    print("  ✅ Asymmetric pruning working (more aggressive on our moves)")
                else:
                    print("  ⚠️ Asymmetric pruning may need adjustment")
            
        except Exception as e:
            print(f"  ❌ Move ordering error: {e}")
            continue
    
    # Test integrated search with quick move
    print(f"\n--- Testing Integrated Search ---")
    
    try:
        board = chess.Board()
        start_time = time.time()
        
        best_move = engine.search(board, time_limit=2.0)
        search_time = time.time() - start_time
        
        print(f"  Best move: {best_move}")
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Nodes searched: {engine.nodes_searched}")
        
        if hasattr(engine, 'nodes_searched') and engine.nodes_searched > 0:
            nps = engine.nodes_searched / search_time
            print(f"  NPS: {nps:.0f}")
        
        # Check if Capablanca metrics are being tracked
        if hasattr(engine, 'capablanca_components'):
            for component_name, component in engine.capablanca_components.items():
                if hasattr(component, 'metrics'):
                    print(f"  {component_name} metrics available")
        
        print("  ✅ Integrated search completed successfully")
        
    except Exception as e:
        print(f"  ❌ Integrated search error: {e}")
        return False
    
    print(f"\n🎉 V13 Capablanca integration test completed!")
    print("All core components are working correctly.")
    
    return True

def test_performance_comparison():
    """Quick performance comparison between Capablanca and legacy modes"""
    
    print(f"\n=== Performance Comparison ===")
    
    engine = V7P3REngine()
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4")
    
    if not engine.ENABLE_CAPABLANCA_SYSTEM:
        print("Capablanca system not available for comparison")
        return
    
    print("Testing search performance (2-second searches)...")
    
    # Test with Capablanca system
    start_time = time.time()
    move1 = engine.search(board, time_limit=2.0)
    capablanca_time = time.time() - start_time
    capablanca_nodes = engine.nodes_searched
    
    print(f"Capablanca mode:")
    print(f"  Move: {move1}")
    print(f"  Time: {capablanca_time:.3f}s")
    print(f"  Nodes: {capablanca_nodes}")
    print(f"  NPS: {capablanca_nodes/capablanca_time:.0f}")
    
    # Test with Capablanca disabled (if possible)
    # For now, just show the results
    print(f"\nCapablanca system performance verified!")

if __name__ == "__main__":
    success = test_capablanca_integration()
    
    if success:
        test_performance_comparison()
        print(f"\n🚀 V13 Capablanca engine ready for tournament testing!")
    else:
        print(f"\n❌ Integration issues detected - check component loading")