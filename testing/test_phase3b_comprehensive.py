#!/usr/bin/env python3
"""
V7P3R Chess Engine - Phase 3B Comprehensive Validation Suite
Tests the complete adaptive evaluation and move ordering system
"""

import sys
import os
import time
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_engine_integration():
    """Test that the engine initializes and works with new adaptive systems"""
    
    print("V7P3R Phase 3B Integration Test")
    print("=" * 50)
    
    try:
        # Initialize engine
        engine = V7P3REngine()
        print("✓ Engine initialization successful")
        
        # Test positions
        test_positions = [
            {
                'name': 'Opening Position',
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                'target_depth': 3
            },
            {
                'name': 'Volatile Position (User Example)',
                'fen': 'r4rk1/ppp2ppp/8/1q5n/3p4/3P1P1P/PPPQ1P2/R4RK1 w - - 0 16',
                'target_depth': 3
            },
            {
                'name': 'Middlegame Tactical',
                'fen': 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5',
                'target_depth': 3
            }
        ]
        
        total_nodes = 0
        total_time = 0
        test_count = 0
        
        for i, test_case in enumerate(test_positions):
            print(f"\nTest {i+1}: {test_case['name']}")
            print(f"FEN: {test_case['fen']}")
            
            board = chess.Board(test_case['fen'])
            depth = test_case['target_depth']
            
            start_time = time.time()
            
            # Test search
            try:
                best_move = engine.search(board, depth=depth)
                end_time = time.time()
                search_time = end_time - start_time
                
                nodes = engine.nodes_searched
                nps = nodes / search_time if search_time > 0 else 0
                
                print(f"  Best move: {best_move}")
                print(f"  Nodes: {nodes:,}")
                print(f"  Time: {search_time:.3f}s")
                print(f"  NPS: {nps:,.0f}")
                
                # Test evaluation directly
                eval_score = engine._evaluate_position(board)
                print(f"  Position eval: {eval_score:.3f}")
                
                # Get adaptive evaluation stats
                adaptive_stats = engine.adaptive_evaluator.get_evaluation_stats()
                print(f"  Adaptive eval calls: {adaptive_stats['calls']}")
                print(f"  Posture breakdown: {adaptive_stats['posture_breakdown']}")
                
                total_nodes += nodes
                total_time += search_time
                test_count += 1
                
                print("  ✓ Test passed")
                
            except Exception as e:
                print(f"  ✗ Test failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Reset for next test
            engine.nodes_searched = 0
            engine.depth_reached = 0
            engine.evaluation_cache.clear()
        
        # Overall performance summary
        if test_count > 0:
            avg_nps = total_nodes / total_time if total_time > 0 else 0
            print(f"\n" + "=" * 50)
            print("OVERALL PERFORMANCE SUMMARY:")
            print(f"Tests completed: {test_count}/{len(test_positions)}")
            print(f"Total nodes: {total_nodes:,}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Average NPS: {avg_nps:,.0f}")
            
            # Performance target
            target_nps = 20000  # Conservative target for new system
            performance_pass = avg_nps >= target_nps
            print(f"Performance target: {target_nps:,} NPS")
            print(f"Performance result: {'PASS' if performance_pass else 'FAIL'}")
            
            return test_count == len(test_positions) and performance_pass
        
        return False
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_posture_driven_decisions():
    """Test that the engine makes different decisions based on position posture"""
    
    print(f"\n" + "=" * 50)
    print("POSTURE-DRIVEN DECISION TEST")
    print("=" * 50)
    
    try:
        engine = V7P3REngine()
        
        # Test the volatile position where defensive posture should be clear
        board = chess.Board('r4rk1/ppp2ppp/8/1q5n/3p4/3P1P1P/PPPQ1P2/R4RK1 w - - 0 16')
        
        # Get posture assessment
        volatility, posture = engine.posture_assessor.assess_position_posture(board)
        print(f"Position assessment: {volatility.value} volatility, {posture.value} posture")
        
        # Get adaptive move ordering
        legal_moves = list(board.legal_moves)
        ordered_moves = engine._order_moves_advanced(board, legal_moves, depth=0)
        
        print(f"Total legal moves: {len(legal_moves)}")
        print("Top 10 moves from adaptive ordering:")
        
        for i, move in enumerate(ordered_moves[:10]):
            # Get move classification
            classification = engine.adaptive_move_orderer.get_move_classification(board, move)
            print(f"  {i+1:2d}. {move} - {classification}")
        
        # Test that defensive moves are prioritized
        defensive_types = ['escape_moves', 'blocking_moves', 'defensive_captures', 'defensive_moves']
        defensive_count = 0
        
        for move in ordered_moves[:10]:
            move_type, _ = engine.adaptive_move_orderer._classify_move(board, move, posture, volatility)
            if move_type in defensive_types:
                defensive_count += 1
        
        defensive_ratio = defensive_count / 10
        print(f"\nDefensive moves in top 10: {defensive_count}/10 ({defensive_ratio:.1%})")
        
        # In emergency posture, should be mostly defensive
        success = defensive_ratio >= 0.7 if posture.value == 'emergency' else True
        print(f"Posture-appropriate decisions: {'PASS' if success else 'FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"Posture decision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare performance with and without adaptive systems"""
    
    print(f"\n" + "=" * 50)
    print("PERFORMANCE COMPARISON TEST")
    print("=" * 50)
    
    try:
        engine = V7P3REngine()
        board = chess.Board('rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3')
        
        # Test multiple evaluations for timing
        iterations = 100
        
        start_time = time.time()
        for _ in range(iterations):
            eval_score = engine._evaluate_position(board)
        end_time = time.time()
        
        avg_eval_time = ((end_time - start_time) / iterations) * 1000  # ms
        
        print(f"Adaptive evaluation average time: {avg_eval_time:.2f}ms")
        
        # Get adaptive evaluation statistics
        adaptive_stats = engine.adaptive_evaluator.get_evaluation_stats()
        print(f"Adaptive evaluation calls: {adaptive_stats['calls']}")
        print(f"Average total time: {adaptive_stats.get('avg_total_time', 0)*1000:.2f}ms")
        
        # Performance target: under 10ms per evaluation
        performance_target = 10.0  # ms
        performance_pass = avg_eval_time <= performance_target
        
        print(f"Performance target: {performance_target}ms")
        print(f"Performance result: {'PASS' if performance_pass else 'FAIL'}")
        
        return performance_pass
        
    except Exception as e:
        print(f"Performance comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting V7P3R Phase 3B Comprehensive Validation...")
    
    try:
        # Run all tests
        integration_pass = test_engine_integration()
        posture_pass = test_posture_driven_decisions()
        performance_pass = test_performance_comparison()
        
        print(f"\n" + "=" * 50)
        print("FINAL VALIDATION RESULTS:")
        print(f"Engine Integration: {'PASS' if integration_pass else 'FAIL'}")
        print(f"Posture-Driven Decisions: {'PASS' if posture_pass else 'FAIL'}")
        print(f"Performance: {'PASS' if performance_pass else 'FAIL'}")
        
        overall_pass = integration_pass and posture_pass and performance_pass
        print(f"Overall Phase 3B: {'PASS' if overall_pass else 'FAIL'}")
        
        if overall_pass:
            print("\n✓ Phase 3B adaptive evaluation system is ready for deployment!")
            print("  - Posture assessment working correctly")
            print("  - Adaptive move ordering prioritizing correctly")
            print("  - Performance targets met")
            print("  - Full engine integration successful")
        else:
            print("\n✗ Phase 3B needs improvements before deployment.")
            
    except Exception as e:
        print(f"Validation suite failed with error: {e}")
        import traceback
        traceback.print_exc()