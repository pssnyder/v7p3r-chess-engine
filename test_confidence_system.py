#!/usr/bin/env python3
"""
Test script for V7P3R v9.1 Confidence-Based Evaluation System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import chess
import time
from src.v7p3r import V7P3RCleanEngine

def test_confidence_system():
    """Test the confidence-based evaluation system"""
    print("=== V7P3R v9.1 Confidence System Test ===")
    
    # Initialize engine
    engine = V7P3RCleanEngine()
    
    # Test positions
    test_positions = [
        # Starting position
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"),
        
        # Tactical position with mate threat
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7+"),
        
        # Endgame position  
        ("8/8/8/8/8/3k4/3P4/3K4 w - - 0 1", "d4"),
    ]
    
    for i, (fen, expected_move) in enumerate(test_positions):
        print(f"\n--- Test Position {i+1} ---")
        print(f"FEN: {fen}")
        print(f"Expected move: {expected_move}")
        
        board = chess.Board(fen)
        print(f"Position: {board}")
        
        # Test with multithreading enabled
        print("\n>> Testing with multithreaded confidence evaluation:")
        engine.set_multithreaded_evaluation(True)
        start_time = time.time()
        best_move = engine.search(board, time_limit=2.0)
        mt_time = time.time() - start_time
        
        print(f"Best move (MT): {best_move}")
        print(f"Time taken: {mt_time:.2f}s")
        
        # Get confidence stats
        stats = engine.get_confidence_stats()
        print(f"Confidence stats: {stats}")
        
        # Test with traditional evaluation
        print("\n>> Testing with traditional evaluation:")
        engine.set_multithreaded_evaluation(False)
        start_time = time.time()
        best_move_traditional = engine.search(board, time_limit=2.0)
        trad_time = time.time() - start_time
        
        print(f"Best move (Traditional): {best_move_traditional}")
        print(f"Time taken: {trad_time:.2f}s")
        
        # Compare results
        print(f"\n>> Comparison:")
        print(f"Moves match: {best_move == best_move_traditional}")
        print(f"Time difference: {mt_time - trad_time:.2f}s")
        
        engine.new_game()  # Reset for next test

def test_confidence_calculation():
    """Test the confidence calculation directly"""
    print("\n=== Confidence Calculation Test ===")
    
    try:
        from src.v7p3r_confidence_engine import (
            ConfidenceWeightedEvaluation, EvaluationMetrics, MoveCategory
        )
        
        # Test different scenarios
        scenarios = [
            # High-quality mate evaluation
            {
                'raw_eval': 25000,
                'metrics': EvaluationMetrics(
                    search_depth=6, time_allocated=1.0, time_spent=0.9,
                    thread_count=4, beta_cutoffs=10, move_ordering_hits=8,
                    move_ordering_attempts=10, nodes_searched=1000
                ),
                'category': MoveCategory.MATE,
                'description': 'High-quality mate'
            },
            
            # Low-quality critical position
            {
                'raw_eval': 500,
                'metrics': EvaluationMetrics(
                    search_depth=2, time_allocated=1.0, time_spent=0.3,
                    thread_count=1, beta_cutoffs=1, move_ordering_hits=2,
                    move_ordering_attempts=5, nodes_searched=100
                ),
                'category': MoveCategory.CRITICAL,
                'description': 'Low-quality critical position'
            },
            
            # High-quality positional move
            {
                'raw_eval': 50,
                'metrics': EvaluationMetrics(
                    search_depth=8, time_allocated=1.0, time_spent=1.0,
                    thread_count=4, beta_cutoffs=15, move_ordering_hits=12,
                    move_ordering_attempts=15, nodes_searched=2000
                ),
                'category': MoveCategory.POSITIONAL,
                'description': 'High-quality positional move'
            }
        ]
        
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        
        for scenario in scenarios:
            print(f"\n--- {scenario['description']} ---")
            
            # Create confidence evaluation manually
            eval_result = ConfidenceWeightedEvaluation(
                raw_evaluation=scenario['raw_eval'],
                confidence_weight=0.0,
                final_evaluation=0.0,
                move_category=scenario['category'],
                metrics=scenario['metrics']
            )
            
            # Calculate confidence
            eval_result.confidence_weight = eval_result._calculate_confidence_weight()
            eval_result.final_evaluation = eval_result._apply_confidence_weighting()
            
            print(f"Raw evaluation: {eval_result.raw_evaluation}")
            print(f"Confidence weight: {eval_result.confidence_weight:.3f}")
            print(f"Final evaluation: {eval_result.final_evaluation:.1f}")
            print(f"Category: {eval_result.move_category.value}")
            
            # Verify 50% threshold for mates/critical moves
            if eval_result.is_mate or eval_result.is_critical:
                confidence_ok = eval_result.confidence_weight >= 0.50
                print(f"50% threshold check: {'PASS' if confidence_ok else 'FAIL'}")
        
    except ImportError as e:
        print(f"Could not import confidence engine modules: {e}")
        print("This is expected during development - modules will be available at runtime")

if __name__ == "__main__":
    try:
        test_confidence_system()
        test_confidence_calculation()
        print("\n=== Test completed successfully ===")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            engine = V7P3RCleanEngine()
            engine.cleanup()
        except:
            pass
