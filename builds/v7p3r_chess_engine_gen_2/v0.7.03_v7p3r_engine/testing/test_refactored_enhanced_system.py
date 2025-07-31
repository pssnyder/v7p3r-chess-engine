#!/usr/bin/env python3
"""
Test script for the refactored enhanced metrics system
This will verify that the new dataset-based metrics collection works correctly
"""

import chess
import sys
import os
import time

# Add the v7p3r_engine path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'v7p3r_engine'))

from v7p3r_engine.v7p3r import v7p3rEngine
from metrics.refactored_enhanced_metrics_collector import RefactoredEnhancedMetricsCollector

def test_refactored_system():
    """
    Test the refactored enhanced metrics system with search_dataset and score_dataset
    """
    print("Testing Refactored Enhanced Metrics System")
    print("=" * 50)
    
    try:
        # Create test board and engine
        board = chess.Board()
        engine = v7p3rEngine()
        
        print(f"Engine initialized: {engine.name}")
        print(f"Board position: {board.fen()}")
        
        # Create refactored collector
        collector = RefactoredEnhancedMetricsCollector()
        
        # Test with a few moves
        for move_num in range(1, 4):
            print(f"\n--- Move {move_num} ---")
            
            # Get a legal move
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                print("No legal moves available")
                break
                
            test_move = legal_moves[0]
            print(f"Testing move: {test_move}")
            
            # Perform engine search to populate datasets
            print("Performing engine search...")
            start_time = time.time()
            
            try:
                # This should populate the search_dataset and score_dataset
                search_result = engine.search_engine.search(board, board.turn)
                search_time = time.time() - start_time
                
                print(f"Search completed in {search_time:.3f}s")
                print(f"Search result: {search_result}")
                
                # Verify datasets are populated
                print("\nVerifying dataset population:")
                
                # Check search_dataset
                if hasattr(engine.search_engine, 'search_dataset'):
                    search_data = engine.search_engine.search_dataset
                    print(f"Search dataset keys: {list(search_data.keys())}")
                    print(f"Nodes searched: {search_data.get('nodes_searched', 'N/A')}")
                    print(f"Best move: {search_data.get('best_move', 'N/A')}")
                    print(f"Best score: {search_data.get('best_score', 'N/A')}")
                else:
                    print("ERROR: No search_dataset found!")
                
                # Check score_dataset  
                if hasattr(engine.scoring_calculator, 'score_dataset'):
                    score_data = engine.scoring_calculator.score_dataset
                    print(f"Score dataset keys: {list(score_data.keys())}")
                    non_zero_scores = {k: v for k, v in score_data.items() 
                                     if k.endswith('_score') or k in ['checkmate_threats', 'king_safety'] 
                                     and v != 0.0}
                    print(f"Non-zero scoring components: {len(non_zero_scores)}")
                    for key, value in list(non_zero_scores.items())[:5]:
                        print(f"  {key}: {value}")
                else:
                    print("ERROR: No score_dataset found!")
                
                # Test refactored metrics collection
                print("\nTesting refactored metrics collection:")
                
                # Collect search metrics
                search_metrics = collector.collect_from_search_dataset(engine.search_engine)
                print(f"Search metrics collected: {len(search_metrics)} items")
                
                # Collect scoring metrics
                scoring_metrics = collector.collect_from_score_dataset(engine.scoring_calculator)
                print(f"Scoring metrics collected: {len(scoring_metrics)} items")
                
                # Collect comprehensive metrics
                comprehensive_metrics = collector.collect_comprehensive_metrics(
                    engine, board, test_move, time_taken=search_time
                )
                print(f"Comprehensive metrics collected: {len(comprehensive_metrics)} items")
                
                # Show key metrics
                print(f"\nKey metrics:")
                key_metrics = ['search_algorithm', 'depth_reached', 'nodes_searched', 
                             'total_score', 'material_balance', 'game_phase', 'nps']
                for key in key_metrics:
                    if key in comprehensive_metrics:
                        print(f"  {key}: {comprehensive_metrics[key]}")
                
                # Validate completeness
                validated_metrics = collector.validate_metrics_completeness(comprehensive_metrics)
                print(f"Validated metrics: {len(validated_metrics)} items")
                
                # Actually make the move on the board for next iteration
                board.push(test_move)
                
            except Exception as e:
                print(f"ERROR during search: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"\n--- Final Game State ---")
        print(f"Moves played: {len(board.move_stack)}")
        print(f"Final position: {board.fen()}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_dataset_accessibility():
    """
    Test direct access to search_dataset and score_dataset objects
    """
    print("\nTesting Dataset Accessibility")
    print("=" * 30)
    
    try:
        # Create engine
        engine = v7p3rEngine()
        board = chess.Board()
        
        # Check if datasets exist before search
        print("Before search:")
        print(f"  Search engine has search_dataset: {hasattr(engine.search_engine, 'search_dataset')}")
        print(f"  Scoring calculator has score_dataset: {hasattr(engine.scoring_calculator, 'score_dataset')}")
        
        # Perform a simple search
        print("\nPerforming search...")
        result = engine.search_engine.search(board, chess.WHITE)
        
        # Check datasets after search
        print("\nAfter search:")
        if hasattr(engine.search_engine, 'search_dataset'):
            dataset = engine.search_engine.search_dataset
            print(f"  Search dataset type: {type(dataset)}")
            print(f"  Search dataset size: {len(dataset) if isinstance(dataset, dict) else 'N/A'}")
            print(f"  Sample keys: {list(dataset.keys())[:5] if isinstance(dataset, dict) else 'N/A'}")
        
        if hasattr(engine.scoring_calculator, 'score_dataset'):
            dataset = engine.scoring_calculator.score_dataset
            print(f"  Score dataset type: {type(dataset)}")
            print(f"  Score dataset size: {len(dataset) if isinstance(dataset, dict) else 'N/A'}")
            print(f"  Sample keys: {list(dataset.keys())[:5] if isinstance(dataset, dict) else 'N/A'}")
        
    except Exception as e:
        print(f"Dataset accessibility test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_accessibility()
    test_refactored_system()
