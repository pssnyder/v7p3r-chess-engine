#!/usr/bin/env python3
"""
Quick performance test for GA engine to identify bottlenecks.
Run this to verify the optimizations are working.
"""

import os
import sys
import time
import yaml
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v7p3r_ga_engine.v7p3r_ga import v7p3rGeneticAlgorithm

def test_ga_performance():
    """Test GA performance with minimal configuration."""
    print("=== GA Performance Test ===")
    
    # Load minimal config for testing
    config_path = os.path.join(os.path.dirname(__file__), 'ga_config.yaml')
    
    print(f"Loading config from: {config_path}")
    
    try:
        start_time = time.time()
        
        # Initialize GA
        print("1. Initializing GA...")
        init_start = time.time()
        ga = v7p3rGeneticAlgorithm(config_path)
        init_time = time.time() - init_start
        print(f"   GA initialization: {init_time:.2f}s")
        
        # Test single evaluation
        print("2. Testing single evaluation...")
        eval_start = time.time()
        if ga.population:
            score = ga._evaluate(ga.population[0])
            eval_time = time.time() - eval_start
            print(f"   Single evaluation: {eval_time:.2f}s, Score: {score:.4f}")
        
        # Test full generation
        print("3. Testing full generation evaluation...")
        gen_start = time.time()
        scores = []
        for i, individual in enumerate(ga.population):
            print(f"   Evaluating individual {i+1}/{len(ga.population)}...")
            score = ga._evaluate(individual)
            scores.append(score)
            print(f"   Individual {i+1} score: {score:.4f}")
        gen_time = time.time() - gen_start
        print(f"   Full generation evaluation: {gen_time:.2f}s")
        
        # Calculate average time per individual
        avg_time = gen_time / len(ga.population)
        print(f"   Average time per individual: {avg_time:.2f}s")
        
        # Estimate total training time
        generations = ga.config.get('generations', 2)
        estimated_total = gen_time * generations
        print(f"   Estimated total training time: {estimated_total:.2f}s ({estimated_total/60:.1f} minutes)")
        
        total_time = time.time() - start_time
        print(f"\nTotal test time: {total_time:.2f}s")
        
        # Performance assessment
        if avg_time > 30:
            print("ΓÜá∩╕Å  WARNING: Very slow performance detected!")
            print("   Consider further reducing positions_count or population_size")
        elif avg_time > 10:
            print("ΓÜá∩╕Å  CAUTION: Slow performance detected")
            print("   Training may take a while")
        else:
            print("Γ£à Performance looks reasonable")
        
        # Cache performance
        if hasattr(ga, 'evaluation_cache'):
            print(f"\nCache performance:")
            print(f"   Cache size: {len(ga.evaluation_cache)}")
            print(f"   Cache enabled: {ga.cache_enabled}")
        
        # Cleanup
        ga.cleanup()
        print("\nΓ£à Performance test completed successfully!")
        
    except Exception as e:
        print(f"Γ¥î Error during performance test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_stockfish_performance():
    """Test Stockfish evaluation performance."""
    print("\n=== Stockfish Performance Test ===")
    
    try:
        from v7p3r_engine.stockfish_handler import StockfishHandler
        import chess
        
        # Test config
        stockfish_config = {
            "stockfish_path": "s:/Maker Stuff/Programming/V7P3R Chess Engine/viper_chess_engine/v7p3r_engine/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
            "elo_rating": 100,
            "skill_level": 1,
            "debug_mode": False,
            "depth": 1,
            "max_depth": 1,
            "movetime": 25,
        }
        
        print("1. Initializing Stockfish...")
        stockfish = StockfishHandler(stockfish_config)
        
        # Test positions
        test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        ]
        
        print("2. Testing position evaluations...")
        total_eval_time = 0
        for i, fen in enumerate(test_fens):
            board = chess.Board(fen)
            start = time.time()
            eval_score = stockfish.evaluate_position(board)
            eval_time = time.time() - start
            total_eval_time += eval_time
            print(f"   Position {i+1}: {eval_time:.3f}s, Score: {eval_score}")
        
        avg_eval_time = total_eval_time / len(test_fens)
        print(f"   Average evaluation time: {avg_eval_time:.3f}s")
        
        if avg_eval_time > 1.0:
            print("ΓÜá∩╕Å  WARNING: Stockfish evaluations are very slow!")
        elif avg_eval_time > 0.5:
            print("ΓÜá∩╕Å  CAUTION: Stockfish evaluations are slow")
        else:
            print("Γ£à Stockfish performance looks good")
        
        stockfish.close()
        print("Γ£à Stockfish test completed!")
        
    except Exception as e:
        print(f"Γ¥î Error during Stockfish test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("Starting GA performance analysis...")
    
    # Run tests
    success = True
    success &= test_stockfish_performance()
    success &= test_ga_performance()
    
    if success:
        print("\n≡ƒÄë All performance tests passed!")
        print("\nYou can now run the full GA training with:")
        print("python v7p3r_ga_training.py")
    else:
        print("\nΓ¥î Some performance tests failed.")
        print("Please check the errors above before running full training.")
