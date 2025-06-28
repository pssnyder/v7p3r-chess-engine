#!/usr/bin/env python3
"""
Test Stockfish cleanup to ensure processes are properly terminated
"""
import subprocess
import time
import psutil
import yaml
import sys
import os
sys.path.append(os.path.abspath('.'))

def count_stockfish_processes():
    """Count running Stockfish processes"""
    count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'stockfish' in proc.info['name'].lower():
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return count

def test_stockfish_cleanup():
    """Test that Stockfish processes are properly cleaned up"""
    print("=== Stockfish Cleanup Test ===")
    
    initial_count = count_stockfish_processes()
    print(f"Initial Stockfish processes: {initial_count}")
    
    try:
        # Load config
        with open("v7p3r_ga_engine/ga_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        # Create PositionEvaluator (this will start Stockfish)
        from v7p3r_ga_engine.position_evaluator import PositionEvaluator
        print("Creating PositionEvaluator...")
        
        evaluator = PositionEvaluator(
            stockfish_config=config["stockfish_config"],
            use_cuda=False,
            use_nn_evaluator=False
        )
        
        after_create_count = count_stockfish_processes()
        print(f"After creating evaluator: {after_create_count} Stockfish processes")
        
        # Do a quick evaluation to ensure Stockfish is working
        positions = evaluator.load_positions("random", 2)
        print("Testing evaluation...")
        evals = evaluator.batch_stockfish_evaluation([positions[0]])
        print(f"Evaluation result: {evals[0]}")
        
        # Explicitly cleanup
        print("Calling cleanup...")
        evaluator.cleanup()
        time.sleep(1)  # Give it a moment to terminate
        
        after_cleanup_count = count_stockfish_processes()
        print(f"After cleanup: {after_cleanup_count} Stockfish processes")
        
        # Test destructor cleanup
        print("Deleting evaluator...")
        del evaluator
        time.sleep(1)
        
        final_count = count_stockfish_processes()
        print(f"Final count: {final_count} Stockfish processes")
        
        if final_count <= initial_count:
            print("✅ Stockfish cleanup test PASSED!")
            return True
        else:
            print("❌ Stockfish cleanup test FAILED - processes not cleaned up!")
            return False
            
    except Exception as e:
        print(f"❌ Error in cleanup test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ga_cleanup():
    """Test GA training cleanup"""
    print("\n=== GA Training Cleanup Test ===")
    
    initial_count = count_stockfish_processes()
    print(f"Initial Stockfish processes: {initial_count}")
    
    try:
        # Run minimal GA training
        from v7p3r_ga_engine.v7p3r_ga_training import TrainingRunner
        
        config = {
            "population_size": 2,
            "generations": 1,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "elitism_rate": 0.5,
            "adaptive_mutation": False,
            "positions_source": "random",
            "positions_count": 2,
            "max_stagnation": 1,
            "use_cuda": False,
            "cuda_batch_size": 8,
            "use_multiprocessing": False,
            "max_workers": 1,
            "use_neural_evaluator": False,
            "neural_model_path": None,
            "stockfish_config": {
                "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
                "elo_rating": 100,
                "skill_level": 1,
                "debug_mode": False,
                "depth": 1,
                "max_depth": 1,
                "movetime": 50,
            }
        }
        
        print("Starting minimal GA training...")
        trainer = TrainingRunner(config)
        trainer.prepare_environment()
        
        during_count = count_stockfish_processes()
        print(f"During training setup: {during_count} Stockfish processes")
        
        # This should complete quickly with minimal config
        trainer.run_training()
        trainer.save_results()
        
        # Explicit cleanup
        trainer.position_evaluator.cleanup()
        del trainer
        
        time.sleep(2)  # Give processes time to terminate
        
        final_count = count_stockfish_processes()
        print(f"After GA training: {final_count} Stockfish processes")
        
        if final_count <= initial_count:
            print("✅ GA cleanup test PASSED!")
            return True
        else:
            print("❌ GA cleanup test FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Error in GA cleanup test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Stockfish process cleanup...\n")
    
    # Run tests
    stockfish_ok = test_stockfish_cleanup()
    ga_ok = test_ga_cleanup()
    
    print(f"\n=== Test Summary ===")
    print(f"Stockfish cleanup: {'✅' if stockfish_ok else '❌'}")
    print(f"GA cleanup: {'✅' if ga_ok else '❌'}")
    
    if stockfish_ok and ga_ok:
        print("\n🎉 All cleanup tests passed! Stockfish processes should now terminate properly.")
    else:
        print("\n⚠️  Some cleanup tests failed. Check the issues above.")
        
    print(f"\nFinal process count: {count_stockfish_processes()} Stockfish processes")
