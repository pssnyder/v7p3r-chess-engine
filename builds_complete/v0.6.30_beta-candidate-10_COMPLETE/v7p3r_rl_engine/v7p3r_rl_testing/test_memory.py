#!/usr/bin/env python3
"""
Minimal GA test with reduced resource usage to identify the issue
"""
import yaml
import sys
import os
import psutil
import time
sys.path.append(os.path.abspath('.'))

def memory_usage():
    """Get current memory usage in GB"""
    return psutil.virtual_memory().used / (1024**3)

def test_minimal_ga():
    """Test GA with minimal configuration to identify memory leaks"""
    print("=== Minimal GA Test ===")
    print(f"Initial memory: {memory_usage():.1f}GB")
    
    try:
        # Load minimal config
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
            "use_cuda": False,  # Disable CUDA for this test
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
        
        print("Loading components...")
        print(f"Memory after config: {memory_usage():.1f}GB")
        
        # Import components
        from v7p3r_ga_engine.v7p3r_ga_training import TrainingRunner
        print(f"Memory after imports: {memory_usage():.1f}GB")
        
        # Create trainer
        trainer = TrainingRunner(config)
        print(f"Memory after trainer creation: {memory_usage():.1f}GB")
        
        # Prepare environment
        trainer.prepare_environment()
        print(f"Memory after environment prep: {memory_usage():.1f}GB")
        
        # Run minimal training
        print("Starting minimal training...")
        stats = trainer.run_training()
        print(f"Memory after training: {memory_usage():.1f}GB")
        
        # Clean up
        trainer.save_results()
        del trainer
        print(f"Memory after cleanup: {memory_usage():.1f}GB")
        
        print("Γ£à Minimal test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Γ¥î Error in minimal test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_memory_leak():
    """Test if CUDA is causing memory leaks"""
    print("\n=== CUDA Memory Leak Test ===")
    
    try:
        import torch
        print(f"Initial system memory: {memory_usage():.1f}GB")
        
        if torch.cuda.is_available():
            print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
            
            # Create some tensors
            for i in range(100):
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda')
                z = torch.matmul(x, y)
                
                # Force cleanup
                del x, y, z
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                    print(f"Iteration {i}: GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB, System: {memory_usage():.1f}GB")
            
            torch.cuda.empty_cache()
            print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
            print(f"Final system memory: {memory_usage():.1f}GB")
            
        else:
            print("CUDA not available for memory test")
            
    except Exception as e:
        print(f"Γ¥î Error in CUDA test: {e}")

if __name__ == "__main__":
    print("=== GA Memory Debug Test ===\n")
    
    # Test CUDA memory first
    test_cuda_memory_leak()
    
    # Test minimal GA
    test_minimal_ga()
    
    print(f"\nFinal system memory: {memory_usage():.1f}GB")
