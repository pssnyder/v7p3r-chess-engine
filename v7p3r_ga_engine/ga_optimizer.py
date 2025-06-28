"""
Optimization utility for finding optimal GA parameters and system settings.
Includes hyperparameter tuning and performance benchmarking.
"""

import yaml
import time
import os
import sys
from typing import Dict, List, Tuple, Any
import itertools
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from v7p3r_ga_training import TrainingRunner

class GAOptimizer:
    """Hyperparameter optimization for GA training."""
    
    def __init__(self, base_config_path: str = "v7p3r_ga_engine/ga_config.yaml"):
        """Initialize optimizer with base configuration."""
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.results = []
        self.benchmark_results = {}
    
    def run_parameter_sweep(self, param_grid: Dict[str, List[Any]], 
                          max_experiments: int = 10) -> List[Dict]:
        """
        Run parameter sweep to find optimal GA settings.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values to try
            max_experiments: Maximum number of experiments to run
            
        Returns:
            List of experiment results sorted by best fitness
        """
        print(f"[Optimizer] Starting parameter sweep with {len(param_grid)} parameters")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        # Limit number of experiments
        if len(combinations) > max_experiments:
            print(f"[Optimizer] Limiting to {max_experiments} experiments (out of {len(combinations)} possible)")
            import random
            combinations = random.sample(combinations, max_experiments)
        
        results = []
        
        for i, param_combo in enumerate(combinations):
            print(f"\n{'='*80}")
            print(f"[Optimizer] Experiment {i+1}/{len(combinations)}")
            print(f"{'='*80}")
            
            # Create experiment config
            exp_config = self.base_config.copy()
            for param_name, param_value in zip(param_names, param_combo):
                exp_config[param_name] = param_value
                print(f"  {param_name}: {param_value}")
            
            # Run training
            try:
                start_time = time.time()
                trainer = TrainingRunner(exp_config)
                trainer.prepare_environment()
                final_stats = trainer.run_training(generations=exp_config.get("generations", 5))
                trainer.save_results()
                
                experiment_time = time.time() - start_time
                
                # Record results
                if final_stats is None:
                    final_stats = {}
                
                result = {
                    'experiment_id': i + 1,
                    'parameters': dict(zip(param_names, param_combo)),
                    'best_fitness': final_stats.get('best_fitness', float('-inf')),
                    'generations_completed': final_stats.get('generations_completed', 0),
                    'avg_generation_time': final_stats.get('avg_generation_time', 0),
                    'total_experiment_time': experiment_time,
                    'stagnation_counter': final_stats.get('stagnation_counter', 0),
                    'fitness_std': final_stats.get('fitness_std', 0)
                }
                
                results.append(result)
                print(f"[Optimizer] Experiment {i+1} completed: Best fitness = {result['best_fitness']:.6f}")
                
            except Exception as e:
                print(f"[Optimizer] Experiment {i+1} FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        # Sort by best fitness
        results.sort(key=lambda x: x['best_fitness'], reverse=True)
        
        # Save results
        self.save_optimization_results(results, param_grid)
        
        return results
    
    def benchmark_system_performance(self) -> Dict[str, Any]:
        """
        Benchmark system performance with different settings.
        """
        print("[Optimizer] Running system performance benchmark...")
        
        benchmark_configs = [
            {"name": "CPU_Only", "use_cuda": False, "population_size": 10, "positions_count": 20},
            {"name": "CUDA_Small", "use_cuda": True, "population_size": 10, "positions_count": 20},
            {"name": "CUDA_Medium", "use_cuda": True, "population_size": 20, "positions_count": 50},
            {"name": "CUDA_Large", "use_cuda": True, "population_size": 30, "positions_count": 100},
        ]
        
        benchmark_results = {}
        
        for bench_config in benchmark_configs:
            config_name = bench_config.pop("name")
            print(f"\n[Benchmark] Testing {config_name}...")
            
            # Create test config
            test_config = self.base_config.copy()
            test_config.update(bench_config)
            test_config["generations"] = 2  # Short test
            
            try:
                start_time = time.time()
                trainer = TrainingRunner(test_config)
                trainer.prepare_environment()
                trainer.run_training()
                
                benchmark_time = time.time() - start_time
                
                # Get performance stats
                perf_stats = trainer.position_evaluator.get_performance_stats()
                
                benchmark_results[config_name] = {
                    'total_time': benchmark_time,
                    'performance_stats': perf_stats,
                    'config': bench_config
                }
                
                print(f"[Benchmark] {config_name}: {benchmark_time:.2f}s")
                
            except Exception as e:
                print(f"[Benchmark] {config_name} FAILED: {e}")
                benchmark_results[config_name] = {'error': str(e)}
        
        self.benchmark_results = benchmark_results
        return benchmark_results
    
    def find_optimal_cuda_settings(self) -> Dict[str, Any]:
        """Find optimal CUDA batch size and memory settings."""
        if not self._check_cuda_available():
            return {"error": "CUDA not available"}
        
        print("[Optimizer] Finding optimal CUDA settings...")
        
        batch_sizes = [16, 32, 64, 128, 256]
        optimal_settings = {}
        
        for batch_size in batch_sizes:
            print(f"[CUDA] Testing batch size: {batch_size}")
            
            config = self.base_config.copy()
            config.update({
                "use_cuda": True,
                "cuda_batch_size": batch_size,
                "population_size": 20,
                "positions_count": 50,
                "generations": 1
            })
            
            try:
                start_time = time.time()
                trainer = TrainingRunner(config)
                trainer.prepare_environment()
                trainer.run_training()
                
                test_time = time.time() - start_time
                memory_stats = trainer.position_evaluator.get_performance_stats().get('memory_usage', {})
                
                optimal_settings[batch_size] = {
                    'time': test_time,
                    'memory_used': memory_stats.get('allocated', 0),
                    'success': True
                }
                
                print(f"[CUDA] Batch {batch_size}: {test_time:.2f}s, {memory_stats.get('allocated', 0):.2f}GB")
                
            except Exception as e:
                print(f"[CUDA] Batch {batch_size} FAILED: {e}")
                optimal_settings[batch_size] = {'error': str(e), 'success': False}
        
        # Find best batch size (fastest with no errors)
        successful_tests = {k: v for k, v in optimal_settings.items() if v.get('success', False)}
        if successful_tests:
            best_batch_size = min(successful_tests.keys(), key=lambda k: successful_tests[k]['time'])
            print(f"[CUDA] Optimal batch size: {best_batch_size}")
            return {
                'optimal_batch_size': best_batch_size,
                'all_results': optimal_settings,
                'recommendation': successful_tests[best_batch_size]
            }
        
        return {'error': 'No successful CUDA tests', 'all_results': optimal_settings}
    
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def save_optimization_results(self, results: List[Dict], param_grid: Dict):
        """Save optimization results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = f"v7p3r_ga_engine/ga_results/optimization_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(results_dir, "optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'parameter_grid': param_grid,
                'results': results,
                'benchmark_results': self.benchmark_results
            }, f, indent=2)
        
        # Save summary report
        report_path = os.path.join(results_dir, "optimization_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== GA Optimization Report ===\n\n")
            
            if results:
                f.write("=== Best Parameters ===\n")
                best_result = results[0]
                f.write(f"Best Fitness: {best_result['best_fitness']:.6f}\n")
                f.write(f"Parameters:\n")
                for param, value in best_result['parameters'].items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"Generations: {best_result['generations_completed']}\n")
                f.write(f"Avg Generation Time: {best_result['avg_generation_time']:.2f}s\n\n")
                
                f.write("=== Top 5 Results ===\n")
                for i, result in enumerate(results[:5]):
                    f.write(f"{i+1}. Fitness: {result['best_fitness']:.6f} | ")
                    f.write(f"Params: {result['parameters']}\n")
            
            if self.benchmark_results:
                f.write("\n=== Performance Benchmark ===\n")
                for config_name, bench_result in self.benchmark_results.items():
                    if 'total_time' in bench_result:
                        f.write(f"{config_name}: {bench_result['total_time']:.2f}s\n")
                    else:
                        f.write(f"{config_name}: FAILED\n")
        
        print(f"[Optimizer] Results saved to {results_dir}")
        return results_dir

def main():
    """Main optimization workflow."""
    optimizer = GAOptimizer()
    
    # Define parameter grid for optimization
    param_grid = {
        'population_size': [15, 20, 25],
        'mutation_rate': [0.1, 0.2, 0.3],
        'crossover_rate': [0.6, 0.7, 0.8],
        'elitism_rate': [0.05, 0.1, 0.15],
        'positions_count': [30, 50, 70]
    }
    
    print("GA Optimization Suite")
    print("1. Parameter sweep")
    print("2. System benchmark")
    print("3. CUDA optimization")
    print("4. Full optimization")
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == '1':
        results = optimizer.run_parameter_sweep(param_grid, max_experiments=8)
        print(f"\nBest parameters found:")
        if results:
            best = results[0]
            print(f"Fitness: {best['best_fitness']:.6f}")
            for param, value in best['parameters'].items():
                print(f"  {param}: {value}")
    
    elif choice == '2':
        benchmark_results = optimizer.benchmark_system_performance()
        print(f"\nBenchmark completed. Results:")
        for config, result in benchmark_results.items():
            if 'total_time' in result:
                print(f"{config}: {result['total_time']:.2f}s")
    
    elif choice == '3':
        if optimizer._check_cuda_available():
            cuda_results = optimizer.find_optimal_cuda_settings()
            if 'optimal_batch_size' in cuda_results:
                print(f"\nOptimal CUDA batch size: {cuda_results['optimal_batch_size']}")
        else:
            print("CUDA not available")
    
    elif choice == '4':
        print("Running full optimization...")
        benchmark_results = optimizer.benchmark_system_performance()
        if optimizer._check_cuda_available():
            cuda_results = optimizer.find_optimal_cuda_settings()
        results = optimizer.run_parameter_sweep(param_grid, max_experiments=6)
        print("Full optimization completed!")

if __name__ == "__main__":
    main()
