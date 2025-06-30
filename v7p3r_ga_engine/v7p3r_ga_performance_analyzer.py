"""
Performance profiling and analysis tools for GA training.
Helps identify bottlenecks and optimization opportunities.
"""

import time
import psutil
import os
import sys
import cProfile
import pstats
from typing import Dict, List, Any, Optional
import threading
import queue
import matplotlib.pyplot as plt
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class PerformanceProfiler:
    """Comprehensive performance profiler for GA training."""
    
    def __init__(self, sample_interval: float = 1.0):
        """
        Initialize profiler.
        
        Args:
            sample_interval: How often to sample system metrics (seconds)
        """
        self.sample_interval = sample_interval
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
        self.start_time = None
        self.profile_data = {}
        
    def start_monitoring(self):
        """Start system monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("[Profiler] System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("[Profiler] System monitoring stopped")
    
    def _monitor_system(self):
        """Monitor system resources in background thread."""
        while self.monitoring:
            try:
                timestamp = time.time() - (self.start_time or 0)
                
                # CPU and memory metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                # GPU metrics (if available)
                gpu_metrics = self._get_gpu_metrics()
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                
                metrics = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'memory_available_gb': memory_info.available / (1024**3),
                    'process_memory_mb': process_memory.rss / (1024**2),
                    'gpu_metrics': gpu_metrics
                }
                
                self.metrics_queue.put(metrics)
                time.sleep(self.sample_interval)
                
            except Exception as e:
                print(f"[Profiler] Error in monitoring: {e}")
                break
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'gpu_memory_cached_gb': torch.cuda.memory_reserved() / (1024**3),
                    'gpu_utilization': self._get_gpu_utilization()
                }
        except Exception:
            pass
        return {}
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage using nvidia-ml-py if available."""
        try:
            # Optional dependency - only import if available
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except (ImportError, Exception):
            return None
    
    def profile_function(self, func, *args, **kwargs):
        """Profile a specific function using cProfile."""
        print(f"[Profiler] Profiling function: {func.__name__}")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # Save profile data
        profile_filename = f"profile_{func.__name__}_{int(time.time())}.prof"
        profiler.dump_stats(profile_filename)
        
        # Generate readable stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Store top functions
        self.profile_data[func.__name__] = {
            'filename': profile_filename,
            'top_functions': self._extract_top_functions(stats)
        }
        
        print(f"[Profiler] Profile saved: {profile_filename}")
        return result
    
    def _extract_top_functions(self, stats, limit: int = 20) -> List[Dict]:
        """Extract top functions from profile stats."""
        top_functions = []
        
        for func_key, (call_count, total_time, cumulative_time, callers) in list(stats.stats.items())[:limit]:
            filename, line_num, func_name = func_key
            
            top_functions.append({
                'function': f"{os.path.basename(filename)}:{line_num}({func_name})",
                'calls': call_count,
                'total_time': total_time,
                'cumulative_time': cumulative_time,
                'time_per_call': total_time / call_count if call_count > 0 else 0
            })
        
        return top_functions
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        metrics_list = []
        
        # Collect all metrics from queue
        while not self.metrics_queue.empty():
            try:
                metrics_list.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        
        if not metrics_list:
            return {}
        
        # Calculate summary statistics
        cpu_values = [m['cpu_percent'] for m in metrics_list]
        memory_values = [m['memory_percent'] for m in metrics_list]
        process_memory_values = [m['process_memory_mb'] for m in metrics_list]
        
        summary = {
            'monitoring_duration': metrics_list[-1]['timestamp'] if metrics_list else 0,
            'samples_collected': len(metrics_list),
            'cpu_stats': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_stats': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'process_memory_stats': {
                'avg': sum(process_memory_values) / len(process_memory_values),
                'max': max(process_memory_values),
                'min': min(process_memory_values)
            }
        }
        
        # GPU stats if available
        gpu_memory_values = [m['gpu_metrics'].get('gpu_memory_allocated_gb', 0) 
                           for m in metrics_list if m.get('gpu_metrics')]
        
        if gpu_memory_values:
            summary['gpu_stats'] = {
                'avg_memory_gb': sum(gpu_memory_values) / len(gpu_memory_values),
                'max_memory_gb': max(gpu_memory_values),
                'min_memory_gb': min(gpu_memory_values)
            }
        
        return summary
    
    def generate_performance_report(self, output_dir: str = "performance_analysis"):
        """Generate comprehensive performance report."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[Profiler] Generating performance report in {output_dir}")
        
        # Get metrics summary
        metrics_summary = self.get_metrics_summary()
        
        # Generate text report
        report_path = os.path.join(output_dir, "performance_report.txt")
        with open(report_path, 'w') as f:
            f.write("=== GA Training Performance Analysis ===\n\n")
            
            if metrics_summary:
                f.write("=== System Resource Usage ===\n")
                f.write(f"Monitoring Duration: {metrics_summary['monitoring_duration']:.2f}s\n")
                f.write(f"Samples Collected: {metrics_summary['samples_collected']}\n\n")
                
                cpu_stats = metrics_summary['cpu_stats']
                f.write(f"CPU Usage:\n")
                f.write(f"  Average: {cpu_stats['avg']:.1f}%\n")
                f.write(f"  Maximum: {cpu_stats['max']:.1f}%\n")
                f.write(f"  Minimum: {cpu_stats['min']:.1f}%\n\n")
                
                mem_stats = metrics_summary['memory_stats']
                f.write(f"System Memory Usage:\n")
                f.write(f"  Average: {mem_stats['avg']:.1f}%\n")
                f.write(f"  Maximum: {mem_stats['max']:.1f}%\n")
                f.write(f"  Minimum: {mem_stats['min']:.1f}%\n\n")
                
                proc_mem_stats = metrics_summary['process_memory_stats']
                f.write(f"Process Memory Usage:\n")
                f.write(f"  Average: {proc_mem_stats['avg']:.1f}MB\n")
                f.write(f"  Maximum: {proc_mem_stats['max']:.1f}MB\n")
                f.write(f"  Minimum: {proc_mem_stats['min']:.1f}MB\n\n")
                
                if 'gpu_stats' in metrics_summary:
                    gpu_stats = metrics_summary['gpu_stats']
                    f.write(f"GPU Memory Usage:\n")
                    f.write(f"  Average: {gpu_stats['avg_memory_gb']:.2f}GB\n")
                    f.write(f"  Maximum: {gpu_stats['max_memory_gb']:.2f}GB\n")
                    f.write(f"  Minimum: {gpu_stats['min_memory_gb']:.2f}GB\n\n")
            
            # Function profiling results
            if self.profile_data:
                f.write("=== Function Profiling Results ===\n")
                for func_name, profile_info in self.profile_data.items():
                    f.write(f"\nFunction: {func_name}\n")
                    f.write(f"Profile file: {profile_info['filename']}\n")
                    f.write("Top time-consuming functions:\n")
                    
                    for i, func_info in enumerate(profile_info['top_functions'][:10]):
                        f.write(f"  {i+1:2d}. {func_info['function']}\n")
                        f.write(f"      Calls: {func_info['calls']}, ")
                        f.write(f"Total: {func_info['total_time']:.4f}s, ")
                        f.write(f"Per call: {func_info['time_per_call']:.6f}s\n")
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, "metrics_summary.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        print(f"[Profiler] Report generated:")
        print(f"  - Text report: {report_path}")
        print(f"  - Metrics data: {metrics_path}")
        
        return report_path


class GAPerformanceTester:
    """Specialized performance tester for GA training components."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.test_results = {}
    
    def test_evaluation_performance(self, config_variations: List[Dict]):
        """Test evaluation performance with different configurations."""
        print("[Tester] Testing evaluation performance...")
        
        from v7p3r_ga_training import TrainingRunner
        
        for i, config in enumerate(config_variations):
            print(f"\n[Tester] Configuration {i+1}/{len(config_variations)}: {config}")
            
            # Start monitoring
            self.profiler.start_monitoring()
            
            try:
                # Create trainer and run one generation
                trainer = TrainingRunner(config)
                trainer.prepare_environment()
                
                # Profile the evaluation step
                def single_evaluation():
                    trainer.ga.initialize_population()
                    trainer.ga.evaluate_population(trainer.positions)
                    return trainer.ga.get_convergence_stats()
                
                stats = self.profiler.profile_function(single_evaluation)
                
                # Stop monitoring
                self.profiler.stop_monitoring()
                
                # Collect results
                metrics_summary = self.profiler.get_metrics_summary()
                
                self.test_results[f"config_{i+1}"] = {
                    'config': config,
                    'stats': stats,
                    'metrics': metrics_summary
                }
                
                print(f"[Tester] Config {i+1} completed - Best fitness: {stats.get('best_fitness', 0):.4f}")
                
            except Exception as e:
                print(f"[Tester] Config {i+1} FAILED: {e}")
                self.profiler.stop_monitoring()
    
    def benchmark_cuda_vs_cpu(self):
        """Compare CUDA vs CPU performance."""
        print("[Tester] Benchmarking CUDA vs CPU...")
        
        test_configs = [
            {
                "name": "CPU",
                "use_cuda": False,
                "population_size": 15,
                "positions_count": 30,
                "generations": 2
            },
            {
                "name": "CUDA",
                "use_cuda": True,
                "population_size": 15,
                "positions_count": 30,
                "generations": 2
            }
        ]
        
        results = {}
        
        for config in test_configs:
            config_name = config.pop("name")
            print(f"\n[Benchmark] Testing {config_name}...")
            
            # Add base config
            full_config = {
                "stockfish_config": {
                    "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
                    "depth": 2, "movetime": 200
                },
                **config
            }
            
            self.profiler.start_monitoring()
            start_time = time.time()
            
            try:
                from v7p3r_ga_training import TrainingRunner
                trainer = TrainingRunner(full_config)
                trainer.prepare_environment()
                trainer.run_training()
                
                end_time = time.time()
                self.profiler.stop_monitoring()
                
                results[config_name] = {
                    'total_time': end_time - start_time,
                    'metrics': self.profiler.get_metrics_summary(),
                    'config': config
                }
                
                print(f"[Benchmark] {config_name}: {end_time - start_time:.2f}s")
                
            except Exception as e:
                print(f"[Benchmark] {config_name} FAILED: {e}")
                self.profiler.stop_monitoring()
                results[config_name] = {'error': str(e)}
        
        return results
    
    def generate_comparison_report(self, output_dir: str = "performance_comparison"):
        """Generate comparison report from test results."""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "performance_comparison.txt")
        with open(report_path, 'w') as f:
            f.write("=== GA Performance Comparison Report ===\n\n")
            
            for test_name, result in self.test_results.items():
                f.write(f"=== {test_name.upper()} ===\n")
                f.write(f"Configuration: {result['config']}\n")
                
                if 'stats' in result:
                    stats = result['stats']
                    f.write(f"Best Fitness: {stats.get('best_fitness', 0):.6f}\n")
                    f.write(f"Generations: {stats.get('generations_completed', 0)}\n")
                
                if 'metrics' in result:
                    metrics = result['metrics']
                    if 'cpu_stats' in metrics:
                        f.write(f"CPU Usage: {metrics['cpu_stats']['avg']:.1f}% avg, {metrics['cpu_stats']['max']:.1f}% max\n")
                    if 'process_memory_stats' in metrics:
                        f.write(f"Memory Usage: {metrics['process_memory_stats']['max']:.1f}MB peak\n")
                    if 'gpu_stats' in metrics:
                        f.write(f"GPU Memory: {metrics['gpu_stats']['max_memory_gb']:.2f}GB peak\n")
                
                f.write("\n")
        
        print(f"[Tester] Comparison report saved: {report_path}")
        return report_path


def main():
    """Main performance testing workflow."""
    tester = GAPerformanceTester()
    
    print("GA Performance Testing Suite")
    print("1. CUDA vs CPU benchmark")
    print("2. Configuration comparison")
    print("3. Full performance analysis")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == '1':
        results = tester.benchmark_cuda_vs_cpu()
        print("\nBenchmark Results:")
        for config_name, result in results.items():
            if 'total_time' in result:
                print(f"{config_name}: {result['total_time']:.2f}s")
            else:
                print(f"{config_name}: FAILED")
    
    elif choice == '2':
        config_variations = [
            {"population_size": 10, "positions_count": 20, "use_cuda": True, "generations": 2},
            {"population_size": 20, "positions_count": 40, "use_cuda": True, "generations": 2},
            {"population_size": 30, "positions_count": 60, "use_cuda": True, "generations": 2},
        ]
        
        # Add base stockfish config to all
        for config in config_variations:
            config["stockfish_config"] = {
                "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
                "depth": 2, "movetime": 200
            }
        
        tester.test_evaluation_performance(config_variations)
        tester.generate_comparison_report()
    
    elif choice == '3':
        print("Running full performance analysis...")
        
        # CUDA benchmark
        cuda_results = tester.benchmark_cuda_vs_cpu()
        
        # Configuration testing
        config_variations = [
            {"population_size": 15, "positions_count": 30, "use_cuda": True, "generations": 2},
            {"population_size": 25, "positions_count": 50, "use_cuda": True, "generations": 2}
        ]
        
        for config in config_variations:
            config["stockfish_config"] = {
                "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
                "depth": 2, "movetime": 200
            }
        
        tester.test_evaluation_performance(config_variations)
        
        # Generate reports
        tester.generate_comparison_report()
        tester.profiler.generate_performance_report()
        
        print("Full analysis completed!")

if __name__ == "__main__":
    main()
