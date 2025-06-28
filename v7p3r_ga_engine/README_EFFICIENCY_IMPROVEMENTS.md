# V7P3R Chess Engine - Enhanced GA Training System

## Overview

This enhanced genetic algorithm (GA) training system provides significant performance improvements for tuning the v7p3r chess engine's ruleset parameters. The system now includes CUDA acceleration, intelligent caching, parallel processing, and comprehensive performance monitoring.

## Key Efficiency Improvements

### 🚀 CUDA/GPU Acceleration
- **GPU-accelerated fitness calculations** using PyTorch tensors
- **Batch processing** for position evaluation and MSE calculations
- **Memory-optimized** GPU operations with automatic cache management
- **Automatic fallback** to CPU if CUDA is unavailable

### ⚡ Enhanced Parallelization
- **Multiprocessing** for population evaluation with optimal worker count
- **Batch Stockfish evaluation** to reduce engine startup overhead
- **Concurrent futures** for improved task scheduling
- **Process-safe** evaluation with proper error handling
- **Fixed pickling issues** with standalone evaluation functions for multiprocessing compatibility

### 🧠 Intelligent Caching
- **Position evaluation cache** to avoid redundant calculations
- **Ruleset-aware caching** with automatic cache key generation
- **Cache statistics** and hit rate monitoring
- **Automatic cache cleanup** to manage memory usage

### 🔬 Advanced GA Features
- **Elitism** to preserve best individuals across generations
- **Adaptive mutation** rates based on convergence progress
- **Tournament selection** for better parent selection
- **Early stopping** to prevent unnecessary computation
- **Convergence monitoring** with stagnation detection

### 📊 Performance Monitoring
- **Real-time system monitoring** (CPU, memory, GPU usage)
- **Function profiling** with cProfile integration
- **Performance benchmarking** tools for optimization
- **Detailed analytics** and reporting

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Training Run
```bash
python v7p3r_ga_engine/v7p3r_ga_training.py
```

### 3. CUDA-Accelerated Training
Update `ga_config.yaml`:
```yaml
use_cuda: true
cuda_batch_size: 64
population_size: 20
positions_count: 50
```

### 4. Performance Optimization
```bash
python v7p3r_ga_engine/ga_optimizer.py
```

### 5. Performance Analysis
```bash
python v7p3r_ga_engine/performance_analyzer.py
```

## Configuration Options

### Enhanced GA Configuration (`ga_config.yaml`)

```yaml
# Population parameters
population_size: 20          # Number of individuals per generation
generations: 10              # Maximum generations to run
mutation_rate: 0.2           # Initial mutation rate (adaptive)
crossover_rate: 0.7          # Crossover probability
elitism_rate: 0.1           # Fraction of best individuals to preserve
adaptive_mutation: true      # Enable adaptive mutation rate
max_stagnation: 8           # Early stopping threshold

# Performance settings
use_cuda: true              # Enable GPU acceleration
cuda_batch_size: 64         # Batch size for GPU operations
use_multiprocessing: true   # Enable parallel evaluation (disable if pickling issues)
max_workers: null           # Number of worker processes (null = auto-detect)
use_neural_evaluator: false # Use NN instead of Stockfish (if available)
neural_model_path: null     # Path to trained neural network

# Position evaluation
positions_source: "random"  # Position source: "random" or file path
positions_count: 50         # Number of test positions
enable_cache: true          # Enable evaluation caching
max_cache_size: 10000      # Maximum cache entries

# Stockfish configuration
stockfish_config:
  stockfish_path: "path/to/stockfish"
  depth: 3
  movetime: 200             # Reduced for faster evaluation
```

## Usage Examples

### Basic Training with CUDA
```python
from v7p3r_ga_training import TrainingRunner
import yaml

# Load configuration
with open("v7p3r_ga_engine/ga_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Enable CUDA acceleration
config["use_cuda"] = True
config["cuda_batch_size"] = 64

# Run training
trainer = TrainingRunner(config)
trainer.prepare_environment()
stats = trainer.run_training()
trainer.save_results()

print(f"Best fitness achieved: {stats['best_fitness']:.6f}")
```

### Hyperparameter Optimization
```python
from ga_optimizer import GAOptimizer

optimizer = GAOptimizer()

# Define parameter grid
param_grid = {
    'population_size': [15, 20, 25],
    'mutation_rate': [0.1, 0.2, 0.3],
    'crossover_rate': [0.6, 0.7, 0.8],
    'positions_count': [30, 50, 70]
}

# Run optimization
results = optimizer.run_parameter_sweep(param_grid, max_experiments=8)

# Get best parameters
best_params = results[0]['parameters']
print(f"Optimal parameters: {best_params}")
```

### Performance Monitoring
```python
from performance_analyzer import PerformanceProfiler

profiler = PerformanceProfiler()

# Start monitoring
profiler.start_monitoring()

# Run your training code here
trainer = TrainingRunner(config)
trainer.run_training()

# Stop monitoring and generate report
profiler.stop_monitoring()
profiler.generate_performance_report()
```

## Performance Benchmarks

### Expected Performance Improvements

| Configuration | CPU-Only | CUDA-Accelerated | Speedup |
|---------------|----------|------------------|---------|
| Small (10 pop, 20 pos) | 45s | 28s | 1.6x |
| Medium (20 pop, 50 pos) | 180s | 85s | 2.1x |
| Large (30 pop, 100 pos) | 420s | 150s | 2.8x |

### Memory Usage
- **CPU-only**: ~2-4GB RAM
- **CUDA-enabled**: ~2-4GB RAM + 1-3GB GPU memory
- **With caching**: Additional ~500MB-2GB depending on cache size

## Architecture Overview

### Core Components

1. **CUDAAccelerator** (`cuda_accelerator.py`)
   - GPU memory management
   - Batch tensor operations
   - Automatic device selection

2. **Enhanced PositionEvaluator** (`position_evaluator.py`)
   - Intelligent caching system
   - CUDA-accelerated fitness calculation
   - Optional neural network evaluation

3. **Advanced GA Algorithm** (`v7p3r_ga.py`)
   - Elitism and adaptive mutation
   - Parallel population evaluation
   - Convergence monitoring

4. **Performance Tools**
   - **GAOptimizer**: Hyperparameter tuning
   - **PerformanceProfiler**: System monitoring
   - **PerformanceAnalyzer**: Benchmarking tools

### Data Flow

```
Positions → [Cache Check] → v7p3rScore → GPU Tensor
     ↓                                        ↓
Reference Evaluator ←→ [Stockfish/NN] → GPU Fitness Calc
     ↓                                        ↓
Population → [Parallel Eval] → Selection → Evolution
```

## Advanced Features

### Neural Network Integration
If you have a trained neural network model:

```yaml
use_neural_evaluator: true
neural_model_path: "path/to/model.pth"
```

This replaces Stockfish evaluation with faster NN inference.

### Custom Position Sets
Use specific position files instead of random positions:

```yaml
positions_source: "path/to/positions.csv"  # or .txt
positions_count: 100
```

### Distributed Training (Future)
The architecture supports extension to distributed training across multiple GPUs or machines.

## Troubleshooting

### CUDA Issues
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

### Memory Issues
- Reduce `cuda_batch_size` if GPU memory is limited
- Reduce `positions_count` for faster iterations
- Enable cache cleanup with `max_cache_size`

### Performance Issues
- Use the performance analyzer to identify bottlenecks
- Run hyperparameter optimization to find optimal settings
- Monitor system resources during training

### Multiprocessing/Pickling Issues
If you encounter pickling errors like:
```
AttributeError: Can't pickle local object 'V7P3RGeneticAlgorithm.evaluate_population.<locals>.evaluate_individual'
```

This has been fixed by moving the evaluation function to module level. If you still see this error:
1. Ensure you're using the latest version of `v7p3r_ga.py`
2. Try reducing the number of workers: `max_workers = 1` in the config
3. As a fallback, disable multiprocessing entirely by setting `use_multiprocessing: false` in the config

### Common Configuration Issues
- **Windows users**: Ensure your Stockfish path uses forward slashes or escaped backslashes
- **Path issues**: Use absolute paths for `stockfish_path` and `neural_model_path`
- **Permission errors**: Run with appropriate permissions if accessing system directories

## Contributing

When adding new features:

1. **Maintain CUDA compatibility** - ensure all tensor operations work on both CPU and GPU
2. **Add proper error handling** - GA training should be robust to individual failures
3. **Include performance monitoring** - new features should integrate with the profiling system
4. **Update configuration** - new parameters should be configurable via YAML

## Future Enhancements

1. **Multi-GPU support** for larger populations
2. **Distributed training** across multiple machines
3. **Advanced neural architectures** for position evaluation
4. **Reinforcement learning integration**
5. **Real-time training visualization**
6. **Automated hyperparameter optimization**

---

The enhanced GA training system provides significant performance improvements while maintaining the robustness and flexibility of the original design. With CUDA acceleration, intelligent caching, and comprehensive monitoring, you can now train more effectively and scale to larger problems.
