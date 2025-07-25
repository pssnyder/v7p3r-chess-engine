# v7p3r Chess Engine

A sophisticated chess engine featuring multiple AI approaches including traditional search algorithms, neural networks, and genetic algorithm optimization.

## Core Engine Features

- **Multiple AI Engines:** Traditional v7p3r engine, neural network (NN), and genetic algorithm (GA) variants
- **Advanced Search:** Alpha-beta pruning, move ordering, and time management
- **Position Evaluation:** Multiple evaluation functions with piece-square tables
- **Opening Book:** Built-in opening book support for improved early game play
- **Flexible Configuration:** YAML-based configuration for all engine components
- **Performance Analytics:** Comprehensive metrics and benchmarking tools

## Quick Start

### Core Engine

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Play against the engine:
   ```bash
   python v7p3r_engine/play_v7p3r.py
   ```

3. Run the web interface:
   ```bash
   python web_applications/v7p3r_webapp.py
   ```

### Neural Network Engine

1. Train a new NN model:
   ```bash
   python v7p3r_nn_engine/v7p3r_nn_training.py
   ```

2. Validate NN performance:
   ```bash
   python v7p3r_nn_engine/v7p3r_nn_validation.py
   ```

### Genetic Algorithm Engine

1. Run GA optimization:
   ```bash
   python v7p3r_ga_engine/ga_optimizer.py
   ```

2. Analyze GA performance:
   ```bash
   python v7p3r_ga_engine/performance_analyzer.py
   ```

### ELO Testing with Stockfish

Determine the ELO strength of your v7p3r configuration:

1. Quick run with default settings:
   ```bash
   python engine_utilities/run_elo_finder.py
   ```

2. Customize parameters:
   ```bash
   python engine_utilities/run_elo_finder.py --initial-elo 1500 --v7p3r-depth 4 --v7p3r-ruleset aggressive_evaluation
   ```

3. For more options and advanced usage:
   ```bash
   python engine_utilities/run_elo_finder.py --help
   ```

See [TEST_GUIDE.md](docs/TEST_GUIDE.md) for detailed instructions and interpretation of ELO results.

---

## Core Engine Components

### v7p3r_engine (Traditional Engine)
- `v7p3r.py` ΓÇö Main engine logic and UCI interface
- `v7p3r_search.py` ΓÇö Alpha-beta search implementation  
- `v7p3r_score.py` ΓÇö Position evaluation and scoring
- `v7p3r_ordering.py` ΓÇö Move ordering for search optimization
- `v7p3r_book.py` ΓÇö Opening book implementation
- `v7p3r_time.py` ΓÇö Time management for tournament play
- `v7p3r_pst.py` ΓÇö Piece-square tables for evaluation
- `stockfish_handler.py` ΓÇö Interface for Stockfish integration
- `play_v7p3r.py` ΓÇö Interactive play interface
- `rulesets.yaml` ΓÇö Configuration for different playing styles

### v7p3r_nn_engine (Neural Network Engine)
- `v7p3r_nn.py` ΓÇö Neural network engine implementation
- `v7p3r_nn_training.py` ΓÇö Training pipeline for NN models
- `v7p3r_nn_validation.py` ΓÇö Model validation and testing
- `v7p3r_nn_models/` ΓÇö Trained neural network models
- `v7p3r_nn_move_vocab/` ΓÇö Move vocabulary for NN training

### v7p3r_ga_engine (Genetic Algorithm Engine)
- `v7p3r_ga.py` ΓÇö Genetic algorithm engine implementation
- `ga_optimizer.py` ΓÇö GA optimization and evolution logic
- `v7p3r_ga_training.py` ΓÇö GA training and population management
- `position_evaluator.py` ΓÇö GA-optimized position evaluation
- `performance_analyzer.py` ΓÇö Performance analysis tools
- `ruleset_manager.py` ΓÇö Dynamic ruleset management
- `cuda_accelerator.py` ΓÇö GPU acceleration for GA operations
- `v7p3r_ga_models/` ΓÇö Evolved GA models and configurations

### Support Systems
- `metrics/chess_metrics.py` ΓÇö Engine performance metrics dashboard
- `metrics/metrics_store.py` ΓÇö Metrics database and storage
- `engine_utilities/` ΓÇö Benchmarking, monitoring, and utility tools
- `config/` ΓÇö YAML configuration files for all components

---

## Testing

Run comprehensive tests for engine components:

- **Unit Tests:** Each engine component has corresponding test files
- **Integration Tests:** Full engine testing with various configurations
- **Performance Tests:** ELO rating and benchmark comparisons

Run individual component tests:
```bash
# Test traditional engine
python -m pytest v7p3r_engine/ -v

# Test neural network engine  
python -m pytest v7p3r_nn_engine/ -v

# Test genetic algorithm engine
python -m pytest v7p3r_ga_engine/ -v

# Test utilities and metrics
python -m pytest engine_utilities/ metrics/ -v
```

See [UNIT_TESTING_GUIDE.md](docs/UNIT_TESTING_GUIDE.md) for detailed testing procedures.

---

## Configuration

The engine uses YAML-based configuration files in the `config/` directory:

- `v7p3r_nn_config.yaml` ΓÇö Neural network training and model parameters
- `v7p3r_ga_config.yaml` ΓÇö Genetic algorithm evolution settings  
- `v7p3r_rl_config.yaml` ΓÇö Reinforcement learning configuration
- `stockfish_config.yaml` ΓÇö Stockfish integration settings
- `engine_utilities_config.yaml` ΓÇö Benchmarking and utility settings
- `chess_metrics_config.yaml` ΓÇö Metrics collection and analysis

Customize engine behavior by editing the appropriate configuration files before running.

---

## Engine Comparison

The v7p3r engine offers three distinct approaches:

1. **Traditional Engine (v7p3r_engine):** Classic alpha-beta search with hand-crafted evaluation
   - Fast and deterministic
   - Well-suited for tactical positions
   - Configurable search depth and time controls

2. **Neural Network Engine (v7p3r_nn_engine):** Deep learning-based evaluation
   - Learns from large datasets of master games
   - Strong positional understanding
   - Requires training time but adapts to playing styles

3. **Genetic Algorithm Engine (v7p3r_ga_engine):** Evolutionary optimization
   - Self-improving through evolution
   - Discovers novel evaluation strategies
   - GPU-accelerated for faster evolution

## Advanced Features

- **Multi-Engine Support:** Run multiple engine variants simultaneously
- **Performance Analytics:** Detailed metrics on search efficiency and evaluation accuracy  
- **Opening Book Integration:** Comprehensive opening theory database
- **Time Management:** Tournament-ready time control handling
- **UCI Protocol:** Compatible with standard chess interfaces
- **Cloud Storage:** Firebase integration for model and game storage

---

## Example Usage

### Traditional Engine
```bash
# Play a game against the traditional engine
python v7p3r_engine/play_v7p3r.py

# Analyze a position
python v7p3r_engine/v7p3r.py --analyze "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

### Neural Network Engine  
```bash
# Train a new model
python v7p3r_nn_engine/v7p3r_nn_training.py --epochs 100 --batch-size 64

# Validate model performance
python v7p3r_nn_engine/v7p3r_nn_validation.py --model latest
```

### Genetic Algorithm Engine
```bash  
# Run evolution for 50 generations
python v7p3r_ga_engine/ga_optimizer.py --generations 50 --population 100

# Analyze best performers
python v7p3r_ga_engine/performance_analyzer.py --top 10
```

### Benchmarking
```bash
# Compare all engines against Stockfish
python engine_utilities/engine_benchmark.py --engines all --depth 6

# Find ELO ratings
python engine_utilities/run_elo_finder.py --games 100
```

---

## License

Open source ΓÇö feel free to use and modify!

### Analytics ETL System

The v7p3r Chess Engine includes a robust ETL (Extract, Transform, Load) system for analytics:

1. Run the ETL process to transform raw game data into analytics-ready format:

    ```bash
    python -m engine_utilities.etl_processor
    ```

2. Set up scheduled ETL jobs:

    ```bash
    python -m engine_utilities.etl_scheduler --setup-local
    ```

3. Monitor ETL performance:

    ```bash
    python -m engine_utilities.etl_monitor --job-history
    ```

See [ETL System Documentation](docs/etl_system.md) for details on the analytics architecture.
