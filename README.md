# v7p3r Chess Engine

A sophisticated chess engine featuring advanced search algorithms, position evaluation, and opening book support.

## Core Engine Features

- **Advanced Search:** Alpha-beta pruning, move ordering, and time management
- **Position Evaluation:** Multiple evaluation functions with piece-square tables
- **Opening Book:** Built-in opening book support for improved early game play
- **Interactive Play:** Play against the engine using a simple interface
- **Flexible Configuration:** YAML-based configuration for engine components

## Quick Start

### Core Engine

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Play against the engine:
   ```bash
   python v7p3r_play.py
   ```

3. Analyze a position:
   ```bash
   python v7p3r.py --analyze "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
   ```

---

## Core Engine Components

### v7p3r_engine (Traditional Engine)
- `v7p3r.py` — Main engine logic and UCI interface
- `v7p3r_search.py` — Alpha-beta search implementation  
- `v7p3r_score.py` — Position evaluation and scoring
- `v7p3r_ordering.py` — Move ordering for search optimization
- `v7p3r_book.py` — Opening book implementation
- `v7p3r_time.py` — Time management for tournament play
- `v7p3r_pst.py` — Piece-square tables for evaluation
- `stockfish_handler.py` — Interface for Stockfish integration
- `v7p3r_play.py` — Interactive play interface
- `rulesets.yaml` — Configuration for different playing styles

---

## Testing

Run comprehensive tests for engine components:

- **Unit Tests:** Each engine component has corresponding test files
- **Integration Tests:** Full engine testing with various configurations

Run individual component tests:
```bash
# Test traditional engine
python -m pytest ./ -v
```

---

## Configuration

The engine uses YAML-based configuration files in the `config/` directory:

- `rulesets.yaml` — Configuration for different playing styles

Customize engine behavior by editing the appropriate configuration files before running.

---

## Example Usage

### Traditional Engine
```bash
# Play a game against the traditional engine
python v7p3r_play.py

# Analyze a position
python v7p3r.py --analyze "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

---

## License

Open source — feel free to use and modify!
2. Customize parameters:
   ```bash
   python engine_utilities/run_elo_finder.py --initial-elo 1500 --v7p3r-depth 4 --v7p3r-ruleset aggressive_ruleset
   ```

3. For more options and advanced usage:
   ```bash
   python engine_utilities/run_elo_finder.py --help
   ```

See [TEST_GUIDE.md](docs/TEST_GUIDE.md) for detailed instructions and interpretation of ELO results.

---

## Core Engine Components

### v7p3r_engine (Traditional Engine)
- `v7p3r.py` — Main engine logic and UCI interface
- `v7p3r_search.py` — Alpha-beta search implementation  
- `v7p3r_score.py` — Position evaluation and scoring
- `v7p3r_ordering.py` — Move ordering for search optimization
- `v7p3r_book.py` — Opening book implementation
- `v7p3r_time.py` — Time management for tournament play
- `v7p3r_pst.py` — Piece-square tables for evaluation
- `stockfish_handler.py` — Interface for Stockfish integration
- `v7p3r_play.py` — Interactive play interface
- `rulesets.yaml` — Configuration for different playing styles

### v7p3r_nn_engine (Neural Network Engine)
- `v7p3r_nn.py` — Neural network engine implementation
- `v7p3r_nn_training.py` — Training pipeline for NN models
- `v7p3r_nn_validation.py` — Model validation and testing
- `v7p3r_nn_models/` — Trained neural network models
- `v7p3r_nn_move_vocab/` — Move vocabulary for NN training

### v7p3r_ga_engine (Genetic Algorithm Engine)
- `v7p3r_ga.py` — Genetic algorithm engine implementation
- `ga_optimizer.py` — GA optimization and evolution logic
- `v7p3r_ga_training.py` — GA training and population management
- `position_evaluator.py` — GA-optimized position evaluation
- `performance_analyzer.py` — Performance analysis tools
- `ruleset_manager.py` — Dynamic ruleset management
- `cuda_accelerator.py` — GPU acceleration for GA operations
- `v7p3r_ga_models/` — Evolved GA models and configurations

### v7p3r_rl_engine (Reinforcement Learning Engine)
- `v7p3r_rl.py` — RL engine implementation
- `v7p3r_rl_training.py` — Training pipeline for RL models
- `v7p3r_rl_evaluation.py` — Evaluation and testing of RL models
- `v7p3r_rl_models/` — Trained RL models and configurations

### Metrics
- `metrics/v7p3r_chess_metrics.py` — Engine performance metrics dashboard
- `metrics/metrics_store.py` — Metrics database and storage
- `metrics/elo_tracker.py` — ELO tracking and analysis tools

### Puzzles
- `puzzle_solver.py` — Puzzle-solving engine
- `puzzle_generator.py` — Puzzle generation tools
- `sample_puzzles.yaml` — Example puzzles for testing
- `generated_puzzles.yaml` — Generated puzzles for training

### Support Systems
- `engine_utilities/` — Benchmarking, monitoring, and utility tools
- `config/` — YAML configuration files for all components

---

## Testing

Run comprehensive tests for engine components:

- **Unit Tests:** Each engine component has corresponding test files
- **Integration Tests:** Full engine testing with various configurations
- **Performance Tests:** ELO rating and benchmark comparisons

Run individual component tests:
```bash
# Test traditional engine
python -m pytest ./ -v

# Test neural network engine  
python -m pytest v7p3r_nn_engine/ -v

# Test genetic algorithm engine
python -m pytest v7p3r_ga_engine/ -v

# Test reinforcement learning engine
python -m pytest v7p3r_rl_engine/ -v

# Test puzzles
python -m pytest puzzles/ -v

# Test utilities and metrics
python -m pytest engine_utilities/ metrics/ -v
```

See [UNIT_TESTING_GUIDE.md](docs/UNIT_TESTING_GUIDE.md) for detailed testing procedures.

---

## Configuration

JSON config

---

## Engine Comparison

The v7p3r engine offers four distinct approaches:

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

4. **Reinforcement Learning Engine (v7p3r_rl_engine):** RL-based evaluation and decision-making
   - Learns through self-play and reward systems
   - Adapts dynamically to opponents
   - Requires extensive training but offers high adaptability

## Advanced Features

- **Multi-Engine Support:** Run multiple engine variants simultaneously
- **Performance Analytics:** Detailed metrics on search efficiency and evaluation accuracy  
- **Opening Book Integration:** Comprehensive opening theory database
- **Puzzle Solver:** Tactical puzzle-solving and generation tools
- **Time Management:** Tournament-ready time control handling
- **UCI Protocol:** Compatible with standard chess interfaces
- **Cloud Storage:** Firebase integration for model and game storage

---

## Example Usage

### Traditional Engine
```bash
# Play a game against the traditional engine
python v7p3r_play.py

# Analyze a position
python v7p3r.py --analyze "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
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

### Reinforcement Learning Engine
```bash
# Train RL models
python v7p3r_rl_engine/v7p3r_rl_training.py --episodes 1000

# Evaluate RL models
python v7p3r_rl_engine/v7p3r_rl_evaluation.py --model latest
```

### Puzzle Solver
```bash
# Solve puzzles
python puzzles/puzzle_solver.py --input puzzles/sample_puzzles.yaml

# Generate puzzles
python puzzles/puzzle_generator.py --output puzzles/generated_puzzles.yaml
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

Open source — feel free to use and modify!

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
