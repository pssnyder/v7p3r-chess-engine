# engine_utilities/engine_benchmark.py
# TODO - Implement the engine benchmark suite to evaluate the v7p3r chess engine's performance across various configurations on various tasks such as adaptive ELO finding and puzzle solving.

import os
import yaml
from engine_utilities.puzzle_solver import PuzzleSolver
from engine_utilities.adaptive_elo_finder import AdaptiveEloSimulator

class EngineBenchmark:
    def __init__(self, config_path="config/engine_benchmark_config.yaml"):
        self.config = self._load_config(config_path)
        self.engine_config = self.config.get('engine_config', {})
        self.puzzle_solver = PuzzleSolver(config_path=self.config.get('puzzle_config', 'config/puzzle_config.yaml'))
        self.elo_simulator = AdaptiveEloSimulator(
            v7p3r_config=self.engine_config.get('v7p3r', {}),
            game_config=self.engine_config.get('chess_game', {})
        )

    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def run_benchmark(self):
        print("Running Stockfish ELO (adaptive) benchmark...")
        stockfish_elo_result = self.elo_simulator.run_simulation()
        print("Stockfish ELO result:", stockfish_elo_result)

        print("Running Puzzle ELO benchmark...")
        puzzle_elo_results = self.puzzle_solver.solve_puzzles()
        print("Puzzle ELO results:", puzzle_elo_results)

        # Output results to file
        out_dir = self.config.get('output_dir', 'benchmark_results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"benchmark_{self.engine_config.get('name', 'engine')}_{self._timestamp()}.yaml")
        with open(out_path, 'w') as f:
            yaml.dump({
                'stockfish_elo': stockfish_elo_result,
                'puzzle_elo': puzzle_elo_results,
                'engine_config': self.engine_config
            }, f)
        print(f"Benchmark results saved to {out_path}")

    def _timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    benchmark = EngineBenchmark()
    benchmark.run_benchmark()