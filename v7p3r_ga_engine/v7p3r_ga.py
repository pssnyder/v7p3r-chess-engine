"""
v7p3r_ga.py
Genetic Algorithm for v7p3r Chess Engine Configuration Optimization

This module implements a genetic algorithm (GA) to optimize the configuration
dictionary for the v7p3r chess engine. The GA evolves config dicts, using self-play
(v7p3r vs. v7p3r_opponent) as the fitness function. Only wins count as fitness.
The best config is exported as YAML, including the specified config sections.

Author: Pat Snyder
"""
import random
import copy
import yaml
import concurrent.futures
import os
import multiprocessing
from typing import Dict, Any, List
from v7p3r_engine.v7p3r_engine import v7p3rEngine

# --- Config Section Names to Export ---
V7P3R_CONFIG_SECTION = "v7p3r"
BEST_EVAL_SECTION = "best_evaluation"

class V7P3RGeneticAlgorithm:
    """
    Genetic Algorithm for optimizing v7p3r engine configuration.
    Each individual is a config dict. Fitness is the number of wins in self-play
    against a static opponent config. Only wins count as fitness.

    Parallelism: The number of games played in parallel is controlled by max_workers.
    Set this to the number of physical CPU cores or less to avoid overloading your system.
    """
    def __init__(self, base_config: Dict[str, Any],
                 opponent_config: Dict[str, Any],
                 population_size: int = 20,
                 mutation_rate: float = 0.2,
                 elite_count: int = 2,
                 games_per_individual: int = 4,
                 max_workers: int = 4):
        self.base_config = base_config
        self.opponent_config = opponent_config
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.games_per_individual = games_per_individual
        # Limit max_workers to available CPU cores
        cpu_count = multiprocessing.cpu_count()
        if max_workers > cpu_count:
            print(f"[WARNING] max_workers ({max_workers}) > CPU cores ({cpu_count}). Limiting to {cpu_count}.")
            max_workers = cpu_count
        self.max_workers = max_workers
        self.population = []  # List[Dict[str, Any]]

    def initialize_population(self):
        """Create initial population by mutating the base config."""
        self.population = [self._mutate_config(copy.deepcopy(self.base_config), force_mutate=True)
                           for _ in range(self.population_size)]

    def evaluate_fitness(self, config: Dict[str, Any]) -> int:
        """
        Run self-play games (v7p3r(config) vs. v7p3r_opponent) and return win count.
        Only wins count as fitness. Draws/losses = 0.
        """
        wins = 0
        for _ in range(self.games_per_individual):
            result = self._run_self_play_game(config, self.opponent_config)
            if result == 1:
                wins += 1
        return wins

    def evaluate_population(self) -> List[int]:
        """Evaluate fitness for the entire population in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fitnesses = list(executor.map(self.evaluate_fitness, self.population))
        return fitnesses

    def evolve_population(self, fitnesses: List[int]):
        """
        Evolve population using selection, crossover, and mutation.
        Elitism: keep top N configs. Rest are offspring.
        """
        # Pair configs with fitness and sort
        paired = list(zip(self.population, fitnesses))
        paired.sort(key=lambda x: x[1], reverse=True)
        elites = [copy.deepcopy(cfg) for cfg, _ in paired[:self.elite_count]]
        new_population = elites[:]
        # Fill rest of population
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(paired)
            parent2 = self._tournament_selection(paired)
            child = self._crossover_configs(parent1, parent2)
            child = self._mutate_config(child)
            new_population.append(child)
        self.population = new_population

    def _tournament_selection(self, paired, k=3):
        """Select one config using tournament selection."""
        sample = random.sample(paired, k)
        sample.sort(key=lambda x: x[1], reverse=True)
        return copy.deepcopy(sample[0][0])

    def _crossover_configs(self, cfg1: Dict[str, Any], cfg2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two configs (single-point or uniform crossover)."""
        child = copy.deepcopy(cfg1)
        for key in child:
            if key in cfg2 and isinstance(child[key], (int, float)):
                if random.random() < 0.5:
                    child[key] = cfg2[key]
        return child

    def _mutate_config(self, config: Dict[str, Any], force_mutate=False) -> Dict[str, Any]:
        """Randomly mutate config values (weights, bonuses, etc)."""
        new_config = copy.deepcopy(config)
        for key, value in new_config.items():
            if isinstance(value, (int, float)):
                if force_mutate or random.random() < self.mutation_rate:
                    if isinstance(value, int):
                        new_config[key] = max(1, value + random.choice([-1, 1]))
                    else:
                        new_config[key] = value * random.uniform(0.8, 1.2)
            elif isinstance(value, dict):
                new_config[key] = self._mutate_config(value, force_mutate)
        return new_config

    def _run_self_play_game(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> int:
        """
        Run a single self-play game between v7p3r(config1) and v7p3r_opponent(config2).
        Returns 1 if config1 wins, 0 otherwise (draw/loss).
        """
        import chess
        board = chess.Board()
        engine1 = v7p3rEngine(board=board.copy(), player=True)
        engine2 = v7p3rEngine(board=board.copy(), player=False)
        move_count = 0
        max_moves = 200  # Prevent infinite games
        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                move = engine1.search(board, chess.WHITE)
            else:
                move = engine2.search(board, chess.BLACK)
            if move is None or move == chess.Move.null():
                break  # No legal moves
            board.push(move)
            move_count += 1
        # Determine result
        result = board.result()
        # If config1 played as White, win is '1-0'. If as Black, win is '0-1'.
        # Here, config1 always plays White.
        if result == '1-0':
            return 1  # config1 wins
        else:
            return 0  # draw or loss

    def get_best_config(self, fitnesses: List[int]) -> Dict[str, Any]:
        """Return the config with the highest fitness."""
        idx = fitnesses.index(max(fitnesses))
        return self.population[idx]

    def export_best_config_yaml(self, best_config: Dict[str, Any], out_path: str):
        """
        Export the best config as YAML, including only the v7p3r and best_evaluation sections.
        """
        export_dict = {}
        if V7P3R_CONFIG_SECTION in best_config:
            export_dict[V7P3R_CONFIG_SECTION] = best_config[V7P3R_CONFIG_SECTION]
        if BEST_EVAL_SECTION in best_config:
            export_dict[BEST_EVAL_SECTION] = best_config[BEST_EVAL_SECTION]
        with open(out_path, "w") as f:
            yaml.dump(export_dict, f, default_flow_style=False)

# --- End of V7P3RGeneticAlgorithm class ---

# Usage example and further logic should be implemented in v7p3r_ga_training.py
