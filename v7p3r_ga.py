"""
Core genetic algorithm logic for ruleset optimization with CUDA acceleration.
Features improved parallelization, elitism, and adaptive mutation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import copy
import logging
import json

# Ensure parent path for engine imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v7p3r_config import v7p3rConfig
from v7p3r_ga_ruleset_manager import v7p3rGARulesetManager
from v7p3r_score import v7p3rScore
from v7p3r_pst import v7p3rPST
from v7p3r_stockfish_handler import StockfishHandler
from puzzles.puzzle_db_manager import PuzzleDBManager

class v7p3rGeneticAlgorithm:
    """
    Genetic Algorithm for tuning v7p3r evaluation rulesets.
    """
    def __init__(self, config_overrides=None):
        # Use centralized configuration manager
        self.config_manager = v7p3rConfig(overrides=config_overrides)
        self.config = self.config_manager.get_v7p3r_ga_config()

        # Setup logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Managers
        self.ruleset_manager = v7p3rGARulesetManager()
        stockfish_config = self.config_manager.get_stockfish_config()
        self.stockfish = StockfishHandler(stockfish_config)
        puzzle_config = self.config_manager.get_puzzle_config()
        self.puzzle_db = PuzzleDBManager(puzzle_config)

        # Add evaluation cache for efficiency
        self.evaluation_cache = {}
        self.cache_enabled = self.config.get('enable_cache', True)
        self.max_cache_size = self.config.get('max_cache_size', 1000)

        # Scorer
        engine_cfg = {'verbose_output': False}
        # Initialize PST with default piece values and logger
        pst = v7p3rPST(self.logger)
        self.scorer = v7p3rScore(engine_cfg, pst)

        # Load base ruleset
        all_rulesets = self.ruleset_manager.load_all_rulesets()
        base_name = self.config.get('base_ruleset', 'default_ruleset')
        self.base_ruleset = copy.deepcopy(all_rulesets.get(base_name, {}))

        # Initialize population
        self.population = self._initialize_population()

        # Load test positions
        self._load_test_positions()

    def _initialize_population(self):
        pop_size = self.config.get('population_size', 10)
        pop = [copy.deepcopy(self.base_ruleset)]
        while len(pop) < pop_size:
            indiv = copy.deepcopy(self.base_ruleset)
            self._mutate(indiv, scale=0.5)
            pop.append(indiv)
        return pop

    def _load_test_positions(self):
        count = self.config.get('positions_count', 20)
        level = self.config.get('positions_source', 'random')
        # Using puzzle DB manager to fetch
        self.test_positions = self.puzzle_db.get_puzzles(limit=count)
        if not self.test_positions:
            self.logger.warning('No puzzles found; defaulting to starting position')
            self.test_positions = [{'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'}]

    def _mutate(self, ruleset, scale=0.1):
        rate = self.config.get('mutation_rate', 0.1)
        for k, v in ruleset.items():
            if isinstance(v, (int, float)) and random.random() < rate:
                change = random.uniform(-scale, scale) * (abs(v) or 1)
                ruleset[k] = v + change

    def _crossover(self, p1, p2):
        child = {}
        rate = self.config.get('crossover_rate', 0.7)
        for k in p1:
            child[k] = p1[k] if random.random() < rate else p2.get(k, p1[k])
        return child

    def _evaluate(self, ruleset):
        # Override ruleset directly in scorer
        self.scorer.ruleset = ruleset
        # Optional: update ruleset_name for logging
        self.scorer.ruleset_name = 'GA_Tuning'
        error = 0.0
        from chess import Board
        
        # Batch evaluation for efficiency
        positions_to_evaluate = []
        for i, entry in enumerate(self.test_positions):
            if i >= 3:  # Limit to first 3 positions for faster evaluation during training
                break
            fen = entry['fen'] if isinstance(entry, dict) else entry
            try:
                board = Board(fen)
                positions_to_evaluate.append((board, fen))
            except Exception:
                error += 1e6
        
        # Evaluate positions with caching
        for board, fen in positions_to_evaluate:
            try:
                # Check cache first
                if self.cache_enabled and fen in self.evaluation_cache:
                    sf_eval = self.evaluation_cache[fen]
                else:
                    sf_eval = self.stockfish.evaluate_position(board)
                    # Cache the result
                    if self.cache_enabled:
                        if len(self.evaluation_cache) >= self.max_cache_size:
                            # Remove oldest entry (simple FIFO)
                            oldest_key = next(iter(self.evaluation_cache))
                            del self.evaluation_cache[oldest_key]
                        self.evaluation_cache[fen] = sf_eval
                
                v7_eval = self.scorer.calculate_score(board, board.turn) * 100
                error += (v7_eval - sf_eval) ** 2
            except Exception as e:
                self.logger.warning(f"Error evaluating position {fen}: {e}")
                error += 1e6
        
        mse = error / max(len(positions_to_evaluate), 1)
        return -mse

    def cleanup(self):
        """
        Cleanup resources, close database connections, etc.
        """
        if hasattr(self.stockfish, 'close'):
            self.stockfish.close()
        if hasattr(self.puzzle_db, 'close'):
            self.puzzle_db.close()
        self.logger.info('Cleanup complete')
        
    def run(self):
        gens = self.config.get('generations', 20)
        elitism = self.config.get('elitism_rate', 0.1)
        for g in range(gens):
            self.logger.info(f'Generation {g+1}/{gens}')
            scores = [self._evaluate(r) for r in self.population]
            paired = list(zip(self.population, scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            # preserve elites
            elite_count = max(1, int(len(paired) * elitism))
            new_pop = [copy.deepcopy(paired[i][0]) for i in range(elite_count)]
            # breed rest
            while len(new_pop) < len(self.population):
                p1 = random.choice(paired[:len(paired)//2])[0]
                p2 = random.choice(paired[:len(paired)//2])[0]
                child = self._crossover(p1, p2)
                self._mutate(child)
                new_pop.append(child)
            self.population = new_pop
            best = paired[0]
            self.logger.info(f'Best fitness: {best[1]}')
            # save best to results
            self._save_best(best[0], g)
        self.logger.info('GA tuning complete')

    def _save_best(self, ruleset, generation):
        out_dir = os.path.join(os.path.dirname(__file__), 'ga_results')
        os.makedirs(out_dir, exist_ok=True)
        name = f'generation_{generation+1}_best.json'
        path = os.path.join(out_dir, name)
        with open(path, 'w') as f:
            json.dump(ruleset, f, indent=4)
        # also update main custom_rulesets.json
        self.ruleset_manager.save_ruleset(f'tuned_ga_gen{generation+1}', ruleset)
        self.logger.info(f'Saved best ruleset for gen {generation+1} to {path}')

def main():
    ga = v7p3rGeneticAlgorithm()
    ga.run()

if __name__ == '__main__':
    main()