"""
Orchestrates the GA training process.
"""

import os
import sys
import yaml
import json
import logging
from v7p3r_ga import v7p3rGeneticAlgorithm
from v7p3r_ga_ruleset_manager import GARulesetManager
from puzzles.puzzle_db_manager import PuzzleDBManager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'ga_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize managers
    ruleset_manager = GARulesetManager()
    puzzle_db = PuzzleDBManager(config.get('puzzle_db_config', {}))
    ga = v7p3rGeneticAlgorithm(config_path)

    results_dir = os.path.join(os.path.dirname(__file__), 'ga_results')
    os.makedirs(results_dir, exist_ok=True)

    generations = config.get('generations', 3)
    positions_per_gen = config.get('positions_count', 5)

    for gen in range(generations):
        logger.info(f'=== Generation {gen+1}/{generations} ===')
        # Fetch new random FENs for this generation
        fens = puzzle_db.get_random_fens(count=positions_per_gen)
        ga.test_positions = [{'fen': fen} for fen in fens]

        # Evaluate and evolve
        scores = [ga._evaluate(r) for r in ga.population]
        paired = list(zip(ga.population, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        best_ruleset, best_score = paired[0]

        # Record results
        gen_result = {
            'generation': gen+1,
            'fens': fens,
            'ruleset': best_ruleset,
            'fitness': best_score,
            'evals': []
        }
        from chess import Board
        for fen in fens:
            board = Board(fen)
            sf_eval = ga.stockfish.evaluate_position(board)
            v7_eval = ga.scorer.calculate_score(board, board.turn) * 100
            gen_result['evals'].append({'fen': fen, 'stockfish': sf_eval, 'v7p3r': v7_eval})

        # Save generation result
        with open(os.path.join(results_dir, f'generation_{gen+1}_results.json'), 'w') as f:
            json.dump(gen_result, f, indent=2)

        # Evolve population
        ga.population = ga._initialize_population()  # Or use your evolve logic

        # Save best ruleset
        ruleset_manager.save_ruleset(f'tuned_ga_gen{gen+1}', best_ruleset)

    logger.info('GA training complete.')

if __name__ == "__main__":
    main()
