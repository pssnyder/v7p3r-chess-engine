"""
Orchestrates the GA training process.
"""

import os
import sys
import yaml
import json
import logging
import copy
import random

# Add the parent directory to the Python path first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v7p3r_ga_engine.v7p3r_ga import v7p3rGeneticAlgorithm
from v7p3r_ga_engine.v7p3r_ga_ruleset_manager import v7p3rGARulesetManager
from puzzles.puzzle_db_manager import PuzzleDBManager

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'ga_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize managers
    ruleset_manager = v7p3rGARulesetManager()
    # Pass the path to a YAML config file or None for default
    puzzle_db_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'puzzle_config.yaml'))
    print(f"[DEBUG] Using puzzle DB config: {puzzle_db_config_path}")
    puzzle_db = PuzzleDBManager(puzzle_db_config_path)
    ga = v7p3rGeneticAlgorithm(config_path)

    # Debug: check if test_positions were loaded with fallback
    if hasattr(ga, 'test_positions') and ga.test_positions:
        if len(ga.test_positions) == 1 and ga.test_positions[0].get('fen', '').startswith('rnbqkbnr'):
            print("[DEBUG] v7p3rGeneticAlgorithm loaded fallback starting position as test_positions. This may indicate no puzzles were found at GA init.")
        else:
            print(f"[DEBUG] v7p3rGeneticAlgorithm loaded {len(ga.test_positions)} test positions at init.")

    results_dir = os.path.join(os.path.dirname(__file__), 'ga_results')
    os.makedirs(results_dir, exist_ok=True)

    generations = config.get('generations', 3)
    positions_per_gen = config.get('positions_count', 5)

    try:
        for gen in range(generations):
            logger.info(f'=== Generation {gen+1}/{generations} ===')
            # Fetch new random FENs for this generation (reduced count)
            positions_per_gen_actual = min(positions_per_gen, 3)  # Cap at 3 for speed
            fens = puzzle_db.get_random_fens(count=positions_per_gen_actual)
            print(f"[DEBUG] Generation {gen+1}: Retrieved {len(fens)} FENs from DB.")
            if not fens:
                logger.error(f"No FENs found in the database for generation {gen+1}. Check DB path and data.")
                # Try to fetch a few puzzles directly for debug
                sample_puzzles = puzzle_db.get_puzzles(limit=5)
                print(f"[DEBUG] Sample puzzles from DB: {sample_puzzles}")
                raise RuntimeError("No FENs found in the puzzle database!")
            ga.test_positions = [{'fen': fen} for fen in fens]

            print(f"[DEBUG] Generation {gen+1}: Starting evaluation of {len(ga.population)} individuals...")
            import time
            eval_start = time.time()
            scores = []
            for idx, r in enumerate(ga.population):
                print(f"[DEBUG] Evaluating individual {idx+1}/{len(ga.population)}...")
                s = ga._evaluate(r)
                print(f"[DEBUG] Individual {idx+1} score: {s}")
                scores.append(s)
            eval_end = time.time()
            print(f"[DEBUG] Generation {gen+1}: Evaluation complete in {eval_end - eval_start:.2f} seconds.")

            paired = list(zip(ga.population, scores))
            paired.sort(key=lambda x: x[1], reverse=True)
            best_ruleset, best_score = paired[0]
            print(f"[DEBUG] Generation {gen+1}: Best score: {best_score}")

            # Record results (simplified)
            gen_result = {
                'generation': gen+1,
                'fens': fens,
                'ruleset': best_ruleset,
                'fitness': best_score,
                'evals': []
            }
            # Skip detailed evaluation recording for speed during debugging
            if len(fens) <= 3:  # Only record if small number of positions
                from chess import Board
                for fen in fens:
                    try:
                        board = Board(fen)
                        sf_eval = ga.stockfish.evaluate_position(board)
                        v7_eval = ga.scorer.calculate_score(board, board.turn) * 100
                        gen_result['evals'].append({'fen': fen, 'stockfish': sf_eval, 'v7p3r': v7_eval})
                    except Exception as e:
                        print(f"[DEBUG] Error evaluating {fen}: {e}")

            # Save generation result
            with open(os.path.join(results_dir, f'generation_{gen+1}_results.json'), 'w') as f:
                json.dump(gen_result, f, indent=2)

            print(f"[DEBUG] Generation {gen+1}: Evolving population...")
            elitism = ga.config.get('elitism_rate', 0.1)
            elite_count = max(1, int(len(paired) * elitism))
            new_pop = [copy.deepcopy(paired[i][0]) for i in range(elite_count)]
            while len(new_pop) < len(ga.population):
                p1 = random.choice(paired[:len(paired)//2])[0]
                p2 = random.choice(paired[:len(paired)//2])[0]
                child = ga._crossover(p1, p2)
                ga._mutate(child)
                new_pop.append(child)
            ga.population = new_pop
            print(f"[DEBUG] Generation {gen+1}: Population evolved. New population size: {len(ga.population)}")

            # Save best ruleset
            ruleset_manager.save_ruleset(f'tuned_ga_gen{gen+1}', best_ruleset)
            print(f"[DEBUG] Generation {gen+1}: Best ruleset saved.")

        logger.info('GA training complete.')
    finally:
        # Clean up Stockfish process if possible
        if hasattr(ga, 'stockfish') and hasattr(ga.stockfish, 'close'):
            ga.stockfish.close()

if __name__ == "__main__":
    main()
