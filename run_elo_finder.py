#!/usr/bin/env python
# run_elo_finder.py
"""
Simplified script to run the Stockfish ELO Finder simulation.
This allows users to quickly determine the ELO rating of their V7P3R engine configuration.
"""

import argparse
import logging
import yaml
import os
from engine_utilities.adaptive_elo_finder import AdaptiveEloSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run the Stockfish ELO Finder simulation for V7P3R chess engine')
    
    # Core parameters
    parser.add_argument('--initial-elo', type=int, default=1500,
                        help='Starting ELO for Stockfish (default: 1500)')
    parser.add_argument('--min-elo', type=int, default=800,
                        help='Minimum ELO to test (default: 800)')
    parser.add_argument('--max-elo', type=int, default=3200,
                        help='Maximum ELO to test (default: 3200)')
    parser.add_argument('--max-games', type=int, default=100,
                        help='Maximum number of games to play (default: 100)')
    parser.add_argument('--min-games', type=int, default=20,
                        help='Minimum games before checking convergence (default: 20)')
    
    # Advanced parameters
    parser.add_argument('--adjustment-factor', type=float, default=1.0,
                        help='Controls how aggressively ELO changes (default: 1.0)')
    parser.add_argument('--convergence-threshold', type=float, default=0.05,
                        help='Win rate stability threshold (default: 0.05)')
    parser.add_argument('--use-central-storage', action='store_true',
                        help='Use central database storage for results')
    
    # V7P3R configuration
    parser.add_argument('--v7p3r-depth', type=int, default=None,
                        help='Search depth for V7P3R engine')
    parser.add_argument('--v7p3r-ruleset', type=str, default=None,
                        choices=['aggressive_evaluation', 'conservative_evaluation', 'balanced_evaluation'],
                        help='Evaluation ruleset for V7P3R engine')
    parser.add_argument('--v7p3r-config', type=str, default=None,
                        help='Path to custom V7P3R configuration YAML file')
    
    args = parser.parse_args()
    
    # Load custom V7P3R config if provided
    v7p3r_config = {}
    if args.v7p3r_config and os.path.exists(args.v7p3r_config):
        with open(args.v7p3r_config, 'r') as f:
            v7p3r_config = yaml.safe_load(f)
    
    # Override with command line options if provided
    if not 'v7p3r' in v7p3r_config:
        v7p3r_config['v7p3r'] = {}
    
    if args.v7p3r_depth:
        v7p3r_config['v7p3r']['depth'] = args.v7p3r_depth
    
    if args.v7p3r_ruleset:
        v7p3r_config['v7p3r']['ruleset'] = args.v7p3r_ruleset
    
    # Create and run the simulator
    simulator = AdaptiveEloSimulator(
        initial_elo=args.initial_elo,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        adjustment_factor=args.adjustment_factor,
        convergence_threshold=args.convergence_threshold,
        min_games_for_convergence=args.min_games,
        max_games=args.max_games,
        v7p3r_config=v7p3r_config,
        use_central_storage=args.use_central_storage
    )
    
    logger.info("Starting Stockfish ELO Finder simulation")
    logger.info(f"V7P3R config: {v7p3r_config}")
    
    results = simulator.run_simulation()
    
    # Print summary
    print("\n" + "="*60)
    print("STOCKFISH ELO FINDER RESULTS")
    print("="*60)
    print(f"V7P3R estimated ELO: {results['v7p3r_estimated_elo']}")
    print(f"Games played: {results['games_played']}")
    print(f"Win rate: {results['win_rate']*100:.1f}%")
    print(f"Final Stockfish ELO: {results['final_elo']}")
    print(f"Converged: {'Yes' if results['converged'] else 'No - max games reached'}")
    print("="*60)
    print(f"Detailed results saved to: games/{results['id']}_results.yaml")
    
if __name__ == "__main__":
    main()
