<<<<<<< HEAD
# v7p3r_ga_engine/v7p3r_ga.py
# v7p3r Chess Engine Genetic Algorithm Module
# TODO: refactor this code pulled from another project to fit the goals set out in the enhancement issue #84 [v7p3r AI Models] Phase 2: Genetic Algorithm Engine: Implement v7p3r_ga.py for automated engine tuning

import random
import numpy as np
import chess
import chess.pgn
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ChessDataset(Dataset):
    def __init__(self, pgn_path, username):
        self.positions = []
        self.moves = []
        
        pgn = open(pgn_path)
        while True:
            game = chess.pgn.read_game(pgn)
            if not game:
                break
            
            if game.headers["White"] == username or game.headers["Black"] == username:
                board = game.board()
                for move in game.mainline_moves():
                    if (board.turn == chess.WHITE and game.headers["White"] == username) or \
                       (board.turn == chess.BLACK and game.headers["Black"] == username):
                        self.positions.append(self.board_to_tensor(board))
                        self.moves.append(move.uci())
                    board.push(move)

    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                tensor[channel][7 - square//8][square%8] = 1
        return tensor

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.moves[idx]

class ChessAI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Convolutional layers for spatial pattern recognition
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Fully connected layers for move prediction
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Value head for position evaluation
        self.value_fc1 = nn.Linear(64 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize piece-square tables
        self.initialize_piece_tables()
        
        # Genetic parameters (will be evolved)
        self.genetic_params = {
            'material_weight': 1.0,
            'position_weight': 0.5,
            'search_depth': 2
        }
    
    def forward(self, x):
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x_flat = x.view(x.size(0), -1)
        
        # Policy head (move prediction)
        policy = F.relu(self.fc1(x_flat))
        policy = F.relu(self.fc2(policy))
        policy = self.fc3(policy)  # Raw logits
        
        # Value head (position evaluation)
        value = F.relu(self.value_fc1(x_flat))
        value = torch.tanh(self.value_fc2(value))  # Value between -1 and 1
        
        return policy, value


class GeneticAlgorithm:
    def __init__(self, population_size=30, mutation_rate=0.2, elite_count=3):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.population = []
        
    def initialize_population(self, model_template):
        """Initialize a population of models with random genetic parameters"""
        self.population = []
        for _ in range(self.population_size):
            # Create a copy of the template model
            model = copy.deepcopy(model_template)
            
            # Randomize genetic parameters
            model.genetic_params = {
                'material_weight': random.uniform(0.5, 1.5),
                'position_weight': random.uniform(0.2, 0.8),
                'search_depth': random.randint(1, 3)
            }
            
            self.population.append(model)
    
    def evaluate_fitness(self, model, games):
        """Calculate fitness based on v7p3r's games performance"""
        fitness = 0
        
        for game in games:
            # Check if v7p3r played this game
            played_white = game.headers.get("White") == "v7p3r"
            played_black = game.headers.get("Black") == "v7p3r"
            
            if not (played_white or played_black):
                continue
                
            # Parse result
            result = game.headers.get("Result")
            if result == "1-0":
                result_score = 1.0  # White win
            elif result == "0-1":
                result_score = 0.0  # Black win
            else:
                result_score = 0.5  # Draw
            
            # Higher fitness if v7p3r won
            if (played_white and result_score == 1.0) or (played_black and result_score == 0.0):
                fitness += 10
            elif result_score == 0.5:
                fitness += 2
                
            # Test model's ability to predict v7p3r's moves
            board = chess.Board()
            move_count = 0
            
            for move in game.mainline_moves():
                # If it's v7p3r's turn
                if (board.turn == chess.WHITE and played_white) or (board.turn == chess.BLACK and played_black):
                    # Predict move
                    predicted_move, _ = model.select_move(board)
                    
                    # Compare with actual move
                    if predicted_move == move:
                        fitness += 2
                
                # Apply move
                board.push(move)
                move_count += 1
                
                # Limit to first 20 moves for performance
                if move_count >= 20:
                    break
        
        return fitness
    
    def evolve_population(self, games):
        """Evolve population through selection, crossover, and mutation"""
        # Evaluate fitness for each model
        fitness_scores = []
        for model in self.population:
            fitness = self.evaluate_fitness(model, games)
            fitness_scores.append(fitness)
        
        # Pair models with their fitness scores
        model_fitness = list(zip(self.population, fitness_scores))
        
        # Sort by fitness (descending)
        model_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Keep elite models
        new_population = [model for model, _ in model_fitness[:self.elite_count]]
        
        # Create offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(model_fitness)
            parent2 = self._tournament_selection(model_fitness)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child)
            
            # Add to new population
            new_population.append(child)
        
        # Update population
        self.population = new_population
        
        # Return best model and its fitness
        return self.population[0], model_fitness[0][1]
    
    def _tournament_selection(self, model_fitness, tournament_size=3):
        """Tournament selection - select best from random subset"""
        tournament = random.sample(model_fitness, tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]  # Return the model with highest fitness
    
    def _crossover(self, parent1, parent2):
        """Crossover genetic parameters from two parents"""
        child = copy.deepcopy(parent1)
        
        # For each genetic parameter, randomly choose from parent1 or parent2
        for key in child.genetic_params:
            if random.random() < 0.5:
                child.genetic_params[key] = parent2.genetic_params[key]
        
        return child
    
    def _mutate(self, model):
        """Mutate genetic parameters with probability mutation_rate"""
        for key in model.genetic_params:
            if random.random() < self.mutation_rate:
                # Mutate the parameter
                if key == 'search_depth':
                    model.genetic_params[key] = random.randint(1, 3)
                else:
                    # For continuous parameters, add random noise
                    model.genetic_params[key] *= random.uniform(0.8, 1.2)
        
        return model
=======
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
from v7p3r_engine.v7p3r import v7p3rEvaluationEngine

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
        engine1 = v7p3rEvaluationEngine(board=board.copy(), player=True, engine_config=config1)
        engine2 = v7p3rEvaluationEngine(board=board.copy(), player=False, engine_config=config2)
        move_count = 0
        max_moves = 200  # Prevent infinite games
        while not board.is_game_over() and move_count < max_moves:
            if board.turn == chess.WHITE:
                move = engine1.search(board, chess.WHITE, engine_config=config1)
            else:
                move = engine2.search(board, chess.BLACK, engine_config=config2)
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
>>>>>>> 07a8bd8b88a40e25c3039c45e202a1c15bd0bce9
