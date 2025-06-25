<<<<<<< HEAD

import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import numpy as np
import yaml
import pickle
import os
from v7p3r_ga_engine.v7p3r_ga import ChessDataset, GeneticAlgorithm, ChessAI

torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner

# Load configuration
with open("../config/v7p3r_ga_config.yaml") as f:
    config = yaml.safe_load(f)

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_games(pgn_path):
    """Load games from PGN file"""
    games = []
    with open(pgn_path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games

def filter_v7p3r_games(games):
    """Filter games played by v7p3r"""
    v7p3r_games = []
    for game in games:
        if game.headers.get("White") == "v7p3r" or game.headers.get("Black") == "v7p3r":
            v7p3r_games.append(game)
    return v7p3r_games

def train_supervised_model(dataset, num_classes, epochs=10, batch_size=64):
    """Train a supervised model on the dataset before genetic evolution"""
    model = ChessAI(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for positions, moves in dataloader:
            positions = positions.to(device)
            moves = moves.to(device)
            
            # Forward pass
            policy, _ = model(positions)
            loss = criterion(policy, moves)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(policy.data, 1)
            total += moves.size(0)
            correct += (predicted == moves).sum().item()
            total_loss += loss.item()
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, "
              f"Accuracy: {100 * correct / total:.2f}%")
    
    return model

def train_model():
    """Complete training pipeline with supervised learning and genetic evolution"""
    # Load data
    dataset = ChessDataset("games.pgn", "v7p3r")
    
    # Create move vocabulary
    move_to_index = {move: idx for idx, move in enumerate(np.unique(dataset.moves))}
    num_classes = len(move_to_index)
    
    # Save the move vocabulary
    with open("move_vocab.pkl", "wb") as f:
        pickle.dump(move_to_index, f)
    
    # Convert string moves to indices
    move_indices = [move_to_index[move] for move in dataset.moves]
    dataset.moves = move_indices
    
    print(f"Dataset created with {len(dataset)} positions and {num_classes} possible moves")
    
    # First, train with supervised learning to get a decent starting point
    pretrained_model = train_supervised_model(dataset, num_classes, epochs=5)
    
    # Load games for genetic algorithm fitness evaluation
    all_games = load_games("games.pgn")
    v7p3r_games = filter_v7p3r_games(all_games)
    print(f"Found {len(v7p3r_games)} games played by v7p3r")
    
    # Initialize genetic algorithm
    ga = GeneticAlgorithm(population_size=20)
    ga.initialize_population(pretrained_model)
    
    # Run genetic evolution
    num_generations = 30
    best_fitness_history = []
    
    for generation in range(num_generations):
        print(f"Generation {generation+1}/{num_generations}")
        
        # Evolve population
        best_model, best_fitness = ga.evolve_population(v7p3r_games)
        best_fitness_history.append(best_fitness)
        
        print(f"Best fitness: {best_fitness}")
        print(f"Best model genetic params: {best_model.genetic_params}")
        
        # Save best model every 5 generations
        if (generation + 1) % 5 == 0:
            torch.save(best_model.state_dict(), f"v7p3r_chess_gen_{generation+1}.pth")
    
    # Save final model
    torch.save(best_model.state_dict(), "v7p3r_chess_genetic_model.pth")
    print("Training complete!")
    
    # Plot fitness history
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_generations+1), best_fitness_history)
        plt.title("Best Fitness per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.savefig("fitness_history.png")
        print("Fitness history plot saved")
    except:
        print("Could not create fitness history plot")

if __name__ == "__main__":
    train_model()
=======
import yaml
import os
from v7p3r_ga_engine.v7p3r_ga import V7P3RGeneticAlgorithm

# --- Load base and opponent configs from YAML ---
CONFIG_PATH = os.path.join("..", "config", "v7p3r_config.yaml")
EXPORT_PATH = "v7p3r_ga_best_config.yaml"

with open(CONFIG_PATH, "r") as f:
    config_data = yaml.safe_load(f)

base_config = config_data.get("v7p3r", {})
opponent_config = config_data.get("v7p3r_opponent", {})

# Optionally, also include best_evaluation section in base config for mutation
if "best_evaluation" in config_data:
    base_config = {**base_config, "best_evaluation": config_data["best_evaluation"]}

# --- GA Parameters ---
POP_SIZE = 20
MUT_RATE = 0.2
ELITE_COUNT = 2
GAMES_PER_INDIV = 4
GENERATIONS = 30
MAX_WORKERS = 4


def main():
    print("Initializing v7p3r Genetic Algorithm...")
    ga = V7P3RGeneticAlgorithm(
        base_config=base_config,
        opponent_config=opponent_config,
        population_size=POP_SIZE,
        mutation_rate=MUT_RATE,
        elite_count=ELITE_COUNT,
        games_per_individual=GAMES_PER_INDIV,
        max_workers=MAX_WORKERS,
    )
    ga.initialize_population()
    best_fitness_history = []

    for gen in range(GENERATIONS):
        print(f"Generation {gen+1}/{GENERATIONS}")
        fitnesses = ga.evaluate_population()
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)
        print(f"  Best fitness: {best_fitness}")
        ga.evolve_population(fitnesses)

    # Export best config
    best_config = ga.get_best_config(fitnesses)
    ga.export_best_config_yaml(best_config, EXPORT_PATH)
    print(f"Best config exported to {EXPORT_PATH}")

    # Optionally plot fitness history
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, GENERATIONS+1), best_fitness_history)
        plt.title("Best Fitness per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Fitness (Wins)")
        plt.grid(True)
        plt.savefig("fitness_history.png")
        print("Fitness history plot saved as fitness_history.png")
    except ImportError:
        print("matplotlib not installed, skipping plot.")

if __name__ == "__main__":
    main()
>>>>>>> 07a8bd8b88a40e25c3039c45e202a1c15bd0bce9
