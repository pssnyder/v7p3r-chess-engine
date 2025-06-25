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
