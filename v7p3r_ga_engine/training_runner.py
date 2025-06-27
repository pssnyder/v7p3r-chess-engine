"""
Orchestrates the GA training process.
"""


import os
from ruleset_manager import RulesetManager
from position_evaluator import PositionEvaluator
from v7p3r_ga import V7P3RGeneticAlgorithm

class TrainingRunner:
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.ruleset_manager = RulesetManager(config.get("ruleset_path", "../v7p3r_engine/rulesets.yaml"))
        self.stockfish_config = config.get("stockfish_config", {})
        # Import v7p3rScore dynamically to avoid circular imports
        from v7p3r_engine.v7p3r_score import v7p3rScore
        self.position_evaluator = PositionEvaluator(self.stockfish_config, v7p3r_score_class=v7p3rScore)
        self.base_ruleset = self.ruleset_manager.load_ruleset(config.get("base_ruleset", "default_evaluation"))
        self.ga = V7P3RGeneticAlgorithm(
            base_ruleset=self.base_ruleset,
            ruleset_manager=self.ruleset_manager,
            position_evaluator=self.position_evaluator,
            population_size=config.get("population_size", 20),
            mutation_rate=config.get("mutation_rate", 0.2),
            crossover_rate=config.get("crossover_rate", 0.5)
        )
        self.generations = config.get("generations", 30)
        self.positions_source = config.get("positions_source", "random")
        self.positions_count = config.get("positions_count", 50)
        self.results_dir = config.get("results_dir", "ga_results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.positions = []

    def prepare_environment(self):
        """Set up environment for training (load positions, etc)."""
        self.positions = self.position_evaluator.load_positions(
            source=self.positions_source, count=self.positions_count)

    def run_training(self, generations=None):
        """Run the GA training process."""
        if generations is None:
            generations = self.generations
        self.ga.initialize_population()
        for gen in range(generations):
            print(f"[GA] Generation {gen+1}/{generations}")
            self.ga.evaluate_population(self.positions)
            print(f"  Best fitness: {self.ga.best_fitness:.4f}")
            self.ga.evolve_population()
        print("[GA] Training complete.")

    def save_results(self):
        """Save training results and metrics."""
        # Save best ruleset
        best_name = "ga_optimized"
        self.ga.export_best_ruleset(best_name)
        # Save fitness history
        fitness_path = os.path.join(self.results_dir, "fitness_history.txt")
        with open(fitness_path, "w") as f:
            for val in self.ga.best_fitness_history:
                f.write(f"{val}\n")
        print(f"[GA] Results saved to {self.results_dir}")
