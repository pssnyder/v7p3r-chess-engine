
"""
Core genetic algorithm logic for ruleset optimization (Stockfish similarity fitness).
"""
import random
import copy

class V7P3RGeneticAlgorithm:
    def __init__(self, base_ruleset, ruleset_manager, position_evaluator, population_size=20, mutation_rate=0.2, crossover_rate=0.5):
        """
        base_ruleset: dict
        ruleset_manager: RulesetManager instance
        position_evaluator: PositionEvaluator instance
        """
        self.base_ruleset = base_ruleset
        self.ruleset_manager = ruleset_manager
        self.position_evaluator = position_evaluator
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness_scores = []
        self.best_ruleset = None
        self.best_fitness = float('-inf')
        self.best_fitness_history = []

    def initialize_population(self):
        """Create initial population by mutating the base ruleset."""
        self.population = [copy.deepcopy(self.base_ruleset)]
        for _ in range(self.population_size - 1):
            mutated = self.ruleset_manager.mutate_ruleset(self.base_ruleset, mutation_rate=self.mutation_rate)
            self.population.append(mutated)

    def evaluate_population(self, positions):
        """Evaluate all rulesets in the population and store fitness scores."""
        self.fitness_scores = []
        for ruleset in self.population:
            v7p3r_evals, stockfish_evals = self.position_evaluator.evaluate_ruleset(ruleset, positions)
            fitness = self.position_evaluator.calculate_fitness(v7p3r_evals, stockfish_evals)
            self.fitness_scores.append(fitness)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_ruleset = copy.deepcopy(ruleset)
        self.best_fitness_history.append(self.best_fitness)

    def select_parents(self):
        """Select parents for breeding based on fitness (tournament selection)."""
        selected = []
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            winner = self.population[i] if self.fitness_scores[i] > self.fitness_scores[j] else self.population[j]
            selected.append(copy.deepcopy(winner))
        return selected

    def evolve_population(self):
        """Create next generation through selection, crossover, mutation."""
        new_population = []
        parents = self.select_parents()
        for i in range(0, self.population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i+1] if i+1 < self.population_size else parents[0]
            if random.random() < self.crossover_rate:
                child1 = self.ruleset_manager.crossover_rulesets(parent1, parent2)
                child2 = self.ruleset_manager.crossover_rulesets(parent2, parent1)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            child1 = self.ruleset_manager.mutate_ruleset(child1, mutation_rate=self.mutation_rate)
            child2 = self.ruleset_manager.mutate_ruleset(child2, mutation_rate=self.mutation_rate)
            new_population.extend([child1, child2])
        self.population = new_population[:self.population_size]

    def export_best_ruleset(self, filename):
        """Export the best ruleset to a file."""
        self.ruleset_manager.save_ruleset(self.best_ruleset, name=filename)
