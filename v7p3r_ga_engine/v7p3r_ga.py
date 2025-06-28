
"""
Core genetic algorithm logic for ruleset optimization with CUDA acceleration.
Features improved parallelization, elitism, and adaptive mutation.
"""
import random
import copy
import time
from typing import List, Tuple
import concurrent.futures
import multiprocessing

def _evaluate_individual_worker(args):
    """
    Standalone function for evaluating individual rulesets in parallel.
    This needs to be at module level to be picklable for multiprocessing.
    """
    i, ruleset, positions, position_evaluator = args
    try:
        # This will use CUDA acceleration internally if available
        v7p3r_evals, reference_evals = position_evaluator.evaluate_ruleset(ruleset, positions)
        fitness = position_evaluator.calculate_fitness(v7p3r_evals, reference_evals)
        print(f"    Individual {i+1}: fitness = {fitness:.4f}")
        return i, fitness, ruleset
    except Exception as e:
        print(f"    Individual {i+1}: ERROR = {e}")
        return i, float('-inf'), ruleset

class V7P3RGeneticAlgorithm:
    def __init__(self, base_ruleset, ruleset_manager, position_evaluator, 
                 population_size=20, mutation_rate=0.2, crossover_rate=0.5,
                 elitism_rate=0.1, adaptive_mutation=True, use_multiprocessing=True, max_workers=None):
        """
        base_ruleset: dict
        ruleset_manager: RulesetManager instance
        position_evaluator: PositionEvaluator instance
        elitism_rate: Fraction of best individuals to preserve each generation
        adaptive_mutation: Whether to adapt mutation rate based on progress
        use_multiprocessing: Whether to use multiprocessing for evaluation
        max_workers: Maximum number of worker processes (None = auto-detect)
        """
        self.base_ruleset = base_ruleset
        self.ruleset_manager = ruleset_manager
        self.position_evaluator = position_evaluator
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.adaptive_mutation = adaptive_mutation
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers
        self.population = []
        self.fitness_scores = []
        self.best_ruleset = None
        self.best_fitness = float('-inf')
        self.best_fitness_history = []
        self.generation_times = []
        self.stagnation_counter = 0
        
        # Elite preservation
        self.elite_size = max(1, int(population_size * elitism_rate))

    def initialize_population(self):
        """Create initial population by mutating the base ruleset."""
        self.population = [copy.deepcopy(self.base_ruleset)]
        for _ in range(self.population_size - 1):
            mutated = self.ruleset_manager.mutate_ruleset(self.base_ruleset, mutation_rate=self.mutation_rate)
            self.population.append(mutated)

    def evaluate_population(self, positions):
        """
        Evaluate all rulesets in the population with improved parallelization and progress tracking.
        Falls back to sequential evaluation if multiprocessing is disabled.
        """
        start_time = time.time()
        self.fitness_scores = []
        print(f"    Evaluating {len(self.population)} individuals {'in parallel' if self.use_multiprocessing else 'sequentially'}...")
        
        if not self.use_multiprocessing:
            # Sequential evaluation fallback
            individual_results = []
            for i, ruleset in enumerate(self.population):
                result = _evaluate_individual_worker((i, ruleset, positions, self.position_evaluator))
                individual_results.append(result)
                
                if (i + 1) % max(1, len(self.population) // 4) == 0:
                    progress = ((i + 1) / len(self.population)) * 100
                    elapsed = time.time() - start_time
                    print(f"    Progress: {progress:.1f}% ({i + 1}/{len(self.population)}) - {elapsed:.1f}s elapsed")
        else:
            # Parallel evaluation with multiprocessing
            max_workers = self.max_workers
            if max_workers is None:
                max_workers = min(multiprocessing.cpu_count() - 1, len(self.population), 8)
            
            # Process individuals in parallel
            individual_results = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all individuals for evaluation
                futures = {executor.submit(_evaluate_individual_worker, 
                                         (i, ruleset, positions, self.position_evaluator)): i 
                          for i, ruleset in enumerate(self.population)}
                
                # Collect results with progress tracking
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    i, fitness, ruleset = future.result()
                    individual_results.append((i, fitness, ruleset))
                    completed += 1
                    
                    if completed % max(1, len(self.population) // 4) == 0:
                        progress = (completed / len(self.population)) * 100
                        elapsed = time.time() - start_time
                        print(f"    Progress: {progress:.1f}% ({completed}/{len(self.population)}) - {elapsed:.1f}s elapsed")
        
        # Sort results by original index and extract fitness scores
        individual_results.sort(key=lambda x: x[0])
        self.fitness_scores = [result[1] for result in individual_results]
        
        # Update best individual
        best_idx = max(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i])
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_ruleset = copy.deepcopy(self.population[best_idx])
            self.stagnation_counter = 0
            print(f"    NEW BEST! Fitness: {self.best_fitness:.4f}")
        else:
            self.stagnation_counter += 1
        
        self.best_fitness_history.append(self.best_fitness)
        
        # Adaptive mutation rate
        if self.adaptive_mutation:
            self._adapt_mutation_rate()
        
        eval_time = time.time() - start_time
        self.generation_times.append(eval_time)
        print(f"    Evaluation completed in {eval_time:.2f}s")
        
        # Show performance stats periodically
        if len(self.best_fitness_history) % 5 == 0:
            stats = self.position_evaluator.get_performance_stats()
            print(f"    Performance: Cache hits: {stats['cache_hits']}, "
                  f"Cache size: {stats['cache_size']}, GPU memory: {stats.get('memory_usage', {})}")
    
    def _adapt_mutation_rate(self):
        """Adapt mutation rate based on stagnation."""
        if self.stagnation_counter > 3:
            # Increase mutation if stagnating
            self.mutation_rate = min(0.8, self.mutation_rate * 1.2)
            print(f"    Adapting mutation rate to {self.mutation_rate:.3f} (stagnation: {self.stagnation_counter})")
        elif self.stagnation_counter == 0:
            # Decrease mutation if improving
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
        
        # Reset to initial rate if extreme
        if self.mutation_rate > 0.8:
            self.mutation_rate = self.initial_mutation_rate
            print(f"    Reset mutation rate to {self.mutation_rate:.3f}")
    
    def select_parents(self):
        """Enhanced parent selection with elitism and tournament selection."""
        selected = []
        
        # First, preserve elite individuals
        fitness_indices = list(enumerate(self.fitness_scores))
        fitness_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Add elite individuals
        for i in range(self.elite_size):
            idx = fitness_indices[i][0]
            selected.append(copy.deepcopy(self.population[idx]))
        
        # Fill the rest with tournament selection
        tournament_size = 3
        while len(selected) < self.population_size:
            # Tournament selection
            tournament_indices = random.sample(range(self.population_size), tournament_size)
            winner_idx = max(tournament_indices, key=lambda i: self.fitness_scores[i])
            selected.append(copy.deepcopy(self.population[winner_idx]))
        
        return selected

    def evolve_population(self):
        """Create next generation through selection, crossover, mutation with elitism."""
        new_population = []
        parents = self.select_parents()
        
        # Elitism: directly carry over best individuals
        fitness_indices = list(enumerate(self.fitness_scores))
        fitness_indices.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(self.elite_size):
            idx = fitness_indices[i][0]
            new_population.append(copy.deepcopy(self.population[idx]))
        
        # Fill the rest through crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            if random.random() < self.crossover_rate:
                child = self.ruleset_manager.crossover_rulesets(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            child = self.ruleset_manager.mutate_ruleset(child, mutation_rate=self.mutation_rate)
            new_population.append(child)
        
        self.population = new_population[:self.population_size]

    def get_convergence_stats(self):
        """Get statistics about algorithm convergence."""
        if len(self.best_fitness_history) < 2:
            return {}
        
        recent_improvement = (self.best_fitness_history[-1] - 
                            self.best_fitness_history[max(0, len(self.best_fitness_history) - 5)])
        
        avg_generation_time = sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        
        return {
            'best_fitness': self.best_fitness,
            'generations_completed': len(self.best_fitness_history),
            'stagnation_counter': self.stagnation_counter,
            'current_mutation_rate': self.mutation_rate,
            'recent_improvement': recent_improvement,
            'avg_generation_time': avg_generation_time,
            'fitness_std': self._calculate_fitness_std()
        }
    
    def _calculate_fitness_std(self):
        """Calculate standard deviation of current population fitness."""
        if not self.fitness_scores:
            return 0.0
        
        valid_scores = [f for f in self.fitness_scores if f != float('-inf')]
        if len(valid_scores) < 2:
            return 0.0
        
        mean_fitness = sum(valid_scores) / len(valid_scores)
        variance = sum((f - mean_fitness) ** 2 for f in valid_scores) / len(valid_scores)
        return variance ** 0.5

    def export_best_ruleset(self, filename):
        """Export the best ruleset to a file."""
        if self.best_ruleset is not None:
            self.ruleset_manager.save_ruleset(self.best_ruleset, name=filename)
            print(f"    Best ruleset exported as '{filename}' with fitness {self.best_fitness:.4f}")
        else:
            print("    No best ruleset available to export")
    
    def should_terminate_early(self, max_stagnation=10):
        """Check if algorithm should terminate early due to convergence."""
        return self.stagnation_counter >= max_stagnation
