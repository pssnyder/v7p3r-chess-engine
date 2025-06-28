"""
Orchestrates the GA training process.
"""


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../v7p3r_engine')))
from ruleset_manager import RulesetManager
from position_evaluator import PositionEvaluator
from v7p3r_ga import V7P3RGeneticAlgorithm

class TrainingRunner:
    def __init__(self, config):
        """Initialize with enhanced configuration."""
        self.config = config
        self.ruleset_manager = RulesetManager(config.get("ruleset_path", "v7p3r_engine/rulesets.yaml"))
        self.stockfish_config = config.get("stockfish_config", {})
        
        # Import v7p3rScore dynamically to avoid circular imports
        from v7p3r_engine.v7p3r_score import v7p3rScore
        
        # Create enhanced position evaluator with CUDA support
        self.position_evaluator = PositionEvaluator(
            self.stockfish_config, 
            v7p3r_score_class=v7p3rScore,
            use_cuda=config.get("use_cuda", True),
            use_nn_evaluator=config.get("use_neural_evaluator", False),
            nn_model_path=config.get("neural_model_path")
        )
        
        self.base_ruleset = self.ruleset_manager.load_ruleset(config.get("base_ruleset", "default_evaluation"))
        
        # Create enhanced GA with new features
        self.ga = V7P3RGeneticAlgorithm(
            base_ruleset=self.base_ruleset,
            ruleset_manager=self.ruleset_manager,
            position_evaluator=self.position_evaluator,
            population_size=config.get("population_size", 20),
            mutation_rate=config.get("mutation_rate", 0.2),
            crossover_rate=config.get("crossover_rate", 0.5),
            elitism_rate=config.get("elitism_rate", 0.1),
            adaptive_mutation=config.get("adaptive_mutation", True),
            use_multiprocessing=config.get("use_multiprocessing", True),
            max_workers=config.get("max_workers", None)
        )
        
        self.generations = config.get("generations", 30)
        self.positions_source = config.get("positions_source", "random")
        self.positions_count = config.get("positions_count", 50)
        self.max_stagnation = config.get("max_stagnation", 10)
        self.results_dir = config.get("results_dir", "v7p3r_ga_engine/ga_results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.positions = []
        self.training_start_time = None

    def prepare_environment(self):
        """Set up environment for training with enhanced diagnostics."""
        print("[GA] Preparing training environment...")
        
        # Load positions
        self.positions = self.position_evaluator.load_positions(
            source=self.positions_source, count=self.positions_count)
        
        # Print system information
        if hasattr(self.position_evaluator, 'cuda_accelerator'):
            import torch
            print(f"[GA] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[GA] GPU: {torch.cuda.get_device_name()}")
                print(f"[GA] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        print(f"[GA] Loaded {len(self.positions)} test positions")
        print(f"[GA] Population size: {self.ga.population_size}")
        print(f"[GA] Max generations: {self.generations}")
        print(f"[GA] Elitism rate: {self.ga.elitism_rate}")
        print(f"[GA] Adaptive mutation: {self.ga.adaptive_mutation}")

    def run_training(self, generations=None):
        """Run the GA training process with enhanced monitoring and early stopping."""
        import time
        if generations is None:
            generations = self.generations
        
        self.training_start_time = time.time()
        print(f"\n[GA] Starting training with {generations} max generations...")
        print(f"[GA] Early stopping enabled (max stagnation: {self.max_stagnation})")
        
        print(f"[GA] Initializing population of size {self.ga.population_size}...")
        self.ga.initialize_population()
        print(f"[GA] Population initialized. Starting evolution...")
        
        for gen in range(generations):
            gen_start_time = time.time()
            print(f"\n{'='*60}")
            print(f"[GA] Generation {gen+1}/{generations} START")
            print(f"{'='*60}")
            
            # Evaluate population
            try:
                self.ga.evaluate_population(self.positions)
            except Exception as e:
                print(f"  ERROR during evaluation: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Get detailed statistics
            stats = self.ga.get_convergence_stats()
            perf_stats = self.position_evaluator.get_performance_stats()
            
            # Display progress
            best_fitness = stats.get('best_fitness', float('-inf'))
            avg_fitness = sum(self.ga.fitness_scores) / len(self.ga.fitness_scores) if self.ga.fitness_scores else 0
            fitness_std = stats.get('fitness_std', 0)
            
            print(f"\n  GENERATION {gen+1} RESULTS:")
            print(f"  Best fitness:      {best_fitness:.6f}")
            print(f"  Average fitness:   {avg_fitness:.6f}")
            print(f"  Fitness std dev:   {fitness_std:.6f}")
            print(f"  Mutation rate:     {stats.get('current_mutation_rate', 0):.4f}")
            print(f"  Stagnation:        {stats.get('stagnation_counter', 0)} generations")
            
            # Performance metrics
            print(f"\n  PERFORMANCE:")
            print(f"  Cache hit rate:    {perf_stats['cache_hits']}/{perf_stats['cache_hits'] + perf_stats['cache_misses']}")
            print(f"  Cache size:        {perf_stats['cache_size']} entries")
            memory_info = perf_stats.get('memory_usage', {})
            if 'allocated' in memory_info:
                print(f"  GPU memory:        {memory_info['allocated']:.2f}GB allocated")
            
            # Population diversity
            unique_fitnesses = len(set(f for f in self.ga.fitness_scores if f != float('-inf')))
            print(f"  Population diversity: {unique_fitnesses}/{len(self.ga.fitness_scores)} unique fitness values")
            
            # Evolution step
            print(f"\n  Evolving population...")
            try:
                self.ga.evolve_population()
            except Exception as e:
                print(f"  ERROR during evolution: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Generation timing
            gen_time = time.time() - gen_start_time
            total_time = time.time() - self.training_start_time
            estimated_remaining = (gen_time * (generations - gen - 1)) if gen > 0 else 0
            
            print(f"\n  TIMING:")
            print(f"  Generation time:   {gen_time:.2f}s")
            print(f"  Total elapsed:     {total_time:.2f}s")
            print(f"  Est. remaining:    {estimated_remaining:.2f}s")
            
            print(f"{'='*60}")
            print(f"[GA] Generation {gen+1} COMPLETE")
            
            # Early stopping check
            if self.ga.should_terminate_early(self.max_stagnation):
                print(f"\n[GA] Early stopping: {self.max_stagnation} generations without improvement")
                break
            
            # Periodic cache cleanup
            if (gen + 1) % 5 == 0:
                print(f"  Clearing caches to free memory...")
                self.position_evaluator.clear_cache()
        
        total_training_time = time.time() - self.training_start_time
        print(f"\n[GA] Training completed in {total_training_time:.2f}s")
        print(f"[GA] Best fitness achieved: {self.ga.best_fitness:.6f}")
        
        return self.ga.get_convergence_stats()

    def save_results(self):
        """Save training results and metrics with enhanced analytics."""
        print(f"\n[GA] Saving results to {self.results_dir}...")
        
        # Save best ruleset
        best_name = "ga_optimized_cuda"
        self.ga.export_best_ruleset(best_name)
        
        # Save detailed fitness history
        fitness_path = os.path.join(self.results_dir, "fitness_history.txt")
        with open(fitness_path, "w") as f:
            f.write("# Generation, Best_Fitness, Avg_Fitness, Std_Fitness\n")
            for i, best_fit in enumerate(self.ga.best_fitness_history):
                # Calculate average fitness for this generation if available
                if i < len(self.ga.generation_times):
                    # This is approximate since we don't store per-generation avg
                    f.write(f"{i+1}, {best_fit:.6f}\n")
        
        # Save performance metrics
        metrics_path = os.path.join(self.results_dir, "performance_metrics.txt")
        final_stats = self.ga.get_convergence_stats()
        perf_stats = self.position_evaluator.get_performance_stats()
        
        with open(metrics_path, "w") as f:
            f.write("=== GA Training Performance Report ===\n\n")
            f.write(f"Final Best Fitness: {final_stats.get('best_fitness', 0):.6f}\n")
            f.write(f"Generations Completed: {final_stats.get('generations_completed', 0)}\n")
            f.write(f"Final Mutation Rate: {final_stats.get('current_mutation_rate', 0):.4f}\n")
            f.write(f"Final Stagnation Counter: {final_stats.get('stagnation_counter', 0)}\n")
            f.write(f"Average Generation Time: {final_stats.get('avg_generation_time', 0):.2f}s\n")
            f.write(f"Recent Improvement: {final_stats.get('recent_improvement', 0):.6f}\n")
            f.write(f"Population Diversity (final): {final_stats.get('fitness_std', 0):.6f}\n\n")
            
            f.write("=== Cache Performance ===\n")
            f.write(f"Cache Hits: {perf_stats['cache_hits']}\n")
            f.write(f"Cache Misses: {perf_stats['cache_misses']}\n")
            f.write(f"Final Cache Size: {perf_stats['cache_size']}\n")
            
            memory_info = perf_stats.get('memory_usage', {})
            if 'allocated' in memory_info:
                f.write(f"\n=== GPU Memory Usage ===\n")
                f.write(f"Allocated: {memory_info['allocated']:.2f}GB\n")
                f.write(f"Cached: {memory_info.get('cached', 0):.2f}GB\n")
                f.write(f"Max Allocated: {memory_info.get('max_allocated', 0):.2f}GB\n")
        
        # Save generation timing data
        timing_path = os.path.join(self.results_dir, "generation_times.txt")
        with open(timing_path, "w") as f:
            f.write("# Generation, Time_Seconds\n")
            for i, time_sec in enumerate(self.ga.generation_times):
                f.write(f"{i+1}, {time_sec:.2f}\n")
        
        # Save configuration used
        config_path = os.path.join(self.results_dir, "training_config.yaml")
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"[GA] Results saved:")
        print(f"  - Best ruleset: {best_name}")
        print(f"  - Fitness history: {fitness_path}")
        print(f"  - Performance metrics: {metrics_path}")
        print(f"  - Generation times: {timing_path}")
        print(f"  - Training config: {config_path}")
        
        # Final cleanup
        self.position_evaluator.clear_cache()
        self.position_evaluator.cleanup()  # Ensure Stockfish is terminated
        print(f"[GA] Training complete and cleaned up!")

def main():
    import yaml
    
    trainer = None
    try:
        with open("v7p3r_ga_engine/ga_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            trainer = TrainingRunner(config)
            trainer.prepare_environment()
            trainer.run_training(generations=config.get("generations", 30))
            trainer.save_results()
    finally:
        # Ensure cleanup even if there's an error
        if trainer:
            try:
                trainer.position_evaluator.cleanup()
                print("[GA] Final cleanup completed")
            except:
                pass

if __name__ == "__main__":
    main()
