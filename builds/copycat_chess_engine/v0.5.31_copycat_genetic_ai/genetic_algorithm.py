# V0.5.31 Copycat Genetic AI - Genetic Algorithm Component
# "Acting Brain" - Uses RL model guidance for optimal move selection

import torch
import numpy as np
import chess
import random
import yaml
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from reinforcement_training import ReinforcementTrainer, ChessPositionEncoder

@dataclass
class MoveCandidate:
    """Represents a candidate move with genetic properties"""
    move: chess.Move
    rl_score: float          # Score from RL model
    eval_score: float        # Score from evaluation engine
    fitness: float           # Combined fitness score
    genes: List[float]       # Genetic representation (move characteristics)

class GeneticMoveSelector:
    """Genetic Algorithm for intelligent move selection using RL guidance"""
    
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, encoding='utf-8-sig') as f:
            config_data = yaml.safe_load(f)
        
        if isinstance(config_data, dict):
            self.config = config_data
        else:
            self.config = {}
        
        self.ga_config = self.config.get('genetic_algorithm', {})
        self.performance_config = self.config.get('performance', {})
        
        print(f"🧬 Initializing Genetic Algorithm Move Selector")
        
        # Load trained RL models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.position_encoder = ChessPositionEncoder()
        self.rl_actor = None
        self.rl_critic = None
        self.evaluation_engine = None
        
        # Load RL models if they exist
        self._load_rl_models()
        
        # Genetic Algorithm parameters
        self.population_size = self.ga_config.get('population_size', 50)
        self.elite_size = self.ga_config.get('elite_size', 10)
        self.mutation_rate = self.ga_config.get('mutation_rate', 0.1)
        self.crossover_rate = self.ga_config.get('crossover_rate', 0.7)
        self.max_generations = self.ga_config.get('max_generations', 20)
        self.fitness_weights = self.ga_config.get('fitness_weights', {
            'rl_weight': 0.6,       # Weight for RL model score
            'eval_weight': 0.3,     # Weight for evaluation engine score
            'diversity_weight': 0.1  # Weight for move diversity
        })
        
        # Performance tracking
        self.move_selection_times = []
        self.generation_stats = []
        
        print(f"   Population size: {self.population_size}")
        print(f"   Elite size: {self.elite_size}")
        print(f"   Device: {self.device}")
    
    def _load_rl_models(self):
        """Load trained RL models if available"""
        try:
            from reinforcement_training import ActorNetwork, CriticNetwork
            
            # Load actor model
            if Path("rl_actor_model.pth").exists():
                self.rl_actor = ActorNetwork().to(self.device)
                self.rl_actor.load_state_dict(torch.load("rl_actor_model.pth", map_location=self.device))
                self.rl_actor.eval()
                print("   ✅ RL Actor model loaded")
            else:
                print("   ⚠️ RL Actor model not found")
            
            # Load critic model
            if Path("rl_critic_model.pth").exists():
                self.rl_critic = CriticNetwork().to(self.device)
                self.rl_critic.load_state_dict(torch.load("rl_critic_model.pth", map_location=self.device))
                self.rl_critic.eval()
                print("   ✅ RL Critic model loaded")
            else:
                print("   ⚠️ RL Critic model not found")
                
        except Exception as e:
            print(f"   ⚠️ Could not load RL models: {e}")
    
    def get_rl_move_scores(self, board, legal_moves):
        """Get move scores from RL model"""
        if self.rl_actor is None:
            # Fallback to random scores if no RL model
            return {move: random.random() for move in legal_moves}
        
        try:
            state = self.position_encoder.encode_position(board)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                move_probs = self.rl_actor(state_tensor)
            
            # Map legal moves to scores
            move_scores = {}
            for i, move in enumerate(legal_moves):
                if i < len(move_probs[0]):
                    move_scores[move] = move_probs[0][i].item()
                else:
                    move_scores[move] = 0.0
            
            return move_scores
            
        except Exception as e:
            print(f"   ⚠️ RL scoring error: {e}")
            return {move: random.random() for move in legal_moves}
    
    def get_eval_move_scores(self, board, legal_moves):
        """Get move scores from evaluation engine"""
        try:
            from evaluation_engine import EvaluationEngine
            
            move_scores = {}
            for move in legal_moves:
                board.push(move)
                evaluator = EvaluationEngine(board)
                score = evaluator.evaluate_position()
                move_scores[move] = score
                board.pop()
            
            return move_scores
            
        except Exception as e:
            print(f"   ⚠️ Evaluation scoring error: {e}")
            return {move: random.random() for move in legal_moves}
    
    def encode_move_genes(self, move, board):
        """Encode move characteristics as genes"""
        genes = []
        
        # Gene 1: From square (normalized 0-1)
        genes.append(move.from_square / 63.0)
        
        # Gene 2: To square (normalized 0-1)
        genes.append(move.to_square / 63.0)
        
        # Gene 3: Piece type (normalized)
        piece = board.piece_at(move.from_square)
        if piece:
            genes.append(piece.piece_type / 6.0)  # 6 piece types
        else:
            genes.append(0.0)
        
        # Gene 4: Capture value (normalized)
        captured = board.piece_at(move.to_square)
        if captured:
            genes.append(captured.piece_type / 6.0)
        else:
            genes.append(0.0)
        
        # Gene 5: Special move type
        special_gene = 0.0
        if move.promotion:
            special_gene = 0.8
        elif board.is_castling(move):
            special_gene = 0.6
        elif board.is_en_passant(move):
            special_gene = 0.4
        genes.append(special_gene)
        
        # Gene 6: Move direction (simplified)
        from_rank, from_file = divmod(move.from_square, 8)
        to_rank, to_file = divmod(move.to_square, 8)
        direction = (to_rank - from_rank) / 7.0  # Normalized rank change
        genes.append(max(-1.0, min(1.0, direction)))
        
        return genes
    
    def calculate_fitness(self, candidate: MoveCandidate, board, diversity_bonus=0.0):
        """Calculate fitness score for a move candidate"""
        weights = self.fitness_weights
        
        # Normalize scores to 0-1 range
        rl_score = max(0.0, min(1.0, candidate.rl_score))
        eval_score = (candidate.eval_score + 1.0) / 2.0  # Assuming eval scores in [-1, 1]
        eval_score = max(0.0, min(1.0, eval_score))
        
        # Combined fitness
        fitness = (
            weights['rl_weight'] * rl_score +
            weights['eval_weight'] * eval_score +
            weights['diversity_weight'] * diversity_bonus
        )
        
        return fitness
    
    def create_initial_population(self, board) -> List[MoveCandidate]:
        """Create initial population of move candidates"""
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return []
        
        # Get scores from both RL and evaluation engines
        rl_scores = self.get_rl_move_scores(board, legal_moves)
        eval_scores = self.get_eval_move_scores(board, legal_moves)
        
        population = []
        for move in legal_moves:
            genes = self.encode_move_genes(move, board)
            
            candidate = MoveCandidate(
                move=move,
                rl_score=rl_scores.get(move, 0.0),
                eval_score=eval_scores.get(move, 0.0),
                fitness=0.0,
                genes=genes
            )
            
            # Calculate initial fitness
            candidate.fitness = self.calculate_fitness(candidate, board)
            population.append(candidate)
        
        # If we have fewer legal moves than population size, duplicate best ones
        while len(population) < self.population_size and population:
            # Add variations of the best moves
            best_candidates = sorted(population, key=lambda x: x.fitness, reverse=True)
            for candidate in best_candidates:
                if len(population) >= self.population_size:
                    break
                
                # Create a slightly mutated version
                mutated_genes = candidate.genes.copy()
                for i in range(len(mutated_genes)):
                    if random.random() < 0.1:  # Small mutation chance
                        mutated_genes[i] += random.uniform(-0.05, 0.05)
                        mutated_genes[i] = max(0.0, min(1.0, mutated_genes[i]))
                
                mutated_candidate = MoveCandidate(
                    move=candidate.move,
                    rl_score=candidate.rl_score,
                    eval_score=candidate.eval_score,
                    fitness=candidate.fitness,
                    genes=mutated_genes
                )
                population.append(mutated_candidate)
        
        return population[:self.population_size]
    
    def crossover(self, parent1: MoveCandidate, parent2: MoveCandidate) -> Tuple[MoveCandidate, MoveCandidate]:
        """Create offspring through crossover"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Single-point crossover on genes
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        
        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]
        
        # Create children (inherit moves from parents)
        child1 = MoveCandidate(
            move=parent1.move,
            rl_score=parent1.rl_score,
            eval_score=parent1.eval_score,
            fitness=0.0,
            genes=child1_genes
        )
        
        child2 = MoveCandidate(
            move=parent2.move,
            rl_score=parent2.rl_score,
            eval_score=parent2.eval_score,
            fitness=0.0,
            genes=child2_genes
        )
        
        return child1, child2
    
    def mutate(self, candidate: MoveCandidate) -> MoveCandidate:
        """Apply mutation to a candidate"""
        if random.random() > self.mutation_rate:
            return candidate
        
        # Mutate genes
        mutated_genes = candidate.genes.copy()
        for i in range(len(mutated_genes)):
            if random.random() < 0.3:  # 30% chance to mutate each gene
                mutation_strength = random.uniform(-0.1, 0.1)
                mutated_genes[i] += mutation_strength
                mutated_genes[i] = max(0.0, min(1.0, mutated_genes[i]))
        
        mutated_candidate = MoveCandidate(
            move=candidate.move,
            rl_score=candidate.rl_score,
            eval_score=candidate.eval_score,
            fitness=0.0,
            genes=mutated_genes
        )
        
        return mutated_candidate
    
    def select_best_move(self, board, time_limit=None) -> Tuple[chess.Move, Dict]:
        """Select the best move using genetic algorithm"""
        start_time = time.time()
        
        # Create initial population
        population = self.create_initial_population(board)
        
        if not population:
            # Fallback: return a random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves), {"error": "No population created"}
            else:
                # This should never happen in a real game
                raise ValueError("No legal moves available")
        
        best_fitness_history = []
        avg_fitness_history = []
        
        # Genetic Algorithm evolution
        for generation in range(self.max_generations):
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            # Calculate diversity bonuses
            diversity_scores = self._calculate_diversity_scores(population)
            
            # Update fitness with diversity
            for i, candidate in enumerate(population):
                candidate.fitness = self.calculate_fitness(
                    candidate, board, diversity_scores[i]
                )
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track statistics
            best_fitness = population[0].fitness
            avg_fitness = sum(c.fitness for c in population) / len(population)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            # Elite selection
            elite = population[:self.elite_size]
            
            # Create new generation
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, 3)
                parent2 = self._tournament_selection(population, 3)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Select best move
        final_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        best_candidate = final_population[0]
        
        # Record timing
        selection_time = time.time() - start_time
        self.move_selection_times.append(selection_time)
        
        # Prepare statistics
        stats = {
            "best_fitness": best_candidate.fitness,
            "rl_score": best_candidate.rl_score,
            "eval_score": best_candidate.eval_score,
            "generations": len(best_fitness_history),
            "selection_time": selection_time,
            "fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "population_size": len(final_population)
        }
        
        print(f"🧬 GA selected move: {best_candidate.move}")
        print(f"   Fitness: {best_candidate.fitness:.3f}")
        print(f"   RL Score: {best_candidate.rl_score:.3f}")
        print(f"   Eval Score: {best_candidate.eval_score:.3f}")
        print(f"   Time: {selection_time:.3f}s")
        
        return best_candidate.move, stats
    
    def _calculate_diversity_scores(self, population) -> List[float]:
        """Calculate diversity bonus for each candidate"""
        diversity_scores = []
        
        for i, candidate in enumerate(population):
            diversity_score = 0.0
            
            # Calculate distance from other candidates
            for j, other in enumerate(population):
                if i != j:
                    gene_distance = sum(
                        abs(candidate.genes[k] - other.genes[k])
                        for k in range(len(candidate.genes))
                    )
                    diversity_score += gene_distance
            
            # Normalize by population size
            if len(population) > 1:
                diversity_score /= (len(population) - 1)
            
            diversity_scores.append(diversity_score)
        
        return diversity_scores
    
    def _tournament_selection(self, population, tournament_size=3) -> MoveCandidate:
        """Select candidate using tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.move_selection_times:
            return {"error": "No move selections recorded"}
        
        return {
            "total_moves": len(self.move_selection_times),
            "avg_selection_time": np.mean(self.move_selection_times),
            "min_selection_time": np.min(self.move_selection_times),
            "max_selection_time": np.max(self.move_selection_times),
            "total_time": np.sum(self.move_selection_times)
        }
    
    def save_performance_data(self, filename="ga_performance.json"):
        """Save performance data to file"""
        data = {
            "config": self.ga_config,
            "performance_stats": self.get_performance_stats(),
            "move_selection_times": self.move_selection_times,
            "generation_stats": self.generation_stats
        }
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 GA performance data saved to {filename}")

# Example usage and testing
if __name__ == "__main__":
    print("🧬 Testing Genetic Algorithm Move Selector...")
    
    # Create a test position
    board = chess.Board()
    board.push_san("e4")  # 1. e4
    board.push_san("e5")  # 1... e5
    
    # Initialize genetic selector
    selector = GeneticMoveSelector("config_test.yaml")
    
    # Select best move
    best_move, stats = selector.select_best_move(board, time_limit=5.0)
    
    print(f"\n🎯 Best move selected: {best_move}")
    print(f"📊 Selection stats: {stats}")
    
    # Save performance data
    selector.save_performance_data()
