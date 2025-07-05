"""
Enhanced v7p3r Reinforcement Learning Engine

This engine implements a modern RL approach for chess using:
- PPO (Proximal Policy Optimization) algorithm
- Integration with v7p3r_engine scoring system
- Ruleset-based reward/penalty system
- CUDA acceleration for training
- Stockfish validation during training
- Self-play and opponent training modes

Compatible with the updated v7p3r engine architecture.

Author: v7p3r Project
Date: 2025-06-27
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import chess
import chess.engine
import numpy as np
import random
import logging
from typing import List, Tuple, Dict, Optional
from collections import deque
import time

# Add paths for v7p3r engine integration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../v7p3r_engine')))

from v7p3r_config import v7p3rConfig

class ChessPositionEncoder:
    """Enhanced position encoding for chess positions."""
    
    def __init__(self):
        self.feature_dim = 8 * 8 * 15 + 8  # 15 channels + 8 auxiliary features
        
    def encode_position(self, board: chess.Board) -> torch.Tensor:
        """
        Encode chess position as a tensor.
        
        Returns:
            Tensor of shape (feature_dim,) representing the position
        """
        # 15 channels: 6 piece types x 2 colors + 3 auxiliary channels
        features = np.zeros((15, 8, 8), dtype=np.float32)
        
        # Piece positions (12 channels)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                channel = piece.piece_type - 1
                if piece.color == chess.BLACK:
                    channel += 6
                features[channel, row, col] = 1.0
        
        # Auxiliary channels
        # Channel 12: Attackers/defenders
        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            attackers = len(board.attackers(chess.WHITE, square))
            defenders = len(board.attackers(chess.BLACK, square))
            features[12, row, col] = attackers - defenders
        
        # Channel 13: Mobility
        white_moves = len(list(board.generate_legal_moves()))
        board.push(chess.Move.null())  # Switch perspective
        black_moves = len(list(board.generate_legal_moves()))
        board.pop()
        features[13, :, :] = (white_moves - black_moves) / 100.0
        
        # Channel 14: Turn and game phase
        features[14, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
        
        # Flatten spatial features
        spatial_features = features.flatten()
        
        # Additional features (8 values)
        aux_features = np.array([
            float(board.turn),  # Side to move
            float(board.has_kingside_castling_rights(chess.WHITE)),
            float(board.has_queenside_castling_rights(chess.WHITE)),
            float(board.has_kingside_castling_rights(chess.BLACK)),
            float(board.has_queenside_castling_rights(chess.BLACK)),
            float(board.is_check()),
            len(board.move_stack) / 200.0,  # Game progress
            float(board.is_repetition())
        ], dtype=np.float32)
        
        # Combine all features
        combined_tensor = torch.cat([torch.tensor(spatial_features), torch.tensor(aux_features)])
        return combined_tensor.to(torch.float32)

class PolicyValueNetwork(nn.Module):
    """Combined policy and value network for PPO."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4096)  # Max possible moves
        )
        
        # Value head (position evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x):
        shared = self.shared_layers(x)
        policy_logits = self.policy_head(shared)
        value = self.value_head(shared)
        return policy_logits, value

class PPOTrainer:
    """Proximal Policy Optimization trainer."""
    
    def __init__(self, network: PolicyValueNetwork, lr: float = 3e-4, 
                 clip_ratio: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        
    def update(self, states, actions, old_log_probs, rewards, values, advantages):
        """Perform PPO update."""
        # Convert to tensors and move to device
        states = torch.stack(states).to(self.network.device if hasattr(self.network, 'device') else 'cuda')
        actions = torch.tensor(actions, device=states.device)
        old_log_probs = torch.tensor(old_log_probs, device=states.device)
        rewards = torch.tensor(rewards, device=states.device)
        advantages = torch.tensor(advantages, device=states.device)
        
        # Forward pass
        policy_logits, new_values = self.network(states)
        
        # Policy loss
        action_probs = F.softmax(policy_logits, dim=-1)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values.squeeze(), rewards)
        
        # Entropy loss (for exploration)
        entropy_loss = -dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }

class V7P3RRLAgent:
    """Enhanced RL agent using v7p3r scoring system and modern RL techniques."""
    
    def __init__(self, config_overrides=None):
        # Use centralized configuration manager
        self.config_manager = v7p3rConfig(overrides=config_overrides)
        self.config = self.config_manager.get_v7p3r_rl_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.position_encoder = ChessPositionEncoder()
        self.network = PolicyValueNetwork(
            input_dim=self.position_encoder.feature_dim,
            hidden_dim=self.config.get('hidden_dim', 512),
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)
        
        self.trainer = PPOTrainer(
            self.network,
            lr=self.config.get('learning_rate', 3e-4),
            clip_ratio=self.config.get('clip_ratio', 0.2),
            vf_coef=self.config.get('vf_coef', 0.5),
            ent_coef=self.config.get('ent_coef', 0.01)
        )
        
        # Initialize v7p3r scoring system
        self._init_v7p3r_scoring()
        
        # Initialize Stockfish for validation
        self._init_stockfish()
        
        # Training state
        self.episode_buffer = []
        self.training_stats = {
            'episodes': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_reward': 0.0,
            'stockfish_wins': 0,
            'stockfish_games': 0
        }
        
        print(f"[RL] Initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"[RL] GPU: {torch.cuda.get_device_name()}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from centralized config manager (deprecated - keeping for compatibility)"""
        logging.warning("_load_config is deprecated. Using centralized configuration manager.")
        return self.config
    
    def _init_v7p3r_scoring(self):
        """Initialize v7p3r scoring system with ruleset."""
        try:
            from v7p3r_score import v7p3rScore
            from v7p3r_pst import v7p3rPST
            from v7p3r_ga_ruleset_manager import v7p3rGARulesetManager
            
            # Load ruleset for rewards
            self.ruleset_manager = v7p3rGARulesetManager()
            self.reward_ruleset = self.ruleset_manager.load_ruleset(
                self.config.get('reward_ruleset', 'default_evaluation')
            )
            
            # Initialize scoring
            engine_config = {
                'verbose_output': False,
                'engine_ruleset': 'rl_reward_system'
            }
            
            logger = logging.getLogger("v7p3r_rl")
            pst = v7p3rPST()
            self.scorer = v7p3rScore(engine_config, pst)
            
            print(f"[RL] v7p3r scoring initialized with ruleset: {self.reward_ruleset.get('name', 'unknown')}")
            
        except Exception as e:
            print(f"[RL] Warning: Could not initialize v7p3r scoring: {e}")
            self.scorer = None
    
    def _init_stockfish(self):
        """Initialize Stockfish for validation."""
        try:
            from stockfish_handler import StockfishHandler
            
            stockfish_config = self.config_manager.get_stockfish_config()
            
            self.stockfish = StockfishHandler(stockfish_config)
            print(f"[RL] Stockfish initialized for validation")
            
        except Exception as e:
            print(f"[RL] Warning: Could not initialize Stockfish: {e}")
            self.stockfish = None
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'stockfish') and self.stockfish:
                self.stockfish.quit()
                print("[RL] Stockfish process terminated")
        except Exception as e:
            print(f"[RL] Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
    
    def move_to_index(self, move: chess.Move) -> int:
        """Convert move to unique index for action space."""
        # Simple hash-based mapping for now
        # In production, use a proper move encoding (e.g., from_square * 64 + to_square + promotion)
        return hash(move.uci()) % 4096
    
    def get_legal_moves_with_indices(self, board: chess.Board) -> List[Tuple[chess.Move, int]]:
        """Get legal moves with their corresponding action indices."""
        legal_moves = list(board.legal_moves)
        return [(move, self.move_to_index(move)) for move in legal_moves]
    
    def select_action(self, board: chess.Board, training: bool = True) -> Tuple[Optional[chess.Move], float, float]:
        """
        Select action using the policy network.
        
        Returns:
            Tuple of (move, log_prob, value)
        """
        # Encode position
        state = self.position_encoder.encode_position(board).to(self.device)
        
        # Get network predictions
        with torch.no_grad():
            policy_logits, value = self.network(state.unsqueeze(0))
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)
        
        # Get legal moves and their indices
        legal_moves_with_indices = self.get_legal_moves_with_indices(board)
        if not legal_moves_with_indices:
            # Should not happen but safety check
            return None, 0.0, value.item()
        
        # Extract indices for legal moves
        legal_moves, legal_indices = zip(*legal_moves_with_indices)
        legal_moves = list(legal_moves)  # Convert to list for indexing
        
        # Mask illegal moves
        masked_logits = torch.full((4096,), float('-inf'), device=self.device)
        for idx in legal_indices:
            masked_logits[idx] = policy_logits[idx]
        
        # Create distribution over legal moves
        legal_probs = F.softmax(masked_logits[list(legal_indices)], dim=0)
        
        if training:
            # Sample from distribution
            dist = Categorical(legal_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            selected_move = legal_moves[int(action_idx.item())]
        else:
            # Take best move
            action_idx = torch.argmax(legal_probs)
            log_prob = torch.log(legal_probs[action_idx])
            selected_move = legal_moves[int(action_idx.item())]
        
        return selected_move, log_prob.item(), value.item()
    
    def calculate_reward(self, board: chess.Board, move: chess.Move, game_outcome: Optional[str] = None) -> float:
        """
        Calculate reward using v7p3r scoring system and game outcome.
        
        Args:
            board: Current board position (after move)
            move: Move that was played
            game_outcome: Game result if game ended
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Base reward from v7p3r scoring system
        if self.scorer:
            try:
                position_score = self.scorer.calculate_score(board, board.turn)
                reward += position_score * 0.01  # Scale down position score
            except Exception as e:
                print(f"[RL] Error calculating position score: {e}")
        
        # Reward shaping from ruleset
        if hasattr(self, 'reward_ruleset') and self.reward_ruleset:
            try:
                # Apply ruleset-based rewards/penalties
                for rule_name, rule_config in self.reward_ruleset.get('ruleset', {}).items():
                    if rule_config.get('enabled', False):
                        rule_reward = self._apply_rule_reward(board, move, rule_name, rule_config)
                        reward += rule_reward
            except Exception as e:
                print(f"[RL] Error applying ruleset rewards: {e}")
        
        # Terminal rewards for game outcome
        if game_outcome:
            if game_outcome == "1-0" and not board.turn:  # White wins, we're black
                reward += 10.0
            elif game_outcome == "0-1" and board.turn:  # Black wins, we're white
                reward += 10.0
            elif game_outcome == "1/2-1/2":  # Draw
                reward += 1.0
            else:  # Loss
                reward -= 10.0
        
        # Encourage quick victories/discourage long games
        if board.fullmove_number > 100:
            reward -= 0.1
        
        return reward
    
    def _apply_rule_reward(self, board: chess.Board, move: chess.Move, rule_name: str, rule_config: dict) -> float:
        """Apply individual rule-based reward."""
        reward = 0.0
        weight = rule_config.get('weight', 1.0)
        
        # Check specific rule conditions
        if rule_name == 'center_control':
            # Reward controlling center squares
            center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
            controlled = sum(1 for sq in center_squares if board.is_attacked_by(board.turn, sq))
            reward += controlled * weight * 0.1
            
        elif rule_name == 'piece_development':
            # Reward developing pieces in opening
            if board.fullmove_number <= 10:
                if move.from_square in [chess.B1, chess.G1, chess.B8, chess.G8]:  # Knights
                    reward += weight * 0.2
                elif move.from_square in [chess.C1, chess.F1, chess.C8, chess.F8]:  # Bishops
                    reward += weight * 0.15
                    
        elif rule_name == 'king_safety':
            # Reward castling
            if board.is_castling(move):
                reward += weight * 0.5
            # Penalty for exposing king
            elif board.piece_at(move.from_square):
                piece = board.piece_at(move.from_square)
                if piece and piece.piece_type == chess.KING:
                    if board.fullmove_number <= 15:
                        reward -= weight * 0.3
                    
        elif rule_name == 'material_balance':
            # Reward captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}  # Pawn, Knight, Bishop, Rook, Queen, King
                    reward += piece_values.get(captured_piece.piece_type, 0) * weight * 0.1
        
        return reward
    
    def play_self_game(self, max_moves: int = 200) -> Dict:
        """
        Play a self-play game for training.
        
        Returns:
            Game statistics and trajectory data
        """
        board = chess.Board()
        trajectory = []
        total_reward = 0.0
        
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            # Select action
            move, log_prob, value = self.select_action(board, training=True)
            
            if move is None:
                break
                
            # Store state before move
            state = self.position_encoder.encode_position(board)
            trajectory.append({
                'state': state,
                'action': self.move_to_index(move),
                'log_prob': log_prob,
                'value': value
            })
            
            # Make move
            board.push(move)
            move_count += 1
            
            # Calculate immediate reward
            game_outcome = board.result() if board.is_game_over() else None
            reward = self.calculate_reward(board, move, game_outcome)
            trajectory[-1]['reward'] = reward
            total_reward += reward
        
        # Game outcome
        result = board.result()
        final_score = self._result_to_score(result)
        
        # Backpropagate final outcome to trajectory
        for i, step in enumerate(trajectory):
            # Add final outcome bonus/penalty
            if result == "1-0":  # White wins
                step['final_outcome'] = 1.0 if i % 2 == 0 else -1.0
            elif result == "0-1":  # Black wins
                step['final_outcome'] = -1.0 if i % 2 == 0 else 1.0
            else:  # Draw
                step['final_outcome'] = 0.1
        
        # Update statistics
        self.training_stats['episodes'] += 1
        self.training_stats['total_reward'] += total_reward
        
        if result == "1-0":
            self.training_stats['wins'] += 1
        elif result == "0-1":
            self.training_stats['losses'] += 1
        else:
            self.training_stats['draws'] += 1
        
        return {
            'trajectory': trajectory,
            'result': result,
            'moves': move_count,
            'total_reward': total_reward,
            'final_score': final_score
        }
    
    def _result_to_score(self, result: str) -> float:
        """Convert game result to score."""
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0
    
    def validate_against_stockfish(self, num_games: int = 10) -> Dict:
        """
        Validate agent performance against Stockfish.
        
        Returns:
            Validation statistics
        """
        if not self.stockfish:
            print("[RL] Stockfish not available for validation")
            return {}
        
        wins = 0
        losses = 0
        draws = 0
        
        for game_idx in range(num_games):
            board = chess.Board()
            agent_is_white = game_idx % 2 == 0
            
            move_count = 0
            max_moves = 100  # Shorter games for validation
            
            while not board.is_game_over() and move_count < max_moves:
                if (board.turn == chess.WHITE) == agent_is_white:
                    # Agent's turn
                    move, _, _ = self.select_action(board, training=False)
                    if move is None:
                        break
                else:
                    # Stockfish's turn
                    try:
                        engine_config = {'depth': 3, 'movetime': 100}
                        move = self.stockfish.search(board, board.turn, engine_config)
                        if move is None:
                            break
                    except Exception as e:
                        print(f"[RL] Stockfish error: {e}")
                        break
                
                board.push(move)
                move_count += 1
            
            # Analyze result
            result = board.result()
            if result == "1-0":
                if agent_is_white:
                    wins += 1
                else:
                    losses += 1
            elif result == "0-1":
                if agent_is_white:
                    losses += 1
                else:
                    wins += 1
            else:
                draws += 1
        
        # Update validation stats
        self.training_stats['stockfish_wins'] += wins
        self.training_stats['stockfish_games'] += num_games
        
        win_rate = wins / num_games if num_games > 0 else 0.0
        
        print(f"[RL] Stockfish validation: {wins}W-{losses}L-{draws}D (WR: {win_rate:.2f})")
        
        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'total_games': num_games
        }
    
    def train_episode_batch(self, batch_size: int = 32) -> Dict:
        """
        Train on a batch of self-play episodes.
        
        Returns:
            Training statistics
        """
        batch_trajectories = []
        
        # Collect batch of episodes
        for _ in range(batch_size):
            episode_data = self.play_self_game()
            batch_trajectories.extend(episode_data['trajectory'])
        
        if not batch_trajectories:
            return {}
        
        # Prepare training data
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        
        for step in batch_trajectories:
            states.append(step['state'].to(self.device))
            actions.append(step['action'])
            old_log_probs.append(step['log_prob'])
            rewards.append(step['reward'] + step.get('final_outcome', 0.0))
            values.append(step['value'])
        
        # Calculate advantages using GAE
        advantages = self._calculate_gae(rewards, values)
        
        # Perform PPO update
        train_stats = self.trainer.update(states, actions, old_log_probs, rewards, values, advantages)
        
        # Clear CUDA cache to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return train_stats
    
    def _calculate_gae(self, rewards: List[float], values: List[float], gamma: float = 0.99, lam: float = 0.95) -> List[float]:
        """Calculate Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def save_model(self, path: str):
        """Save the trained model."""
        save_data = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }
        torch.save(save_data, path)
        print(f"[RL] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        if os.path.exists(path):
            save_data = torch.load(path, map_location=self.device)
            self.network.load_state_dict(save_data['network_state_dict'])
            self.trainer.optimizer.load_state_dict(save_data['optimizer_state_dict'])
            self.training_stats = save_data.get('training_stats', self.training_stats)
            print(f"[RL] Model loaded from {path}")
        else:
            print(f"[RL] Model file not found: {path}")

# Entry point for integration with chess_game.py
class v7p3rRLEngine:
    """Wrapper class for compatibility with chess_game.py."""
    
    def __init__(self, config_overrides=None):
        self.agent = V7P3RRLAgent(config_overrides)
        self.config = self.agent.config
        
        # Load model if specified
        model_path = self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.agent.load_model(model_path)

    def reset(self, board=None):
        """Reset for new game (compatibility method)."""
        pass

    def evaluate_position_from_perspective(self, board: chess.Board, perspective: chess.Color) -> float:
        """Evaluate position from given perspective."""
        if self.agent.scorer:
            return self.agent.scorer.calculate_score(board, perspective)
        else:
            # Fallback to basic material evaluation
            return self._basic_evaluation(board, perspective)
    
    def _basic_evaluation(self, board: chess.Board, perspective: chess.Color) -> float:
        """Basic material-based evaluation."""
        piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == perspective:
                    score += value
                else:
                    score -= value
        
        return score

    def search(self, board: chess.Board, player_color: chess.Color, engine_config=None) -> Optional[chess.Move]:
        """Search for best move (main engine interface)."""
        move, _, _ = self.agent.select_action(board, training=False)
        return move

    def train_self_play(self, episodes: int = 100, batch_size: int = 16):
        """Train the agent with self-play."""
        print(f"[RL] Starting self-play training for {episodes} episodes")
        
        for episode in range(0, episodes, batch_size):
            current_batch_size = min(batch_size, episodes - episode)
            train_stats = self.agent.train_episode_batch(current_batch_size)
            
            if episode % (batch_size * 5) == 0:  # Print stats every 5 batches
                stats = self.agent.training_stats
                print(f"[RL] Episode {episode}: "
                      f"Total: {stats['episodes']}, "
                      f"W/L/D: {stats['wins']}/{stats['losses']}/{stats['draws']}, "
                      f"Avg Reward: {stats['total_reward']/max(stats['episodes'], 1):.3f}")
                
                if train_stats:
                    print(f"[RL] Training: Policy Loss: {train_stats['policy_loss']:.4f}, "
                          f"Value Loss: {train_stats['value_loss']:.4f}")

    def validate_performance(self, num_games: int = 10):
        """Validate against Stockfish."""
        return self.agent.validate_against_stockfish(num_games)

    def save(self, path: str):
        """Save trained model."""
        self.agent.save_model(path)

    def load(self, path: str):
        """Load trained model."""
        self.agent.load_model(path)
    
    def cleanup(self):
        """Clean up resources."""
        self.agent.cleanup()
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass
