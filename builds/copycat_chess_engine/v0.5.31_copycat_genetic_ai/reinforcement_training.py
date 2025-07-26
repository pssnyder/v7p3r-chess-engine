# V0.5.31 Copycat Genetic AI - Reinforcement Learning Training
# "Thinking Brain" - Learns strategic move preferences from evaluation engine rewards

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import chess
import chess.pgn
import yaml
import os
import random
from pathlib import Path
from collections import deque, namedtuple
from evaluation_engine import EvaluationEngine

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ChessPositionEncoder:
    """Encodes chess positions into tensor format for neural networks"""
    
    def __init__(self):
        self.piece_to_channel = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }
    
    def encode_position(self, board):
        """Convert chess board to 12-channel tensor (6 pieces x 2 colors)"""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                # White pieces: channels 0-5, Black pieces: channels 6-11
                channel = self.piece_to_channel[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6
                tensor[channel][7-row][col] = 1.0
        
        return tensor
    
    def encode_move(self, move, board):
        """Encode move as action index"""
        legal_moves = list(board.legal_moves)
        try:
            return legal_moves.index(move)
        except ValueError:
            return -1  # Invalid move

class ActorNetwork(nn.Module):
    """Actor network - suggests move probabilities"""
    
    def __init__(self, action_size=4096, hidden_size=512):
        super(ActorNetwork, self).__init__()
        
        # Convolutional layers for board position understanding
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Use softmax to get move probabilities
        return F.softmax(self.fc3(x), dim=1)

class CriticNetwork(nn.Module):
    """Critic network - evaluates position value"""
    
    def __init__(self, hidden_size=512):
        super(CriticNetwork, self).__init__()
        
        # Convolutional layers for board position understanding
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Single value output
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        return self.fc3(x)  # Position value estimate

class ReplayBuffer:
    """Experience replay buffer for training stability"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ReinforcementTrainer:
    """Main reinforcement learning trainer - the 'Thinking Brain'"""
    
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, encoding='utf-8-sig') as f:
            config_data = yaml.safe_load(f)
        
        # Ensure we have a dictionary
        if isinstance(config_data, dict):
            self.config = config_data
        else:
            self.config = {}
        
        self.rl_config = self.config.get('reinforcement_learning', {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🧠 Initializing Reinforcement Learning Trainer")
        print(f"   Device: {self.device}")
        
        # Initialize components
        self.position_encoder = ChessPositionEncoder()
        self.evaluation_engine = None  # Will be initialized when needed
        
        # Neural networks
        self.actor = ActorNetwork(
            hidden_size=self.rl_config['actor_hidden_size']
        ).to(self.device)
        
        self.critic = CriticNetwork(
            hidden_size=self.rl_config['critic_hidden_size']
        ).to(self.device)
        
        # Target networks for stable training
        self.target_critic = CriticNetwork(
            hidden_size=self.rl_config['critic_hidden_size']
        ).to(self.device)
        
        # Copy weights to target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.rl_config['learning_rate']
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.rl_config['learning_rate']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.rl_config['replay_buffer_size'])
        
        # Training statistics
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'average_rewards': [],
            'validation_scores': {}  # Changed to dict to store validation results
        }
    
    def get_move_candidates(self, board, top_k=10):
        """Get top-k move candidates from actor network"""
        state = self.position_encoder.encode_position(board)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            move_probs = self.actor(state_tensor)
        
        # Get legal moves and their probabilities
        legal_moves = list(board.legal_moves)
        move_candidates = []
        
        for i, move in enumerate(legal_moves):
            if i < len(move_probs[0]):
                prob = move_probs[0][i].item()
                move_candidates.append((move, prob))
        
        # Sort by probability and return top-k
        move_candidates.sort(key=lambda x: x[1], reverse=True)
        return move_candidates[:top_k]
    
    def validate_evaluation_alignment(self, training_files):
        """Validate that evaluation scores align with winning moves"""
        print("\n🔍 Validating evaluation alignment with winning moves...")
        
        validation_results = {}
        sample_size = self.rl_config['validation_sample_size']
        
        for pgn_file in training_files[:3]:  # Test first 3 files
            print(f"   📄 Validating {pgn_file.name}...")
            
            scores = {'winner_scores': [], 'loser_scores': []}
            positions_checked = 0
            
            try:
                with open(pgn_file, encoding='utf-8', errors='ignore') as f:
                    while positions_checked < sample_size:
                        game = chess.pgn.read_game(f)
                        if not game:
                            break
                        
                        result = game.headers.get("Result", "*")
                        if result == "*":  # Skip games without clear result
                            continue
                        
                        winner_color = None
                        if result == "1-0":
                            winner_color = chess.WHITE
                        elif result == "0-1":
                            winner_color = chess.BLACK
                        else:
                            continue  # Skip draws for this validation
                        
                        board = game.board()
                        moves_analyzed = 0
                        
                        for move in game.mainline_moves():
                            if moves_analyzed >= 10:  # Limit moves per game
                                break
                            
                            # Evaluate position before move
                            self.evaluation_engine = EvaluationEngine(board)
                            score = self.evaluation_engine.evaluate_position()
                            
                            # Categorize score based on whose turn it is
                            if board.turn == winner_color:
                                scores['winner_scores'].append(score)
                            else:
                                scores['loser_scores'].append(score)
                            
                            board.push(move)
                            moves_analyzed += 1
                            positions_checked += 1
                            
                            if positions_checked >= sample_size:
                                break
                    
            except Exception as e:
                print(f"      ⚠️ Error validating {pgn_file.name}: {e}")
                continue
            
            # Calculate statistics
            if scores['winner_scores'] and scores['loser_scores']:
                winner_avg = np.mean(scores['winner_scores'])
                loser_avg = np.mean(scores['loser_scores'])
                
                validation_results[pgn_file.name] = {
                    'winner_avg_score': winner_avg,
                    'loser_avg_score': loser_avg,
                    'score_difference': winner_avg - loser_avg,
                    'alignment_good': winner_avg > loser_avg
                }
                
                status = "✅" if winner_avg > loser_avg else "❌"
                print(f"      {status} Winner avg: {winner_avg:.3f}, Loser avg: {loser_avg:.3f}")
        
        return validation_results
    
    def load_training_data(self):
        """Load and process training data from PGN files"""
        print("\n📚 Loading training data...")
        
        training_dir = Path("training_positions")
        if not training_dir.exists():
            raise FileNotFoundError("Training positions directory not found")
        
        pgn_files = list(training_dir.glob("*.pgn"))
        print(f"   Found {len(pgn_files)} PGN files")
        
        # Validate evaluation alignment if configured
        if self.rl_config.get('validate_evaluations', False):
            validation_results = self.validate_evaluation_alignment(pgn_files)
            self.training_stats['validation_scores'] = validation_results
        
        experiences = []
        max_games = self.rl_config['max_games_per_file']
        
        for pgn_file in pgn_files:
            print(f"   📄 Processing {pgn_file.name}...")
            file_experiences = self._process_pgn_file(pgn_file, max_games)
            experiences.extend(file_experiences)
            print(f"      ✅ Added {len(file_experiences)} experiences")
        
        print(f"🎯 Total training experiences: {len(experiences)}")
        return experiences
    
    def _process_pgn_file(self, pgn_file, max_games):
        """Process a single PGN file and extract training experiences"""
        experiences = []
        games_processed = 0
        
        try:
            with open(pgn_file, encoding='utf-8', errors='ignore') as f:
                while games_processed < max_games:
                    game = chess.pgn.read_game(f)
                    if not game:
                        break
                    
                    # Process game and extract experiences
                    game_experiences = self._extract_game_experiences(game)
                    experiences.extend(game_experiences)
                    games_processed += 1
                    
        except Exception as e:
            print(f"      ⚠️ Error processing {pgn_file.name}: {e}")
        
        return experiences
    
    def _extract_game_experiences(self, game):
        """Extract training experiences from a single game"""
        experiences = []
        
        # Get game result for reward weighting
        result = game.headers.get("Result", "*")
        winner_color = None
        if result == "1-0":
            winner_color = chess.WHITE
        elif result == "0-1":
            winner_color = chess.BLACK
        
        board = game.board()
        moves = list(game.mainline_moves())
        
        # Sample positions based on sampling rate
        sampling_rate = self.rl_config['position_sampling_rate']
        sampled_indices = random.sample(
            range(len(moves)), 
            int(len(moves) * sampling_rate)
        )
        
        for i in sampled_indices:
            try:
                move = moves[i]
                current_board = board.copy()
                
                # Replay game up to this position
                for j in range(i):
                    current_board.push(moves[j])
                
                # Get current state
                state = self.position_encoder.encode_position(current_board)
                
                # Get action (move)
                action = self.position_encoder.encode_move(move, current_board)
                if action == -1:  # Invalid move
                    continue
                
                # Calculate reward using evaluation engine
                self.evaluation_engine = EvaluationEngine(current_board)
                base_reward = self.evaluation_engine.evaluate_position()
                
                # Apply winner multiplier if this move was by the winning player
                multiplier = 1.0
                if (winner_color == current_board.turn and 
                    winner_color is not None):
                    multiplier = self.rl_config['winner_reward_multiplier']
                
                reward = base_reward * multiplier
                
                # Get next state
                next_board = current_board.copy()
                next_board.push(move)
                next_state = self.position_encoder.encode_position(next_board)
                
                # Check if game is done
                done = next_board.is_game_over()
                
                # Create experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                experiences.append(experience)
                
            except Exception as e:
                continue  # Skip problematic positions
        
        return experiences
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting Reinforcement Learning Training")
        print(f"   Epochs: {self.rl_config['epochs']}")
        print(f"   Batch size: {self.rl_config['batch_size']}")
        
        # Load training data
        training_experiences = self.load_training_data()
        
        # Add experiences to replay buffer
        for exp in training_experiences:
            self.replay_buffer.push(exp)
        
        print(f"   Replay buffer size: {len(self.replay_buffer)}")
        
        # Training loop
        for epoch in range(self.rl_config['epochs']):
            print(f"\n   🔥 Epoch {epoch+1}/{self.rl_config['epochs']}")
            
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_rewards = []
            
            # Training batches
            num_batches = len(self.replay_buffer) // self.rl_config['batch_size']
            
            for batch_idx in range(num_batches):
                if len(self.replay_buffer) < self.rl_config['batch_size']:
                    break
                
                # Sample batch
                batch = self.replay_buffer.sample(self.rl_config['batch_size'])
                
                # Train on batch
                actor_loss, critic_loss, avg_reward = self._train_batch(batch)
                
                epoch_actor_losses.append(actor_loss)
                epoch_critic_losses.append(critic_loss)
                epoch_rewards.append(avg_reward)
                
                # Progress update
                if batch_idx % 100 == 0:
                    print(f"      Batch {batch_idx}: Actor Loss = {actor_loss:.4f}, "
                          f"Critic Loss = {critic_loss:.4f}, Avg Reward = {avg_reward:.4f}")
            
            # Epoch statistics
            if epoch_actor_losses:
                avg_actor_loss = np.mean(epoch_actor_losses)
                avg_critic_loss = np.mean(epoch_critic_losses)
                avg_reward = np.mean(epoch_rewards)
                
                self.training_stats['actor_losses'].append(avg_actor_loss)
                self.training_stats['critic_losses'].append(avg_critic_loss)
                self.training_stats['average_rewards'].append(avg_reward)
                
                print(f"      📊 Epoch {epoch+1} - Actor Loss: {avg_actor_loss:.4f}, "
                      f"Critic Loss: {avg_critic_loss:.4f}, Avg Reward: {avg_reward:.4f}")
            
            # Update target network
            if epoch % self.rl_config['target_update_frequency'] == 0:
                self._soft_update_target_network()
        
        # Save trained models
        self._save_models()
        print("\n✅ Reinforcement Learning training complete!")
    
    def _train_batch(self, batch):
        """Train on a single batch of experiences"""
        # Convert batch to tensors
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Train Critic
        current_q_values = self.critic(states).squeeze()
        next_q_values = self.target_critic(next_states).squeeze()
        target_q_values = rewards + (self.rl_config['gamma'] * next_q_values * ~dones)
        
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train Actor
        action_probs = self.actor(states)
        state_values = self.critic(states).squeeze()
        
        # Calculate advantage
        advantages = target_q_values.detach() - state_values.detach()
        
        # Policy gradient loss
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        actor_loss = -(log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), rewards.mean().item()
    
    def _soft_update_target_network(self):
        """Soft update of target network"""
        tau = self.rl_config['tau']
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def _save_models(self):
        """Save trained models"""
        torch.save(self.actor.state_dict(), "rl_actor_model.pth")
        torch.save(self.critic.state_dict(), "rl_critic_model.pth")
        
        # Save training statistics (convert numpy types to native Python)
        import json
        
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        stats_to_save = convert_numpy_types(self.training_stats)
        
        with open("training_stats.json", "w") as f:
            json.dump(stats_to_save, f, indent=2)
        
        print("💾 Models and statistics saved!")

if __name__ == "__main__":
    trainer = ReinforcementTrainer()
    trainer.train()
