# v7p3r_rl_training.py
"""
Enhanced v7p3r Reinforcement Learning Training Script

This script implements a comprehensive training pipeline for the v7p3r RL engine including:
- Self-play training with PPO
- Periodic validation against Stockfish
- Model checkpointing and metrics logging
- Resource management and cleanup

Author: v7p3r Project
Date: 2025-06-27
"""

import os
import sys
import yaml
import torch
import time
import argparse
from typing import Dict, Any

# Add paths for v7p3r engine integration
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v7p3r_rl import v7p3rRLEngine
from v7p3r_debug import v7p3rLogger

# Setup centralized logging for this module
v7p3r_rl_training_logger = v7p3rLogger.setup_logger("v7p3r_rl_training")

def create_model_directory(config: Dict[str, Any]):
    """Create directory for model storage."""
    model_path = config.get('model_path', 'v7p3r_rl_engine/v7p3r_rl_models/v7p3r_rl_model.pt')
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"[Training] Created model directory: {model_dir}")

def run_training_session(engine: v7p3rRLEngine, config: Dict[str, Any], logger):
    """Run a complete training session."""
    
    # Training parameters
    total_episodes = config.get('total_episodes', 1000)
    batch_size = config.get('batch_size', 16)
    validation_frequency = config.get('episodes_per_validation', 50)
    save_frequency = config.get('save_frequency', 100)
    
    v7p3r_rl_training_logger.info(f"Starting training session: {total_episodes} episodes, batch size {batch_size}")
    
    # Training loop
    start_time = time.time()
    
    try:
        for episode_batch in range(0, total_episodes, batch_size):
            current_batch_size = min(batch_size, total_episodes - episode_batch)
            
            # Train batch
            v7p3r_rl_training_logger.info(f"Training episodes {episode_batch + 1}-{episode_batch + current_batch_size}")
            engine.train_self_play(current_batch_size, current_batch_size)
            
            # Periodic validation
            if (episode_batch + current_batch_size) % validation_frequency == 0:
                v7p3r_rl_training_logger.info("Running Stockfish validation...")
                validation_stats = engine.validate_performance(num_games=10)
                
                if validation_stats:
                    v7p3r_rl_training_logger.info(f"Validation results: {validation_stats['wins']}W-{validation_stats['losses']}L-{validation_stats['draws']}D "
                              f"(Win rate: {validation_stats['win_rate']:.2%})")
            
            # Periodic model saving
            if (episode_batch + current_batch_size) % save_frequency == 0:
                model_path = config.get('model_path', f'v7p3r_rl_engine/v7p3r_rl_models/v7p3r_rl_checkpoint_{episode_batch + current_batch_size}.pt')
                v7p3r_rl_training_logger.info(f"Saving model checkpoint to {model_path}")
                engine.save(model_path)
            
            # Print progress
            elapsed_time = time.time() - start_time
            episodes_per_hour = (episode_batch + current_batch_size) / (elapsed_time / 3600)
            v7p3r_rl_training_logger.info(f"Progress: {episode_batch + current_batch_size}/{total_episodes} episodes "
                       f"({episodes_per_hour:.1f} episodes/hour)")
    
    except KeyboardInterrupt:
        v7p3r_rl_training_logger.info("Training interrupted by user")
    except Exception as e:
        v7p3r_rl_training_logger.error(f"Training error: {e}")
        raise
    finally:
        # Final model save
        final_model_path = config.get('model_path', 'v7p3r_rl_engine/v7p3r_rl_models/v7p3r_rl_final.pt')
        v7p3r_rl_training_logger.info(f"Saving final model to {final_model_path}")
        engine.save(final_model_path)
        
        # Final validation
        v7p3r_rl_training_logger.info("Running final validation against Stockfish...")
        final_validation = engine.validate_performance(num_games=20)
        if final_validation:
            v7p3r_rl_training_logger.info(f"Final validation: {final_validation['wins']}W-{final_validation['losses']}L-{final_validation['draws']}D "
                       f"(Win rate: {final_validation['win_rate']:.2%})")
        
        # Print training summary
        total_time = time.time() - start_time
        v7p3r_rl_training_logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Print agent statistics
        stats = engine.agent.training_stats
        v7p3r_rl_training_logger.info(f"Training statistics:")
        v7p3r_rl_training_logger.info(f"  Total episodes: {stats['episodes']}")
        v7p3r_rl_training_logger.info(f"  Game results: {stats['wins']}W-{stats['losses']}L-{stats['draws']}D")
        v7p3r_rl_training_logger.info(f"  Average reward: {stats['total_reward']/max(stats['episodes'], 1):.3f}")
        if stats['stockfish_games'] > 0:
            sf_wr = stats['stockfish_wins'] / stats['stockfish_games']
            v7p3r_rl_training_logger.info(f"  Stockfish performance: {stats['stockfish_wins']}/{stats['stockfish_games']} ({sf_wr:.2%})")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="v7p3r RL Engine Training")
    parser.add_argument('--config', default='config/v7p3r_rl_config.yaml', 
                       help='Path to RL config file')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to pre-trained model to continue training')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation, no training')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = v7p3r_rl_training_logger
    v7p3r_rl_training_logger.info("Starting v7p3r RL training script")
    
    # Load configuration
    if not os.path.exists(args.config):
        v7p3r_rl_training_logger.error(f"Config file not found: {args.config}")
        return 1
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override episodes from command line
    config['total_episodes'] = args.episodes
    
    # Create model directory
    create_model_directory(config)
    
    # Initialize engine
    v7p3r_rl_training_logger.info("Initializing v7p3r RL engine...")
    engine = v7p3rRLEngine(args.config)
    
    # Load pre-trained model if specified
    if args.load_model:
        if os.path.exists(args.load_model):
            v7p3r_rl_training_logger.info(f"Loading pre-trained model from {args.load_model}")
            engine.load(args.load_model)
        else:
            v7p3r_rl_training_logger.warning(f"Pre-trained model not found: {args.load_model}")
    
    try:
        if args.validate_only:
            # Only run validation
            v7p3r_rl_training_logger.info("Running validation only...")
            validation_stats = engine.validate_performance(num_games=20)
            if validation_stats:
                v7p3r_rl_training_logger.info(f"Validation results: {validation_stats}")
        else:
            # Run full training
            run_training_session(engine, config, logger)
        
        return 0
    
    except Exception as e:
        v7p3r_rl_training_logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        v7p3r_rl_training_logger.info("Cleaning up resources...")
        engine.cleanup()

if __name__ == "__main__":
    # Enable CUDA optimization
    torch.backends.cudnn.benchmark = True
    
    # Run training
    exit_code = main()
    sys.exit(exit_code)