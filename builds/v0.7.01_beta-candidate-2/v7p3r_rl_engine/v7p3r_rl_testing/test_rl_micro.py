#!/usr/bin/env python3
"""
Quick test of v7p3r RL engine functionality
"""

import sys
import os

# Add path for imports
sys.path.insert(0, os.path.abspath('.'))

def test_rl_engine():
    """Test basic RL engine functionality."""
    print("Testing v7p3r RL Engine...")
    
    try:
        from v7p3r_rl_engine.v7p3r_rl import v7p3rRLEngine
        
        # Initialize engine
        print("1. Initializing engine...")
        engine = v7p3rRLEngine()
        
        # Test move selection
        print("2. Testing move selection...")
        import chess
        board = chess.Board()
        move = engine.search(board, chess.WHITE)
        print(f"   Selected move: {move}")
        
        # Test self-play (very short)
        print("3. Testing self-play...")
        game_data = engine.agent.play_self_game(max_moves=3)
        print(f"   Game result: {game_data['result']} in {game_data['moves']} moves")
        print(f"   Reward: {game_data['total_reward']:.3f}")
        
        # Test training (micro batch)
        print("4. Testing training...")
        train_stats = engine.agent.train_episode_batch(batch_size=1)
        if train_stats:
            print(f"   Policy loss: {train_stats.get('policy_loss', 'N/A'):.4f}")
        
        # Cleanup
        print("5. Cleaning up...")
        engine.cleanup()
        
        print("Γ£ô All tests passed!")
        return True
        
    except Exception as e:
        print(f"Γ£ù Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rl_engine()
    sys.exit(0 if success else 1)
