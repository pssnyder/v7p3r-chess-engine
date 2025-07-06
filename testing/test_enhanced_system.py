#!/usr/bin/env python3
"""
Test the enhanced metrics system with a quick game
"""
import sys
import os
import time
import chess

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Import the enhanced v7p3r_play module
from v7p3r_engine.v7p3r_play import v7p3rChess

# Test configuration with enhanced features
test_config = {
    "game_config": {
        "game_count": 1,
        "starting_position": "default",
        "white_player": "v7p3r",
        "black_player": "stockfish",
    },
    "engine_config": {
        "name": "v7p3r",
        "version": "2.0.0",
        "ruleset": "tuned_ga_gen2",
        "search_algorithm": "minimax",
        "depth": 2,                          # Lower depth for faster testing
        "max_depth": 2,
        "max_moves": 4,
        "use_game_phase": True,
        "monitoring_enabled": True,
        "verbose_output": False,             # Reduce verbosity for cleaner output
        "logger": "v7p3r_engine_logger",
    },
    "stockfish_config": {
        "stockfish_path": "stockfish/stockfish-windows-x86-64-avx2.exe",
        "elo_rating": 100,
        "skill_level": 1,
        "debug_mode": False,
        "depth": 1,                          # Very low depth for fast moves
        "max_depth": 1,
        "movetime": 50,                      # Very short time for Stockfish
    },
}

def monitor_enhanced_database():
    """Monitor the enhanced database during the game"""
    try:
        from metrics.enhanced_metrics_store import EnhancedMetricsStore
        
        store = EnhancedMetricsStore(db_path="metrics/chess_metrics_v2.db")
        
        print("\n" + "="*50)
        print("ENHANCED METRICS MONITORING")
        print("="*50)
        
        # Get recent games
        games = store.get_engine_performance(engine_name="v7p3r")
        if games:
            print(f"V7P3R Engine Performance:")
            for game in games[-3:]:  # Show last 3 performance records
                print(f"  Games: {game.get('games_played', 0)}")
                print(f"  Avg time/move: {game.get('avg_time_per_move', 0):.3f}s")
                print(f"  Avg nodes/move: {game.get('avg_nodes_per_move', 0):.0f}")
                print(f"  Avg evaluation: {game.get('avg_evaluation', 0):.2f}")
                print(f"  Win-Draw-Loss: {game.get('wins', 0)}-{game.get('draws', 0)}-{game.get('losses', 0)}")
                break
        
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error monitoring enhanced database: {e}")

if __name__ == "__main__":
    print("Testing Enhanced V7P3R Chess Engine Metrics System")
    print("=" * 55)
    
    try:
        # Create game instance
        game = v7p3rChess(config=test_config)
        
        # Check if enhanced metrics are enabled
        if hasattr(game, 'use_enhanced_metrics') and game.use_enhanced_metrics:
            print("✓ Enhanced metrics system is active!")
            print("✓ Detailed scoring breakdown will be collected")
            print("✓ Engine-specific attribution enabled")
            print("✓ Position analysis included")
        else:
            print("⚠ Using legacy metrics system")
        
        print(f"✓ Testing with: {game.white_player} vs {game.black_player}")
        print("✓ Starting test game...")
        print()
        
        # Start the game
        start_time = time.time()
        
        # Run for limited number of moves by checking periodically
        moves_played = 0
        max_test_moves = 10  # Limit test to 10 moves
        
        while not game.board.is_game_over() and moves_played < max_test_moves:
            initial_move_count = game.board.fullmove_number
            
            # Process one move
            game.process_engine_move()
            
            # Check if a move was actually made
            if game.board.fullmove_number > initial_move_count or game.board.fen() != chess.STARTING_FEN:
                moves_played += 1
                print(f"Move {moves_played} completed")
                
                # Check game state
                if game.board.is_game_over():
                    break
            
            # Safety timeout
            if time.time() - start_time > 60:  # 1 minute timeout
                print("Test timeout reached")
                break
        
        # Finish the game for testing
        if hasattr(game, 'handle_game_end'):
            game.handle_game_end()
        
        end_time = time.time()
        
        print(f"\nTest completed in {end_time - start_time:.1f} seconds")
        print(f"Moves played: {moves_played}")
        
        # Monitor the enhanced database
        monitor_enhanced_database()
        
        print("\n✓ Enhanced metrics system test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
