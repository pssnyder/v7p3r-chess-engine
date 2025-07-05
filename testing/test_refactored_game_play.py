#!/usr/bin/env python3
"""
Test the refactored enhanced metrics system in actual game play
"""

import sys
import os
import chess

# Add the v7p3r_engine path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'v7p3r_engine'))

from v7p3r_engine.v7p3r_play import v7p3rChess

def test_game_with_refactored_metrics():
    """
    Test a short game using the refactored metrics system
    """
    print("Testing Game with Refactored Enhanced Metrics")
    print("=" * 50)
    
    try:
        # Create a game configuration for a short test
        test_config = {
            "game_config": {
                "game_count": 1,
                "white_player": "v7p3r",
                "black_player": "v7p3r",
                "starting_position": "default"
            },
            "engine_config": {
                "monitoring_enabled": True,
                "verbose_output": False,
                "depth": 2,
                "max_depth": 3
            },
            "stockfish_config": {
                "stockfish_path": "engines/stockfish/stockfish.exe",
                "depth": 2,
                "skill_level": 5
            }
        }
        
        # Initialize the game with refactored metrics
        print("Initializing game...")
        game = v7p3rChess(config=test_config)
        
        # Verify refactored system is available
        print(f"Refactored metrics available: {game.use_refactored_metrics}")
        print(f"Enhanced metrics available: {game.use_enhanced_metrics}")
        
        if game.use_refactored_metrics:
            print("✓ Refactored enhanced metrics system is active")
        else:
            print("✗ Refactored enhanced metrics system is NOT active")
        
        # Play a few moves to test metrics collection
        moves_played = 0
        max_moves = 6  # Play 3 moves for each side
        
        print(f"\nPlaying {max_moves} moves to test metrics collection...")
        
        while moves_played < max_moves and not game.board.is_game_over():
            print(f"\n--- Move {moves_played + 1} ---")
            print(f"Current player: {'White' if game.board.turn == chess.WHITE else 'Black'}")
            print(f"Current FEN: {game.board.fen()}")
            
            # Process one engine move
            initial_move_count = len(game.board.move_stack)
            game.process_engine_move()
            
            # Check if move was actually made
            if len(game.board.move_stack) > initial_move_count:
                moves_played += 1
                last_move = game.board.peek()
                print(f"Move played: {last_move}")
                
                # Check if datasets are populated after the move
                if hasattr(game.engine.search_engine, 'search_dataset'):
                    search_data = game.engine.search_engine.search_dataset
                    print(f"Search data updated - nodes: {search_data.get('nodes_searched', 'N/A')}")
                
                if hasattr(game.engine.scoring_calculator, 'score_dataset'):
                    score_data = game.engine.scoring_calculator.score_dataset
                    print(f"Score data updated - total components: {len(score_data)}")
                    print(f"Total score: {score_data.get('score', 'N/A')}")
            else:
                print("No move was made - ending test")
                break
        
        print(f"\n--- Game Summary ---")
        print(f"Total moves played: {len(game.board.move_stack)}")
        print(f"Final position: {game.board.fen()}")
        print(f"Game over: {game.board.is_game_over()}")
        
        # Test direct metrics collection on final position
        if game.use_refactored_metrics and len(game.board.move_stack) > 0:
            print(f"\n--- Testing Direct Metrics Collection ---")
            last_move = game.board.peek()
            
            # Get comprehensive metrics for the last move
            metrics = game.refactored_collector.collect_comprehensive_metrics(
                game.engine, game.board, last_move, time_taken=0.1
            )
            
            print(f"Comprehensive metrics collected: {len(metrics)} components")
            
            # Show key metrics
            key_metrics = [
                'search_algorithm', 'depth_reached', 'nodes_searched', 'total_score',
                'material_balance', 'game_phase', 'position_type', 'engine_name'
            ]
            
            print("Key metrics:")
            for key in key_metrics:
                if key in metrics:
                    print(f"  {key}: {metrics[key]}")
            
            # Show scoring breakdown
            scoring_metrics = {k: v for k, v in metrics.items() 
                             if k.endswith('_score') and v != 0.0}
            print(f"\nNon-zero scoring components: {len(scoring_metrics)}")
            for key, value in list(scoring_metrics.items())[:10]:
                print(f"  {key}: {value}")
        
        # Clean up
        game.cleanup_engines()
        print(f"\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_database_storage():
    """
    Test that metrics are being properly stored in the database
    """
    print(f"\n--- Testing Database Storage ---")
    
    try:
        # Check if enhanced metrics database exists
        from metrics.enhanced_metrics_store import EnhancedMetricsStore
        
        store = EnhancedMetricsStore()
        print(f"Enhanced metrics database: {store.db_path}")
        
        # Try to query recent game data
        import sqlite3
        conn = sqlite3.connect(store.db_path)
        cursor = conn.cursor()
        
        # Check games table
        cursor.execute("SELECT COUNT(*) FROM games")
        game_count = cursor.fetchone()[0]
        print(f"Games in database: {game_count}")
        
        # Check move metrics table
        cursor.execute("SELECT COUNT(*) FROM move_metrics")
        move_count = cursor.fetchone()[0]
        print(f"Move metrics in database: {move_count}")
        
        # Get most recent moves
        if move_count > 0:
            cursor.execute("""
                SELECT game_id, move_number, move_uci, search_algorithm, total_score
                FROM move_metrics 
                ORDER BY id DESC 
                LIMIT 5
            """)
            recent_moves = cursor.fetchall()
            print(f"\nMost recent moves:")
            for move in recent_moves:
                print(f"  Game: {move[0]}, Move: {move[1]}, UCI: {move[2]}, Algorithm: {move[3]}, Score: {move[4]}")
        
        conn.close()
        
    except Exception as e:
        print(f"Database storage test failed: {e}")

if __name__ == "__main__":
    test_game_with_refactored_metrics()
    test_database_storage()
