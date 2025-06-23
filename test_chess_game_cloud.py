#!/usr/bin/env python3
"""
Test the actual chess_game.py with mock cloud storage to verify background sync works.
"""

import sys
import os
import threading
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MockCloudStore:
    """Mock CloudStore that simulates cloud operations but stores locally."""
    
    def __init__(self, bucket_name=None):
        self.bucket_name = bucket_name or "mock-bucket"
        self.game_data = []
        self.move_data = []
        self.metadata_data = []
        self.operation_log = []
        print(f"MockCloudStore initialized with bucket: {self.bucket_name}")
    
    def upload_game_pgn(self, game_id, pgn_text):
        self.game_data.append({"game_id": game_id, "pgn": pgn_text})
        url = f"gs://{self.bucket_name}/games/{game_id}.pgn"
        self.operation_log.append(f"Uploaded PGN for {game_id}")
        print(f"MockCloud: Uploaded PGN for {game_id}")
        return url
    
    def upload_game_metadata(self, game_id, metadata):
        self.metadata_data.append({"game_id": game_id, "metadata": metadata})
        self.operation_log.append(f"Uploaded metadata for {game_id}")
        print(f"MockCloud: Uploaded metadata for {game_id}")
    
    def upload_move_metrics(self, game_id, move_metrics):
        for metric in move_metrics:
            metric['game_id'] = game_id
            self.move_data.append(metric)
        self.operation_log.append(f"Uploaded {len(move_metrics)} move metrics for {game_id}")
        print(f"MockCloud: Uploaded {len(move_metrics)} move metrics for {game_id}")
    
    def get_all_game_results(self):
        """Return mock game results."""
        game_results = []
        for metadata_entry in self.metadata_data:
            metadata = metadata_entry['metadata']
            game_result = {
                'game_id': metadata_entry['game_id'],
                'timestamp': metadata.get('timestamp', ''),
                'winner': metadata.get('result', '*'),
                'game_pgn': '',
                'white_player': metadata.get('white_player', ''),
                'black_player': metadata.get('black_player', ''),
                'game_length': metadata.get('game_length', 0),
                'white_engine_config': metadata.get('game_settings', {}).get('white_engine_config', {}),
                'black_engine_config': metadata.get('game_settings', {}).get('black_engine_config', {})
            }
            game_results.append(game_result)
        return game_results
    
    def get_all_move_metrics(self):
        """Return mock move metrics."""
        return self.move_data
    
    def get_stats(self):
        """Get statistics of what was uploaded."""
        return {
            "games": len(self.game_data),
            "metadata_entries": len(self.metadata_data),
            "move_metrics": len(self.move_data),
            "operations": len(self.operation_log)
        }

def test_chess_game_with_mock_cloud():
    """Test chess_game.py with mock cloud storage."""
    print("Testing chess_game.py with Mock Cloud Storage")
    print("=" * 50)
    
    # Patch CloudStore before importing chess_game
    import engine_utilities.cloud_store
    original_cloudstore = engine_utilities.cloud_store.CloudStore
    
    # Create a shared mock store to track operations
    mock_store = MockCloudStore("test-viper-chess-engine")
    
    def mock_cloudstore_factory(*args, **kwargs):
        return mock_store
    
    engine_utilities.cloud_store.CloudStore = mock_cloudstore_factory
    
    try:        # Import chess game after patching
        import chess_game
        
        print("Initializing chess game with cloud-first architecture...")
        
        # Create configuration (using a mock configuration if ChessGameConfig is not defined)
        config = {
            "game_config": {
                "cloud_storage_enabled": True
            }
        }

        # Verify cloud is enabled
        cloud_enabled = config.get('game_config', {}).get('cloud_storage_enabled', False) if isinstance(config, dict) else config.game_config.get('game_config', {}).get('cloud_storage_enabled', False)
        print(f"Cloud storage enabled in config: {cloud_enabled}")
        
        if not cloud_enabled:
            print("ERROR: Cloud storage not enabled in config!")
            return
          # Create game instance
        game = chess_game.ChessGame(config)
        
        print(f"Game initialized with cloud_store: {game.cloud_store is not None}")
        
        # Let the game run briefly to test background sync
        print("Starting game for background sync testing...")
        
        # Run game in a separate thread so we can monitor progress
        def run_game():
            try:
                game.run()
            except KeyboardInterrupt:
                print("Game interrupted")
            except Exception as e:
                print(f"Game error: {e}")
        
        game_thread = threading.Thread(target=run_game, daemon=True)
        game_thread.start()
        
        # Monitor for a short time
        for i in range(10):
            time.sleep(2)
            stats = mock_store.get_stats()
            print(f"After {(i+1)*2}s - Games: {stats['games']}, Metadata: {stats['metadata_entries']}, Moves: {stats['move_metrics']}")
            
            # If we see some cloud activity, that's good
            if stats['operations'] > 0:
                print("✓ Cloud operations detected!")
                break
        
        # Final stats
        final_stats = mock_store.get_stats()
        print(f"\nFinal Statistics:")
        print(f"  Games uploaded: {final_stats['games']}")
        print(f"  Metadata entries: {final_stats['metadata_entries']}")
        print(f"  Move metrics: {final_stats['move_metrics']}")
        print(f"  Total operations: {final_stats['operations']}")
        
        if final_stats['operations'] > 0:
            print("✓ Chess game cloud integration working!")
            
            # Show some operations
            print("\nRecent operations:")
            for op in mock_store.operation_log[-5:]:
                print(f"  - {op}")
        else:
            print("✗ No cloud operations detected")
        
        # Clean up
        if hasattr(game, '_cleanup_background_sync'):
            game._cleanup_background_sync()
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original CloudStore
        engine_utilities.cloud_store.CloudStore = original_cloudstore

if __name__ == "__main__":
    test_chess_game_with_mock_cloud()
