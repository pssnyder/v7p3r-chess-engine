"""
Firebase Cloud Functions Integration for V7P3R Chess Engine

This module provides easy integration with the deployed Firebase Cloud Functions
for data ingestion, ETL processing, and analytics.

Usage Examples:
    # Submit game data
    firebase_client = FirebaseCloudClient()
    firebase_client.submit_game_data(game_id, pgn, winner, engine_white, engine_black)
    
    # Run ETL processing
    result = firebase_client.run_etl_processing(limit=100)
    
    # Check ETL status
    status = firebase_client.get_etl_status()
"""

import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FirebaseCloudClient:
    """Client for interacting with V7P3R Firebase Cloud Functions."""
    
    def __init__(self):
        """Initialize the Firebase Cloud Functions client."""
        self.base_urls = {
            'data_ingestion': 'https://submit-game-data-rcocsjdzja-uc.a.run.app',
            'etl_processing': 'https://us-central1-v7p3r-chess-engine.cloudfunctions.net/run_etl_processing',
            'etl_status': 'https://us-central1-v7p3r-chess-engine.cloudfunctions.net/get_etl_status'
        }
        
    def submit_game_data(self, game_id: str, pgn: str, winner: str, 
                        engine_white: str, engine_black: str, 
                        additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Submit game data to Firebase for storage.
        
        Args:
            game_id: Unique identifier for the game
            pgn: PGN string of the game
            winner: Winner of the game ('white', 'black', 'draw')
            engine_white: Name of the white engine
            engine_black: Name of the black engine
            additional_data: Optional additional game metadata
            
        Returns:
            Response dictionary from the Cloud Function
        """
        data = {
            'game_id': game_id,
            'pgn': pgn,
            'winner': winner,
            'engine_white': engine_white,
            'engine_black': engine_black
        }
        
        if additional_data:
            data.update(additional_data)
            
        try:
            response = requests.post(
                self.base_urls['data_ingestion'],
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error submitting game data: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_etl_processing(self, limit: int = 100, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Trigger ETL processing on raw game data.
        
        Args:
            limit: Maximum number of records to process
            force_reprocess: Whether to reprocess already processed records
            
        Returns:
            ETL job result dictionary
        """
        data = {
            'limit': limit,
            'force_reprocess': force_reprocess
        }
        
        try:
            response = requests.post(
                self.base_urls['etl_processing'],
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=60  # ETL might take longer
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error running ETL processing: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_etl_status(self) -> Dict[str, Any]:
        """
        Get the current ETL processing status.
        
        Returns:
            ETL status dictionary
        """
        try:
            response = requests.get(
                self.base_urls['etl_status'],
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting ETL status: {e}")
            return {"status": "error", "error": str(e)}
    
    def submit_and_process_game(self, game_id: str, pgn: str, winner: str,
                               engine_white: str, engine_black: str,
                               process_immediately: bool = True) -> Dict[str, Any]:
        """
        Submit game data and optionally trigger immediate ETL processing.
        
        Args:
            game_id: Unique identifier for the game
            pgn: PGN string of the game
            winner: Winner of the game
            engine_white: Name of the white engine
            engine_black: Name of the black engine
            process_immediately: Whether to trigger ETL processing after submission
            
        Returns:
            Combined result dictionary
        """
        # Submit the game data
        submit_result = self.submit_game_data(game_id, pgn, winner, engine_white, engine_black)
        
        if submit_result.get('status') != 'success':
            return submit_result
        
        # Optionally trigger ETL processing
        if process_immediately:
            etl_result = self.run_etl_processing(limit=1)
            return {
                "submit_result": submit_result,
                "etl_result": etl_result
            }
        
        return submit_result


# Example usage functions for integration with existing chess engine code
def integrate_with_existing_game_loop():
    """
    Example of how to integrate Firebase backend with existing game simulation loop.
    """
    firebase_client = FirebaseCloudClient()
    
    # This would be called after each game in your simulation
    def on_game_completed(game_id, pgn, winner, white_engine, black_engine):
        result = firebase_client.submit_game_data(
            game_id=game_id,
            pgn=pgn,
            winner=winner,
            engine_white=white_engine,
            engine_black=black_engine
        )
        
        if result.get('status') == 'success':
            logger.info(f"Game {game_id} submitted to Firebase successfully")
        else:
            logger.error(f"Failed to submit game {game_id}: {result.get('error')}")
        
        return result
    
    # Periodic ETL processing (could be called every N games or on a schedule)
    def run_periodic_etl():
        firebase_client = FirebaseCloudClient()
        
        # Check current status
        status = firebase_client.get_etl_status()
        unprocessed = status.get('unprocessed_games', 0)
        
        if unprocessed > 0:
            logger.info(f"Processing {unprocessed} unprocessed games...")
            result = firebase_client.run_etl_processing()
            
            if result.get('status') == 'success':
                processed = result.get('records_processed', 0)
                logger.info(f"ETL processing completed: {processed} games processed")
            else:
                logger.error(f"ETL processing failed: {result.get('error')}")
        else:
            logger.info("No unprocessed games found")


if __name__ == "__main__":
    # Test the Firebase integration
    firebase_client = FirebaseCloudClient()
    
    # Test data submission
    test_result = firebase_client.submit_game_data(
        game_id="integration-test-001",
        pgn="1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4 1-0",
        winner="white",
        engine_white="V7P3R",
        engine_black="TestEngine"
    )
    print(f"Data submission test: {test_result}")
    
    # Test ETL processing
    etl_result = firebase_client.run_etl_processing(limit=5)
    print(f"ETL processing test: {etl_result}")
    
    # Test status check
    status = firebase_client.get_etl_status()
    print(f"ETL status: {status}")
