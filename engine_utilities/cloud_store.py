# engine_utilities/cloud_store.py
"""
CloudStore provides helpers to upload game data to Google Cloud Storage and Firestore.
"""
import os
import time
from google.cloud import storage, firestore
from google.api_core import retry, exceptions
import logging
import json

logger = logging.getLogger(__name__)

class CloudStore:
    def __init__(self, bucket_name=None, firestore_collection='games', max_retries=5, retry_delay=1):
        # GCS bucket
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError('GCS_BUCKET_NAME env var or bucket_name argument required')
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Firestore
        self.db = firestore.Client()
        self.collection = self.db.collection(firestore_collection)
        
        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _retry_operation(self, operation, *args, **kwargs):
        """Execute an operation with exponential backoff retry logic."""
        retries = 0
        last_exception = None
        
        while retries < self.max_retries:
            try:
                return operation(*args, **kwargs)
            except (exceptions.GoogleAPIError, exceptions.RetryError, exceptions.ServiceUnavailable) as e:
                last_exception = e
                wait_time = self.retry_delay * (2 ** retries)  # Exponential backoff
                logger.warning(f"Operation failed: {e}. Retrying in {wait_time}s (attempt {retries+1}/{self.max_retries})")
                time.sleep(wait_time)
                retries += 1
          # If we get here, all retries failed
        logger.error(f"Operation failed after {self.max_retries} attempts: {last_exception}")
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Operation failed after {self.max_retries} attempts")

    def upload_game_pgn(self, game_id: str, pgn_text: str) -> str:
        """Upload PGN text to GCS under games/{game_id}.pgn and return public URL."""
        def _upload():
            blob = self.bucket.blob(f'games/{game_id}.pgn')
            blob.upload_from_string(pgn_text, content_type='text/plain')
            url = blob.public_url
            logger.info(f'Uploaded PGN for {game_id} to {url}')
            return url
        
        return self._retry_operation(_upload)

    def upload_game_metadata(self, game_id: str, metadata: dict):
        """Store game metadata in Firestore under document game_id."""
        def _upload():
            doc_ref = self.collection.document(game_id)
            doc_ref.set(metadata)
            logger.info(f'Uploaded metadata for {game_id} to Firestore')
        
        self._retry_operation(_upload)

    def upload_move_metrics(self, game_id: str, move_metrics: list):
        """Batch upload move-level metrics under subcollection 'moves' for a game."""
        def _upload():
            subcol = self.collection.document(game_id).collection('moves')
            batch = self.db.batch()
            for metric in move_metrics:
                doc = subcol.document()
                batch.set(doc, metric)
            batch.commit()
            logger.info(f'Uploaded {len(move_metrics)} move metrics for {game_id}')
        
        self._retry_operation(_upload)
    
    def bulk_upload_games(self, games_data: list):
        """
        Bulk upload multiple games data (PGN and metadata) to cloud storage.
        
        Args:
            games_data: List of dictionaries containing game data with keys:
                        - game_id: Unique identifier for the game
                        - pgn_text: PGN representation of the game
                        - metadata: Dictionary of game metadata
                        - move_metrics: Optional list of move metrics
        """
        logger.info(f"Starting bulk upload of {len(games_data)} games")
        
        for game in games_data:
            game_id = game.get('game_id')
            if not game_id:
                logger.warning("Skipping game with missing game_id")
                continue
                
            try:
                # Upload PGN if present
                if 'pgn_text' in game:
                    self.upload_game_pgn(game_id, game['pgn_text'])
                
                # Upload metadata if present
                if 'metadata' in game:
                    self.upload_game_metadata(game_id, game['metadata'])
                
                # Upload move metrics if present
                if 'move_metrics' in game and game['move_metrics']:
                    self.upload_move_metrics(game_id, game['move_metrics'])
                    
            except Exception as e:
                logger.error(f"Error uploading game {game_id}: {e}")
    
    def upload_raw_simulation_data(self, simulation_id: str, raw_data: dict):
        """
        Upload raw simulation data to a dedicated raw data collection.
        This allows quick storage of simulation results for later processing.
        
        Args:
            simulation_id: Unique identifier for the simulation run
            raw_data: Dictionary containing all raw simulation data
        """
        def _upload():
            # Store in a separate 'raw_simulations' collection
            doc_ref = self.db.collection('raw_simulations').document(simulation_id)
            doc_ref.set({
                'timestamp': firestore.SERVER_TIMESTAMP,
                'data': raw_data
            })
            logger.info(f'Uploaded raw simulation data for {simulation_id}')
        
        self._retry_operation(_upload)
    
    def upload_json_file(self, file_path: str, json_data: dict, content_type='application/json'):
        """
        Upload any JSON data to GCS.
        
        Args:
            file_path: Path within the bucket (e.g., 'simulations/my_sim.json')
            json_data: Dictionary to be stored as JSON
            content_type: Content type of the file
        
        Returns:
            Public URL of the uploaded file
        """
        def _upload():
            blob = self.bucket.blob(file_path)
            blob.upload_from_string(
                json.dumps(json_data, indent=2),
                content_type=content_type
            )
            url = blob.public_url
            logger.info(f'Uploaded JSON file to {url}')
            return url
        
        return self._retry_operation(_upload)
