# engine_utilities/cloud_store.py
"""
CloudStore provides helpers to upload game data to Google Cloud Storage and Firestore.
"""
import os
from google.cloud import storage, firestore
import logging

logger = logging.getLogger(__name__)

class CloudStore:
    def __init__(self, bucket_name=None, firestore_collection='games'):
        # GCS bucket
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError('GCS_BUCKET_NAME env var or bucket_name argument required')
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Firestore
        self.db = firestore.Client()
        self.collection = self.db.collection(firestore_collection)

    def upload_game_pgn(self, game_id: str, pgn_text: str) -> str:
        """Upload PGN text to GCS under games/{game_id}.pgn and return public URL."""
        blob = self.bucket.blob(f'games/{game_id}.pgn')
        blob.upload_from_string(pgn_text, content_type='text/plain')
        url = blob.public_url
        logger.info(f'Uploaded PGN for {game_id} to {url}')
        return url

    def upload_game_metadata(self, game_id: str, metadata: dict):
        """Store game metadata in Firestore under document game_id."""
        doc_ref = self.collection.document(game_id)
        doc_ref.set(metadata)
        logger.info(f'Uploaded metadata for {game_id} to Firestore')

    def upload_move_metrics(self, game_id: str, move_metrics: list):
        """Batch upload move-level metrics under subcollection 'moves' for a game."""
        subcol = self.collection.document(game_id).collection('moves')
        batch = self.db.batch()
        for metric in move_metrics:
            doc = subcol.document()
            batch.set(doc, metric)
        batch.commit()
        logger.info(f'Uploaded {len(move_metrics)} move metrics for {game_id}')
