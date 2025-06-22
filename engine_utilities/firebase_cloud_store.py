import os
import json
import logging
from datetime import datetime
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.configure_firebase import initialize_firebase

logger = logging.getLogger(__name__)

class FirebaseCloudStore:
    """Handles uploading and retrieving data from Firebase Cloud Storage and Firestore"""
    
    def __init__(self):
        """Initialize the CloudStore"""
        self.db, self.bucket = initialize_firebase()
        self.is_connected = self.db is not None and self.bucket is not None
        if not self.is_connected:
            logger.warning("Failed to connect to Firebase backend")
        
    def upload_pgn(self, pgn_content, game_id):
        """Upload a PGN file to Cloud Storage"""
        if not self.is_connected or self.bucket is None:
            logger.error("Not connected to Firebase or bucket is None")
            return False
            
        try:
            # Create a storage reference
            blob = self.bucket.blob(f"pgns/{game_id}.pgn")
            blob.upload_from_string(pgn_content)
            logger.info(f"Uploaded PGN for game {game_id}")
            return True
        except Exception as e:
            logger.error(f"Error uploading PGN: {e}")
            return False
    
    def download_pgn(self, game_id):
        """Download a PGN file from Cloud Storage"""
        if not self.is_connected or self.bucket is None:
            logger.error("Not connected to Firebase or bucket is None")
            return None
            
        try:
            blob = self.bucket.blob(f"pgns/{game_id}.pgn")
            if not blob.exists():
                logger.error(f"PGN file for game {game_id} not found")
                return None                
            return blob.download_as_text()
        except Exception as e:
            logger.error(f"Error downloading PGN: {e}")
            return None
    
    def upload_game_metadata(self, game_id, metadata):
        """Upload game metadata to Firestore"""
        if not self.is_connected or self.db is None:
            logger.error("Not connected to Firebase or database is None")
            return False
            
        try:
            # Add timestamp
            metadata['uploaded_at'] = datetime.now()
            
            # Store in Firestore
            self.db.collection('games').document(game_id).set(metadata)
            logger.info(f"Uploaded metadata for game {game_id}")
            return True
        except Exception as e:
            logger.error(f"Error uploading game metadata: {e}")
            return False
    
    def upload_metrics(self, metrics_data):
        """Upload metrics data to Firestore"""
        if not self.is_connected or self.db is None:
            logger.error("Not connected to Firebase or database is None")
            return False
            
        try:
            # Add timestamp if not present
            if 'timestamp' not in metrics_data:
                metrics_data['timestamp'] = datetime.now()
            
            # Store in Firestore
            doc_ref = self.db.collection('metrics').document()
            doc_ref.set(metrics_data)
            logger.info(f"Uploaded metrics with ID {doc_ref.id}")
            return True
        except Exception as e:
            logger.error(f"Error uploading metrics: {e}")
            return False
    
    def get_game_metadata(self, game_id=None, limit=100):
        """Retrieve game metadata from Firestore"""
        if not self.is_connected or self.db is None:
            logger.error("Not connected to Firebase or database is None")
            return []
            
        try:            
            if game_id:
                # Get a specific game
                doc = self.db.collection('games').document(game_id).get()
                return doc.to_dict() if doc.exists else None
            else:
                # Get recent games
                docs = self.db.collection('games').order_by('uploaded_at', direction='DESCENDING').limit(limit).stream()
                return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving game metadata: {e}")
            return [] if not game_id else None
    
    def get_metrics(self, engine_name=None, start_date=None, end_date=None, limit=100):
        """Retrieve metrics from Firestore with optional filtering"""
        if not self.is_connected or self.db is None:
            logger.error("Not connected to Firebase or database is None")
            return []
            
        try:
            query = self.db.collection('metrics')
            
            # Apply filters
            if engine_name:
                query = query.where('engine_name', '==', engine_name)
            if start_date:
                query = query.where('timestamp', '>=', start_date)
            if end_date:
                query = query.where('timestamp', '<=', end_date)
                
            # Execute query
            docs = query.order_by('timestamp', direction='DESCENDING').limit(limit).stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving metrics: {e}")
            return []
    
    def upload_model(self, model_path, model_name, version, metadata=None):
        """Upload a trained model file to Cloud Storage"""
        if not self.is_connected or self.bucket is None or self.db is None:
            logger.error("Not connected to Firebase or storage/database is None")
            return False
            
        try:
            # Ensure bucket is valid
            if self.bucket is None:
                logger.error("Bucket is None, cannot upload model")
                return False
            
            # Create a storage reference
            blob = self.bucket.blob(f"models/{model_name}/v{version}.pth")
            blob.upload_from_filename(model_path)
            
            # Store metadata
            if metadata:
                metadata['uploaded_at'] = datetime.now()
                metadata['version'] = version
                metadata['model_name'] = model_name
                self.db.collection('models').document(f"{model_name}_v{version}").set(metadata)
                
            logger.info(f"Uploaded model {model_name} v{version}")
            return True
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return False
    
    def download_model(self, model_name, version=None):
        """Download the latest or specific version of a model from Cloud Storage"""
        if not self.is_connected or self.db is None:
            logger.error("Not connected to Firebase or database is None")
            return None
            
        try:
            # Ensure bucket is valid
            if self.bucket is None:
                logger.error("Bucket is None, cannot download model")
                return None
            
            # If version not specified, get the latest
            if not version:
                if self.db is None:
                    logger.error("Database connection is None")
                    return None
                
                models = self.db.collection('models').where('model_name', '==', model_name).order_by('version', direction='DESCENDING').limit(1).stream()
                models_list = list(models)
                if not models_list:
                    logger.error(f"No models found for {model_name}")
                    return None
                    
                version = models_list[0].get('version')
            
            # Download the model file
            local_path = f"downloaded_{model_name}_v{version}.pth"
            blob = self.bucket.blob(f"models/{model_name}/v{version}.pth")
            if not blob.exists():
                logger.error(f"Model {model_name} v{version} not found")
                return None
                
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded model {model_name} v{version}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return None
