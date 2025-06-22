import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# This file will manage Firebase configuration
# Note: You'll need to install the firebase-admin package: pip install firebase-admin

# Path to service account key
SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), 'firebase_service_account.json')

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        # Import here to avoid errors if the package is not installed
        import firebase_admin
        from firebase_admin import credentials, firestore, storage
        
        # Check if already initialized
        if not len(firebase_admin._apps):
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'v7p3r-chess-engine.appspot.com',
                'projectId': 'v7p3r-chess-engine'
            })
        
        # Initialize Firestore and Storage clients
        db = firestore.client()
        bucket = storage.bucket()
        
        logger.info("Firebase initialization successful")
        return db, bucket
    except ImportError:
        logger.error("Firebase Admin SDK not installed. Run: pip install firebase-admin")
        return None, None
    except Exception as e:
        logger.error(f"Error initializing Firebase: {e}")
        return None, None
