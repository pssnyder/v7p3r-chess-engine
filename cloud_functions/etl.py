# ETL Cloud Functions for V7P3R Chess Engine
# This file contains standalone ETL functions

import json
import logging
import datetime
import uuid
from typing import Dict, List, Any

from firebase_functions import https_fn
from firebase_functions.options import CorsOptions
from firebase_admin import firestore
from google.cloud.firestore import SERVER_TIMESTAMP

logger = logging.getLogger(__name__)


@https_fn.on_request(cors=CorsOptions(cors_origins="*", cors_methods=["post"]))
def run_etl_processing(req):
    """Cloud Function to perform ETL processing on raw game data."""
    if req.method != 'POST':
        return https_fn.Response("Method not allowed", status=405)

    try:
        request_data = req.get_json(silent=True) or {}
        limit = request_data.get('limit', 100)
        
        db = firestore.client()
        raw_games_ref = db.collection('raw_games')
        query = raw_games_ref.limit(limit)
        games = query.stream()
        
        processed_count = 0
        failed_count = 0
        
        for game_doc in games:
            try:
                game_data = game_doc.to_dict()
                game_id = game_doc.id
                
                # Basic analytics
                analytics_data = {
                    'game_id': game_id,
                    'winner': game_data.get('winner'),
                    'engine_white': game_data.get('engine_white'),
                    'engine_black': game_data.get('engine_black'),
                    'processed_at': datetime.datetime.now().isoformat()
                }
                
                # Store analytics
                analytics_ref = db.collection('game_analytics').document(game_id)
                analytics_ref.set(analytics_data)
                  # Mark as processed
                game_doc.reference.update({
                    'processed': True, 
                    'processed_at': SERVER_TIMESTAMP
                })
                
                processed_count += 1
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to process game {game_doc.id}: {e}")
        
        response_data = {
            "status": "success",
            "job_id": str(uuid.uuid4()),
            "records_processed": processed_count,
            "records_failed": failed_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return https_fn.Response(json.dumps(response_data), status=200, mimetype="application/json")

    except Exception as e:
        error_msg = f"Error in ETL processing: {str(e)}"
        logger.error(error_msg)
        
        response_data = {
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return https_fn.Response(json.dumps(response_data), status=500, mimetype="application/json")


@https_fn.on_request(cors=CorsOptions(cors_origins="*", cors_methods=["get"]))
def get_etl_status(req):
    """Cloud Function to get ETL processing status."""
    if req.method != 'GET':
        return https_fn.Response("Method not allowed", status=405)

    try:
        db = firestore.client()
        
        # Count raw games
        raw_games_ref = db.collection('raw_games')
        total_raw = len(list(raw_games_ref.stream()))
        
        # Count processed games  
        processed_query = raw_games_ref.where('processed', '==', True)
        processed_count = len(list(processed_query.stream()))
        
        response_data = {
            "status": "success",
            "total_raw_games": total_raw,
            "processed_games": processed_count,
            "unprocessed_games": total_raw - processed_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return https_fn.Response(json.dumps(response_data), status=200, mimetype="application/json")

    except Exception as e:
        error_msg = f"Error getting ETL status: {str(e)}"
        logger.error(error_msg)
        
        response_data = {
            "status": "error", 
            "error": error_msg,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return https_fn.Response(json.dumps(response_data), status=500, mimetype="application/json")
