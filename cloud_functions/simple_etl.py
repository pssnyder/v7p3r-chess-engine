"""
Simplified ETL Cloud Function that doesn't depend on the complex local ETL processor.

This provides basic ETL functionality for the Firebase backend without requiring
all the local dependencies of the full ChessAnalyticsETL system.
"""

import json
import logging
import datetime
import uuid
from typing import Dict, List, Any, Optional

from firebase_functions import https_fn
from firebase_functions.options import set_global_options, CorsOptions
from firebase_admin import initialize_app, firestore
from google.cloud import firestore as gcs_firestore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For cost control
set_global_options(max_instances=10)

# Initialize Firebase Admin SDK
initialize_app()


@https_fn.on_request(cors=CorsOptions(cors_origins="*", cors_methods=["post"]))
def run_etl_processing(req):
    """
    Cloud Function to perform basic ETL processing on raw game data.
    
    This function processes raw game data from Firestore and creates
    analytics-ready summaries.
    """
    if req.method != 'POST':
        return https_fn.Response(f"Method {req.method} not allowed. Please use POST.", status=405)  # type: ignore

    try:
        # Get optional parameters from request
        request_data = req.get_json(silent=True) or {}
        limit = request_data.get('limit', 100)
        force_reprocess = request_data.get('force_reprocess', False)
        
        logger.info(f"Starting ETL processing with limit: {limit}")
        
        # Initialize Firestore client
        db = firestore.client()
        
        # Extract raw game data
        raw_games_ref = db.collection('raw_games')
        
        if force_reprocess:
            # Get all games
            query = raw_games_ref.limit(limit)
        else:
            # Get only unprocessed games (add a processed flag later)
            query = raw_games_ref.where('processed', '==', False).limit(limit)
            
        games = query.stream()
        
        processed_count = 0
        failed_count = 0
        errors = []
        
        # Process each game
        for game_doc in games:
            try:
                game_data = game_doc.to_dict()
                game_id = game_doc.id
                
                # Basic analytics extraction
                analytics_data = extract_game_analytics(game_data)
                
                # Store analytics data
                analytics_ref = db.collection('game_analytics').document(game_id)
                analytics_ref.set(analytics_data)
                
                # Mark as processed
                game_doc.reference.update({'processed': True, 'processed_at': firestore.SERVER_TIMESTAMP})  # type: ignore
                
                processed_count += 1
                logger.info(f"Processed game {game_id}")
                
            except Exception as e:
                failed_count += 1
                error_msg = f"Failed to process game {game_doc.id}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Prepare response
        response_data = {
            "status": "success",
            "job_id": str(uuid.uuid4()),
            "records_processed": processed_count,
            "records_failed": failed_count,
            "errors": errors[:10],  # Limit error details
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"ETL processing completed: {processed_count} processed, {failed_count} failed")
        return https_fn.Response(json.dumps(response_data), status=200, mimetype="application/json")  # type: ignore

    except Exception as e:
        error_msg = f"Error in ETL processing: {str(e)}"
        logger.error(error_msg)
        
        response_data = {
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return https_fn.Response(json.dumps(response_data), status=500, mimetype="application/json")  # type: ignore


@https_fn.on_request(cors=CorsOptions(cors_origins="*", cors_methods=["get"]))
def get_etl_status(req):
    """
    Cloud Function to get the status of ETL processing.
    """
    if req.method != 'GET':
        return https_fn.Response(f"Method {req.method} not allowed. Please use GET.", status=405)  # type: ignore

    try:
        db = firestore.client()
        
        # Count raw games
        raw_games_ref = db.collection('raw_games')
        total_raw = len(list(raw_games_ref.stream()))
        
        # Count processed games
        processed_query = raw_games_ref.where('processed', '==', True)
        processed_count = len(list(processed_query.stream()))
        
        # Count analytics records
        analytics_ref = db.collection('game_analytics')
        analytics_count = len(list(analytics_ref.stream()))
        
        response_data = {
            "status": "success",
            "total_raw_games": total_raw,
            "processed_games": processed_count,
            "unprocessed_games": total_raw - processed_count,
            "analytics_records": analytics_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return https_fn.Response(json.dumps(response_data), status=200, mimetype="application/json")  # type: ignore

    except Exception as e:
        error_msg = f"Error getting ETL status: {str(e)}"
        logger.error(error_msg)
        
        response_data = {
            "status": "error", 
            "error": error_msg,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return https_fn.Response(json.dumps(response_data), status=500, mimetype="application/json")  # type: ignore


def extract_game_analytics(game_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract basic analytics from raw game data.
    
    Args:
        game_data: Raw game data dictionary
        
    Returns:
        Analytics data dictionary
    """
    analytics = {
        'game_id': game_data.get('game_id'),
        'winner': game_data.get('winner'),
        'engine_white': game_data.get('engine_white'),
        'engine_black': game_data.get('engine_black'),
        'processed_at': datetime.datetime.now().isoformat()
    }
    
    # Parse PGN if available
    pgn = game_data.get('pgn', '')
    if pgn:
        analytics.update(parse_pgn_analytics(pgn))
    
    # Add timing data if available
    if 'received_at' in game_data:
        analytics['received_at'] = game_data['received_at']
    
    return analytics


def parse_pgn_analytics(pgn: str) -> Dict[str, Any]:
    """
    Parse basic analytics from PGN data.
    
    Args:
        pgn: PGN string
        
    Returns:
        Dictionary with basic PGN analytics
    """
    try:
        # Count moves (very basic parsing)
        moves = pgn.split()
        move_count = len([m for m in moves if '.' in m and m[0].isdigit()])
        
        # Check for common patterns
        contains_castling = 'O-O' in pgn
        contains_promotion = '=' in pgn
        contains_capture = 'x' in pgn
        
        return {
            'move_count': move_count,
            'contains_castling': contains_castling,
            'contains_promotion': contains_promotion,
            'contains_capture': contains_capture,
            'pgn_length': len(pgn)
        }
    except Exception as e:
        logger.warning(f"Failed to parse PGN analytics: {e}")
        return {
            'move_count': 0,
            'pgn_length': len(pgn),
            'parse_error': str(e)
        }
