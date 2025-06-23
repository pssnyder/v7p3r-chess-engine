# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_functions.options import set_global_options, CorsOptions
from firebase_admin import initialize_app, firestore
from google.cloud.firestore import SERVER_TIMESTAMP
import json

# For cost control, you can set the maximum number of containers that can be
# running at the same time. This helps mitigate the impact of unexpected
# traffic spikes by instead downgrading performance. This limit is a per-function
# limit. You can override the limit for each function using the max_instances
# parameter in the decorator.
set_global_options(max_instances=10)

# Initialize the Firebase Admin SDK.
initialize_app()


@https_fn.on_request(cors=CorsOptions(cors_origins="*", cors_methods=["get", "post"]))
def submit_game_data(req):
    """
    An HTTP-triggered Cloud Function to receive and store raw game data from simulations.
    It expects a POST request with a JSON body containing the game details.
    """
    # We only want to handle POST requests for data submission.
    if req.method != 'POST':
        return https_fn.Response(f"Method {req.method} not allowed. Please use POST.", status=405)  # type: ignore

    # Get the JSON data from the request body.
    try:
        data = req.get_json(silent=False)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return https_fn.Response("Invalid JSON format.", status=400)  # type: ignore

    # Validate that the essential data fields are present.
    required_fields = ['game_id', 'pgn', 'winner', 'engine_white', 'engine_black']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return https_fn.Response(f"Missing required fields: {', '.join(missing_fields)}", status=400)  # type: ignore

    # Store the validated data in Firestore.
    try:
        db = firestore.client()
        game_ref = db.collection('raw_games').document(data['game_id'])
        
        # Add a server timestamp to track when the data was received.
        data['received_at'] = SERVER_TIMESTAMP
        
        game_ref.set(data)

        response_data = {"status": "success", "message": f"Game {data['game_id']} received and stored."}
        return https_fn.Response(json.dumps(response_data), status=200, mimetype="application/json")  # type: ignore

    except Exception as e:
        print(f"Error writing to Firestore: {e}")
        return https_fn.Response("Internal Server Error: Could not write data to the database.", status=500)  # type: ignore


@https_fn.on_request(cors=CorsOptions(cors_origins="*", cors_methods=["post"]))
def run_etl_processing(req):
    """Cloud Function to perform ETL processing on raw game data."""
    if req.method != 'POST':
        return https_fn.Response("Method not allowed", status=405)

    try:
        import datetime
        import uuid
        
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
                print(f"Failed to process game {game_doc.id}: {e}")
        
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
        print(error_msg)
        
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
        import datetime
        
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
        print(error_msg)
        
        response_data = {
            "status": "error", 
            "error": error_msg,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return https_fn.Response(json.dumps(response_data), status=500, mimetype="application/json")