# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_functions.options import set_global_options, CorsOptions
from firebase_admin import initialize_app, firestore
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
        data['received_at'] = firestore.SERVER_TIMESTAMP  # type: ignore
        
        game_ref.set(data)

        response_data = {"status": "success", "message": f"Game {data['game_id']} received and stored."}
        return https_fn.Response(json.dumps(response_data), status=200, mimetype="application/json")  # type: ignore

    except Exception as e:
        print(f"Error writing to Firestore: {e}")
        return https_fn.Response("Internal Server Error: Could not write data to the database.", status=500)  # type: ignore