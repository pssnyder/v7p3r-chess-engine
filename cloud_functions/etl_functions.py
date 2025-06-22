"""
Firebase Cloud Function integration for the existing V7P3R ETL system.

This module provides Cloud Functions that integrate with the existing
ETL processor located in engine_utilities/etl_processor.py
"""

import sys
import os
import json
import logging
import traceback

# Add the parent directory to the Python path to import from engine_utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from firebase_functions import https_fn
from firebase_functions.options import set_global_options, CorsOptions
from firebase_admin import initialize_app, firestore
import json
import logging

# Import your existing ETL system
from engine_utilities.etl_processor import ChessAnalyticsETL

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
    Cloud Function to trigger ETL processing using the existing ChessAnalyticsETL system.
    
    This function can be called manually or scheduled to process raw game data
    into analytics-ready format using your existing ETL infrastructure.
    """
    if req.method != 'POST':
        return https_fn.Response(f"Method {req.method} not allowed. Please use POST.", status=405)  # type: ignore

    try:
        # Get optional parameters from request
        request_data = req.get_json(silent=True) or {}
        config_path = request_data.get('config_path', 'config/etl_config.yaml')
        force_reprocess = request_data.get('force_reprocess', False)
        
        logger.info(f"Starting ETL processing with config: {config_path}")
        
        # Initialize your existing ETL processor
        etl_processor = ChessAnalyticsETL(config_path=config_path)
        
        # Run the ETL process
        if force_reprocess:
            logger.info("Force reprocessing requested")
            # You can add specific logic for force reprocessing if needed
        
        # Execute the ETL process
        result = etl_processor.run_etl_job()
          # Get job metrics
        metrics = etl_processor.job_metrics
        
        response_data = {
            "status": "success",
            "job_id": metrics.job_id,
            "records_processed": metrics.records_processed,
            "records_failed": metrics.records_failed,
            "records_skipped": metrics.records_skipped,
            "errors": metrics.errors,
            "warnings": metrics.warnings,
            "start_time": metrics.start_time.isoformat(),
            "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
            "extraction_time": metrics.extraction_time_seconds,
            "transform_time": metrics.transform_time_seconds,
            "load_time": metrics.load_time_seconds
        }
        
        logger.info(f"ETL processing completed successfully. Job ID: {metrics.job_id}")
        return https_fn.Response(json.dumps(response_data), status=200, mimetype="application/json")  # type: ignore

    except Exception as e:
        error_msg = f"Error in ETL processing: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        response_data = {
            "status": "error",
            "error": error_msg
        }
        
        return https_fn.Response(json.dumps(response_data), status=500, mimetype="application/json")  # type: ignore


@https_fn.on_request(cors=CorsOptions(cors_origins="*", cors_methods=["get"]))
def get_etl_status(req):
    """
    Cloud Function to get the status of recent ETL jobs.
    
    Returns information about recent ETL job runs, including metrics and status.
    """
    if req.method != 'GET':
        return https_fn.Response(f"Method {req.method} not allowed. Please use GET.", status=405)  # type: ignore

    try:
        # You can implement logic to query your ETL job metrics from the database
        # For now, return a simple status
        
        response_data = {
            "status": "success",
            "message": "ETL status endpoint available",
            "available_endpoints": {
                "run_etl": "/run_etl_processing (POST)",
                "status": "/get_etl_status (GET)"
            }
        }
        
        return https_fn.Response(json.dumps(response_data), status=200, mimetype="application/json")  # type: ignore

    except Exception as e:
        error_msg = f"Error getting ETL status: {str(e)}"
        logger.error(error_msg)
        
        response_data = {
            "status": "error", 
            "error": error_msg
        }
        
        return https_fn.Response(json.dumps(response_data), status=500, mimetype="application/json")  # type: ignore
