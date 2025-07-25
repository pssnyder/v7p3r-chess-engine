# engine_utilities/engine_db_manager.py
# This module manages the database of chess engine configurations and their metrics.
# It can be expanded as the engines db utility toolkit needs expand as well.
import os
import sys
import threading
import time
import json
import yaml
import uuid
import logging
import requests
# Ensure project root is in sys.path for local script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from http.server import BaseHTTPRequestHandler, HTTPServer
from metrics.metrics_store import MetricsStore



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EngineDBManager:
    """
    Manages the database of chess engine configurations and their metrics.
    Local-only version: all data is stored in local SQLite DBs.
    """
    def __init__(self, db_path="metrics/chess_metrics.db", config_path="config/engine_utilities.yaml"):
        self.db_path = db_path
        self.metrics_store = MetricsStore(db_path=db_path)
        self.config = self._load_config(config_path)
        self.server_thread = None
        self.httpd = None
        self.running = False

    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def start_server(self, host="0.0.0.0", port=8080):
        """Start an HTTP server to receive incoming data from remote engines."""
        self.running = True
        handler = self._make_handler()
        self.httpd = HTTPServer((host, port), handler)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        logger.info(f"EngineDBManager server running at http://{host}:{port}")

    def stop_server(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.running = False
            logger.info("EngineDBManager server stopped.")

    def _make_handler(self):
        manager = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    response_data = {'status': 'success', 'message': ''}
                    # Process different request types
                    if data.get('type') == 'game':
                        manager._handle_game_data(data['game_data'])
                        response_data['message'] = 'Game data stored'
                    elif data.get('type') == 'move':
                        manager._handle_move_data(data['move_data'])
                        response_data['message'] = 'Move data stored'
                    elif data.get('type') == 'bulk':
                        manager.bulk_upload(data['data_list'])
                        response_data['message'] = f'Bulk data stored ({len(data["data_list"])} items)'
                    elif data.get('type') == 'raw_simulation':
                        manager._handle_raw_simulation(data['simulation_data'])
                        response_data['message'] = 'Raw simulation data stored'
                    else:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'status': 'error', 'message': 'Unknown data type'}).encode('utf-8'))
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response_data).encode('utf-8'))
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    error_message = str(e)
                    self.wfile.write(json.dumps({'status': 'error', 'message': f'Error: {error_message}'}).encode('utf-8'))
            def log_message(self, format, *args):
                # Override to use our logger instead of printing to stderr
                logger.info(f"{self.client_address[0]} - {format%args}")
        return Handler

    def _handle_game_data(self, game_data):
        """Handle incoming game data: store locally only."""
        self.metrics_store.add_game_result(**game_data)

    def _handle_move_data(self, move_data):
        """Handle incoming move data: store locally only."""
        self.metrics_store.add_move_metric(**move_data)

    def _handle_raw_simulation(self, simulation_data):
        """Handle raw simulation data by storing locally only."""
        with open(f"logging/raw_simulation_{simulation_data.get('id', uuid.uuid4())}.json", 'w') as f:
            json.dump(simulation_data, f, indent=2)

    def bulk_upload(self, data_list):
        """Accept a list of game/move data for bulk upload."""
        games_data = []
        moves_data = {}  # game_id -> list of moves
        for item in data_list:
            if item.get('type') == 'game':
                games_data.append(item['game_data'])
            elif item.get('type') == 'move':
                move = item['move_data']
                game_id = move.get('game_id')
                if game_id:
                    if game_id not in moves_data:
                        moves_data[game_id] = []
                    moves_data[game_id].append(move)
        for game_data in games_data:
            self._handle_game_data(game_data)
        for game_id, moves in moves_data.items():
            for move in moves:
                self.metrics_store.add_move_metric(**move)

    def listen_and_store(self):
        """Main loop for local/remote data collection."""
        self.start_server()
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_server()


# Client functionality for remote engines to send data to the server
class EngineDBClient:
    """Client for sending chess engine data to an EngineDBManager server."""
    
    def __init__(self, server_url=None, config_path="config/engine_utilities.yaml"):
        self.config = self._load_config(config_path)
        self.server_url = server_url or self.config.get('client', {}).get('server_url')
        if not self.server_url:
            logger.warning("No server URL specified, using default localhost:8080")
            self.server_url = "http://localhost:8080"
        
        self.max_retries = self.config.get('client', {}).get('max_retries', 3)
        self.retry_delay = self.config.get('client', {}).get('retry_delay', 1)
        
        # Initialize local storage for offline operation
        self.offline_buffer = []
        self.offline_mode = False
        
        logger.info(f"EngineDBClient initialized with server URL: {self.server_url}")
    
    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _send_with_retry(self, data):
        """Send data to server with retry logic."""
        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.post(
                    self.server_url,
                    json=data,
                    headers={'Content-Type': 'application/json'},
                    timeout=10  # 10 second timeout
                )
                if response.status_code == 200:
                    return True
                logger.warning(f"Server returned status {response.status_code}: {response.text}")
            except requests.RequestException as e:
                logger.warning(f"Request failed: {e}")
                # Switch to offline mode after first failure
                if not self.offline_mode:
                    logger.info("Switching to offline mode")
                    self.offline_mode = True
            
            # Backoff before retry
            wait_time = self.retry_delay * (2 ** retries)
            logger.info(f"Retrying in {wait_time}s (attempt {retries+1}/{self.max_retries})")
            time.sleep(wait_time)
            retries += 1
        
        # If we're here, all retries failed
        return False
    
    def send_game_data(self, game_data):
        """Send game result data to server."""
        data = {
            'type': 'game',
            'game_data': game_data
        }
        
        # Try to send, buffer if failed
        if self.offline_mode or not self._send_with_retry(data):
            self.offline_buffer.append(data)
            logger.info("Game data stored in offline buffer")
            return False
        return True
    
    def send_move_data(self, move_data):
        """Send move metrics data to server."""
        data = {
            'type': 'move',
            'move_data': move_data
        }
        
        # Try to send, buffer if failed
        if self.offline_mode or not self._send_with_retry(data):
            self.offline_buffer.append(data)
            logger.info("Move data stored in offline buffer")
            return False
        return True
    
    def send_raw_simulation(self, simulation_data):
        """Send raw simulation data to server."""
        data = {
            'type': 'raw_simulation',
            'simulation_data': simulation_data
        }
        
        # Raw simulation data is too important to lose, always buffer
        self.offline_buffer.append(data)
        
        # Try to send immediately, but don't worry if it fails
        if not self.offline_mode:
            self._send_with_retry(data)
        return True
    
    def flush_offline_buffer(self):
        """Try to send all buffered data to the server."""
        if not self.offline_buffer:
            logger.info("No offline data to flush")
            return True
        
        logger.info(f"Attempting to flush {len(self.offline_buffer)} buffered items")
        
        # First try to reconnect if in offline mode
        if self.offline_mode:
            try:
                # Simple ping to check if server is back
                response = requests.get(self.server_url, timeout=5)
                if response.status_code == 200:
                    self.offline_mode = False
                    logger.info("Reconnected to server")
            except requests.RequestException:
                logger.warning("Server still unreachable, staying in offline mode")
                return False
        
        # If we have many items, use bulk upload
        if len(self.offline_buffer) > 10:
            bulk_data = {
                'type': 'bulk',
                'data_list': self.offline_buffer
            }
            
            if self._send_with_retry(bulk_data):
                logger.info(f"Successfully flushed {len(self.offline_buffer)} buffered items")
                self.offline_buffer = []
                return True
            return False
        
        # Otherwise send items one by one
        successful = 0
        remaining = []
        
        for item in self.offline_buffer:
            if self._send_with_retry(item):
                successful += 1
            else:
                remaining.append(item)
        
        self.offline_buffer = remaining
        logger.info(f"Flushed {successful}/{successful + len(remaining)} buffered items")
        return len(remaining) == 0
    
    def save_offline_buffer(self, file_path="logging/offline_buffer.json"):
        """Save offline buffer to a file for later recovery."""
        if not self.offline_buffer:
            return
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.offline_buffer, f, indent=2)
            logger.info(f"Saved {len(self.offline_buffer)} offline items to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save offline buffer: {e}")
    
    def load_offline_buffer(self, file_path="logging/offline_buffer.json"):
        """Load offline buffer from a file."""
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r') as f:
                buffer_data = json.load(f)
            self.offline_buffer.extend(buffer_data)
            logger.info(f"Loaded {len(buffer_data)} offline items from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load offline buffer: {e}")
    
    def __enter__(self):
        """Context manager support for auto-loading offline buffer."""
        self.load_offline_buffer()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support for auto-saving and flushing."""
        # Try to flush buffer first
        if not self.offline_mode:
            self.flush_offline_buffer()
        
        # Save any remaining items
        if self.offline_buffer:
            self.save_offline_buffer()


# Example usage (for local dev/testing):
if __name__ == "__main__":
    # Choose whether to run as server or client
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        # Run as client (for testing)
        client = EngineDBClient()
        client.send_game_data({
            'game_id': 'test_game_1',
            'winner': '1-0',
            'white_player': 'Test Engine',
            'black_player': 'Test Opponent',
            'game_pgn': '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6'
        })
        print("Test data sent to server")
    else:
        # Run as server
        manager = EngineDBManager()
        manager.listen_and_store()
