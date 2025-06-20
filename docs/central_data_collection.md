# Central Data Collection System

## Overview

The central data collection system allows the V7P3R Chess Engine to collect game results, move metrics, and raw simulation data from multiple computers running simulations, without requiring git commits to synchronize data.

## Architecture

1. **Server Component**: `engine_db_manager.py` provides an HTTP server that listens for incoming data from remote clients.
2. **Client Component**: Also in `engine_db_manager.py`, the `EngineDBClient` class sends data to the central server.
3. **Cloud Storage**: `cloud_store.py` provides integration with Google Cloud Storage and Firestore for persistent data storage.
4. **Game Integration**: `chess_game.py` uses the data collector pattern to send data to the central server during games.
5. **Simulation Manager**: `game_simulation_manager.py` manages game simulations and ensures data is sent to the central server.

## Configuration

### Server Configuration

To configure the central data server, edit `config/engine_utilities.yaml`:

```yaml
# Cloud storage configuration
cloud:
  enabled: true  # Set to true to enable cloud storage
  bucket_name: "your-bucket-name"  # GCS bucket name
  
# Server configuration
server:
  host: "0.0.0.0"  # Listen on all interfaces
  port: 8080  # Port to listen on
```

### Client Configuration

To configure clients to send data to the central server:

```yaml
# Client configuration
client:
  server_url: "http://your-server-ip:8080"  # Central server URL
  max_retries: 3  # Number of retry attempts for failed uploads
```

### Simulation Configuration

To enable central data collection in simulations, edit `simulation_config.yaml`:

```yaml
# Enable central data storage for simulation results
use_central_storage: true
```

## Google Cloud Setup

1. Create a Google Cloud project
2. Create a Cloud Storage bucket for game data
3. Set up Firestore for game metadata
4. Generate a service account key with permissions for Cloud Storage and Firestore
5. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account key file

## Usage

### Running the Central Server

To run the central data server:

```bash
python -m engine_utilities.engine_db_manager
```

### Running Simulations with Central Data Collection

To run simulations that send data to the central server:

```bash
python -m engine_utilities.game_simulation_manager
```

## Data Structure

### Game Results

```json
{
  "game_id": "unique_game_id",
  "timestamp": "20250620_123456",
  "winner": "1-0",
  "game_pgn": "1. e4 e5 2. Nf3 Nc6...",
  "white_player": "V7P3R",
  "black_player": "Stockfish",
  "game_length": 40,
  "white_engine_id": "v7p3r-1.0",
  "black_engine_id": "stockfish-15",
  "white_ai_type": "deepsearch",
  "black_ai_type": "uci"
}
```

### Move Metrics

```json
{
  "game_id": "unique_game_id",
  "move_number": 12,
  "player_color": "white",
  "move_uci": "e2e4",
  "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "evaluation": 0.32,
  "ai_type": "deepsearch",
  "depth": 4,
  "nodes_searched": 10245,
  "time_taken": 1.24,
  "pv_line": "e2e4 e7e5 g1f3"
}
```

### Raw Simulation Data

```json
{
  "id": "sim_20250620_123456_abcd1234",
  "timestamp": "2025-06-20T12:34:56.789Z",
  "config": {
    "max_concurrent_simulations": 4,
    "simulations": [...]
  },
  "results": [...]
}
```

## Troubleshooting

### Server Not Responding

- Check that the server is running
- Verify that the port is not blocked by a firewall
- Ensure the client is using the correct server URL

### Data Not Being Stored in Cloud

- Check that cloud storage is enabled in configuration
- Verify that the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
- Ensure the service account has proper permissions
- Check the server logs for any errors

### Offline Mode

If the client cannot reach the server, it will operate in offline mode:

1. Data is stored locally in a buffer
2. The client will periodically attempt to reconnect to the server
3. When the connection is restored, buffered data will be sent to the server
4. To manually flush the buffer, call `client.flush_offline_buffer()`

## Extending the System

The central data collection system can be extended in several ways:

1. **Additional Data Types**: Add new data types by extending the server handler
2. **New Storage Backends**: Implement new storage backends beyond Google Cloud
3. **Real-time Analysis**: Add real-time analysis capabilities to the server
4. **Access Control**: Implement authentication and authorization for the server
