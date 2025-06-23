# V7P3R Chess Engine Analytics ETL System

This document describes the ETL (Extract, Transform, Load) system for the V7P3R Chess Engine analytics layer. The ETL system is responsible for processing raw game and simulation data from various sources, transforming it into an analytics-optimized schema, and loading it into a reporting database for analysis.

## Overview

The V7P3R Chess ETL system is designed to:

1. Extract data from multiple sources (local SQLite, cloud Firestore, PGN files)
2. Validate and clean the data
3. Transform it into a normalized, analytics-optimized schema
4. Load it into a dedicated reporting database
5. Track all job metrics and performance data
6. Support versioning, backfilling, and reprocessing

The ETL system consists of several components:

- **ETL Processor** (`etl_processor.py`): Core data processing engine
- **ETL Scheduler** (`etl_scheduler.py`): Job scheduling and orchestration
- **ETL Monitor** (`etl_monitor.py`): Performance monitoring and reporting

## Architecture

![ETL Architecture](../images/etl_architecture.png)

The ETL system is designed with the following principles:

- **Idempotency**: Jobs can be safely rerun without duplicating data
- **Scalability**: Batch processing and parallelization for performance
- **Fault Tolerance**: Error handling and recovery mechanisms
- **Monitoring**: Comprehensive metrics collection and reporting
- **Flexibility**: Support for multiple data sources and formats

## Analytics Schema

The reporting database (`chess_analytics.db`) uses the following schema:

### Games Table
- `game_id`: Unique identifier for the game
- `simulation_id`: ID of the simulation that generated this game
- `timestamp`: When the game was played
- `white_player`: Name of the white player/engine
- `black_player`: Name of the black player/engine
- `result`: Game result (1-0, 0-1, 1/2-1/2)
- `game_length`: Number of moves in the game
- Additional metadata (time control, event, site, etc.)

### Engine Configurations Table
- `config_id`: Unique identifier for the configuration
- `game_id`: Associated game
- `side`: White or black
- `engine_id`: Engine identifier
- `engine_name`: Engine name
- `engine_version`: Engine version
- Additional configuration details (search depth, evaluation ruleset, etc.)

### Moves Table
- `move_id`: Unique identifier for the move
- `game_id`: Associated game
- `move_number`: Move number in the game
- `side`: White or black
- `move_san`: Move in Standard Algebraic Notation (SAN)
- `move_uci`: Move in Universal Chess Interface (UCI) format
- `fen_before`: FEN string before the move
- `fen_after`: FEN string after the move
- Additional move metrics (evaluation, depth, nodes searched, etc.)

### Engine Performance Table
- `id`: Unique identifier
- `game_id`: Associated game
- `config_id`: Associated engine configuration
- `move_id`: Associated move (if applicable)
- `metric_name`: Name of the performance metric
- `metric_value`: Value of the metric
- `phase`: Game phase (opening, middlegame, endgame)

### ETL Job Metrics Table
- `job_id`: Unique identifier for the ETL job
- `start_time`: When the job started
- `end_time`: When the job completed
- `status`: Job status (running, completed, failed)
- `records_processed`: Number of records processed
- `records_failed`: Number of records that failed processing
- Additional job metrics (extraction time, transformation time, etc.)

## Usage

### Running the ETL Process

To run the ETL process manually:

```bash
python -m engine_utilities.etl_processor --config config/etl_config.yaml
```

Options:
- `--limit N`: Process only N records (for testing)
- `--start-date YYYY-MM-DD`: Process data from this date
- `--end-date YYYY-MM-DD`: Process data until this date
- `--no-parallel`: Disable parallel processing

### Scheduling ETL Jobs

To set up scheduled ETL jobs:

```bash
python -m engine_utilities.etl_scheduler --setup-local
```

For Google Cloud Scheduler integration:

```bash
python -m engine_utilities.etl_scheduler --setup-gcp
```

### Monitoring ETL Performance

To monitor ETL performance:

```bash
python -m engine_utilities.etl_monitor --job-history
```

To monitor system resources during ETL execution:

```bash
python -m engine_utilities.etl_monitor --monitor --duration 300
```

To generate a resource usage report:

```bash
python -m engine_utilities.etl_monitor --monitor --duration 300 --report
```

## Configuration

The ETL system is configured through `config/etl_config.yaml`. Key configuration sections include:

### Reporting Database
```yaml
reporting_db:
  path: "metrics/chess_analytics.db"
  backup_enabled: true
  backup_frequency_hours: 24
```

### Cloud Storage
```yaml
cloud:
  enabled: false  # Set to true to enable cloud-based ETL
  bucket_name: "viper-chess-engine-data"
```

### Processing Settings
```yaml
processing:
  batch_size: 100  # Number of games to process in a batch
  max_workers: 4  # Maximum number of parallel worker threads
```

### Scheduling
```yaml
schedule:
  cron: "0 2 * * *"  # Run daily at 2 AM (cron format)
  timezone: "UTC"
```

## Resource Usage

The ETL system is designed to run efficiently on local compute resources. It can be configured to limit resource usage:

- **CPU**: The `max_workers` setting controls parallelization
- **Memory**: The `batch_size` setting controls memory usage
- **Disk**: The system performs sequential I/O to minimize disk impact

## Versioning and Reprocessing

The ETL system supports versioning and reprocessing of data. Each ETL job is tracked with a unique job ID and schema version, allowing for:

- **Schema Evolution**: Track schema changes over time
- **Data Reprocessing**: Reprocess historical data with updated logic
- **Backfilling**: Fill in missing data from specific time periods

## Monitoring and Alerts

The ETL Monitor provides comprehensive monitoring and alerting capabilities:

- **Job Status**: Track success/failure of ETL jobs
- **Performance Metrics**: CPU, memory, and disk usage
- **Data Quality**: Validation errors and data completeness
- **Processing Time**: Extraction, transformation, and loading times

## Legacy System Compatibility

The ETL system is designed to work alongside the legacy metrics system. The original metrics functionality is preserved but marked as deprecated in the code.

## Future Enhancements

Planned enhancements for the ETL system include:

1. Enhanced data validation and cleansing
2. More sophisticated data quality metrics
3. Integration with additional data sources
4. Automated schema migration tools
5. Real-time processing capabilities

## Troubleshooting

Common issues and their solutions:

1. **ETL Job Failures**: Check the job error log in the ETL job metrics table
2. **Performance Issues**: Adjust batch size and worker count in configuration
3. **Data Quality Issues**: Run the ETL Monitor's data quality checks
4. **Cloud Connectivity**: Verify GCP credentials and permissions

For more information, contact the V7P3R development team.
