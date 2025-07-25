# Using the Puzzle Database Manager

This document provides instructions on how to use the V7P3R Chess Engine Puzzle Database Manager.

## Overview

The Puzzle Database Manager is a tool for:
- Importing and managing Lichess puzzle data
- Querying puzzles based on various criteria (ELO, themes, etc.)
- Generating test sets for engine evaluation
- Providing an interactive console UI

## Basic Usage

### Importing Puzzles

To import puzzles from a CSV file:

```
python -m engine_utilities.puzzle_db_manager --csv "training_data/csv_data_puzzles/lichess_db_puzzle.csv"
```

### Generating a Test Set (Non-interactive)

To generate a test set without using the UI:

```
python -m engine_utilities.puzzle_db_manager --generate-test --no-ui
```

### Using the Interactive UI

To start the interactive UI:

```
python -m engine_utilities.puzzle_db_manager
```

The UI allows you to:
1. Import puzzles from CSV
2. Query puzzles with various filters
3. Generate test sets
4. View database statistics

## Difficulty Levels

Puzzles are categorized into the following difficulty levels based on ELO rating:

- Beginner: 800-1200
- Intermediate: 1201-1600
- Advanced: 1601-2000
- Expert: 2001-2400
- Master: 2401-3000

## Test Sets

Test sets are saved as JSON files in the `training_data/fen_data_puzzle_lists/` directory with a timestamp and filter information in the filename.

## Configuration

You can customize the behavior by providing a configuration file:

```
python -m engine_utilities.puzzle_db_manager --config "path/to/config.yaml"
```

The default configuration is loaded from `config/puzzle_config.yaml`.
