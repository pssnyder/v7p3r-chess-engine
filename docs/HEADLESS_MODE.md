# V7P3R Chess Engine - Headless Mode

The V7P3R Chess Engine has been optimized to run in headless mode for better performance. This document explains how to use the updated system.

## Key Changes

1. **Removed Visual Display from Engine**: The pygame visualization has been removed from the main engine to improve performance.

2. **Separate PGN Watcher**: The existing `active_game_watcher.py` is now the official way to visualize games without impacting engine performance.

3. **Background Operation**: The engine can now run in the background, allowing you to monitor progress through terminal output.

## How to Use

### Running the Engine

To run the engine in headless mode (default now):

```powershell
python play_chess.py
```

### Running the Engine in Background

To run the engine in the background with output redirected to a log file:

```powershell
# Windows PowerShell
.\run_engine_background.ps1

# Windows CMD
run_engine_background.bat
```

### Visualizing Games

To watch games visually while the engine is running:

```powershell
# In a separate terminal window
python active_game_watcher.py

# Or use the helper script
.\run_pgn_watcher.ps1
```

The PGN watcher monitors the `active_game.pgn` file for changes and provides visual feedback without impacting engine performance.

## Benefits

1. **Improved Performance**: With visualization removed, the engine can focus computational resources on move calculation.

2. **Decoupled Components**: The engine and visualization are now separate, allowing each to be optimized independently.

3. **Better Monitoring**: You can run the engine in the background and check progress in the logs, while optionally watching games visually in a separate window.

4. **Reduced Resource Usage**: The headless mode uses significantly less memory and CPU when visualization isn't needed.

## Command Line Options

The engine supports the following command line options:

```
--config, -c       Configuration file to use (default: config.json)
--games, -g        Number of games to play (overrides config)
--white, -w        White player (v7p3r or stockfish)
--black, -b        Black player (v7p3r or stockfish)
--depth, -d        Search depth for v7p3r
--stockfish-elo    Stockfish ELO rating
--background       Run in background mode (with output redirected)
```
