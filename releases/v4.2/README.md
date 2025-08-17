# V7P3R Chess Engine

A modular chess engine designed for incremental improvement and evaluation.

## Recent Improvements

1. **Created Standardized Utilities Module (`v7p3r_utils.py`)**
   - Standardized piece values and evaluation constants
   - Shared utility functions for material balance, game phase detection, etc.
   - Enhanced tactical evaluation tools (hanging pieces, capture sequences)
   - Unified "capture to escape check" detection

2. **Enhanced Primary Scoring**
   - Improved capture evaluation with exchange analysis
   - Better detection of hanging pieces and favorable exchanges
   - Standardized material balance calculation

3. **Improved Secondary Scoring**
   - Fully implemented "capture to escape check" functionality
   - Added detailed evaluation for escape check tactics
   - Better integration with other scoring components

4. **Consolidated Common Functionality**
   - Removed duplicate code for material evaluation
   - Standardized game phase detection
   - Unified MVV-LVA and exchange evaluation
   - Consistent draw position detection

5. **Tactical Improvements**
   - Better evaluation of piece exchanges
   - Enhanced detection of hanging pieces
   - Improved move ordering for captures that escape check

## Architecture

The engine is designed with the following modular components:

- **Engine Coordinator** (`v7p3r_engine.py`): Main engine interface and orchestration
- **Search** (`v7p3r_search.py`): Search algorithm and depth management
- **Scoring** (`v7p3r_scoring.py`): Coordinates all scoring components
- **Tempo** (`v7p3r_tempo.py`): Critical move detection and tempo evaluation
- **Primary Scoring** (`v7p3r_primary_scoring.py`): Material and piece-square tables
- **Secondary Scoring** (`v7p3r_secondary_scoring.py`): Castling, tactics, and escape check
- **MVV-LVA** (`v7p3r_mvv_lva.py`): Most Valuable Victim-Least Valuable Attacker evaluation
- **Rules** (`v7p3r_rules.py`): Game phase detection and rule-based guidance
- **Utilities** (`v7p3r_utils.py`): Shared constants and utility functions

## Configuration

All engine features can be toggled in `config.json`, allowing for easy experimentation and incremental testing.

## Usage

To play against the engine:
```python
python play_chess.py
```

To run batch analysis:
```python
python batch_analyzer.py
```
