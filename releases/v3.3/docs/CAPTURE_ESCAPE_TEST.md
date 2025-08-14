# Testing Capture-to-Escape-Check Functionality

This document explains how to test the "capture to escape check" functionality in the V7P3R chess engine.

## Overview

The V7P3R engine includes a feature that prioritizes capturing moves that also escape check. This functionality is controlled by the `use_captures_to_escape_check` setting in the config file.

## Test Configuration

We've created a special configuration file (`capture_escape_config.json`) that isolates this functionality:

- Disables opening book to ensure the engine uses its evaluation
- Disables castling and tactics scoring to isolate the capture-to-escape-check logic
- Keeps the capture-to-escape-check feature enabled

## Running the Test

To test this functionality with the provided test script:

```powershell
python test_capture_escape.py
```

This will:
1. Set up a position where white is in check by a queen
2. The best move is to capture the queen with the white queen
3. Run the engine with the special config and see if it finds the capture

## Manual Testing

You can also test manually:

```powershell
python play_chess.py --config capture_escape_config.json
```

This will start a normal game with the special configuration.

## Expected Behavior

When the engine is in check and has an opportunity to capture the checking piece:

- With `use_captures_to_escape_check` enabled: The engine should strongly prefer capturing the checking piece (especially if it's a favorable capture)
- With `use_captures_to_escape_check` disabled: The engine will evaluate escaping check without the special bonus for capturing

## Implementation Details

The capture-to-escape-check logic is implemented in:

1. `v7p3r_utils.py`: The `is_capture_that_escapes_check()` function detects when a move both captures and escapes check
2. `v7p3r_secondary_scoring.py`: The `_evaluate_escape_check()` method provides scoring bonuses for such moves
3. Configuration is passed from `v7p3r_engine.py` through `v7p3r_scoring.py` to the `SecondaryScoring` class

This ensures the engine makes tactically strong moves when under attack.
