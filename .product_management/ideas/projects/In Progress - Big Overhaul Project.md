# V7P3R Chess Engine Big Overhaul Project: Primary Update Documentation

This document outlines the progress and key updates for the V7P3R Chess Engine overhaul project. It will serve as the primary documentation for each phase, including progress tracking, unit tests, and visualizations where applicable.

## Project Outline

The project is divided into three main phases, each with specific goals and deliverables.

### Phase 1: Core Engine Functionality & Data Collection - COMPLETED

**Objective**: Establish robust core engine functionality and ensure accurate data collection for game analysis.

#### Steps:

1.  **Update Configuration File Handling**: Modify the codebase to correctly load settings from the new YAML file structure (`v7p3r.yaml`, `chess_game.yaml`, `stockfish_handler.yaml`).
2.  **Ensure Automated Game Play**: Verify that `chess_game.py` can run an AI vs. AI game (e.g., V7P3R vs. Stockfish) using the updated configurations and save game data (PGN, config, logs).
3.  **Basic Metrics Collection**: Confirm that essential game metrics (result, players, length, configurations) are being saved.

### Phase 2: Configuration GUI

**Objective**: Develop a user-friendly graphical interface for managing engine and game configurations.

#### Steps:

1.  **Design Configuration Data Structure**: Determine how configurations will be stored and managed (e.g., a JSON file or a simple database) to allow for saving, loading, and creating new named configurations.
2.  **Implement GUI for V7P3R Settings**: Build out `v7p3r_gui.app.py` (likely using Flask or Streamlit) to:
    * Load existing configurations.
    * Display current settings from `v7p3r.yaml`.
    * Allow users to modify these settings.
    * Save changes back to `v7p3r.yaml` or a new named configuration.
    * Allow users to select which named configuration is "active" for the engine.
3.  **Implement GUI for Game Settings**: Extend the GUI to manage settings in `chess_game.yaml`.
4.  **Implement GUI for Stockfish Settings**: Extend the GUI to manage settings in `stockfish_handler.yaml`.

### Phase 3: Engine Monitor and TODOs

**Objective**: Refine the engine monitoring dashboard and address outstanding code tasks.

#### Steps:

1.  **Update `engine_monitor.app.py`**: Refactor the Streamlit dashboard based on TODO notes, focusing on historical analysis and removing real-time/log-based features not relevant to its new scope.
2.  **Address Code TODOs**: Systematically go through the codebase, identify all TODO comments, and implement them.
3.  **Adaptive ELO (Stretch Goal)**: If time permits and core functionality is stable, begin planning/implementing the adaptive ELO for opponent AI in `chess_game.py`.

## Phase 1 Progress: Core Engine Functionality & Data Collection

### Step 1: Update Configuration File Handling

**Status: In Progress**

**Details:**
*   **`chess_game.py`**: 
    *   Modified `ChessGame.__init__` to load `chess_game.yaml` (into `self.game_config_data`), `v7p3r.yaml` (into `self.v7p3r_config_data`), and `engine_utilities/stockfish_handler.yaml` (into `self.stockfish_config_data`).
    *   Updated various methods (`_initialize_ai_engines`, `set_headers`, etc.) to access settings from these new configuration attributes.
    *   `save_game_data` now includes a `game_specific_config` dictionary containing all three loaded config data structures, plus the resolved `white_actual_config` and `black_actual_config` used for the game.
*   **`v7p3r.py`**: 
    *   Corrected `reportUndefinedVariable` for `legal_moves` in `V7P3REvaluationEngine._deep_search`.
    *   `V7P3REvaluationEngine.__init__` now loads `v7p3r.yaml` into `self.v7p3r_config_data` and `chess_game.yaml` into `self.game_settings_config_data`.
    *   `V7P3REvaluationEngine._ensure_ai_config` has been significantly refactored to correctly merge configurations in the order: `v7p3r.yaml` (base) -> `chess_game.yaml` (player-specific overrides like `white_ai_config` or `black_ai_config`) -> runtime `ai_config` (highest precedence). This method now performs a deep merge for nested dictionary settings.
    *   `V7P3REvaluationEngine.configure_for_side` updated to use the fully resolved configuration from `_ensure_ai_config` to set all engine parameters (search algorithm, ruleset, depth, scoring modifier, hash size, TT settings, move ordering, quiescence, PSTs, time limits, etc.).
    *   `V7P3REvaluationEngine.search` now calls `_ensure_ai_config` at the beginning of each search to get the most up-to-date resolved configuration for the current player and then calls `configure_for_side`.
    *   Methods like `order_moves`, `_order_move_score`, and `_quiescence_search` updated to pull parameters (e.g., bonuses, max depths) from the resolved `self.ai_config`.
*   **`engine_utilities/v7p3r_scoring_calculation.py`**: 
    *   `V7P3RScoringCalculation.__init__` modified to accept `v7p3r_yaml_config` (the parsed `v7p3r.yaml` data, specifically the `v7p3r` key and its sub-keys) and the resolved `ai_config` for the current context.
    *   It now loads all ruleset definitions from `v7p3r_yaml_config.get('rulesets', {})`.
    *   The active `ruleset_name`, `scoring_modifier`, `pst_enabled`, and `pst_weight` are now determined from the passed `ai_config` (which itself is a result of merging `v7p3r.yaml`, `chess_game.yaml` player specifics, and runtime overrides).
    *   The main scoring method, `calculate_score` (renamed from `_calculate_score`), re-synchronizes its internal state (current ruleset, modifier, PST settings) with the `ai_config` at the start of each call. This ensures that if the `ai_config` changes (e.g., for a different player or a re-configuration), the scoring calculator uses the correct, up-to-date parameters.
    *   `_get_rule_value` updated to fetch values from the `current_ruleset`, with a fallback to `v7p3r_yaml_config.get('default_evaluation', {})` for common rule defaults, and then to a hardcoded default if not found.
    *   PST score application in `calculate_score` made perspective-aware (adjusts score if `color` is `chess.BLACK` and PST scores are White-centric).
    *   `_special_moves` method now accepts a `color` parameter and its logic was refined for en passant and promotion opportunities for that specific color.

**Next Steps for Step 1:**
1.  Thoroughly test the configuration loading and merging logic in `chess_game.py` and `v7p3r.py`.
2.  Verify that `V7P3RScoringCalculation` correctly uses the rulesets and parameters from `v7p3r.yaml` as intended, especially with different player AI configurations in `chess_game.yaml`.

```python
# Code Snippet: Configuration Loading (Initial Implementation/Changes)
# This section will contain the actual code modifications related to loading YAML files.
# For example:
# import yaml
# 
# def load_config(filepath):
#     with open(filepath, 'r') as file:
#         return yaml.safe_load(file)
# 
# chess_game_config = load_config('chess_game.yaml')
# stockfish_config = load_config('stockfish_handler.yaml')
# v7p3r_config = load_config('v7p3r.yaml')
# 
# print("Chess Game Config:", chess_game_config)
# print("Stockfish Config:", stockfish_config)
# print("V7P3R Config:", v7p3r_config)
```

### Phase 1: Core Engine Functionality & Data Collection

**Objective**: Establish robust core engine functionality and ensure accurate data collection for game analysis.

#### Steps:

1.  **Update Configuration File Handling**: **COMPLETED** (as detailed above, pending testing).
2.  **Ensure Automated Game Play**: Verify that `chess_game.py` can run an AI vs. AI game (e.g., V7P3R vs. Stockfish) using the updated configurations and save game data (PGN, config, logs).
    *   **Next Action**: Run a test game between V7P3R and Stockfish using `chess_game.py`.
    *   **Verification**: Check that the game completes, a PGN file is saved, a combined configuration file is saved, and logs are generated without errors related to config access.
3.  **Basic Metrics Collection**: Confirm that essential game metrics (result, players, length, configurations) are being saved correctly in the output files.
    *   **Next Action**: After a successful test game, inspect the saved PGN headers and the combined configuration file.
    *   **Verification**: Ensure all expected data points are present and accurate.
