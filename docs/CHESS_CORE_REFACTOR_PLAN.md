# Chess Core Refactor Plan

## Overview
Refactor the `v7p3rChess` class by extracting generic chess functionality into a new `ChessCore` class in `chess_core.py`. This will reduce file size and improve maintainability while preserving all existing functionality.

## Proposed Architecture

### ChessCore Class (Generic Chess Components)
Location: `chess_core.py`

**Responsibilities:**
- Basic chess board management
- Move validation and execution
- Game state tracking (new_game, game_over detection)
- PGN handling (save, load, export)
- FEN import/export
- Game result determination
- Basic move history tracking
- Time formatting utilities

**Methods to Extract:**
- `new_game()`
- `get_board_result()`
- `handle_game_end()`
- `save_game_data()`
- `quick_save_pgn()` and `quick_save_pgn_to_file()`
- `save_local_game_files()`
- `import_fen()`
- `push_move()`
- `set_headers()`
- `_format_time_for_display()`
- Basic board state properties

### v7p3rChess Class (Engine-Specific Components)
Location: `v7p3r_play.py`

**Responsibilities:**
- V7P3R engine initialization and management
- Engine-specific configurations
- AI/ML engine integration (RL, GA, NN)
- Stockfish integration
- Move processing for different engines
- Metrics recording
- Game loop and pygame management
- Engine evaluation recording
- Move display with engine-specific information

**Methods to Keep:**
- `__init__()` (simplified)
- `process_engine_move()`
- `record_evaluation()`
- `display_move_made()`
- `run()`
- `cleanup_engines()`
- `_create_nn_engine_wrapper()`
- `_get_engine_config_for_player()`
- Engine initialization logic

## Implementation Strategy

### Phase 1: Create ChessCore Base Class
1. Create `ChessCore` class with extracted generic methods
2. Include proper chess board, game, and PGN management
3. Add basic logging support
4. Ensure all methods are engine-agnostic

### Phase 2: Modify v7p3rChess to Inherit from ChessCore
1. Change `v7p3rChess` to inherit from `ChessCore`
2. Remove duplicated methods that are now in the base class
3. Override methods where engine-specific behavior is needed
4. Maintain all existing functionality and interfaces

### Phase 3: Testing and Validation
1. Ensure all existing functionality works identically
2. Verify PGN output is unchanged
3. Test all engine types (v7p3r, stockfish, RL, GA, NN)
4. Validate metrics recording continues to work

## Benefits
- Smaller, more manageable file sizes
- Clear separation of concerns
- Reusable chess components for future development
- Easier testing and debugging
- Better code organization and maintainability

## Risk Mitigation
- Preserve all existing method signatures
- Maintain backward compatibility
- Keep the same public interface for v7p3rChess
- Thorough testing before considering complete

## Files to be Modified
- `chess_core.py` - New core chess functionality
- `v7p3r_play.py` - Simplified to engine-specific logic only

## Dependencies
No new external dependencies required. All existing imports and functionality preserved.
