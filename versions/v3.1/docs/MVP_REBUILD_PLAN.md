# V7P3R Chess Engine MVP Rebuild Plan

## Core Principles
1. Clean slate approach - rewrite all v7p3r_* modules
2. Only implement features enabled in config
3. No duplicate functionality across modules
4. Single responsibility principle
5. Preserve file structure and module names
6. Build for future extensibility

## Currently Enabled Features (from config)
- Basic engine functionality (depth: 6)
- Checkmate detection
- Stalemate detection
- Draw prevention
- Primary scoring
- Secondary scoring
- Material count (weight: 10)
- Material score (weight: 10)
- Piece square tables (weight: 10)
- Game phase detection
- Quiescence search
- MVV-LVA
- Alpha-beta pruning
- Move ordering (max_ordered_moves: 10)

## Module Rebuild Plan

### 1. v7p3r_config.py
- Load default_config.json
- Implement override system using override_config.json
- No modifications to default_config.json

### 2. v7p3r_mvv_lva.py
- Core capture evaluation logic
- Single source of truth for piece values
- MVV-LVA scoring matrix
- Capture calculation functions (to be used by other modules)

### 3. v7p3r_score.py
- Primary scoring system
- Secondary scoring system
- Integration with MVV-LVA for captures
- Material evaluation
- Piece square table evaluation

### 4. v7p3r_pst.py
- Piece square tables definition
- Game phase detection
- Table interpolation based on game phase

### 5. v7p3r_ordering.py
- Move ordering using MVV-LVA scores
- Basic promotion scoring
- No duplicate capture logic

### 6. v7p3r_search.py
- Minimax implementation
- Alpha-beta pruning
- Basic quiescence search
- Depth management

### 7. v7p3r_quiescence.py
- Stand-alone quiescence search
- Integration with MVV-LVA
- Basic delta pruning

### 8. v7p3r_rules.py
- Checkmate detection
- Stalemate detection
- Draw detection
- Basic move validation

### 9. v7p3r_engine.py
- Main engine class
- Search coordination
- Best move selection
- Time management

### 10. v7p3r_utilities.py
- Helper functions
- Common utilities
- Shared constants

## Implementation Order
1. Start with core evaluation (MVV-LVA, scoring)
2. Build up search functionality
3. Add move ordering
4. Implement game rules
5. Complete engine coordination

## Validation Steps
1. Each module will be tested in isolation
2. Integration testing between dependent modules
3. Full engine validation via play_chess.py
4. Performance benchmarking

## Next Steps
1. Begin with v7p3r_mvv_lva.py rebuild
2. Ensure single responsibility for capture evaluation
3. Remove all duplicate functionality
4. Build other modules on this foundation
