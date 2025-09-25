# V7P3R v12.0 Core Engine Files

## Clean src/ Directory Structure

The `src/` directory now contains only the essential v12.0 core engine files:

### **Core Engine Implementation**
- **`v7p3r.py`** - Main chess engine with search, evaluation, and nudge system
- **`v7p3r_uci.py`** - UCI (Universal Chess Interface) protocol implementation

### **Evaluation Components**
- **`v7p3r_bitboard_evaluator.py`** - Core bitboard-based position evaluation
- **`v7p3r_advanced_pawn_evaluator.py`** - Advanced pawn structure analysis
- **`v7p3r_king_safety_evaluator.py`** - King safety evaluation system

### **Data**
- **`v7p3r_enhanced_nudges.json`** - Enhanced nudge database (2160 positions)

## Files Moved to development/

### `development/v11_experiments/`
- All v11.x experimental engine variations
- Enhanced evaluation experiments
- Tactical pattern detectors
- Search and move ordering experiments
- Old scoring calculation systems
- Time management variations

### `development/utilities/`
- Nudge database utilities and mergers
- Old nudge database files
- Duplicate README and requirements files

## Architecture

The clean v12.0 engine follows a modular architecture:

```
v7p3r.py (Main Engine)
├── v7p3r_bitboard_evaluator.py (Core Evaluation)
├── v7p3r_advanced_pawn_evaluator.py (Pawn Analysis)
├── v7p3r_king_safety_evaluator.py (King Safety)
└── v7p3r_enhanced_nudges.json (Position Database)

v7p3r_uci.py (UCI Interface)
└── Imports v7p3r.py engine
```

## Benefits of Clean Structure

1. **Clear Dependencies**: Easy to understand component relationships
2. **Maintainable**: No duplicate or experimental code in production
3. **Stable**: Only proven, essential components included
4. **Professional**: Clean, focused codebase for v12.0 foundation

This represents the cleanest engine structure since the v10.8 recovery baseline, with only the proven enhancements integrated for v12.0.