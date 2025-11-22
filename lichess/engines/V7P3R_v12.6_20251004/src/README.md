# === PAT'S V7P3R v12.3 CORE ENGINE FILES ===

## Clean src/ Directory Structure  

The `src/` directory now contains only the essential v12.3 core engine files with unified evaluator:

## === PAT'S ENGINE FEATURES QUICK MAP ===

**Want to adjust search behavior?** → `v7p3r.py` lines 190-220 (piece values, depth, TT size)
**Want to tune time management?** → `v7p3r.py` lines 1250-1280 + `v7p3r_uci.py` lines 90-120
**Want to modify evaluation?** → `v7p3r_bitboard_evaluator.py` lines 50-90 (tuning constants)
**Want to change move ordering?** → `v7p3r.py` lines 800-950 (advanced move ordering)
**Want to adjust king safety?** → `v7p3r_bitboard_evaluator.py` lines 50-80 (king safety constants)
**Want to tweak pawn evaluation?** → `v7p3r_bitboard_evaluator.py` lines 60-90 (pawn structure)
**Want to enable/disable features?** → `v7p3r.py` lines 180-190 (feature toggles)

### **Core Engine Implementation**
- **`v7p3r.py`** - Main chess engine with search, unified evaluation, and tactical detection
- **`v7p3r_uci.py`** - UCI (Universal Chess Interface) protocol implementation

### **Unified Evaluation System**
- **`v7p3r_bitboard_evaluator.py`** - Comprehensive bitboard evaluation with integrated:
  - Material and positional evaluation  
  - King safety pattern detection
  - Advanced pawn structure analysis
  - Tactical pattern recognition (knight forks, pins/skewers)
### **Data**  
- **`v7p3r_enhanced_nudges.json`** - Enhanced nudge database (2160 positions)

## V12.3 Architectural Changes

### Files Consolidated (No longer separate):
- `v7p3r_advanced_pawn_evaluator.py` → **Integrated into unified bitboard evaluator**
- `v7p3r_king_safety_evaluator.py` → **Integrated into unified bitboard evaluator**

All advanced evaluation features are now part of the single `v7p3r_bitboard_evaluator.py` for maximum performance and code clarity.

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

## V12.3 Architecture

The unified v12.3 engine follows a streamlined architecture:

```
v7p3r.py (Main Engine)
├── v7p3r_bitboard_evaluator.py (Unified Evaluation)
│   ├── Material & Positional Evaluation
│   ├── King Safety Pattern Detection  
│   ├── Advanced Pawn Structure Analysis
│   └── Tactical Pattern Recognition
└── v7p3r_enhanced_nudges.json (Position Database - disabled)

v7p3r_uci.py (UCI Interface)  
└── Imports v7p3r.py engine
```

## Benefits of V12.3 Unified Structure

1. **Single Evaluation Path**: One method call handles all evaluation aspects
2. **Maintainable**: No duplicate or experimental code in production
3. **Stable**: Only proven, essential components included
4. **Professional**: Clean, focused codebase for v12.0 foundation

This represents the cleanest engine structure since the v10.8 recovery baseline, with only the proven enhancements integrated for v12.0.