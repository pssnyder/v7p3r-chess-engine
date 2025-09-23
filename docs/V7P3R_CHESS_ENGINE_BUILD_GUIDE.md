# V7P3R Chess Engine Build Guide

This document outlines the standardized build process and requirements for the V7P3R Chess Engine to prevent confusion and ensure consistent builds.

## Build Requirements

### Prerequisites
- **Python 3.12+** (current development version)
- **PyInstaller 6.9.0+** (for executable creation)
- **python-chess library** (core chess functionality)
- **Required Python packages** (see requirements.txt in root)

### Build Environment
- **Windows 10/11** (primary development platform)
- **8GB+ RAM** recommended for build process
- **2GB+ free disk space**
- **Git** (for version control and repository management)

## CRITICAL: Standardized Build Process

### 🚨 **ALWAYS Use the Official Build Script**
**Never** use manual PyInstaller commands. **Always** use the version-specific build script:

```bash
# For v12.0 and future versions
build_v12.0.bat
```

### 📁 **Required Directory Structure**
```
v7p3r-chess-engine/
├── src/                    # Core engine files ONLY
│   ├── v7p3r.py           # Main engine
│   ├── v7p3r_uci.py       # UCI interface
│   ├── v7p3r_bitboard_evaluator.py
│   ├── v7p3r_advanced_pawn_evaluator.py
│   ├── v7p3r_king_safety_evaluator.py
│   └── v7p3r_enhanced_nudges.json
├── dist/                   # Build output directory
├── build/                  # PyInstaller working directory (auto-generated)
├── build_v12.0.bat        # Current build script
└── V7P3R_v12.0.spec       # PyInstaller specification
```

### 🎯 **Essential Build Parameters**
```bash
pyinstaller --onefile \
    --name V7P3R_v12.0 \
    --distpath dist \
    --workpath build \
    --specpath . \
    --add-data "src\v7p3r_enhanced_nudges.json;." \
    --hidden-import chess \
    --hidden-import chess.engine \
    --hidden-import chess.pgn \
    --hidden-import json \
    --hidden-import time \
    --hidden-import sys \
    --hidden-import threading \
    --hidden-import random \
    --hidden-import math \
    src\v7p3r_uci.py
```

## Build Process Steps

### 1. **Pre-Build Verification**
```bash
# Verify clean src directory (6-7 files only)
ls src/

# Test engine functionality
python test_v12_foundation.py
```

### 2. **Execute Build**
```bash
# Run the official build script
build_v12.0.bat

# OR manual build with exact parameters above
```

### 3. **Build Verification**
```bash
# Test UCI interface
echo "uci" | dist\V7P3R_v12.0.exe

# Verify nudge database loading
echo -e "uci\nisready\nquit" | dist\V7P3R_v12.0.exe
```

### 4. **Deployment**
```bash
# Copy to engine-tester (established workflow)
cp dist\V7P3R_v12.0.exe "engine-tester\engines\V7P3R\"
```

## Critical Build Requirements

### 🔴 **Nudge Database Embedding**
- **MUST** use `--add-data "src\v7p3r_enhanced_nudges.json;."` 
- **MUST** include PyInstaller resource detection in engine code
- **Result**: Standalone executable with embedded 2160-position database

### 🔴 **Directory Standards**
- **Output**: `dist/` directory (NOT `builds/`)
- **Working**: `build/` directory (auto-generated, can be deleted)
- **Source**: `src/` directory (clean, core files only)

### 🔴 **Version Consistency**
- **Engine**: UCI reports correct version (e.g., "V7P3R v12.0")
- **Filename**: Matches version (e.g., V7P3R_v12.0.exe)
- **Build Script**: Version-specific script name

## Troubleshooting

### Common Issues
- **"builds/" folder**: Wrong directory - use `dist/`
- **External JSON files**: Missing `--add-data` parameter
- **Version mismatch**: Update UCI interface version string
- **Import errors**: Missing `--hidden-import` parameters

### Build Verification Checklist
- ✅ Executable in `dist/` directory
- ✅ UCI reports correct version
- ✅ Loads "2160 nudge positions" message
- ✅ No external files required
- ✅ Size approximately 8MB

## Version Management

### Build Script Naming
- `build_v12.0.bat` (current)
- `build_v12.1.bat` (next version)
- Archive old scripts to `development/old_builds/`

### Repository Cleanliness
- Keep `src/` directory clean (core files only)
- Archive experimental code to `development/`
- Remove build artifacts (`build/`, `__pycache__/`)

## Distribution Workflow

1. **Build**: Use version-specific script
2. **Test**: Verify functionality locally
3. **Deploy**: Copy to engine-tester from `dist/`
4. **Archive**: Old versions preserved in `dist/` for reference

---

**Following this guide ensures consistent, reliable builds and prevents the confusion that occurred during v11-v12 transition.**