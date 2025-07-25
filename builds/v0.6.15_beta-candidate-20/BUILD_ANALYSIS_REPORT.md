# Build Analysis Report: v0.6.15_beta-candidate-20

**Generated:** 2025-07-25 11:08:54  
**Version:** v0.6.15  
**Tag:** beta-candidate-20  
**Overall Completeness:** 43.2%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 91 | ✅ |
| Total Size | 432.50 MB | ⚠️ |
| Python Files | 39 | ✅ |
| Critical Files | 2 categories | ⚠️ |
| Code Quality | 22.9% | ⚠️ |
| Syntax Errors | 0 | ✅ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .code-workspace | 1 | 0.00 MB | 0.1 KB | DEVELOPMENT - Viper Chess Engine.code-workspace |
| .csv | 1 | 0.00 MB | 0.0 KB | static_metrics.csv |
| .db | 1 | 431.48 MB | 441840.0 KB | chess_metrics.db |
| .jpg | 1 | 0.00 MB | 1.4 KB | chess_board_theme.jpg |
| .md | 17 | 0.29 MB | 17.2 KB | Ready - Distributed Computing Architecture Project.md |
| .no_extension | 3 | 0.02 MB | 5.5 KB | metrics_revision_20250609_chat |
| .pgn | 1 | 0.00 MB | 0.3 KB | rl_ai_game.pgn |
| .pkl | 2 | 0.18 MB | 92.4 KB | move_vocab_20250531_164812.pkl |
| .png | 12 | 0.09 MB | 7.6 KB | wQ.png |
| .py | 39 | 0.34 MB | 8.9 KB | viper.py |
| .pyc | 2 | 0.06 MB | 29.8 KB | metrics_store.cpython-312.pyc |
| .txt | 3 | 0.01 MB | 4.9 KB | raw_data_examples.txt |
| .yaml | 8 | 0.03 MB | 4.3 KB | viper.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 50.0%  
**Maximum Depth:** 4 levels  
**Total Folders:** 29

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| engine_utilities | 14 | 2 | engine_utilities |
| __pycache__ | 1 | 0 | engine_utilities\__pycache__ |
| templates | 1 | 0 | engine_utilities\templates |
| games | 0 | 0 | games |
| images | 13 | 0 | images |
| logging | 0 | 0 | logging |
| metrics | 7 | 1 | metrics |
| __pycache__ | 1 | 0 | metrics\__pycache__ |
| product_management | 1 | 1 | product_management |
| ideas | 0 | 3 | product_management\ideas |


---

## 🐍 Python Code Analysis

**Total Lines of Code:** 5,570  
**Functions:** 248  
**Classes:** 21  
**Code Quality Score:** 22.9%

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
| viper.py | 69.3 KB | 952 | 34 | 2 | ❌ |
| metrics_store.py | 44.6 KB | 796 | 34 | 1 | ❌ |
| chess_game.py | 43.0 KB | 673 | 19 | 1 | ❌ |
| viper_scoring_calculation.py | 39.1 KB | 473 | 29 | 1 | ❌ |
| chess_metrics.py | 28.6 KB | 405 | 6 | 0 | ❌ |
| stockfish_handler.py | 22.2 KB | 392 | 15 | 1 | ❌ |
| lichess_handler.py | 15.3 KB | 300 | 19 | 1 | ⚠️ |
| engine_monitor.app.py | 17.2 KB | 299 | 10 | 0 | ❌ |
| opening_book.py | 11.6 KB | 204 | 12 | 2 | ⚠️ |
| pgn_watcher.py | 9.2 KB | 197 | 15 | 2 | ✅ |
| piece_square_tables.py | 10.3 KB | 171 | 6 | 1 | ❌ |
| time_manager.py | 8.3 KB | 170 | 9 | 1 | ⚠️ |
| viper_genetic_ai.py | 8.6 KB | 152 | 13 | 3 | ✅ |
| viper_genetic_ai_training.py | 4.9 KB | 101 | 4 | 0 | ❌ |
| engine_db_manager.py | 5.0 KB | 87 | 10 | 2 | ✅ |


---

## 🔗 Dependencies Analysis

**External Packages:** 34  
**Internal Modules:** 37  
**Missing Imports:** 2

### External Dependencies
```

__future__
atexit
chess
chess_core
copy
dash
engine_utilities
genetic_algorithm
glob
hashlib
http
importlib
io
logging
matplotlib
metrics
numpy
pandas
pickle
plotly
psutil
pygame
queue
requests
seaborn
shutil
socket
sqlite3
streamlit
subprocess
threading
torch
yaml
```

### ⚠️ Missing Dependencies
- chess_metrics.py imports missing metrics_store
- chess_metrics.py imports missing metrics_backup


---

## 🎯 Critical Files Assessment

**Completeness Score:** 33.3%

### Entry Points
- ✅ chess_game.py
- ✅ viper.py

### Core Components Status
- ✅ **Main Entry**: chess_game.py, viper.py
- ✅ **Game Logic**: chess_game.py
- ❌ **Engine Core**: Missing all files
- ❌ **Search**: Missing all files
- ❌ **Evaluation**: Missing all files
- ❌ **Config**: Missing all files


---

## ⚠️ Issues & Recommendations

### Critical Issues
- No critical issues detected

### Warnings
- ⚠️ Empty directory: games
- ⚠️ Empty directory: logging
- ⚠️ Empty directory: training\training_data\pgn_data_chess.com_v7p3r
- ⚠️ Empty directory: training\training_data\pgn_data_endgames
- ⚠️ Empty directory: training\training_data\pgn_data_general
- ⚠️ Empty directory: training\training_data\pgn_data_important_games
- ⚠️ Empty directory: training\training_data\pgn_data_openings
- ⚠️ Empty directory: training\training_data\pgn_data_tactics
- ⚠️ Very large file: chess_metrics.db (431.5 MB)

### Suggestions
- 💡 Python cache directories present: 2 directories
- 💡 Temporary files found: .gitattributes, .gitignore


---

## 📏 Size Analysis

**Total Build Size:** 432.50 MB

### Size Distribution
- **Tiny** (< 1KB): 30 files
- **Small** (1-10KB): 37 files  
- **Medium** (10-100KB): 23 files
- **Large** (100KB-1MB): 0 files
- **Huge** (> 1MB): 1 files

### Largest Files
| File | Size | Path |
|------|------|------|
| chess_metrics.db | 431.48 MB | metrics\chess_metrics.db |
| move_vocab_20250531_164812.pkl | 0.09 MB | training\viper_genetic_ai_engine\viper_genetic_ai_moves\move_vocab_20250531_164812.pkl |
| move_vocab_20250601_094824.pkl | 0.09 MB | training\viper_nn_ai_engine\viper_nn_ai_moves\move_vocab_20250601_094824.pkl |
| Ready - Distributed Computing Architecture Project.md | 0.07 MB | product_management\ideas\projects\Ready - Distributed Computing Architecture Project.md |
| Ready - Distributed_Computing Architecture_Project.md | 0.07 MB | product_management\ideas\projects\Ready - Distributed_Computing Architecture_Project.md |
| viper.py | 0.07 MB | viper.py |
| metrics_store.py | 0.04 MB | metrics\metrics_store.py |
| metrics_store.cpython-312.pyc | 0.04 MB | metrics\__pycache__\metrics_store.cpython-312.pyc |
| chess_game.py | 0.04 MB | chess_game.py |
| viper_scoring_calculation.py | 0.04 MB | engine_utilities\viper_scoring_calculation.py |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- 🔧 **LOW PRIORITY**: Significant issues need resolution
- Consider for feature extraction rather than full restoration
- Remove 19 empty files
- Review 7 sets of duplicate files


---

## 📋 Action Items

### Immediate Tasks
1. **Fix Syntax Errors**: 0 files need attention
2. **Resolve Dependencies**: 2 missing imports
3. **Code Review**: 31 files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**v0.6.15_beta-candidate-20** - **NEEDS WORK** 🔧

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:08:54*
