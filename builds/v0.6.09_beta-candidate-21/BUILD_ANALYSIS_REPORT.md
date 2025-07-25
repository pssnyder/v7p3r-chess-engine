# Build Analysis Report: v0.6.09_beta-candidate-21

**Generated:** 2025-07-25 11:08:18  
**Version:** v0.6.09  
**Tag:** beta-candidate-21  
**Overall Completeness:** 44.9%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 586 | ✅ |
| Total Size | 1921.67 MB | ⚠️ |
| Python Files | 15 | ✅ |
| Critical Files | 2 categories | ⚠️ |
| Code Quality | 25.6% | ⚠️ |
| Syntax Errors | 0 | ✅ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .1 | 1 | 10.00 MB | 10240.0 KB | viper_evaluation_engine.log.1 |
| .code-workspace | 1 | 0.00 MB | 0.1 KB | TESTING - Viper Chess Engine.code-workspace |
| .csv | 1 | 0.00 MB | 0.0 KB | static_metrics.csv |
| .db | 1 | 34.81 MB | 35644.0 KB | chess_metrics.db |
| .ipynb | 1 | 0.01 MB | 9.9 KB | updates_20250609_001.ipynb |
| .jpg | 1 | 0.00 MB | 1.4 KB | chess_board_theme.jpg |
| .json | 1 | 0.00 MB | 1.0 KB | devcontainer.json |
| .log | 74 | 1872.95 MB | 25917.6 KB | eval_game_20250607_002251.log |
| .md | 1 | 0.00 MB | 3.0 KB | README.md |
| .no_extension | 2 | 0.00 MB | 1.0 KB | .gitignore |
| .pgn | 259 | 1.34 MB | 5.3 KB | export_all_eval_games_20250606_210841.pgn |
| .png | 12 | 0.09 MB | 7.6 KB | wQ.png |
| .py | 15 | 0.32 MB | 22.1 KB | viper.py |
| .pyc | 12 | 0.34 MB | 29.0 KB | evaluation_engine.cpython-312.pyc |
| .txt | 2 | 0.01 MB | 2.7 KB | raw_data_examples.txt |
| .yaml | 202 | 1.79 MB | 9.1 KB | viper.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 25.0%  
**Maximum Depth:** 2 levels  
**Total Folders:** 10

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| .devcontainer | 1 | 0 | .devcontainer |
| __pycache__ | 6 | 0 | __pycache__ |
| engine_utilities | 11 | 1 | engine_utilities |
| __pycache__ | 5 | 0 | engine_utilities\__pycache__ |
| games | 529 | 0 | games |
| ideas | 1 | 0 | ideas |
| images | 13 | 0 | images |
| logging | 4 | 0 | logging |
| metrics | 5 | 1 | metrics |
| __pycache__ | 1 | 0 | metrics\__pycache__ |


---

## 🐍 Python Code Analysis

**Total Lines of Code:** 5,361  
**Functions:** 218  
**Classes:** 14  
**Code Quality Score:** 25.6%

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
| viper.py | 69.3 KB | 952 | 34 | 2 | ❌ |
| chess_metrics.py | 42.4 KB | 736 | 23 | 1 | ❌ |
| metrics_store.py | 38.9 KB | 709 | 28 | 1 | ❌ |
| chess_game.py | 43.0 KB | 673 | 19 | 1 | ❌ |
| viper_scoring_calculation.py | 38.9 KB | 469 | 29 | 1 | ❌ |
| stockfish_handler.py | 22.2 KB | 392 | 15 | 1 | ❌ |
| lichess_handler.py | 15.3 KB | 300 | 19 | 1 | ⚠️ |
| engine_monitor.app.py | 17.2 KB | 299 | 10 | 0 | ❌ |
| opening_book.py | 11.6 KB | 204 | 12 | 2 | ⚠️ |
| pgn_watcher.py | 7.5 KB | 171 | 13 | 2 | ✅ |
| piece_square_tables.py | 10.3 KB | 171 | 6 | 1 | ❌ |
| time_manager.py | 8.3 KB | 170 | 9 | 1 | ⚠️ |
| viper_evaluation_webapp.py | 4.8 KB | 95 | 0 | 0 | ❌ |
| export_eval_games.py | 1.2 KB | 20 | 1 | 0 | ⚠️ |
| viper_gui.app.py | 0.2 KB | 0 | 0 | 0 | ❌ |


---

## 🔗 Dependencies Analysis

**External Packages:** 26  
**Internal Modules:** 15  
**Missing Imports:** 1

### External Dependencies
```

__future__
atexit
chess
dash
engine_utilities
glob
importlib
io
logging
matplotlib
metrics
numpy
pandas
plotly
psutil
pygame
queue
requests
seaborn
socket
sqlite3
streamlit
subprocess
threading
yaml
```

### ⚠️ Missing Dependencies
- chess_metrics.py imports missing metrics_store


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
- No warnings

### Suggestions
- 💡 Large file: eval_game_20250606_211902.log (33.3 MB)
- 💡 Large file: eval_game_20250606_211906.log (34.8 MB)
- 💡 Large file: eval_game_20250606_211911.log (36.3 MB)
- 💡 Large file: eval_game_20250606_211915.log (37.8 MB)
- 💡 Large file: eval_game_20250606_211920.log (39.3 MB)
- 💡 Large file: eval_game_20250606_211924.log (30.8 MB)
- 💡 Large file: eval_game_20250606_211929.log (32.3 MB)
- 💡 Large file: eval_game_20250606_211933.log (33.8 MB)
- 💡 Large file: eval_game_20250606_211938.log (35.3 MB)
- 💡 Large file: eval_game_20250606_211942.log (36.7 MB)


---

## 📏 Size Analysis

**Total Build Size:** 1921.67 MB

### Size Distribution
- **Tiny** (< 1KB): 86 files
- **Small** (1-10KB): 215 files  
- **Medium** (10-100KB): 208 files
- **Large** (100KB-1MB): 3 files
- **Huge** (> 1MB): 74 files

### Largest Files
| File | Size | Path |
|------|------|------|
| eval_game_20250607_002251.log | 39.95 MB | games\eval_game_20250607_002251.log |
| eval_game_20250606_212213.log | 39.74 MB | games\eval_game_20250606_212213.log |
| eval_game_20250607_002423.log | 39.72 MB | games\eval_game_20250607_002423.log |
| eval_game_20250607_002556.log | 39.48 MB | games\eval_game_20250607_002556.log |
| eval_game_20250606_211920.log | 39.26 MB | games\eval_game_20250606_211920.log |
| eval_game_20250607_002729.log | 39.25 MB | games\eval_game_20250607_002729.log |
| eval_game_20250607_002902.log | 39.01 MB | games\eval_game_20250607_002902.log |
| eval_game_20250607_003039.log | 38.78 MB | games\eval_game_20250607_003039.log |
| eval_game_20250606_212243.log | 38.73 MB | games\eval_game_20250606_212243.log |
| eval_game_20250607_003212.log | 38.54 MB | games\eval_game_20250607_003212.log |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- 🔧 **LOW PRIORITY**: Significant issues need resolution
- Consider for feature extraction rather than full restoration
- Review 17 sets of duplicate files


---

## 📋 Action Items

### Immediate Tasks
1. **Fix Syntax Errors**: 0 files need attention
2. **Resolve Dependencies**: 1 missing imports
3. **Code Review**: 12 files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**v0.6.09_beta-candidate-21** - **NEEDS WORK** 🔧

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:08:18*
