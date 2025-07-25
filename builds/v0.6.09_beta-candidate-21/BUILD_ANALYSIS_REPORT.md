# Build Analysis Report: v0.6.09_beta-candidate-21

**Generated:** 2025-07-25 11:18:42  
**Version:** v0.6.09  
**Tag:** beta-candidate-21  
**Overall Completeness:** 44.9%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 39 | ✅ |
| Total Size | 35.26 MB | ✅ |
| Python Files | 15 | ✅ |
| Critical Files | 2 categories | ⚠️ |
| Code Quality | 25.6% | ⚠️ |
| Syntax Errors | 0 | ✅ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .code-workspace | 1 | 0.00 MB | 0.1 KB | TESTING - Viper Chess Engine.code-workspace |
| .csv | 1 | 0.00 MB | 0.0 KB | static_metrics.csv |
| .db | 1 | 34.81 MB | 35644.0 KB | chess_metrics.db |
| .jpg | 1 | 0.00 MB | 1.4 KB | chess_board_theme.jpg |
| .md | 1 | 0.00 MB | 3.0 KB | README.md |
| .no_extension | 2 | 0.00 MB | 1.0 KB | .gitignore |
| .png | 12 | 0.09 MB | 7.6 KB | wQ.png |
| .py | 15 | 0.32 MB | 22.1 KB | viper.py |
| .txt | 2 | 0.01 MB | 2.7 KB | raw_data_examples.txt |
| .yaml | 3 | 0.03 MB | 10.6 KB | viper.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 25.0%  
**Maximum Depth:** 1 levels  
**Total Folders:** 3

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| engine_utilities | 11 | 0 | engine_utilities |
| images | 13 | 0 | images |
| metrics | 5 | 0 | metrics |


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
- 💡 Large file: chess_metrics.db (34.8 MB)
- 💡 Temporary files found: .gitattributes, .gitignore


---

## 📏 Size Analysis

**Total Build Size:** 35.26 MB

### Size Distribution
- **Tiny** (< 1KB): 5 files
- **Small** (1-10KB): 21 files  
- **Medium** (10-100KB): 12 files
- **Large** (100KB-1MB): 0 files
- **Huge** (> 1MB): 1 files

### Largest Files
| File | Size | Path |
|------|------|------|
| chess_metrics.db | 34.81 MB | metrics\chess_metrics.db |
| viper.py | 0.07 MB | viper.py |
| chess_game.py | 0.04 MB | chess_game.py |
| chess_metrics.py | 0.04 MB | metrics\chess_metrics.py |
| viper_scoring_calculation.py | 0.04 MB | engine_utilities\viper_scoring_calculation.py |
| metrics_store.py | 0.04 MB | metrics\metrics_store.py |
| viper.yaml | 0.02 MB | viper.yaml |
| stockfish_handler.py | 0.02 MB | engine_utilities\stockfish_handler.py |
| engine_monitor.app.py | 0.02 MB | engine_utilities\engine_monitor.app.py |
| lichess_handler.py | 0.01 MB | engine_utilities\lichess_handler.py |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- 🔧 **LOW PRIORITY**: Significant issues need resolution
- Consider for feature extraction rather than full restoration


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
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:18:42*
