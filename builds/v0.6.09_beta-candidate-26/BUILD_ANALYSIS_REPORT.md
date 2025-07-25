# Build Analysis Report: v0.6.09_beta-candidate-26

**Generated:** 2025-07-25 11:18:42  
**Version:** v0.6.09  
**Tag:** beta-candidate-26  
**Overall Completeness:** 63.3%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 45 | ✅ |
| Total Size | 390.06 MB | ⚠️ |
| Python Files | 17 | ✅ |
| Critical Files | 4 categories | ✅ |
| Code Quality | 49.2% | ⚠️ |
| Syntax Errors | 0 | ✅ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .csv | 1 | 0.00 MB | 0.0 KB | static_metrics.csv |
| .db | 1 | 12.19 MB | 12484.0 KB | chess_metrics.db |
| .jpg | 1 | 0.00 MB | 1.4 KB | chess_board_theme.jpg |
| .md | 2 | 0.01 MB | 6.0 KB | testing-scenarios.md |
| .no_extension | 2 | 0.00 MB | 0.4 KB | .gitignore |
| .pgn | 1 | 0.00 MB | 0.3 KB | rl_ai_game.pgn |
| .pkl | 2 | 0.18 MB | 92.4 KB | move_vocab_20250531_164812.pkl |
| .png | 12 | 0.09 MB | 7.6 KB | wQ.png |
| .pth | 2 | 377.26 MB | 193157.0 KB | chess_rl_model.pth |
| .py | 17 | 0.28 MB | 17.0 KB | evaluation_engine.py |
| .txt | 2 | 0.01 MB | 4.7 KB | raw_data_examples.txt |
| .yaml | 2 | 0.03 MB | 15.7 KB | config.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 50.0%  
**Maximum Depth:** 4 levels  
**Total Folders:** 16

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| engine_utilities | 5 | 0 | engine_utilities |
| images | 13 | 0 | images |
| metrics | 4 | 0 | metrics |
| testing | 3 | 0 | testing |
| training | 0 | 4 | training |
| viper_genetic_ai_engine | 2 | 2 | training\viper_genetic_ai_engine |
| viper_genetic_ai_models | 0 | 2 | training\viper_genetic_ai_engine\viper_genetic_ai_models |
| viper_genetic_ai_engine_model_20250530 | 0 | 0 | training\viper_genetic_ai_engine\viper_genetic_ai_models\viper_genetic_ai_engine_model_20250530 |
| viper_genetic_ai_engine_model_20250608 | 2 | 0 | training\viper_genetic_ai_engine\viper_genetic_ai_models\viper_genetic_ai_engine_model_20250608 |
| viper_genetic_ai_moves | 1 | 0 | training\viper_genetic_ai_engine\viper_genetic_ai_moves |


---

## 🐍 Python Code Analysis

**Total Lines of Code:** 5,267  
**Functions:** 248  
**Classes:** 19  
**Code Quality Score:** 49.2%

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
| evaluation_engine.py | 81.1 KB | 1299 | 62 | 2 | ❌ |
| local_metrics_dashboard.py | 39.2 KB | 782 | 24 | 1 | ❌ |
| chess_game.py | 41.2 KB | 710 | 33 | 1 | ❌ |
| metrics_store.py | 21.7 KB | 456 | 23 | 1 | ⚠️ |
| lichess_bot.py | 15.1 KB | 304 | 19 | 1 | ⚠️ |
| engine_metrics_app.py | 17.1 KB | 299 | 10 | 0 | ❌ |
| evaluation_manager.py | 14.3 KB | 262 | 10 | 1 | ❌ |
| opening_book.py | 11.6 KB | 204 | 12 | 2 | ⚠️ |
| pgn_watcher.py | 7.5 KB | 171 | 13 | 2 | ✅ |
| piece_square_tables.py | 8.2 KB | 155 | 6 | 1 | ⚠️ |
| viper_genetic_ai.py | 8.6 KB | 152 | 13 | 3 | ✅ |
| time_manager.py | 7.1 KB | 140 | 8 | 1 | ⚠️ |
| viper_genetic_ai_training.py | 4.9 KB | 101 | 4 | 0 | ❌ |
| evaluation_engine_app.py | 4.7 KB | 95 | 0 | 0 | ❌ |
| viper_nn_ai_training.py | 3.5 KB | 71 | 4 | 1 | ✅ |


---

## 🔗 Dependencies Analysis

**External Packages:** 27  
**Internal Modules:** 17  
**Missing Imports:** 1

### External Dependencies
```

atexit
chess
chess_core
copy
dash
engine_utilities
genetic_algorithm
glob
importlib
io
logging
matplotlib
numpy
pandas
pickle
plotly
psutil
pygame
requests
seaborn
socket
sqlite3
streamlit
threading
torch
yaml
```

### ⚠️ Missing Dependencies
- local_metrics_dashboard.py imports missing metrics_store


---

## 🎯 Critical Files Assessment

**Completeness Score:** 66.7%

### Entry Points
- ✅ chess_game.py

### Core Components Status
- ✅ **Main Entry**: chess_game.py
- ✅ **Engine Core**: evaluation_engine.py
- ✅ **Game Logic**: chess_game.py
- ✅ **Config**: config.yaml
- ❌ **Search**: Missing all files
- ❌ **Evaluation**: Missing all files


---

## ⚠️ Issues & Recommendations

### Critical Issues
- No critical issues detected

### Warnings
- ⚠️ Empty directory: training\viper_hybrid_nn_search_engine
- ⚠️ Empty directory: training\viper_reinforcement_ai_engine
- ⚠️ Empty directory: training\viper_genetic_ai_engine\viper_genetic_ai_models\viper_genetic_ai_engine_model_20250530
- ⚠️ Very large file: chess_rl_model.pth (272.8 MB)
- ⚠️ Very large file: viper_nn_ai_model_20250601.pth (104.4 MB)

### Suggestions
- 💡 Large file: chess_metrics.db (12.2 MB)
- 💡 Temporary files found: .gitattributes, .gitignore


---

## 📏 Size Analysis

**Total Build Size:** 390.06 MB

### Size Distribution
- **Tiny** (< 1KB): 6 files
- **Small** (1-10KB): 24 files  
- **Medium** (10-100KB): 12 files
- **Large** (100KB-1MB): 0 files
- **Huge** (> 1MB): 3 files

### Largest Files
| File | Size | Path |
|------|------|------|
| chess_rl_model.pth | 272.83 MB | training\viper_genetic_ai_engine\viper_genetic_ai_models\viper_genetic_ai_engine_model_20250608\chess_rl_model.pth |
| viper_nn_ai_model_20250601.pth | 104.43 MB | training\viper_nn_ai_engine\viper_nn_ai_models\viper_nn_ai_model_20250601.pth |
| chess_metrics.db | 12.19 MB | metrics\chess_metrics.db |
| move_vocab_20250531_164812.pkl | 0.09 MB | training\viper_genetic_ai_engine\viper_genetic_ai_moves\move_vocab_20250531_164812.pkl |
| move_vocab_20250601_094824.pkl | 0.09 MB | training\viper_nn_ai_engine\viper_nn_ai_moves\move_vocab_20250601_094824.pkl |
| evaluation_engine.py | 0.08 MB | evaluation_engine.py |
| chess_game.py | 0.04 MB | chess_game.py |
| local_metrics_dashboard.py | 0.04 MB | metrics\local_metrics_dashboard.py |
| config.yaml | 0.03 MB | config.yaml |
| metrics_store.py | 0.02 MB | metrics\metrics_store.py |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- ⚠️ **MEDIUM PRIORITY**: Address missing dependencies and syntax errors
- Good candidate for fixes and improvements
- Review 1 sets of duplicate files


---

## 📋 Action Items

### Immediate Tasks
1. **Fix Syntax Errors**: 0 files need attention
2. **Resolve Dependencies**: 1 missing imports
3. **Code Review**: 9 files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**v0.6.09_beta-candidate-26** - **GOOD CANDIDATE** ✅

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:18:42*
