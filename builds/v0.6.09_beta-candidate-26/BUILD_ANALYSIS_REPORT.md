# Build Analysis Report: v0.6.09_beta-candidate-26

**Generated:** 2025-07-25 11:08:40  
**Version:** v0.6.09  
**Tag:** beta-candidate-26  
**Overall Completeness:** 59.8%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 391 | ✅ |
| Total Size | 933.02 MB | ⚠️ |
| Python Files | 17 | ✅ |
| Critical Files | 4 categories | ✅ |
| Code Quality | 49.2% | ⚠️ |
| Syntax Errors | 0 | ✅ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .1 | 2 | 20.00 MB | 10239.7 KB | chess_game.log.1 |
| .2 | 2 | 20.00 MB | 10239.7 KB | chess_game.log.2 |
| .3 | 2 | 20.00 MB | 10239.8 KB | chess_game.log.3 |
| .aspx | 3 | 0.17 MB | 59.2 KB | DefaultWsdlHelpGenerator.aspx |
| .assets | 4 | 8.00 MB | 2047.6 KB | sharedassets0.assets |
| .browser | 3 | 0.00 MB | 1.6 KB | Compat.browser |
| .cff | 1 | 0.00 MB | 0.7 KB | CITATION.cff |
| .code-workspace | 1 | 0.00 MB | 0.1 KB | DEVELOPMENT - Viper Chess Engine.code-workspace |
| .config | 7 | 0.14 MB | 20.4 KB | machine.config |
| .cpp | 23 | 0.40 MB | 17.7 KB | search.cpp |
| .csv | 2 | 0.00 MB | 2.4 KB | viper_issues_20250608.csv |
| .db | 1 | 12.19 MB | 12484.0 KB | chess_metrics.db |
| .dll | 172 | 73.41 MB | 437.1 KB | UnityPlayer.dll |
| .exe | 3 | 77.83 MB | 26566.8 KB | stockfish-windows-x86-64-avx2.exe |
| .h | 36 | 0.27 MB | 7.7 KB | numa.h |
| .html | 1 | 0.00 MB | 0.6 KB | index.html |
| .info | 1 | 0.00 MB | 0.0 KB | app.info |
| .ini | 1 | 0.30 MB | 304.7 KB | browscap.ini |
| .jpg | 1 | 0.00 MB | 1.4 KB | chess_board_theme.jpg |
| .js | 7 | 1.36 MB | 199.0 KB | p5.js |
| .json | 3 | 0.03 MB | 8.6 KB | RuntimeInitializeOnLoads.json |
| .log | 2 | 2.15 MB | 1099.2 KB | viper_evaluation_engine.log |
| .map | 3 | 0.01 MB | 2.6 KB | settings.map |
| .md | 17 | 0.45 MB | 27.2 KB | Regression-Tests.md |
| .no_extension | 11 | 5.54 MB | 515.7 KB | unity default resources |
| .pgn | 2 | 0.01 MB | 3.1 KB | active_game.pgn |
| .pkl | 2 | 0.18 MB | 92.4 KB | move_vocab_20250531_164812.pkl |
| .png | 24 | 0.24 MB | 10.1 KB | 2000px-Chess_Pieces_Sprite_02.png |
| .pth | 2 | 377.26 MB | 193157.0 KB | chess_rl_model.pth |
| .py | 17 | 0.28 MB | 17.0 KB | evaluation_engine.py |
| .pyc | 18 | 0.57 MB | 32.6 KB | evaluation_engine.cpython-313.pyc |
| .resource | 2 | 19.21 MB | 9837.0 KB | sharedassets0.resource |
| .ress | 6 | 292.89 MB | 49987.1 KB | resources.assets.resS |
| .sh | 2 | 0.01 MB | 3.0 KB | get_native_properties.sh |
| .txt | 4 | 0.06 MB | 16.4 KB | Copying.txt |
| .xml | 1 | 0.02 MB | 25.2 KB | config.xml |
| .yaml | 2 | 0.03 MB | 15.7 KB | config.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 50.0%  
**Maximum Depth:** 8 levels  
**Total Folders:** 59

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| .devcontainer | 1 | 0 | .devcontainer |
| __pycache__ | 11 | 0 | __pycache__ |
| engine_utilities | 5 | 3 | engine_utilities |
| __pycache__ | 6 | 0 | engine_utilities\__pycache__ |
| external_engines | 0 | 3 | engine_utilities\external_engines |
| chatfish | 3 | 3 | engine_utilities\external_engines\chatfish |
| Chatfish_BurstDebugInformation_DoNotShip | 0 | 0 | engine_utilities\external_engines\chatfish\Chatfish_BurstDebugInformation_DoNotShip |
| Chatfish_Data | 19 | 3 | engine_utilities\external_engines\chatfish\Chatfish_Data |
| Managed | 166 | 0 | engine_utilities\external_engines\chatfish\Chatfish_Data\Managed |
| Plugins | 0 | 1 | engine_utilities\external_engines\chatfish\Chatfish_Data\Plugins |


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
- ⚠️ Empty directory: ideas
- ⚠️ Empty directory: product_management\roadmap_images
- ⚠️ Empty directory: training\viper_hybrid_nn_search_engine
- ⚠️ Empty directory: training\viper_reinforcement_ai_engine
- ⚠️ Empty directory: engine_utilities\external_engines\chatfish\Chatfish_BurstDebugInformation_DoNotShip
- ⚠️ Empty directory: product_management\issue_reports\error_dumps
- ⚠️ Empty directory: training\viper_genetic_ai_engine\viper_genetic_ai_models\viper_genetic_ai_engine_model_20250530
- ⚠️ Very large file: stockfish-windows-x86-64-avx2.exe (76.1 MB)
- ⚠️ Very large file: resources.assets.resS (148.9 MB)
- ⚠️ Very large file: sharedassets0.assets.resS (84.0 MB)

### Suggestions
- 💡 Large file: chess_metrics.db (12.2 MB)
- 💡 Large file: UnityPlayer.dll (27.6 MB)
- 💡 Large file: globalgamemanagers.assets.resS (35.7 MB)
- 💡 Large file: sharedassets0.resource (18.6 MB)
- 💡 Large file: sharedassets1.assets.resS (24.0 MB)
- 💡 Python cache directories present: 3 directories
- 💡 Temporary files found: .gitattributes, .gitignore


---

## 📏 Size Analysis

**Total Build Size:** 933.02 MB

### Size Distribution
- **Tiny** (< 1KB): 13 files
- **Small** (1-10KB): 108 files  
- **Medium** (10-100KB): 176 files
- **Large** (100KB-1MB): 61 files
- **Huge** (> 1MB): 33 files

### Largest Files
| File | Size | Path |
|------|------|------|
| chess_rl_model.pth | 272.83 MB | training\viper_genetic_ai_engine\viper_genetic_ai_models\viper_genetic_ai_engine_model_20250608\chess_rl_model.pth |
| resources.assets.resS | 148.94 MB | engine_utilities\external_engines\chatfish\Chatfish_Data\resources.assets.resS |
| viper_nn_ai_model_20250601.pth | 104.43 MB | training\viper_nn_ai_engine\viper_nn_ai_models\viper_nn_ai_model_20250601.pth |
| sharedassets0.assets.resS | 83.98 MB | engine_utilities\external_engines\chatfish\Chatfish_Data\sharedassets0.assets.resS |
| stockfish-windows-x86-64-avx2.exe | 76.14 MB | engine_utilities\external_engines\stockfish\stockfish-windows-x86-64-avx2.exe |
| globalgamemanagers.assets.resS | 35.72 MB | engine_utilities\external_engines\chatfish\Chatfish_Data\globalgamemanagers.assets.resS |
| UnityPlayer.dll | 27.56 MB | engine_utilities\external_engines\chatfish\UnityPlayer.dll |
| sharedassets1.assets.resS | 24.00 MB | engine_utilities\external_engines\chatfish\Chatfish_Data\sharedassets1.assets.resS |
| sharedassets0.resource | 18.61 MB | engine_utilities\external_engines\chatfish\Chatfish_Data\sharedassets0.resource |
| chess_metrics.db | 12.19 MB | metrics\chess_metrics.db |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- 🔧 **LOW PRIORITY**: Significant issues need resolution
- Consider for feature extraction rather than full restoration
- Review 4 sets of duplicate files


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
**v0.6.09_beta-candidate-26** - **NEEDS WORK** 🔧

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:08:40*
