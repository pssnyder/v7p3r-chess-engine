# Build Analysis Report: v0.6.09_beta-candidate-11

**Generated:** 2025-07-25 11:08:18  
**Version:** v0.6.09  
**Tag:** beta-candidate-11  
**Overall Completeness:** 38.5%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 22 | ✅ |
| Total Size | 0.36 MB | ✅ |
| Python Files | 14 | ✅ |
| Critical Files | 2 categories | ⚠️ |
| Code Quality | 0.0% | ⚠️ |
| Syntax Errors | 14 | ❌ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .db | 1 | 0.00 MB | 0.1 KB | chess_metrics.db |
| .json | 1 | 0.00 MB | 0.2 KB | EXTRACTION_INFO.json |
| .md | 1 | 0.00 MB | 3.0 KB | README.md |
| .py | 14 | 0.32 MB | 23.6 KB | viper.py |
| .txt | 2 | 0.01 MB | 2.7 KB | raw_data_examples.txt |
| .yaml | 3 | 0.03 MB | 10.6 KB | viper.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 25.0%  
**Maximum Depth:** 1 levels  
**Total Folders:** 2

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| engine_utilities | 4 | 0 | engine_utilities |
| metrics | 4 | 0 | metrics |


---

## 🐍 Python Code Analysis

**Total Lines of Code:** 5,370  
**Functions:** 0  
**Classes:** 0  
**Code Quality Score:** 0.0%

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
| viper.py | 69.3 KB | 953 | 0 | 0 | ❌ |
| chess_metrics.py | 42.4 KB | 737 | 0 | 0 | ❌ |
| metrics_store.py | 38.9 KB | 710 | 0 | 0 | ❌ |
| chess_game.py | 43.0 KB | 675 | 0 | 0 | ❌ |
| viper_scoring_calculation.py | 38.8 KB | 464 | 0 | 0 | ❌ |
| stockfish_handler.py | 22.2 KB | 393 | 0 | 0 | ❌ |
| lichess_handler.py | 15.3 KB | 301 | 0 | 0 | ❌ |
| engine_monitor.app.py | 17.2 KB | 300 | 0 | 0 | ❌ |
| opening_book.py | 11.6 KB | 205 | 0 | 0 | ❌ |
| piece_square_tables.py | 10.3 KB | 172 | 0 | 0 | ❌ |
| pgn_watcher.py | 7.5 KB | 172 | 0 | 0 | ❌ |
| time_manager.py | 8.3 KB | 171 | 0 | 0 | ❌ |
| viper_evaluation_webapp.py | 4.8 KB | 96 | 0 | 0 | ❌ |
| export_eval_games.py | 1.2 KB | 21 | 0 | 0 | ❌ |

### ❌ Syntax Errors (14)
- chess_game.py:1: invalid non-printable character U+FEFF
- opening_book.py:1: invalid non-printable character U+FEFF
- piece_square_tables.py:1: invalid non-printable character U+FEFF
- stockfish_handler.py:1: invalid non-printable character U+FEFF
- time_manager.py:1: invalid non-printable character U+FEFF
- viper.py:1: invalid non-printable character U+FEFF
- viper_evaluation_webapp.py:1: invalid non-printable character U+FEFF
- viper_scoring_calculation.py:1: invalid non-printable character U+FEFF
- engine_utilities\engine_monitor.app.py:1: invalid non-printable character U+FEFF
- engine_utilities\export_eval_games.py:1: invalid non-printable character U+FEFF


---

## 🔗 Dependencies Analysis

**External Packages:** 26  
**Internal Modules:** 14  
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
- No suggestions


---

## 📏 Size Analysis

**Total Build Size:** 0.36 MB

### Size Distribution
- **Tiny** (< 1KB): 3 files
- **Small** (1-10KB): 8 files  
- **Medium** (10-100KB): 11 files
- **Large** (100KB-1MB): 0 files
- **Huge** (> 1MB): 0 files

### Largest Files
| File | Size | Path |
|------|------|------|
| viper.py | 0.07 MB | viper.py |
| chess_game.py | 0.04 MB | chess_game.py |
| chess_metrics.py | 0.04 MB | metrics\chess_metrics.py |
| metrics_store.py | 0.04 MB | metrics\metrics_store.py |
| viper_scoring_calculation.py | 0.04 MB | viper_scoring_calculation.py |
| viper.yaml | 0.02 MB | viper.yaml |
| stockfish_handler.py | 0.02 MB | stockfish_handler.py |
| engine_monitor.app.py | 0.02 MB | engine_utilities\engine_monitor.app.py |
| lichess_handler.py | 0.01 MB | engine_utilities\lichess_handler.py |
| opening_book.py | 0.01 MB | opening_book.py |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- 🗑️ **CLEANUP CANDIDATE**: Multiple critical issues detected
- Consider archiving or using for reference only
- Fix 14 syntax errors


---

## 📋 Action Items

### Immediate Tasks
1. **Fix Syntax Errors**: 14 files need attention
2. **Resolve Dependencies**: 1 missing imports
3. **Code Review**: 14 files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**v0.6.09_beta-candidate-11** - **CLEANUP NEEDED** 🗑️

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:08:18*
