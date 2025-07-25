# Build Analysis Report: v0.6.07_beta-candidate-12

**Generated:** 2025-07-25 11:08:18  
**Version:** v0.6.07  
**Tag:** beta-candidate-12  
**Overall Completeness:** 53.5%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 19 | ✅ |
| Total Size | 0.31 MB | ✅ |
| Python Files | 13 | ✅ |
| Critical Files | 4 categories | ✅ |
| Code Quality | 0.0% | ⚠️ |
| Syntax Errors | 13 | ❌ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .db | 1 | 0.00 MB | 0.1 KB | chess_metrics.db |
| .json | 1 | 0.00 MB | 0.2 KB | EXTRACTION_INFO.json |
| .md | 2 | 0.01 MB | 6.0 KB | testing-scenarios.md |
| .py | 13 | 0.26 MB | 20.8 KB | evaluation_engine.py |
| .txt | 1 | 0.00 MB | 0.1 KB | requirements.txt |
| .yaml | 1 | 0.03 MB | 31.4 KB | config.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 50.0%  
**Maximum Depth:** 1 levels  
**Total Folders:** 4

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| engine_utilities | 2 | 0 | engine_utilities |
| metrics | 3 | 0 | metrics |
| testing | 2 | 0 | testing |
| web_applications | 3 | 0 | web_applications |


---

## 🐍 Python Code Analysis

**Total Lines of Code:** 4,909  
**Functions:** 0  
**Classes:** 0  
**Code Quality Score:** 0.0%

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
| evaluation_engine.py | 81.1 KB | 1300 | 0 | 0 | ❌ |
| local_metrics_dashboard.py | 39.2 KB | 783 | 0 | 0 | ❌ |
| chess_game.py | 41.2 KB | 711 | 0 | 0 | ❌ |
| metrics_store.py | 21.7 KB | 456 | 0 | 0 | ❌ |
| lichess_bot.py | 15.1 KB | 305 | 0 | 0 | ❌ |
| engine_metrics_app.py | 17.1 KB | 300 | 0 | 0 | ❌ |
| evaluation_manager.py | 14.3 KB | 263 | 0 | 0 | ❌ |
| opening_book.py | 11.6 KB | 205 | 0 | 0 | ❌ |
| pgn_watcher.py | 7.5 KB | 172 | 0 | 0 | ❌ |
| piece_square_tables.py | 8.2 KB | 156 | 0 | 0 | ❌ |
| time_manager.py | 7.1 KB | 141 | 0 | 0 | ❌ |
| evaluation_engine_app.py | 4.7 KB | 96 | 0 | 0 | ❌ |
| export_eval_games.py | 1.2 KB | 21 | 0 | 0 | ❌ |

### ❌ Syntax Errors (13)
- chess_game.py:1: invalid non-printable character U+FEFF
- evaluation_engine.py:1: invalid non-printable character U+FEFF
- opening_book.py:1: invalid non-printable character U+FEFF
- piece_square_tables.py:1: invalid non-printable character U+FEFF
- time_manager.py:1: invalid non-printable character U+FEFF
- engine_utilities\export_eval_games.py:1: invalid non-printable character U+FEFF
- engine_utilities\pgn_watcher.py:1: invalid non-printable character U+FEFF
- metrics\local_metrics_dashboard.py:1: invalid non-printable character U+FEFF
- metrics\metrics_store.py:1: invalid non-printable character U+FEFF
- testing\evaluation_manager.py:1: invalid non-printable character U+FEFF


---

## 🔗 Dependencies Analysis

**External Packages:** 22  
**Internal Modules:** 13  
**Missing Imports:** 1

### External Dependencies
```

atexit
chess
dash
engine_utilities
glob
importlib
io
logging
matplotlib
numpy
pandas
plotly
psutil
pygame
requests
seaborn
socket
sqlite3
streamlit
threading
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
- No warnings

### Suggestions
- No suggestions


---

## 📏 Size Analysis

**Total Build Size:** 0.31 MB

### Size Distribution
- **Tiny** (< 1KB): 3 files
- **Small** (1-10KB): 7 files  
- **Medium** (10-100KB): 9 files
- **Large** (100KB-1MB): 0 files
- **Huge** (> 1MB): 0 files

### Largest Files
| File | Size | Path |
|------|------|------|
| evaluation_engine.py | 0.08 MB | evaluation_engine.py |
| chess_game.py | 0.04 MB | chess_game.py |
| local_metrics_dashboard.py | 0.04 MB | metrics\local_metrics_dashboard.py |
| config.yaml | 0.03 MB | config.yaml |
| metrics_store.py | 0.02 MB | metrics\metrics_store.py |
| engine_metrics_app.py | 0.02 MB | web_applications\engine_metrics_app.py |
| lichess_bot.py | 0.01 MB | web_applications\lichess_bot.py |
| evaluation_manager.py | 0.01 MB | testing\evaluation_manager.py |
| opening_book.py | 0.01 MB | opening_book.py |
| testing-scenarios.md | 0.01 MB | testing\testing-scenarios.md |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- 🔧 **LOW PRIORITY**: Significant issues need resolution
- Consider for feature extraction rather than full restoration
- Fix 13 syntax errors


---

## 📋 Action Items

### Immediate Tasks
1. **Fix Syntax Errors**: 13 files need attention
2. **Resolve Dependencies**: 1 missing imports
3. **Code Review**: 13 files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**v0.6.07_beta-candidate-12** - **NEEDS WORK** 🔧

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:08:18*
