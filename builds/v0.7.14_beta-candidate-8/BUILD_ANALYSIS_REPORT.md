# Build Analysis Report: v0.7.14_beta-candidate-8

**Generated:** 2025-07-25 11:09:20  
**Version:** v0.7.14  
**Tag:** beta-candidate-8  
**Overall Completeness:** 60.0%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 28 | ✅ |
| Total Size | 0.11 MB | ✅ |
| Python Files | 18 | ✅ |
| Critical Files | 6 categories | ✅ |
| Code Quality | 0.0% | ⚠️ |
| Syntax Errors | 18 | ❌ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .db | 2 | 0.00 MB | 0.1 KB | puzzle_data.db |
| .json | 5 | 0.00 MB | 0.9 KB | speed_config.json |
| .md | 2 | 0.01 MB | 4.6 KB | V7P3R_CHESS_ENGINE_DESIGN_GUIDE.md |
| .py | 18 | 0.10 MB | 5.6 KB | v7p3r_game.py |
| .txt | 1 | 0.00 MB | 0.1 KB | requirements.txt |


---

## 🗂️ Folder Structure

**Organization Score:** 25.0%  
**Maximum Depth:** 0 levels  
**Total Folders:** 0

- **Flat structure** (no subfolders)


---

## 🐍 Python Code Analysis

**Total Lines of Code:** 2,025  
**Functions:** 0  
**Classes:** 0  
**Code Quality Score:** 0.0%

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
| v7p3r_game.py | 13.2 KB | 275 | 0 | 0 | ❌ |
| metrics.py | 10.8 KB | 213 | 0 | 0 | ❌ |
| active_game_watcher.py | 9.4 KB | 201 | 0 | 0 | ❌ |
| v7p3r_search.py | 6.7 KB | 131 | 0 | 0 | ❌ |
| v7p3r_stockfish.py | 5.5 KB | 123 | 0 | 0 | ❌ |
| v7p3r_rules.py | 5.5 KB | 118 | 0 | 0 | ❌ |
| v7p3r_pst.py | 4.9 KB | 114 | 0 | 0 | ❌ |
| v7p3r_engine.py | 5.6 KB | 111 | 0 | 0 | ❌ |
| v7p3r_tempo.py | 5.1 KB | 97 | 0 | 0 | ❌ |
| v7p3r_book.py | 4.7 KB | 93 | 0 | 0 | ❌ |
| v7p3r_quiescence.py | 4.9 KB | 89 | 0 | 0 | ❌ |
| v7p3r_secondary_scoring.py | 5.8 KB | 89 | 0 | 0 | ❌ |
| v7p3r_primary_scoring.py | 3.7 KB | 80 | 0 | 0 | ❌ |
| v7p3r_move_ordering.py | 4.8 KB | 78 | 0 | 0 | ❌ |
| v7p3r_scoring.py | 3.6 KB | 61 | 0 | 0 | ❌ |

### ❌ Syntax Errors (18)
- active_game_watcher.py:1: invalid non-printable character U+FEFF
- metrics.py:1: invalid non-printable character U+FEFF
- play_chess.py:1: invalid non-printable character U+FEFF
- v7p3r_book.py:1: invalid non-printable character U+FEFF
- v7p3r_config.py:1: invalid non-printable character U+FEFF
- v7p3r_engine.py:1: invalid non-printable character U+FEFF
- v7p3r_game.py:1: invalid non-printable character U+FEFF
- v7p3r_move_ordering.py:1: invalid non-printable character U+FEFF
- v7p3r_mvv_lva.py:1: invalid non-printable character U+FEFF
- v7p3r_primary_scoring.py:1: invalid non-printable character U+FEFF


---

## 🔗 Dependencies Analysis

**External Packages:** 5  
**Internal Modules:** 18  
**Missing Imports:** 0

### External Dependencies
```
argparse
chess
pygame
sqlite3
stockfish
```


---

## 🎯 Critical Files Assessment

**Completeness Score:** 100.0%

### Entry Points
- ✅ play_chess.py

### Core Components Status
- ✅ **Main Entry**: play_chess.py
- ✅ **Engine Core**: v7p3r_engine.py
- ✅ **Game Logic**: v7p3r_game.py, v7p3r_rules.py
- ✅ **Search**: v7p3r_search.py
- ✅ **Evaluation**: v7p3r_scoring.py
- ✅ **Config**: config.json


---

## ⚠️ Issues & Recommendations

### Critical Issues
- No critical issues detected

### Warnings
- No warnings

### Suggestions
- 💡 Temporary files found: .markdownlint.json


---

## 📏 Size Analysis

**Total Build Size:** 0.11 MB

### Size Distribution
- **Tiny** (< 1KB): 7 files
- **Small** (1-10KB): 19 files  
- **Medium** (10-100KB): 2 files
- **Large** (100KB-1MB): 0 files
- **Huge** (> 1MB): 0 files

### Largest Files
| File | Size | Path |
|------|------|------|
| v7p3r_game.py | 0.01 MB | v7p3r_game.py |
| metrics.py | 0.01 MB | metrics.py |
| active_game_watcher.py | 0.01 MB | active_game_watcher.py |
| V7P3R_CHESS_ENGINE_DESIGN_GUIDE.md | 0.01 MB | V7P3R_CHESS_ENGINE_DESIGN_GUIDE.md |
| v7p3r_search.py | 0.01 MB | v7p3r_search.py |
| v7p3r_secondary_scoring.py | 0.01 MB | v7p3r_secondary_scoring.py |
| v7p3r_engine.py | 0.01 MB | v7p3r_engine.py |
| v7p3r_stockfish.py | 0.01 MB | v7p3r_stockfish.py |
| v7p3r_rules.py | 0.01 MB | v7p3r_rules.py |
| v7p3r_tempo.py | 0.01 MB | v7p3r_tempo.py |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- 🔧 **LOW PRIORITY**: Significant issues need resolution
- Consider for feature extraction rather than full restoration
- Remove 1 empty files
- Fix 18 syntax errors


---

## 📋 Action Items

### Immediate Tasks
1. **Fix Syntax Errors**: 18 files need attention
2. **Resolve Dependencies**: 0 missing imports
3. **Code Review**: 18 files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**v0.7.14_beta-candidate-8** - **NEEDS WORK** 🔧

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:09:20*
