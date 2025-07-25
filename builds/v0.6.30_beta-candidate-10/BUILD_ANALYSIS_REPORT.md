# Build Analysis Report: v0.6.30_beta-candidate-10

**Generated:** 2025-07-25 11:18:45  
**Version:** v0.6.30  
**Tag:** beta-candidate-10  
**Overall Completeness:** 47.0%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 85 | ✅ |
| Total Size | 1.41 MB | ✅ |
| Python Files | 52 | ✅ |
| Critical Files | 2 categories | ⚠️ |
| Code Quality | 27.8% | ⚠️ |
| Syntax Errors | 0 | ✅ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .db | 4 | 0.00 MB | 0.1 KB | puzzle_data.db |
| .json | 6 | 0.49 MB | 84.1 KB | puzzle_test_set_Beginner_mate_20250629_032420.json |
| .md | 8 | 0.05 MB | 6.1 KB | UNIT_TESTING_GUIDE.md |
| .py | 52 | 0.74 MB | 14.6 KB | v7p3r_config_gui.py |
| .txt | 1 | 0.00 MB | 0.3 KB | requirements.txt |
| .yaml | 14 | 0.12 MB | 9.0 KB | simulation_config_20250625_1615.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 100.0%  
**Maximum Depth:** 2 levels  
**Total Folders:** 15

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| config | 10 | 1 | config |
| simulation_configs | 2 | 0 | config\simulation_configs |
| docs | 7 | 0 | docs |
| engine_utilities | 8 | 0 | engine_utilities |
| metrics | 13 | 0 | metrics |
| puzzles | 6 | 0 | puzzles |
| v7p3r_engine | 12 | 2 | v7p3r_engine |
| rulesets | 1 | 0 | v7p3r_engine\rulesets |
| saved_configs | 2 | 0 | v7p3r_engine\saved_configs |
| v7p3r_ga_engine | 6 | 1 | v7p3r_ga_engine |


---

## 🐍 Python Code Analysis

**Total Lines of Code:** 13,400  
**Functions:** 513  
**Classes:** 48  
**Code Quality Score:** 27.8%

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
| v7p3r_config_gui.py | 74.2 KB | 1173 | 42 | 1 | ❌ |
| etl_processor.py | 54.0 KB | 986 | 25 | 2 | ❌ |
| metrics_store.py | 49.5 KB | 870 | 34 | 1 | ❌ |
| v7p3r_score.py | 57.7 KB | 795 | 36 | 1 | ❌ |
| engine_monitor.app.py | 43.3 KB | 752 | 16 | 3 | ❌ |
| puzzle_db_manager.py | 30.7 KB | 612 | 22 | 2 | ❌ |
| v7p3r_play.py | 36.9 KB | 593 | 22 | 2 | ❌ |
| v7p3r_nn.py | 33.5 KB | 586 | 31 | 4 | ❌ |
| v7p3r_rl.py | 29.4 KB | 577 | 35 | 5 | ❌ |
| chess_metrics.py | 28.4 KB | 443 | 6 | 0 | ❌ |
| v7p3r_search.py | 29.6 KB | 437 | 10 | 1 | ❌ |
| etl_monitor.py | 21.1 KB | 414 | 12 | 1 | ❌ |
| stockfish_handler.py | 20.6 KB | 378 | 20 | 1 | ⚠️ |
| v7p3r_ga_performance_analyzer.py | 19.3 KB | 366 | 16 | 2 | ❌ |
| etl_scheduler.py | 17.3 KB | 317 | 10 | 1 | ❌ |


---

## 🔗 Dependencies Analysis

**External Packages:** 44  
**Internal Modules:** 52  
**Missing Imports:** 11

### External Dependencies
```

argparse
atexit
cProfile
chess
chess_game
concurrent
copy
csv
dash
dataclasses
engine_utilities
functools
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
plotly
pstats
psutil
puzzles
pygame
queue
requests
seaborn
socket
sqlite3
streamlit
subprocess
threading
tkinter
torch
traceback
uuid
v7p3r_engine
v7p3r_nn_engine
v7p3r_rl_engine
yaml
```

### ⚠️ Missing Dependencies
- chess_metrics.py imports missing metrics_store
- chess_metrics.py imports missing metrics_backup
- v7p3r.py imports missing v7p3r_search
- v7p3r.py imports missing v7p3r_score
- v7p3r.py imports missing v7p3r_ordering
- v7p3r.py imports missing v7p3r_time
- v7p3r.py imports missing v7p3r_book
- v7p3r.py imports missing v7p3r_pst
- v7p3r_ga.py imports missing v7p3r_ga_ruleset_manager
- v7p3r_ga_training.py imports missing v7p3r_ga


---

## 🎯 Critical Files Assessment

**Completeness Score:** 33.3%

### Entry Points
- ✅ v7p3r.py

### Core Components Status
- ✅ **Main Entry**: v7p3r.py
- ✅ **Search**: v7p3r_search.py
- ❌ **Engine Core**: Missing all files
- ❌ **Game Logic**: Missing all files
- ❌ **Evaluation**: Missing all files
- ❌ **Config**: Missing all files


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

**Total Build Size:** 1.41 MB

### Size Distribution
- **Tiny** (< 1KB): 21 files
- **Small** (1-10KB): 36 files  
- **Medium** (10-100KB): 27 files
- **Large** (100KB-1MB): 1 files
- **Huge** (> 1MB): 0 files

### Largest Files
| File | Size | Path |
|------|------|------|
| puzzle_test_set_Beginner_mate_20250629_032420.json | 0.49 MB | puzzles\puzzle_test_set_Beginner_mate_20250629_032420.json |
| v7p3r_config_gui.py | 0.07 MB | v7p3r_engine\v7p3r_config_gui.py |
| v7p3r_score.py | 0.06 MB | v7p3r_engine\v7p3r_score.py |
| etl_processor.py | 0.05 MB | metrics\etl_processor.py |
| metrics_store.py | 0.05 MB | metrics\metrics_store.py |
| engine_monitor.app.py | 0.04 MB | engine_utilities\engine_monitor.app.py |
| v7p3r_play.py | 0.04 MB | v7p3r_engine\v7p3r_play.py |
| v7p3r_nn.py | 0.03 MB | v7p3r_nn_engine\v7p3r_nn.py |
| simulation_config_20250625_1615.yaml | 0.03 MB | config\simulation_configs\simulation_config_20250625_1615.yaml |
| simulation_config_20250625_1312.yaml | 0.03 MB | config\simulation_configs\simulation_config_20250625_1312.yaml |


---

## 🧹 Cleanup Recommendations

### Priority Actions
- 🔧 **LOW PRIORITY**: Significant issues need resolution
- Consider for feature extraction rather than full restoration
- Remove 1 empty files


---

## 📋 Action Items

### Immediate Tasks
1. **Fix Syntax Errors**: 0 files need attention
2. **Resolve Dependencies**: 11 missing imports
3. **Code Review**: 41 files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**v0.6.30_beta-candidate-10** - **NEEDS WORK** 🔧

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:18:45*
