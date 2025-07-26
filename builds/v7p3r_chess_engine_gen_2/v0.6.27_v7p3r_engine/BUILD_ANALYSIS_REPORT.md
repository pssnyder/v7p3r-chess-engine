# Build Analysis Report: v0.6.27_beta-candidate-3

**Generated:** 2025-07-25 11:18:45  
**Version:** v0.6.27  
**Tag:** beta-candidate-3  
**Overall Completeness:** 45.8%

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | 61 | ✅ |
| Total Size | 0.68 MB | ✅ |
| Python Files | 38 | ✅ |
| Critical Files | 2 categories | ⚠️ |
| Code Quality | 37.0% | ⚠️ |
| Syntax Errors | 0 | ✅ |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
| .json | 2 | 0.00 MB | 0.2 KB | .markdownlint.json |
| .md | 7 | 0.04 MB | 6.2 KB | UNIT_TESTING_GUIDE.md |
| .py | 38 | 0.50 MB | 13.5 KB | etl_processor.py |
| .txt | 1 | 0.00 MB | 0.2 KB | requirements.txt |
| .yaml | 13 | 0.13 MB | 10.3 KB | simulation_config_20250625_1615.yaml |


---

## 🗂️ Folder Structure

**Organization Score:** 75.0%  
**Maximum Depth:** 2 levels  
**Total Folders:** 9

| Folder | Files | Subfolders | Path |
|--------|-------|------------|------|
| config | 10 | 1 | config |
| simulation_configs | 2 | 0 | config\simulation_configs |
| docs | 6 | 0 | docs |
| engine_utilities | 11 | 0 | engine_utilities |
| metrics | 10 | 0 | metrics |
| v7p3r_engine | 11 | 0 | v7p3r_engine |
| v7p3r_ga_engine | 2 | 0 | v7p3r_ga_engine |
| v7p3r_nn_engine | 3 | 0 | v7p3r_nn_engine |
| v7p3r_rl_engine | 2 | 0 | v7p3r_rl_engine |


---

## 🐍 Python Code Analysis

**Total Lines of Code:** 8,929  
**Functions:** 365  
**Classes:** 37  
**Code Quality Score:** 37.0%

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
| etl_processor.py | 54.0 KB | 986 | 25 | 2 | ❌ |
| metrics_store.py | 49.5 KB | 870 | 34 | 1 | ❌ |
| engine_monitor.app.py | 43.3 KB | 752 | 16 | 3 | ❌ |
| v7p3r_score.py | 46.2 KB | 650 | 32 | 1 | ❌ |
| v7p3r_nn.py | 33.3 KB | 581 | 30 | 4 | ❌ |
| play_v7p3r.py | 25.3 KB | 448 | 16 | 1 | ❌ |
| chess_metrics.py | 28.4 KB | 443 | 6 | 0 | ❌ |
| etl_monitor.py | 21.1 KB | 414 | 12 | 1 | ❌ |
| stockfish_handler.py | 18.4 KB | 331 | 16 | 1 | ⚠️ |
| etl_scheduler.py | 17.3 KB | 317 | 10 | 1 | ❌ |
| adaptive_elo_finder.py | 18.2 KB | 301 | 8 | 1 | ❌ |
| lichess_handler.py | 15.3 KB | 300 | 19 | 1 | ⚠️ |
| engine_db_manager.py | 14.2 KB | 279 | 23 | 3 | ⚠️ |
| v7p3r_search.py | 17.8 KB | 266 | 9 | 1 | ❌ |
| v7p3r_book.py | 13.3 KB | 230 | 12 | 2 | ⚠️ |


---

## 🔗 Dependencies Analysis

**External Packages:** 43  
**Internal Modules:** 38  
**Missing Imports:** 9

### External Dependencies
```

argparse
atexit
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
multiprocessing
numpy
pandas
pickle
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
torch
traceback
uuid
v7p3r_engine
v7p3r_ga_engine
v7p3r_nn_engine
v7p3r_rl_engine
yaml
```

### ⚠️ Missing Dependencies
- chess_metrics.py imports missing metrics_store
- chess_metrics.py imports missing metrics_backup
- play_v7p3r.py imports missing v7p3r
- v7p3r.py imports missing v7p3r_search
- v7p3r.py imports missing v7p3r_score
- v7p3r.py imports missing v7p3r_ordering
- v7p3r.py imports missing v7p3r_time
- v7p3r.py imports missing v7p3r_book
- v7p3r.py imports missing v7p3r_pst


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

**Total Build Size:** 0.68 MB

### Size Distribution
- **Tiny** (< 1KB): 14 files
- **Small** (1-10KB): 25 files  
- **Medium** (10-100KB): 22 files
- **Large** (100KB-1MB): 0 files
- **Huge** (> 1MB): 0 files

### Largest Files
| File | Size | Path |
|------|------|------|
| etl_processor.py | 0.05 MB | metrics\etl_processor.py |
| metrics_store.py | 0.05 MB | metrics\metrics_store.py |
| v7p3r_score.py | 0.04 MB | v7p3r_engine\v7p3r_score.py |
| engine_monitor.app.py | 0.04 MB | engine_utilities\engine_monitor.app.py |
| v7p3r_nn.py | 0.03 MB | v7p3r_nn_engine\v7p3r_nn.py |
| simulation_config_20250625_1615.yaml | 0.03 MB | config\simulation_configs\simulation_config_20250625_1615.yaml |
| simulation_config_20250625_1312.yaml | 0.03 MB | config\simulation_configs\simulation_config_20250625_1312.yaml |
| simulation_config.yaml | 0.03 MB | config\simulation_config.yaml |
| chess_metrics.py | 0.03 MB | metrics\chess_metrics.py |
| rulesets.yaml | 0.03 MB | v7p3r_engine\rulesets.yaml |


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
2. **Resolve Dependencies**: 9 missing imports
3. **Code Review**: 28 files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**v0.6.27_beta-candidate-3** - **NEEDS WORK** 🔧

---
*Report generated by V7P3R Build Analyzer on 2025-07-25 11:18:45*
