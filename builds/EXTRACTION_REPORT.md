# Beta Candidate Extraction Report
## Extraction Date: July 24, 2025

### Summary
Successfully extracted **17 beta candidates** from git repository history into self-contained build directories.

### Extraction Details

| Build Version | Beta Tag | Date | Files | Python | JSON | YAML/YML | Database | Markdown |
|---------------|----------|------|-------|---------|------|----------|----------|----------|
| v0.5.30 | beta-candidate-6 | unknown | 4 | 3 | 0 | 1 | 0 | 0 |
| v0.5.31 | beta-candidate-5 | unknown | 4 | 3 | 0 | 1 | 0 | 0 |
| v0.6.1 | beta-candidate-16 | unknown | 9 | 6 | 0 | 1 | 0 | 2 |
| v0.6.27 | beta-candidate-3 | unknown | 62 | 40 | 1 | 13 | 0 | 7 |
| v0.6.2 | beta-candidate-15 | unknown | 9 | 6 | 0 | 1 | 0 | 2 |
| v0.6.30 | beta-candidate-10 | unknown | 4387 | 56 | 5 | 4297 | 4 | 22 |
| v0.6.4 | beta-candidate-14 | unknown | 10 | 7 | 0 | 1 | 0 | 2 |
| v0.6.4 | beta-candidate-7 | unknown | 11 | 8 | 0 | 1 | 0 | 2 |
| v0.6.5 | beta-candidate-13 | unknown | 13 | 8 | 0 | 1 | 0 | 3 |
| v0.6.7 | beta-candidate-12 | unknown | 118 | 13 | 1 | 99 | 1 | 2 |
| v0.6.9 | beta-candidate-11 | unknown | 129 | 15 | 1 | 109 | 1 | 1 |
| v0.6.9 | beta-candidate-4 | unknown | 292 | 16 | 1 | 271 | 1 | 1 |
| v0.7.14 | beta-candidate-8 | unknown | 31 | 22 | 4 | 0 | 2 | 2 |
| v0.7.15 | beta-candidate-0 | unknown | 44 | 30 | 5 | 0 | 3 | 5 |
| v0.7.1 | beta-candidate-2 | unknown | 4380 | 46 | 5 | 4297 | 7 | 22 |
| v0.7.3 | beta-candidate-1 | unknown | 224 | 74 | 15 | 99 | 9 | 24 |
| v0.7.7 | beta-candidate-9 | unknown | 143 | 77 | 18 | 0 | 4 | 41 |

### Total Statistics
- **Total Files Extracted**: 9870 files
- **Python Files**: 430
- **JSON Config Files**: 56
- **YAML/YML Config Files**: 9192
- **Database Files**: 32
- **Documentation Files**: 138

### Key Evolution Observations

#### Early Versions (May-June 2025)
- **v0.5.30 - v0.5.31 - v0.6.1**: Basic engine structure with fundamental components
- **Configuration**: YAML-based configurations
- **Architecture**: Simple evaluation-centric engines

#### Middle Versions (June-July 2025)
- **v0.6.27 - v0.6.30 - v0.7.1**: Transition to v7p3r modular architecture
- **Configuration**: Mixed YAML/JSON configurations
- **Architecture**: Modular component-based design

#### Recent Versions (July 2025)
- **v0.7.14 - v0.7.15 - v0.7.7**: Modern architecture with comprehensive module system
- **Configuration**: JSON-based configurations with specialized configs
- **Architecture**: Advanced modular v7p3r framework

### Next Steps for Analysis
1. **Individual Build Testing**: Each build directory contains all necessary files for standalone operation
2. **Feature Comparison**: Compare evaluation functions, search algorithms, and scoring systems across versions
3. **Performance Analysis**: Use database files to analyze engine performance evolution
4. **Code Quality Assessment**: Review coding style evolution and identify best practices
5. **Release Candidate Selection**: Identify most promising builds for further development

### Build Directory Structure
All builds are located in: `builds_complete/`
Each build contains:
- Complete source code for that version
- Configuration files
- Database files (where applicable)
- Documentation (where available)
- All dependencies needed for standalone operation

The builds are ready for:
- Compilation into executable files
- Testing with chess GUI applications (Arena, etc.)
- Performance benchmarking
- Feature extraction and combination
